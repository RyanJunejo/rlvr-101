# Unit 05 — Theory

> **Read this after the labs.** The lecture said three processes run at once and
> that this makes training asynchronous. That asynchrony breaks an assumption in
> Unit 01's derivation. This is the repair, which is where PPO's clipping and
> the KL penalty come from.

---

## 1. The assumption that breaks

Unit 01 §2 derived:

```
∇J = E     [ r(a) · ∇ log π_θ(a) ]
      a ~ π_θ
```

Read the subscript carefully. The expectation is over actions drawn from **the
current policy** `π_θ` — the same parameters you're about to update. That's what
makes the estimator unbiased.

Now look at what the system actually does. Generation and training are separate
processes on separate GPUs. While the trainer computes an update, the inference
server is already generating the next batch — with the weights it has, which are
the ones from before the update.

So the samples you train on came from `π_old`, not `π_θ`. The estimator you're
computing is:

```
E       [ r(a) · ∇ log π_θ(a) ]      ← wrong distribution
 a ~ π_old
```

This is **off-policy**, and it is biased. How badly depends on how far the
policies have drifted, which depends on how many updates happened while that
batch was in flight.

You could avoid it by making the trainer wait for fresh samples every step. That
is *on-policy* and it is correct — and it leaves an expensive GPU idle during
every generation phase, which at 2,048 completions per step is most of the wall
clock. The asynchrony is deliberate; the bias is the price.

## 2. Importance sampling repairs it

The standard correction. For any distributions `p` and `q` where `q` is nonzero
wherever `p` is:

```
E   [ f(x) ]  =  Σ p(x) f(x)  =  Σ q(x) · (p(x)/q(x)) · f(x)  =  E   [ (p/q) · f ]
 p               x               x                                q
```

Multiply by the ratio of how likely the sample was under the distribution you
*want* versus the one you *have*. Applied here:

```
∇J = E       [ ρ(a) · r(a) · ∇ log π_θ(a) ],       ρ(a) = π_θ(a) / π_old(a)
      a ~ π_old
```

`ρ` is the **importance ratio**. A sample the current policy likes more than the
old one had counts for more; one it now disprefers counts for less. With the
ratio, the estimator is unbiased again.

For a language model `ρ` is computable: both policies assign a probability to
the sampled tokens, and the trainer has the logprobs from generation stored in
the trace.

### And introduces a new problem

Importance sampling is unbiased and its variance can be unbounded.

`ρ` is a ratio of probabilities. If the policy has moved such that a sampled
completion is now 100× more likely, `ρ = 100`, and that single sample dominates
the entire batch. The bias is gone; the variance ate it.

Worse, the variance grows with drift — which grows with how asynchronous you let
the system be. So there's a real tension: more asynchrony means better hardware
utilization and a noisier gradient.

## 3. Clipping

PPO's answer is to refuse to trust large ratios. The clipped surrogate objective:

```
L(θ) = E[ min( ρ·A ,  clip(ρ, 1−ε, 1+ε)·A ) ]
```

where `A` is the advantage and `ε` is small, typically 0.2.

Read the `min` carefully, because it isn't symmetric and that's the point.

**When `A > 0`** (this action was better than its group's average, so we want
more of it): the objective is capped at `(1+ε)·A`. Once the policy has raised
this action's probability by 20%, further increase earns nothing. You get no
reward for a large step.

**When `A < 0`** (worse than average, we want less of it): `min` selects the more
negative branch, so the penalty is *not* capped in the direction of decreasing
the probability. You are always free to move away from a bad action.

The asymmetry gives the objective its character: it's a **pessimistic lower
bound** on the true one. Every step is justified by the worst of the two
readings, so you never take a large step on the strength of a large ratio you
have no reason to trust.

Note what clipping is not. It doesn't correct the bias — a clipped estimator is
biased on purpose. It trades a small, bounded bias for a large reduction in
variance, which is the right trade when the alternative is a single sample
detonating your update.

## 4. The KL term

Clipping bounds each step. It does not bound the *cumulative* drift over many
steps, and a policy can walk a long way in small legal increments.

The usual guard is a penalty on the KL divergence from a frozen reference model —
typically the pre-RL checkpoint:

```
objective = E[ clipped surrogate ]  −  β · KL(π_θ ‖ π_ref)
```

**KL divergence** measures how far one distribution has moved from another:

```
KL(p ‖ q) = Σ p(x) · log( p(x) / q(x) )
```

It's zero when they're identical, positive otherwise, and asymmetric — `KL(p‖q)`
is not `KL(q‖p)`, which matters for which failure it penalizes.

Two reasons this term exists:

**Capability retention.** RL post-training optimizes a narrow objective. Left
alone, a model will happily trade away everything the objective doesn't measure
— fluency, general knowledge, calibration — for a few points on the thing it
does. The KL term prices that trade.

**Reward hacking, again.** Unit 02's theory said an optimizer searches for
regions where the proxy exceeds the truth. Those regions are usually *far from
the reference model's distribution* — degenerate text, repeated tokens,
formatting exploits. A KL penalty makes distance itself costly, so the exploit
has to be worth the trip.

That's a useful property: the KL term is a blunt defense against a
failure you can't specify, bought by assuming the exploits live far away. It's
an assumption, not a theorem, and it's the reason `β` is a number people tune
rather than derive.

## 5. What the config's numbers mean now

Reading `reverse-text/rl.toml` with sections 1–4 in hand:

**`group_size = 16`** is `G` from Unit 01 — the baseline is measured from 16
samples of the same prompt. Bigger `G` means a better baseline estimate at
`1/√G`, and linearly more generation.

**`batch_size = 128`** with `group_size = 16` is 2,048 completions per step. That
volume is why generation gets its own GPU, and why the system is asynchronous at
all — which is what makes sections 1–3 necessary.

**`num_train_gpus = 1`, `num_infer_gpus = 1`** is the split that creates the
staleness. One GPU generating against weights that the other is changing.

**`lr = 3e-6`** is small for the same reason the KL term exists. You are nudging a
trained model, and the objective you're nudging it toward is a proxy that gets
less trustworthy the further you go.

## 6. What FSDP does and doesn't change

Fully Sharded Data Parallel splits parameters, gradients and optimizer state
across GPUs so a model that doesn't fit on one can be trained on several.

It changes nothing above. FSDP is a memory and communication strategy; the
gradient it computes is the one a single enormous GPU would have computed. When
`prime-rl` scales to trillion-parameter models, sections 1–4 are unchanged and
only the plumbing differs.

This is worth internalizing because it tells you which parts of the stack are
ideas and which are engineering. The importance ratio is an idea — remove it and
the math is wrong. FSDP is engineering — remove it and you need a bigger GPU.

## Sources

- **Schulman et al. (2017)**, arXiv:1707.06347 — PPO. §3 is the clipped
  objective; the paper is unusually readable.
- **Schulman et al. (2015)**, arXiv:1502.05477 — TRPO, where the trust-region
  idea is derived properly before PPO simplified it.
- **Ouyang et al. (2022)**, arXiv:2203.02155 — InstructGPT; the KL-to-reference
  term in its practiced form.
- **Rajbhandari et al. (2019)**, arXiv:1910.02054 — ZeRO, the sharding scheme
  FSDP implements.
