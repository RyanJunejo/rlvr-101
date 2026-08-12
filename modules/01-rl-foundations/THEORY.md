# Unit 01 — Theory

> **Read this after the labs.** The lecture notes told you what the update does.
> This tells you why it's the right update, and what it's an instance of.
>
> Assumes: algebra, and that you've run all four exercises. Every symbol is
> named in words the first time it appears.

---

## Notation, once

| symbol | says out loud | in the labs |
|---|---|---|
| `θ` | theta — the model's parameters | the five logits |
| `π(a)` | pi of a — the probability the policy assigns to action `a` | `softmax(theta)[a]` |
| `r(a)` | the reward for taking action `a` | `REWARDS[a]` |
| `E[...]` | the average of, over many samples | what a big loop converges to |
| `∇` | nabla — the gradient: the direction that increases what follows | the nudge direction |

`∇_θ f` means "the direction to move `θ` to make `f` go up fastest." When the
subscript is obvious I'll drop it.

---

## 1. What we are actually maximizing

The thing you want is the average reward the policy earns:

```
J(θ) = E     [ r(a) ]  =  Σ  π(a) · r(a)
        a ~ π              a
```

Read the right-hand side as: for every action, multiply its probability by its
reward, and add them up. `J` is a function of `θ`, because `θ` determines `π`.

Training means climbing `J`. To climb it we want `∇J` — and this is where the
lecture said "the chain breaks."

Here is what actually breaks. `J` is perfectly differentiable as written: it's a
weighted sum of rewards, and the weights `π(a)` are smooth in `θ`. The problem
is that we cannot *evaluate* that sum. It ranges over every action — every
possible completion — and there are more of those than atoms. We can only sample
from it.

So the real requirement is sharper than "make it differentiable." We need to
write `∇J` **as an expectation we can sample**.

---

## 2. The log-derivative trick

Start differentiating and see where you get stuck:

```
∇J = ∇ Σ π(a) r(a)
       a

   = Σ r(a) ∇π(a)                    the rewards don't depend on θ
     a
```

This is a sum weighted by `∇π(a)`, and we cannot sample it — sampling gives us
actions distributed according to `π`, not `∇π`.

The fix is one identity. For any positive function,

```
∇ log π(a) = ∇π(a) / π(a)        (the derivative of log)
```

Rearranged:

```
∇π(a) = π(a) · ∇ log π(a)
```

Substitute that in:

```
∇J = Σ r(a) · π(a) · ∇ log π(a)
     a

   = E     [ r(a) · ∇ log π(a) ]
      a ~ π
```

**That last line is the whole method.** The sum turned into an expectation *over
π* — which is exactly the distribution we can sample from. Draw an action,
compute `r(a) · ∇ log π(a)`, and you hold one unbiased sample of a gradient you
could never have computed directly.

This is called the **score function estimator**, or REINFORCE. It's the
`reinforce_step` you wrote in exercise 2.

Two things worth noticing:

**Nothing here is specific to language models.** The derivation needs only that
you can sample from `π` and evaluate `r` on the result. It works for robots and
board games identically.

**The reward never gets differentiated.** `r(a)` appears as a plain multiplier.
That's why your reward function can be an `if` statement, a regex, a compiler, or
a unit-test suite — nothing about it needs to be smooth. This is the property the
whole course rests on, and it falls out of one line of algebra.

### Where the concrete formula came from

For a softmax policy, `π(a) = softmax(θ)[a]`, so

```
log π(a) = θ_a − log Σ exp(θ_j)
                     j
```

Differentiate with respect to one component `θ_i`. The first term contributes 1
when `i = a` and 0 otherwise. The second contributes `exp(θ_i)/Σexp(θ_j)`, which
is just `π_i`. So:

```
∂ log π(a) / ∂θ_i  =  1[i = a] − π_i
```

Stack those over all `i` and you have `onehot(a) − probs` — the vector you
implemented, and the one `WORKED_EXAMPLES.md` verifies numerically to ten
decimal places.

---

## 3. Why a baseline is free

Now the result the lecture asserted. Subtract any constant `b` from the reward:

```
E[ (r(a) − b) · ∇ log π(a) ]
  = E[ r(a) · ∇ log π(a) ]  −  b · E[ ∇ log π(a) ]
```

The claim is that the second term is exactly zero:

```
E[ ∇ log π(a) ] = Σ π(a) · ∇ log π(a)      definition of expectation
                  a

                = Σ ∇π(a)                  π · ∇log π = ∇π, from section 2
                  a

                = ∇ Σ π(a)                 the gradient of a sum is the sum
                    a                       of the gradients

                = ∇ (1)                    probabilities sum to one
                = 0                        the gradient of a constant
```

So the expectation is unchanged: **the estimator with a baseline is still
unbiased, for any `b` that doesn't depend on the sampled action.**

That caveat is load-bearing. If `b` depended on `a`, it couldn't come out of the
expectation in line 1, and the argument collapses. In GRPO the group mean is
computed from *sibling* samples, not from the one being scored — which is what
keeps it legal.

You can see this result three ways, and they're the same fact:

- **Algebraically:** probabilities sum to a constant, so their gradient is zero.
- **Geometrically:** `onehot(a) − probs` always sums to zero (exercise 2), so
  every update lies in a plane, and `b` only pushes along the direction
  perpendicular to it — the direction softmax ignores.
- **Practically:** adding a constant to every reward doesn't change which action
  is best, so it had better not change where training converges.

### But it does change the variance

Unbiased says the estimator is right *on average*. It says nothing about how far
a single sample lands from that average — and that is what determines how many
samples you need.

Write `g(a) = (r(a) − b) · ∇ log π(a)`. Since the mean of `g` doesn't depend on
`b`, minimizing its variance means minimizing `E[‖g‖²]`, which carries a factor
of `(r(a) − b)²`. Choosing `b` near the typical reward shrinks that factor;
choosing `b = 0` when rewards sit around 10 leaves it enormous.

That's the +10 experiment. Rewards became ~10, `(r − 0)² ≈ 100` instead of
`≈ 0.25`, and measured variance rose **359×** while the optimal policy stayed
identical.

**The variance-minimizing baseline** is not the mean reward. It's a
gradient-magnitude-weighted average:

```
b* = E[ ‖∇log π‖² · r ] / E[ ‖∇log π‖² ]
```

(Greensmith, Bartlett & Baxter, 2004.) Essentially nobody uses it: it needs a
second set of statistics, and the plain mean captures most of the benefit. Worth
knowing mainly so you recognize that `E[r]` is a convenient choice rather than
the optimum.

---

## 4. GRPO is a choice of baseline

With section 3 in hand, GRPO stops looking like an algorithm and starts looking
like a decision about where `b` comes from.

You need an estimate of the expected reward for *this prompt*. Three options:

| approach | where `b` comes from | cost |
|---|---|---|
| no baseline | `b = 0` | free, and high variance |
| PPO | a learned value network `V(s)` | a second model, trained alongside |
| GRPO | the mean of `G` samples of the same prompt | `G−1` extra generations |

PPO's value network is a *prediction*: a model that has seen many prompts
guesses this one's expected reward. It's cheap at sample time and expensive in
machinery, and early in training its guesses are poor — which is a bias you
carry silently.

GRPO's group mean is a *measurement*. It costs generations rather than a model,
and it's available because you can sample a language model as many times as you
like on the identical prompt. That's a luxury the setting hands you: you cannot
re-run a robot from the same physical state, which is why value networks
dominate in control.

### The normalization

GRPO divides by the group's standard deviation as well:

```
A_i = (r_i − mean(r)) / std(r)
```

The subtraction is the baseline, justified above. The division is not — it's
scaling, and it does change the objective. Its purpose is practical: it puts
every prompt's advantages on a comparable scale so one learning rate works
across a heterogeneous batch.

It also has a cost, which is worth being honest about. Dividing by `std` gives
*larger* advantages to prompts whose rewards happen to be tightly clustered,
which up-weights low-variance prompts relative to what the plain policy gradient
would do. Some variants (Dr. GRPO among them) drop the division for exactly this
reason. Both are defensible; the point is that the numerator is theory and the
denominator is a knob.

### Why group size 1 gives exactly zero

With `G = 1`, `mean(r) = r₁`, so `A₁ = 0` identically. Not "small" — zero, for
every sample, forever. Your exercise 4 run confirmed it: 600 steps, policy still
uniform at 0.200.

The general statement: GRPO's signal is entirely the *within-group ranking* of
rewards. Any transformation that preserves that ranking and the group's spread
leaves training unchanged, and any prompt whose samples all score identically
contributes nothing regardless of the score's level.

---

## 5. What Units 03 and 05 add on top

Two pieces of real machinery sit above what you implemented, and both are
corrections rather than new ideas.

**The importance ratio.** The derivation in section 2 assumed samples come from
the *current* policy `π_θ`. In async training they don't — they come from a
slightly older `π_old`, because generation and training run as separate
processes. The correction is standard importance sampling: reweight each sample
by `π_θ(a) / π_old(a)`. Unit 05.

**Clipping.** That ratio can be enormous when the policies disagree, and a single
large ratio can wreck an update. PPO clips it to `[1−ε, 1+ε]` and takes the
pessimistic branch, which turns the objective into a lower bound on the true one
— you never get credit for a step you can't justify. Unit 05 again.

**Per-token credit.** Your labs produced one action per episode. A real
completion is thousands of tokens, and the reward arrives once at the end. The
overwhelmingly common answer is to assign every token in the completion the same
advantage. That is theoretically unsatisfying — plainly not every token deserves
equal blame — and it works far better than it should. Unit 03 is where this
starts to bite.

---

## 6. What to take away

The algorithm you implemented is one point in a small space:

```
update  ∝  (weight on this sample) × (direction that makes it more likely)
              ↑                            ↑
        this is where all              this is fixed by the
        the design lives               policy's parametrization
```

REINFORCE puts the raw reward there. Adding a baseline puts the advantage there.
GRPO computes that baseline by sampling. PPO computes it with a network and
multiplies by a clipped ratio. `prime-rl`'s algorithms layer lets you supply your
own.

All of them are the same expectation from section 2, with different answers to
"weighted by what?"

## Sources

- **Williams (1992)**, *Simple statistical gradient-following algorithms* — the
  original REINFORCE paper.
- **Sutton & Barto**, Chapter 13 — the policy gradient theorem, stated for the
  full MDP case rather than our single-step one.
- **Greensmith, Bartlett & Baxter (2004)** — variance bounds and the optimal
  baseline.
- **Schulman et al. (2017)**, arXiv:1707.06347 — PPO; §3 has the clipped
  objective.
- **Shao et al. (2024)**, arXiv:2402.03300 §4.1 — GRPO.
