# Unit 01 — Theory

> **Read this after the labs.** The lecture notes told you what the update does.
> This tells you why it's the right update, and what it's an instance of.
>
> Assumes: algebra, and that you've run all four exercises. Every symbol is
> named in words the first time it appears.
>
> *Math renders on GitHub and in any editor with markdown math preview.*

---

## Notation, once

| symbol | says out loud | in the labs |
|---|---|---|
| $\theta$ | theta — the model's parameters | the five logits |
| $\pi(a)$ | pi of $a$ — the probability the policy gives action $a$ | `softmax(theta)[a]` |
| $r(a)$ | the reward for action $a$ | `REWARDS[a]` |
| $\mathbb{E}[\cdot]$ | the average of, over many samples | what a big loop converges to |
| $\nabla$ | nabla — the direction that increases what follows | the nudge direction |

$\nabla_\theta f$ means "the direction to move $\theta$ to make $f$ go up
fastest." Where the subscript is obvious, it's dropped.

---

## 1. What we are actually maximizing

The thing you want is the average reward the policy earns:

$$
J(\theta) \;=\; \mathbb{E}_{a \sim \pi}\big[\, r(a) \,\big] \;=\; \sum_a \pi(a)\, r(a)
$$

Read the right-hand side as: for every action, multiply its probability by its
reward, and add them up. $J$ depends on $\theta$ because $\theta$ determines
$\pi$.

Training means climbing $J$. To climb it we want $\nabla J$ — and this is where
the lecture said "the chain breaks."

Here is what actually breaks. $J$ is perfectly differentiable as written: it's a
weighted sum of rewards, and the weights $\pi(a)$ are smooth in $\theta$. The
problem is that we cannot **evaluate** that sum. It ranges over every action —
every possible completion — and there are more of those than atoms. We can only
sample from it.

So the real requirement is sharper than "make it differentiable." We need
$\nabla J$ **written as an expectation we can sample.**

---

## 2. The log-derivative trick

Start differentiating and see where you get stuck:

$$
\nabla J \;=\; \nabla \sum_a \pi(a)\, r(a) \;=\; \sum_a r(a)\, \nabla \pi(a)
$$

The rewards don't depend on $\theta$, so they pass through untouched. But this is
a sum weighted by $\nabla\pi(a)$, and we cannot sample it — sampling gives us
actions distributed according to $\pi$, not $\nabla\pi$.

The fix is one identity. For any positive function,

$$
\nabla \log \pi(a) \;=\; \frac{\nabla \pi(a)}{\pi(a)}
\qquad\Longrightarrow\qquad
\nabla \pi(a) \;=\; \pi(a)\, \nabla \log \pi(a)
$$

Substitute that in:

$$
\nabla J \;=\; \sum_a r(a)\, \pi(a)\, \nabla \log \pi(a)
\;=\; \mathbb{E}_{a \sim \pi}\big[\, r(a)\, \nabla \log \pi(a) \,\big]
$$

**That last line is the whole method.** The sum turned into an expectation *over
$\pi$* — exactly the distribution we can sample from. Draw an action, compute
$r(a)\,\nabla \log \pi(a)$, and you hold one unbiased sample of a gradient you
could never have computed directly.

This is the **score function estimator**, or REINFORCE. It's the
`reinforce_step` you wrote in exercise 2.

Two things worth noticing:

**Nothing here is specific to language models.** The derivation needs only that
you can sample from $\pi$ and evaluate $r$ on the result. It works for robots and
board games identically.

**The reward is never differentiated.** $r(a)$ appears as a plain multiplier.
That's why your reward function can be an `if` statement, a regex, a compiler, or
a unit-test suite — nothing about it needs to be smooth. This is the property the
whole course rests on, and it falls out of one line of algebra.

### Where the concrete formula came from

For a softmax policy, $\pi(a) = \mathrm{softmax}(\theta)_a$, so

$$
\log \pi(a) \;=\; \theta_a \;-\; \log \sum_j e^{\theta_j}
$$

Differentiate with respect to one component $\theta_i$. The first term
contributes $1$ when $i = a$ and $0$ otherwise. The second contributes
$e^{\theta_i} / \sum_j e^{\theta_j}$, which is just $\pi_i$:

$$
\frac{\partial \log \pi(a)}{\partial \theta_i} \;=\; \mathbb{1}[i = a] \;-\; \pi_i
$$

Stack that over all $i$ and you have $\mathrm{onehot}(a) - \mathbf{p}$ — the
vector you implemented, and the one `WORKED_EXAMPLES.md` verifies numerically to
ten decimal places.

---

## 3. Why a baseline is free

Now the result the lecture asserted. Subtract any constant $b$ from the reward:

$$
\mathbb{E}\big[(r(a) - b)\, \nabla \log \pi(a)\big]
\;=\;
\mathbb{E}\big[r(a)\, \nabla \log \pi(a)\big]
\;-\;
b\, \mathbb{E}\big[\nabla \log \pi(a)\big]
$$

The claim is that the second term is exactly zero:

$$
\begin{aligned}
\mathbb{E}\big[\nabla \log \pi(a)\big]
&= \sum_a \pi(a)\, \nabla \log \pi(a) && \text{definition of expectation} \\
&= \sum_a \nabla \pi(a) && \pi \nabla \log \pi = \nabla \pi \text{, from §2} \\
&= \nabla \sum_a \pi(a) && \text{gradient of a sum is the sum of gradients} \\
&= \nabla (1) && \text{probabilities sum to one} \\
&= 0 && \text{the gradient of a constant}
\end{aligned}
$$

So the expectation is unchanged: **the estimator with a baseline is still
unbiased, for any $b$ that doesn't depend on the sampled action.**

That caveat is load-bearing. If $b$ depended on $a$, it couldn't come out of the
expectation in the first line, and the argument collapses. In GRPO the group mean
is computed from *sibling* samples, not from the one being scored — which is what
keeps it legal.

You can see this result three ways, and they're the same fact:

- **Algebraically:** probabilities sum to a constant, so their gradient is zero.
- **Geometrically:** $\mathrm{onehot}(a) - \mathbf{p}$ always sums to zero
  (exercise 2), so every update lies in a plane, and $b$ only pushes along the
  direction perpendicular to it — the direction softmax ignores.
- **Practically:** adding a constant to every reward doesn't change which action
  is best, so it had better not change where training converges.

### But it does change the variance

Unbiased says the estimator is right *on average*. It says nothing about how far
a single sample lands from that average — and that is what determines how many
samples you need.

Write $g(a) = (r(a) - b)\,\nabla \log \pi(a)$. Since the mean of $g$ doesn't
depend on $b$, minimizing its variance means minimizing
$\mathbb{E}\big[\lVert g \rVert^2\big]$, which carries a factor of
$(r(a) - b)^2$. Choosing $b$ near the typical reward shrinks that factor;
choosing $b = 0$ when rewards sit around $10$ leaves it enormous.

That's the $+10$ experiment. Rewards became $\approx 10$, so
$(r - 0)^2 \approx 100$ instead of $\approx 0.25$, and measured variance rose
**359×** while the optimal policy stayed identical.

**The variance-minimizing baseline** is not the mean reward. It's a
gradient-magnitude-weighted average:

$$
b^\star \;=\; \frac{\mathbb{E}\big[\lVert \nabla \log \pi \rVert^2\, r\big]}
{\mathbb{E}\big[\lVert \nabla \log \pi \rVert^2\big]}
$$

(Greensmith, Bartlett & Baxter, 2004.) Essentially nobody uses it: it needs a
second set of statistics, and the plain mean captures most of the benefit. Worth
knowing mainly so you recognize that $\mathbb{E}[r]$ is a convenient choice
rather than the optimum.

---

## 4. GRPO is a choice of baseline

With §3 in hand, GRPO stops looking like an algorithm and starts looking like a
decision about where $b$ comes from.

You need an estimate of the expected reward for *this prompt*. Three options:

| approach | where $b$ comes from | cost |
|---|---|---|
| no baseline | $b = 0$ | free, and high variance |
| PPO | a learned value network $V(s)$ | a second model, trained alongside |
| GRPO | the mean of $G$ samples of the same prompt | $G-1$ extra generations |

PPO's value network is a **prediction**: a model that has seen many prompts
guesses this one's expected reward. Cheap at sample time, expensive in machinery,
and early in training its guesses are poor — a bias you carry silently.

GRPO's group mean is a **measurement**. It costs generations rather than a model,
and it's available because you can sample a language model as many times as you
like on the identical prompt. That's a luxury the setting hands you: you cannot
re-run a robot from the same physical state, which is why value networks dominate
in control.

### The normalization

GRPO divides by the group's standard deviation as well:

$$
A_i \;=\; \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)}
$$

The subtraction is the baseline, justified above. The division is not — it's
scaling, and it does change the objective. Its purpose is practical: it puts every
prompt's advantages on a comparable scale so one learning rate works across a
heterogeneous batch.

It also has a cost worth being honest about. Dividing by $\operatorname{std}$
gives *larger* advantages to prompts whose rewards happen to be tightly
clustered, which up-weights low-variance prompts relative to what the plain
policy gradient would do. Some variants (Dr. GRPO among them) drop the division
for exactly this reason. Both are defensible; the point is that the numerator is
theory and the denominator is a knob.

### Why group size 1 gives exactly zero

With $G = 1$, $\operatorname{mean}(r) = r_1$, so $A_1 = 0$ identically. Not
"small" — zero, for every sample, forever. Your exercise 4 run confirmed it: 600
steps, policy still uniform at $0.200$.

The general statement: GRPO's signal is entirely the **within-group ranking** of
rewards. Any transformation preserving that ranking and the group's spread leaves
training unchanged, and any prompt whose samples all score identically
contributes nothing regardless of the score's level.

---

## 5. What Units 03 and 05 add on top

Two pieces of real machinery sit above what you implemented, and both are
corrections rather than new ideas.

**The importance ratio.** The derivation in §2 assumed samples come from the
*current* policy $\pi_\theta$. In async training they don't — they come from a
slightly older $\pi_{\text{old}}$, because generation and training run as
separate processes. The correction is standard importance sampling: reweight each
sample by $\pi_\theta(a) / \pi_{\text{old}}(a)$. Unit 05.

**Clipping.** That ratio can be enormous when the policies disagree, and a single
large ratio can wreck an update. PPO clips it to $[1-\epsilon,\, 1+\epsilon]$ and
takes the pessimistic branch, turning the objective into a lower bound on the
true one — you never get credit for a step you can't justify. Unit 05 again.

**Per-token credit.** Your labs produced one action per episode. A real completion
is thousands of tokens, and the reward arrives once at the end. The overwhelmingly
common answer is to assign every token in the completion the same advantage. That
is theoretically unsatisfying — plainly not every token deserves equal blame — and
it works far better than it should. Unit 03 is where this starts to bite.

---

## 6. What to take away

The algorithm you implemented is one point in a small space:

$$
\text{update} \;\propto\; \underbrace{w(a)}_{\substack{\text{where all the} \\ \text{design lives}}}
\times
\underbrace{\nabla \log \pi(a)}_{\substack{\text{fixed by the policy's} \\ \text{parametrization}}}
$$

REINFORCE puts the raw reward in $w$. Adding a baseline puts the advantage there.
GRPO computes that baseline by sampling. PPO computes it with a network and
multiplies by a clipped ratio. `prime-rl`'s algorithms layer lets you supply your
own.

All of them are the same expectation from §2, with different answers to
"weighted by what?"

## Sources

- **Williams (1992)**, *Simple statistical gradient-following algorithms* — the
  original REINFORCE paper.
- **Sutton & Barto**, Chapter 13 — the policy gradient theorem, stated for the
  full MDP case rather than our single-step one.
- **Greensmith, Bartlett & Baxter (2004)** — variance bounds and the optimal
  baseline.
- **Schulman et al. (2017)**, arXiv:1707.06347 §3 — PPO's clipped objective.
- **Shao et al. (2024)**, arXiv:2402.03300 §4.1 — GRPO.
