# Unit 01 — Problem Set

**100 points.** Write your answers in this file, in your own words, before
looking at `solutions/`. A sentence or two per question is plenty — these test
understanding, not essay-writing.

Scoring guide: **90+** you could teach this unit. **70–89** solid, move on.
**Below 70** reread the lecture notes before Unit 02 — everything after this
builds directly on it.

Terms are in [`GLOSSARY.md`](../../GLOSSARY.md).

---

## Part A — Why any of this is necessary (25 pts)

**A1. (10 pts)** Why can't we train the model the ordinary way, by showing it the
right answer? What do we have instead?

> **Ordinary training (supervised learning)** works by showing the model the ideal
> output and saying "produce this exact text." That requires a dataset of correct
> answers written out — demos, labeled solutions, ideal reasoning traces.
>
> For many things we care about (good reasoning, writing working code, solving
> hard math), we **don't have** those ideal traces. We can't point at one perfect
> chain-of-thought and say "copy this."
>
> **What we have instead is a verifier:** a function that looks at whatever the
> model produced and returns a score (right/wrong, unit tests pass, answer equals
> 391, etc.). So the setup is:
>
> - We **cannot** say "produce this exact text."
> - We **can** say "produce something, and I'll tell you if it was good."
>
> That is why we need policy-gradient / RL-style training: turn scores into
> learning without ever needing the ideal output.

**A2. (10 pts)** Somewhere between the model's internal numbers and the final
score, the math "breaks" and you can't compute a gradient through it. Name the
exact step where it breaks, and explain why that step is different in kind from
the ones around it.

> Trace the chain:
>
> ```
> parameters → logits → probabilities → SAMPLE A TOKEN → text → score
>              (smooth)    (smooth)         ← breaks here
> ```
>
> It breaks at **sampling** — the discrete choice of one token from the
> probability distribution.
>
> Everything before that is smooth calculus: changing a logit a tiny bit changes
> probabilities a tiny bit (softmax is differentiable). The score is a function of
> the finished text. But sampling is a hard jump: you either picked token 4 or
> you didn't. There is no gentle slope through "I almost chose token 3." You
> can't backprop through a coin flip / dice roll.
>
> So ordinary gradient descent (differentiate the score w.r.t. parameters) is
> blocked. We need a different method that *estimates* the right update by trying
> things and scoring them — policy gradients.

**A3. (5 pts)** State the training method in one sentence. If your sentence has
more than one clause you probably haven't simplified it enough.

> Sample an output, score it, and make better-scoring outputs more likely.

> (Expanded version of the same idea: if the score was good, increase the
> probability of that output; if bad, decrease it; repeat. That is policy
> gradient / REINFORCE. Name optional — the behavior is what matters.)

---

## Part B — Mechanics (35 pts)

**B1. (10 pts)** The nudge direction is `onehot(sampled token) − probs`. Read it
out loud as a plain-English instruction — what does it tell you to do?

> **Raise the logit of the token you just sampled, and lower everyone else's.**
>
> Concrete example: probs = `[0.2, 0.2, 0.2, 0.2, 0.2]`, sampled token 3:
>
> ```
> onehot(3) = [0, 0, 0, 1, 0]
> direction = [−0.2, −0.2, −0.2, +0.8, −0.2]
> ```
>
> Token 3 gets a positive nudge; tokens 0,1,2,4 get negative nudges. That is
> exactly "make this output more likely" in logit-space. Scaling by the reward
> (or advantage) then decides *how hard* to push.

**B2. (10 pts)** That vector always sums to zero. Why, and what does that tell
you about what an update can and can't do?

> **Why it sums to zero:** a one-hot vector sums to 1 (one entry is 1). A
> probability vector sums to 1. Difference: `1 − 1 = 0`.
>
> **What that means:** every update is a **redistribution** of probability.
> Probability you add to the sampled token is taken from the others. You cannot
> create probability out of nowhere, and you cannot make all tokens more likely
> at once.
>
> This is also why subtracting a constant baseline is free: the extra term is
> `baseline × (direction)`, and directions that sum to zero cancel out when
> averaged over samples. Softmax also ignores uniform shifts of all logits — same
> intuition.

**B3. (5 pts)** Why do we subtract the maximum logit before exponentiating in
`softmax`? What breaks if we don't?

> Softmax is **shift-invariant**: `softmax(x) = softmax(x − c)` for any constant
> `c`. Subtracting `max(logits)` doesn't change the probabilities, but it keeps
> the largest exponent at `exp(0) = 1`, so nothing blows up.
>
> **If you don't:** once training drives logits large (e.g. 1000, 1001, 1002),
> `exp(1000)` overflows to infinity, then you get `inf/inf → nan`, and training
> dies. Numerical stability trick, not a math change.

**B4. (10 pts)** Exercise 2's reward table is all non-negative. Describe what
goes wrong because of that, and give the one-line fix.

> **What goes wrong:** the REINFORCE update is
> `θ ← θ + lr × reward × direction`. If every reward is ≥ 0, every sampled token
> with a positive score gets pushed **up** — including mediocre ones like token 1
> (reward 0.2). We are reinforcing bad answers, just less hard than good ones.
> Learning still happens eventually (good tokens get bigger pushes), but it's
> wasteful and noisy.
>
> **One-line fix:** use advantage instead of raw reward:
> `θ ← θ + lr × (reward − baseline) × direction`
> so below-average outputs get a **negative** multiplier and are pushed down.

---

## Part C — The baseline (20 pts)

**C1. (10 pts)** You shifted every reward up by +10. The best possible model
didn't change at all, but learning got 359× noisier. Explain what happened.

> Adding +10 to every reward does **not** change which action is best: token 4 is
> still best by the same margin. The optimal policy is identical. So in terms of
> the *task*, nothing meaningful changed.
>
> But without a baseline, the update multiplies by the **raw** reward. Rewards
> that were ~0–1 become ~10–11, so every gradient sample is scaled by a huge
> number. The estimator's **variance** explodes (~359× in the exercise) even
> though the ranking of actions didn't change.
>
> With a baseline, you subtract the new average (~10.44). Advantages stay the
> same size as before (`10.6 − 10.44 = 0.16`, same as `0.6 − 0.44`). Variance is
> unchanged. The baseline cancels the meaningless constant.
>
> **Takeaway:** without a baseline, your learning algorithm is sensitive to an
> arbitrary additive constant that has nothing to do with the task.

**C2. (10 pts)** Following from C1: you write a scoring function that gives 10
points for "replied in valid JSON" and 1 point for "the answer is correct." What
have you just done to yourself, and what's the fix?

> You've built the +10 shift into almost every sample. Nearly every answer that
> formats correctly collects the big 10; the thing you actually care about
> (correctness, worth 1) is a tiny ripple on top. Without a baseline, training is
> dominated by format noise — same disease as C1.
>
> **Fixes:**
> 1. Use a baseline / advantage so constants cancel.
> 2. Better: keep reward components on **comparable scales** (don't make format
>    worth 10× correctness). When you tune a rubric, you are tuning **ratios**,
>    not absolute levels — especially under GRPO, which is scale-free after
>    normalizing within a group.

---

## Part D — GRPO and its consequences (20 pts)

**D1. (5 pts)** GRPO needs the average score for a question. The older approach
trains an entire second neural network to predict it. What does GRPO do instead?

> Older approach (**PPO**): train a **value network** \(V(s)\) whose job is to
> look at a prompt and predict expected reward ("you'll probably score ~0.44
> here"). That is a whole second model, trained alongside the policy, and early
> on its guesses are bad.
>
> **GRPO instead:** don't predict the average — **measure it**. Sample the same
> prompt \(G\) times, score all \(G\) answers, take their mean as the baseline.
> Each answer's advantage is relative to that group:
> \(A_i = (r_i - \bar{r}) / (\sigma + \varepsilon)\).
> No second network, no prediction — you just looked.

**D2. (5 pts)** Why is that approach available when training a language model,
but *not* available when training, say, a robot to walk?

> An LLM prompt is a **resettable, independent trial**. You can ask the same
> question 16 times, get 16 independent rollouts, and average. Each completion
> starts from the same prompt with no leftover physical state.
>
> A walking robot lives in a **sequential physical environment**. The "state"
> evolves continuously (balance, position, momentum). You cannot cheaply fork
> the exact same starting state 16 times in parallel the way you re-query a
> prompt. You also care about long-horizon credit assignment across timesteps,
> which is a different problem than "score this whole answer."
>
> So group-relative baselines are a natural fit for LLM post-training and a poor
> fit for classical robotics RL.

**D3. (5 pts)** GRPO with a group size of 1 didn't learn slowly — it didn't learn
at all, and the model came out exactly as it went in. Why? One sentence.

> With group size 1 the group mean equals that single reward, so every advantage
> is \((r - r) = 0\) and every gradient is exactly zero — GRPO becomes a no-op,
> not "noisy GRPO."

**D4. (5 pts)** A question where all 8 answers score 1.0, and a question where
all 8 score 0.0. What do these have in common, and what does it imply about how
to choose training questions?

> **In common:** zero reward variance inside the group. After subtracting the
> mean, every advantage is 0. Both prompts produce **no learning signal** —
> "always right" and "always wrong" are the same failure mode under GRPO.
>
> **Implication for choosing questions:** train on prompts where the model
> succeeds *sometimes* (spread between 0 and 1 across the group). Too-easy and
> too-hard items waste generation compute. Real pipelines filter for reward
> variance; curriculum design is largely keeping prompts in that band.

---

## Reflection (ungraded, but do it)

What surprised, annoyed, or confused you in this unit? This is usually where the
real misunderstanding is hiding, and it's the most useful thing to carry into
Unit 02.

> The biggest click: **a verifier is not RL** — the verifier only returns a
> score; RL is the update that turns scores into better parameters. And under
> GRPO, absolute scores barely matter: only **relative ranking inside a group**
> does. That means Unit 02's reward design is the real skill — a clever algorithm
> cannot save a verifier that always returns the same number, or a format bonus
> that drowns out correctness.
>
> Also surprising: G=1 isn't "weak GRPO," it's mathematically incapable of
> learning. Sampling breaks the gradient chain; baselines are free; GRPO is just
> "measure the baseline by resampling the prompt." Everything after this unit is
> mostly plumbing and reward design on top of that idea.
