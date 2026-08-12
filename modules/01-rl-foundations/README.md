# Unit 01 — Training on a score

### Lecture notes

> **Time:** 3–4 hours · **Prerequisites:** Unit 00 · **Needs:** nothing (no GPU,
> no API key)
>
> **By the end of this unit you will be able to:**
> 1. Explain precisely where ordinary gradient-based training breaks down when
>    all you have is a score.
> 2. Implement policy gradient training from scratch, and state what the update
>    does in plain English.
> 3. Explain why subtracting a baseline reduces noise *at no cost*, and measure
>    the effect yourself.
> 4. Implement GRPO, and explain why it needs no value network.
> 5. Predict which training questions will produce no learning at all, and say
>    why.
>
> **Deliverables:** 4 labs (`exercise_*.py`), autograder green, problem set in
> `NOTES.md`.

New terms are defined as they appear. Anything you want to look up again is in
[`GLOSSARY.md`](../../GLOSSARY.md).

---

## 1. Start with the problem

Say you have a model that answers math questions, and it's mediocre. You'd like
it to be better. How?

The normal way to train a model is to show it the right answer and say "produce
this." That's ordinary supervised learning, and it works great — when you have
the right answers written out.

But now think about a harder case. You want the model to *reason well*. You don't
have a file of ideal reasoning traces. What you do have is a way to **check** the
final answer: run the code and see if it passes, or compare the number to 391.

So your situation is:

- You **cannot** say "produce this exact text."
- You **can** say "produce something, and I'll tell you if it was good."

That's the entire setup. Everything in this unit follows from it.

## 2. Why this is hard

Normally you train by computing a **gradient** — the direction to nudge each of
the model's parameters to make your score go up. You need an unbroken chain of
smooth math from "parameter" to "score" in order to compute it.

Trace that chain here:

```
parameters  →  logits  →  probabilities  →  SAMPLE A TOKEN  →  text  →  score
              (smooth)      (smooth)          ← ??? →
```

Everything is smooth until you *sample*. Sampling is picking one token out of
many — a discrete jump. There's no gentle slope through "I chose token 4 instead
of token 3." The chain breaks.

> **Gradient:** the direction you'd nudge a set of numbers to make some output go
> up fastest. It's itself a list of numbers, one per input you could nudge.

So you can't just differentiate your way to a better model. You need something
else.

## 3. The trick, in one idea

You can't compute the gradient. But you can **estimate** it by trying things.

1. Have the model produce some output.
2. Score it.
3. If the score was good, adjust the model to make that output **more likely**.
4. If it was bad, make it **less likely**.
5. Repeat a few thousand times.

That's it. That's the whole method. It has a name — **policy gradient** — but the
name is less important than noticing what it actually does: *it turns "I can
score things" into "I can improve things,"* without ever needing to know what the
ideal output was.

Karpathy has a line about this that's worth internalizing: it's just supervised
learning where the training examples are your own samples, weighted by how well
each one did.

## 4. A model small enough to see

Real models are too big to watch. So for this whole unit we use the smallest
model that still has every essential feature:

**A model with a vocabulary of 5 tokens and nothing to read.**

It has 5 numbers inside it, one per token. Those are the **logits**. To produce
output, it converts logits to probabilities and rolls weighted dice.

```
logits (what's inside):   [0.0, 0.0, 0.0, 0.0, 0.0]
probabilities:            [0.2, 0.2, 0.2, 0.2, 0.2]     ← all equally likely
```

Our scoring function is a lookup table. Token 4 is the "right answer":

```
token:    0     1     2     3     4
reward:  0.0   0.2   0.4   0.6   1.0
```

The goal: get the model to output token 4. We're not going to tell it "output
token 4." We're only ever going to let it try, and tell it the score.

This is a real language model doing real RL. It just has a vocabulary of 5
instead of 100,000, and produces 1 token instead of 2,000.

## 5. A single update, worked out by hand

Let's do one step with actual arithmetic, so nothing is mysterious.

**Step 1.** Convert logits to probabilities with `softmax` — exponentiate each
one, then divide by the total so they sum to 1. All our logits are 0, and
`exp(0) = 1`, so:

```
probabilities = [1,1,1,1,1] / 5 = [0.2, 0.2, 0.2, 0.2, 0.2]
```

**Step 2.** Sample. Roll the dice; say we get **token 3**.

**Step 3.** Score it. `reward = 0.6`.

**Step 4.** We want token 3 to be *more likely* next time. Which direction do we
nudge the five logits?

The answer is short:

```
direction = onehot(sampled token) − probabilities
          = [0,0,0,1,0] − [0.2, 0.2, 0.2, 0.2, 0.2]
          = [−0.2, −0.2, −0.2, +0.8, −0.2]
```

Read that vector. It says: **raise the logit of the token you sampled, lower
everyone else's.** Which is exactly what "make this output more likely" should
mean. Note it sums to zero — you're moving probability around, not creating it.

**Step 5.** Take a step, scaled by the reward and a small learning rate
(say `0.1`):

```
new logits = [0,0,0,0,0] + 0.1 × 0.6 × [−0.2, −0.2, −0.2, +0.8, −0.2]
           = [−0.012, −0.012, −0.012, +0.048, −0.012]
```

Convert back to probabilities and token 3 is now at `0.2098` instead of `0.2000`.
The other four each dropped to `0.1976`.

A tiny nudge in the right direction. Do it a few thousand times, with the dice
picking different tokens each round, and the good tokens accumulate more nudges
than the bad ones. That's training.

**Where does `onehot − probs` come from?** It's the derivative of the softmax,
and you can take it on faith for now. But you shouldn't have to:
`WORKED_EXAMPLES.md` in this folder checks it numerically — it wiggles each logit
by a hair, measures what actually happens, and compares to the formula. If the
formula is right, the numbers match to 6 decimal places. They do.

## 6. The problem with what we just built

Look at that reward table again. **Every reward is positive.**

So every update pushes the sampled token *up*. Sample token 1 (reward 0.2)? Push
it up. Sample token 0 (reward 0.0)? Well, times zero, so no push — but token 1,
which is nearly as bad, still gets reinforced.

We're rewarding bad answers. Less than good ones, so it does eventually sort
itself out — but we're making the model work much harder than it needs to.

**The fix is one subtraction.** Instead of using the raw reward, subtract the
average reward first:

```
raw reward for token 1:                     0.2
average reward across all tokens:           0.44
reward − average:                          −0.24     ← negative! push it DOWN
```

Now below-average tokens get actively suppressed instead of weakly encouraged.
The number you subtract is called a **baseline**, and the result — how much
better than typical this output was — is called the **advantage**.

**Subtracting a baseline doesn't distort the answer at all.** It's mathematically
free. You get less noise
and pay nothing. Exercise 3 has you measure exactly how much less noise, and the
result is bigger than you'd guess.

## 7. Where does the average come from?

You need "the average reward for this prompt." How do you know it?

The classical answer is to train a **second neural network** whose only job is to
look at a prompt and predict "you'll probably score about 0.44 here." That's
called a **value network**, and it's what the older algorithm PPO does. It costs
you an entire extra model, and early in training its guesses are bad.

The modern answer is much better, and once you see it you can't unsee it:

> Don't predict the average. **Just sample the prompt several times and measure
> it.**

Ask the same question 8 times, get 8 answers, score all 8. Their mean *is* the
average reward for that prompt. No second network, no prediction, no training.
You just... looked.

That's **GRPO**. A batch of answers to the same prompt is called a **group**, and
each answer's advantage is computed relative to its own group. The name "Group
Relative Policy Optimization" is a complete description of the algorithm.

Worked example — a group of 4 answers to one question:

```
answers:     A      B      C      D
rewards:    1.0    0.0    1.0    0.0        mean = 0.5

advantages: +0.5  −0.5   +0.5   −0.5        (each reward minus the mean)
```

A and C get reinforced. B and D get suppressed. Nothing else was needed.

**One consequence to sit with**, because it's the most useful thing in this
module: the advantages are built by subtracting the mean, so they always sum to
zero. The signal is *entirely* about which answers beat the others in their
group. If all 4 answers score 1.0, the mean is 1.0, and every advantage is
**zero** — that question teaches the model nothing.

Which means a question the model always gets right and a question it always gets
wrong are the *same* failure: no spread, no learning. You'll verify this in the
starkest way possible in exercise 4.

## Labs

Four files, plain numpy, no GPU, no API key. The whole module runs in seconds.

| exercise | what it teaches |
|---|---|
| `exercise_1_bandit.py` | exploring vs. exploiting, and how to estimate a value by averaging |
| `exercise_2_reinforce.py` | the update above, written out and run for real |
| `exercise_3_baseline.py` | why the baseline works — **measured**, not asserted |
| `exercise_4_grpo.py` | GRPO from scratch. The real algorithm. |

Do them in order. Each is the previous one plus one idea.

Exercise 1 might feel like a detour — it's slot machines, not language models.
Stick with it. It's the cleanest place to meet two ideas you need
(you have to *try things* to find out what's good, and you estimate value by
*averaging noisy samples*), and it's 40 lines.

## How to work

```bash
uv run python modules/01-rl-foundations/exercise_1_bandit.py
```

It'll stop at the first `TODO` and tell you what to write. Fill it in, run it
again. When a file runs clean, grade yourself:

```bash
uv run python modules/01-rl-foundations/verify.py
```

The grader checks *properties*, not exact code — there's more than one right way
to write these. When something's wrong it names what broke and shows what it saw.

Worked arithmetic for every exercise is in
[`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md). Fully explained solutions are in
`solutions/01-rl-foundations/` — they're written to explain *why*, so read them
even after you get things working.

## Checkpoint: what you should be able to say

If the module worked, this should sound obvious rather than impressive:

> To train on a score instead of an answer, you sample outputs and make the
> good ones more likely. To know what "good" means you compare each output to
> the average of others from the same prompt. You don't need a network to
> predict that average — you can just sample a few times and look.

That's the algorithm behind essentially every open reasoning model of the last
two years. It is also, as of 2026, the *default* rather than the only option —
`prime-rl` has grown an algorithms layer with multi-agent and hierarchical
variants, and hooks for bringing your own advantage function. Every one of them
computes an advantage and pushes probability toward whatever scored better; the
group mean is one choice of baseline within that family. Unit 08 says more. Once it's boring to you, you're ready for Unit 02, where all the
remaining difficulty lives: **writing the scoring function.**

## Optional reading

Only if you want it — the module stands alone.

- **Karpathy, "Deep Reinforcement Learning: Pong from Pixels"** —
  <http://karpathy.github.io/2016/05/31/rl/>. The best intuition-builder for this
  ever written. Skip the Atari specifics; read the parts about why policy
  gradients work.
- **The GRPO paper** (DeepSeekMath, arXiv:2402.03300), section 4.1 only. Two
  pages. Read it *after* exercise 4 and enjoy recognizing everything.
