# Unit 04 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

### The vocabulary these questions use

Here so you don't have to leave the page. Fuller entries, with examples, in
[`GLOSSARY.md`](../../GLOSSARY.md).

| term | in one line |
|---|---|
| **taskset** | a module exporting exactly one `Taskset` subclass via `__all__` |
| **config** | your `TasksetConfig` subclass — every field is a runner-tunable knob |
| **standard error** | how much a measurement would jump if you re-ran the whole thing |
| **confidence interval** | the range the true rate plausibly sits in, given your sample size |
| **pass@1** | how often the model is right given one attempt. Plain accuracy. |
| **pass@k** | how often at least one of k attempts succeeds |

---

## Part A — Packaging (20 pts)

**A1. (10 pts)** State the taskset packaging contract in one sentence, and
describe what the loader actually does with your module.

> 

**A2. (5 pts)** Why must every field on your `TasksetConfig` have a default?

> 

**A3. (5 pts)** Why does the lab insist on a difficulty knob? Connect your
answer to the Unit 03 result about groups with no spread.

> 

## Part B — Confidence (40 pts)

**B1. (10 pts)** Model A scores 12/20, model B scores 15/20. Write the honest
one-sentence conclusion, and justify it with the intervals you computed.

> 

**B2. (10 pts)** Uncertainty shrinks like `1/√n`. State what that means in
practical terms for someone planning an evaluation, and give the multiplier for
halving your error bars.

> 

**B3. (10 pts)** Detecting a 0.5-point difference needed ~135,000 samples per
model. Explain where that number comes from — specifically, why the required
sample size scales the way it does with the size of the difference.

> 

**B4. (10 pts)** In Part 4 of the lab, two *identical* models produced a 15-point
gap in about a third of experiments. What does that tell you about the last
benchmark comparison you read with n around 20?

> 

---

## Part C — pass@k (30 pts)

**C1. (10 pts)** Explain `pass@k` in one sentence, then explain why it matters
for *training* specifically, beyond its use as a reported score.

> 

**C2. (10 pts)** Two models both have `pass@1` of roughly 5%. One has
`pass@8 = 35%`; the other has `pass@8 = 0%`. One is trainable and one isn't.
Explain which and why, referring to what happens inside a group.

> 

**C3. (10 pts)** The naive estimator (draw k, check if any passed) is *unbiased*
— it gives the right answer on average. Why is the closed-form estimator still
strictly better? Connect this to something from Unit 01.

> 

---

## Part D — Judgment (10 pts)

**D1. (5 pts)** Before publishing a task to the Hub, what's the one question you
should ask about whether it will produce a useful gradient?

> 

**D2. (5 pts)** You're about to spend $40 of GPU credit training on a new task.
Write the checklist you'd run first, using everything from Units 03 and 04.

> 

---

## Reflection (ungraded)

Has this unit changed how you'd read a benchmark table in a paper? How?

> 
