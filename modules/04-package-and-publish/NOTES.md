# Unit 04 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

---

## Part A — Packaging (20 pts)

**A1. (10 pts)** Why is the convention a `load_environment()` *function* rather
than a module-level variable holding an environment? Give all three reasons.

> 

**A2. (5 pts)** Why must every argument to `load_environment()` have a default?

> 

**A3. (5 pts)** Why does forwarding `**kwargs` matter more for a *published* task
than for one you only run yourself?

> 

---

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
