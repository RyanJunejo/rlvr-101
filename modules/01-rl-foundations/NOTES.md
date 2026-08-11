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

> 

**A2. (10 pts)** Somewhere between the model's internal numbers and the final
score, the math "breaks" and you can't compute a gradient through it. Name the
exact step where it breaks, and explain why that step is different in kind from
the ones around it.

> 

**A3. (5 pts)** State the training method in one sentence. If your sentence has
more than one clause you probably haven't simplified it enough.

> 

---

## Part B — Mechanics (35 pts)

**B1. (10 pts)** The nudge direction is `onehot(sampled token) − probs`. Read it
out loud as a plain-English instruction — what does it tell you to do?

> 

**B2. (10 pts)** That vector always sums to zero. Why, and what does that tell
you about what an update can and can't do?

> 

**B3. (5 pts)** Why do we subtract the maximum logit before exponentiating in
`softmax`? What breaks if we don't?

> 

**B4. (10 pts)** Exercise 2's reward table is all non-negative. Describe what
goes wrong because of that, and give the one-line fix.

> 

---

## Part C — The baseline (20 pts)

**C1. (10 pts)** You shifted every reward up by +10. The best possible model
didn't change at all, but learning got 359× noisier. Explain what happened.

> 

**C2. (10 pts)** Following from C1: you write a scoring function that gives 10
points for "replied in valid JSON" and 1 point for "the answer is correct." What
have you just done to yourself, and what's the fix?

> 

---

## Part D — GRPO and its consequences (20 pts)

**D1. (5 pts)** GRPO needs the average score for a question. The older approach
trains an entire second neural network to predict it. What does GRPO do instead?

> 

**D2. (5 pts)** Why is that approach available when training a language model,
but *not* available when training, say, a robot to walk?

> 

**D3. (5 pts)** GRPO with a group size of 1 didn't learn slowly — it didn't learn
at all, and the model came out exactly as it went in. Why? One sentence.

> 

**D4. (5 pts)** A question where all 8 answers score 1.0, and a question where
all 8 score 0.0. What do these have in common, and what does it imply about how
to choose training questions?

> 

---

## Reflection (ungraded, but do it)

What surprised, annoyed, or confused you in this unit? This is usually where the
real misunderstanding is hiding, and it's the most useful thing to carry into
Unit 02.

> 
