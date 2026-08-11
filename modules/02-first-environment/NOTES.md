# Unit 02 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

Scoring guide: **90+** you could teach this unit. **70–89** solid, move on.
**Below 70** reread the lecture notes — Unit 03 assumes all of this.

---

## Part A — The environment object (25 pts)

**A1. (10 pts)** What are the three parts of a `verifiers` environment, and what
does each one own?

> 

**A2. (10 pts)** The same environment object works as a test, as a way to
generate training data, and as a training task, with no changes. Why is that
possible? What does it tell you about the relationship between evaluating a model
and training one?

> 

**A3. (5 pts)** `completion` isn't a string. What is it, and how do you get the
model's text out of it?

> 

---

## Part B — Rubrics and weights (20 pts)

**B1. (10 pts)** You write `def my_reward(completion, answer, **kwargs)`. How
does the library know what to pass in? Why should you always include `**kwargs`?

> 

**B2. (10 pts)** Multiplying every weight in your rubric by 10 changes nothing
about training. Explain why (think about what GRPO does with the numbers), and
then say what *does* change training when you edit a rubric.

> 

---

## Part C — The reward-hacking lab (40 pts)

**C1. (15 pts)** Pick three attacks from the suite. For each, state in one line
why `sloppy_correctness` fell for it, and which rule in your fixed version stops
it.

> 

**C2. (15 pts)** The central argument of this unit: why is an *exploitable*
scoring function categorically worse than a merely *noisy* one? Your answer must
say something about what training is doing as a process.

> 

**C3. (10 pts)** Trace the failure end to end. One answer in a group stumbles
onto "list every number from 380 to 400" and scores 1.0 while the careful answers
score 0.0 on an arithmetic slip. Walk through what happens over the next few
hundred training steps, and describe what the score curve looks like while it
happens.

> 

---

## Part D — Design judgment (15 pts)

**D1. (5 pts)** Why is a *formatting* reward specifically dangerous to weight
heavily? Name the two properties that make any scoring component a magnet for the
model's attention.

> 

**D2. (5 pts)** Your fixed function passes all 12 attacks. Is it safe? Name at
least two ways it could still be fooled, and say what you'd have to give up to
close each one.

> 

**D3. (5 pts)** You're writing a scoring function for a brand-new task tomorrow.
Write your checklist.

> 

---

## Reflection (ungraded)

Which attack did you not see coming? Why do you think you missed it?

> 
