# Unit 06 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

---

## Part A — Why the redesign (30 pts)

**A1. (15 pts)** Describe a task you cannot express with `SingleTurnEnv`, and say
which of the three things bundled into a v0 environment is what blocks you.

> 

**A2. (10 pts)** "Any taskset can run under any compatible harness." Why does
that matter for comparing two systems? What goes wrong if the task definition
contains an assumption about how the model attempts it?

> 

**A3. (5 pts)** Both APIs ship in the same package. Give one situation where
you'd pick v0 anyway.

> 

---

## Part B — Writing a v1 task (35 pts)

**B1. (10 pts)** Map each v0 concept to its v1 equivalent: the dataset row, the
environment, a reward function, the rubric's weights.

> 

**B2. (10 pts)** v1 reward methods must be `async`. What error do you get if you
forget, and why is that error message unhelpful?

> 

**B3. (10 pts)** In v0, weights live in a list parallel to `funcs`. In v1 they
live on the decorator. Describe the specific bug the v1 arrangement makes
impossible.

> 

**B4. (5 pts)** `MathData` is frozen. Why does that matter once a run is over and
you're reading `traces.jsonl`?

> 

---

## Part C — Traces (20 pts)

**C1. (10 pts)** You build a trace by hand, score it, and every reward is 0.0
with no error raised. What did you forget, and why does it fail silently rather
than loudly?

> 

**C2. (10 pts)** Why does the `sampled` flag exist at all? Give a concrete case
where a trace legitimately contains an assistant message the model didn't
produce, and say what scoring it would do to your numbers.

> 

---

## Part D — Tasksets (15 pts)

**D1. (5 pts)** Why is `load()` a generator rather than returning a list? Name
both things it buys you.

> 

**D2. (10 pts)** Connect this to Unit 03. Why does an infinite, procedurally
generated taskset help with the problem you measured there, and what does it
*not* solve?

> 

---

## Reflection (ungraded)

The lecture notes say v1's documentation lags its code, and that three things in
this unit came from reading the installed source rather than the docs. How would
you have found those yourself?

> 
