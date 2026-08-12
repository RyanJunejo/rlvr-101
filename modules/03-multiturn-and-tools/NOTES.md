# Unit 03 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

Scoring guide: **90+** you could teach this unit. **70–89** solid, move on.
**Below 70** reread the lecture notes.

---

## Part A — Multi-turn mechanics (25 pts)

**A1. (10 pts)** Describe the shape of an `Env.run()`: what does a bare
`turn()` do, what does `turn(feedback)` do, and how does the rollout end?

> 

**A2. (10 pts)** The `solved` reward replays the transcript instead of reading
any stored game state. What failure does that discipline make impossible, and
what makes it testable offline?

> 

**A3. (5 pts)** Why is it a bug to keep mutable rollout data on `self` rather
than `self.state`? What would go wrong, and when?

> 

## Part B — The design decisions (20 pts)

**B1. (10 pts)** `max_turns` is 7 because binary search needs 7 guesses for
1–100. Explain why setting it to 10 would *hurt training*, and connect your
answer to Unit 01.

> 

**B2. (10 pts)** When the model sends a message with no valid guess, the
environment replies with a reminder and records nothing — but the turn is still
consumed. Is that the right call? Argue either way, but say what incentive it
creates.

> 

---

## Part C — Sparse rewards (35 pts)

**C1. (10 pts)** State the sparse reward problem in your own words, and derive it
from the Unit 01 result about advantages. Why does a weak model get stuck?

> 

**C2. (10 pts)** In your results table, a beginner (skill 0.0) and an expert
(skill 1.0) both waste most or all of their groups. Explain why these are *the
same failure* despite looking opposite.

> 

**C3. (10 pts)** Shaping took the beginner from 45.2% useful groups to 100%, but
left the expert at 0.0%. Why does shaping fix one end of the curve and not the
other? What *does* fix the other end?

> 

**C4. (5 pts)** Why does measuring *distance from the answer* produce spread even
when no rollout ever wins? State the general principle this suggests.

> 

---

## Part D — What shaping costs (20 pts)

**D1. (10 pts)** The shaped reward caps partial credit at just under 0.5 while
winning pays 1.0. Describe precisely what strategy becomes optimal if you raise
that cap to 1.0 — and say which Unit 02 lesson this is a repeat of.

> 

**D2. (5 pts)** A shaped reward encodes *your theory* of what counts as progress.
Give an example of a task where "getting closer to the goal each step" would be
actively misleading to reward.

> 

**D3. (5 pts)** Why might you *anneal* a shaped reward — weight it heavily early
in training and decay it toward zero later?

> 

---

## Part E — Tools (bonus, 10 pts)

**E1. (5 pts)** Why is a tool's docstring not documentation? What concretely
happens to it?

> 

**E2. (5 pts)** You give a calculator to a model being scored on arithmetic. What
are you now actually training? Is that a problem?

> 

---

## Reflection (ungraded)

Multi-turn debugging is harder than single-turn. What was the first
thing that confused you, and how did you figure out what was happening?

> 
