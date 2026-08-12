# Unit 08 — Theory

> Short, and it's economics rather than mathematics: why a company would give
> away the trainer, and what that implies about where effort is worth spending.

---

## The bottleneck argument

The case for open environments is an argument about which input is scarce.

Training a model needs three things: **compute**, **algorithms**, and **tasks
with checkable rewards**. Ask which one constrains open work:

**Compute** is expensive but purchasable, and the distributed-training line
(INTELLECT-1 through -3) is an argument that it doesn't even have to be
co-located.

**Algorithms** are published. You implemented the core one in an afternoon in
Unit 01, from a paper anyone can read. There is no algorithmic moat.

**Tasks** are the awkward one. Each needs domain knowledge, a verifier someone
thought hard about, and a difficulty band someone measured. They don't benefit
from scale the way compute does — a thousand GPUs don't write you a thousand
tasks. And historically each lab built its own privately and threw them away.

If that reading is right, tasks are the binding constraint on open frontier
models, and the thing most worth sharing is the environment corpus.
That is the Hub's entire thesis, and it's why the same company maintains a free
trainer and sells a hosted platform: the trainer is complementary to the corpus,
and the corpus is what they need to exist.

## Where it could be wrong

Two honest objections.

**Quality may not aggregate.** 365,000 environments is a number about *count*. If
most are saturated, exploitable, or undocumented as to difficulty, the corpus is
large and thin. Unit 02 and Unit 04 both suggest that writing a *good* task is
harder than writing a task, and nothing about a public hub enforces the
difference. This is exactly why the capstone rubric weights the feasibility work
and the honest write-up so heavily.

**The bottleneck may move.** If self-improving agents (Unit 08 §3) can generate
and verify their own tasks, the scarce input stops being human-authored
environments. The RLM work is a bet in that direction, and it sits slightly
awkwardly against the corpus thesis — one says humans should write more tasks,
the other says systems should write their own.

Both could be true at different scales. It's worth noticing that a company is
funding both.

## What follows for you

If the bottleneck argument holds, the transferable skill from this course is not
the API — APIs turn over, as Unit 06 demonstrates — and not the algorithm, which
is a page of numpy.

It's the judgment: can you look at a task and say whether it will teach a model
anything, before spending money finding out? Units 02, 03 and 04 were all
training that one capability from different directions:

- Unit 02: will the grader survive an optimizer?
- Unit 03: is there reward spread to learn from?
- Unit 04: is the difference you're claiming larger than the noise?

Those three questions are the course. Everything else is the machinery that made
them concrete.
