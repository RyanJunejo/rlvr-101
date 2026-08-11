# Unit 05 — Problem Set

**100 points.** Parts A–C can be answered from the labs and lecture notes. Part D
needs an actual training run.

---

## Part A — The system (20 pts)

**A1. (10 pts)** Why do generation and training run as separate processes? Give
the hardware reason: what does each one want from a GPU?

> 

**A2. (10 pts)** The starter configs all need two GPUs. Explain what each one is
doing, and what would happen if you tried to run both roles on one.

> 

---

## Part B — Reading the config (35 pts)

**B1. (10 pts)** `group_size = 16`. What is this number in the algorithm you
wrote in Unit 01, and what specifically breaks if you set it to 1?

> 

**B2. (10 pts)** `batch_size = 128` and `group_size = 16` means 2,048 completions
per step. Walk through why that multiplication is the right one, and say which of
the two you'd reduce first if a run were too slow.

> 

**B3. (5 pts)** `lr = 3e-6` is much smaller than a pretraining learning rate.
Why?

> 

**B4. (10 pts)** The config uses `taskset`, `harness` and `runtime` — the Unit 06
vocabulary — even though most published environments are still v0. What does
`harness.id = "null"` mean for this particular task, and when would you need
something else?

> 

---

## Part C — Money (25 pts)

**C1. (10 pts)** You have $50. A step takes 30 seconds on two H100s. How many
steps can you afford, and what's the arithmetic?

> 

**C2. (5 pts)** In `affordable_steps`, why does `budget // cost_per_step` give
the wrong answer for `$50` at `$0.04` per step? What's actually going on?

> 

**C3. (10 pts)** List the three ways credits get wasted, in order of how often
they happen, and say which unit of this course prevents each one.

> 

---

## Part D — Your run (20 pts)

Answer these from your own training run. If you haven't done one yet, come back.

**D1. (5 pts)** How long did one step take? What did the 20-step `reverse-text`
run cost you?

> 

**D2. (5 pts)** Paste or describe your reward curve. Did it rise? What did
completion length do over the same period?

> 

**D3. (5 pts)** Read five sampled rollouts from the end of the run. Is the model
doing the task, or has it found something cheaper? How can you tell?

> 

**D4. (5 pts)** Extrapolate: what would 500 steps have cost? Is your capstone
feasible within your remaining credit?

> 

---

## Reflection (ungraded)

The lecture notes admit the runbook in section 5 was never executed end to end on
the machine that wrote it. Did anything in it turn out to be wrong or stale? Write
down what, so the next person doesn't hit it.

> 
