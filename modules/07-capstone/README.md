# Unit 07 — Capstone

### Project brief

> **Time:** 8–12 hours · **Prerequisites:** all previous units · **Needs:** an
> API key, and a GPU box for stage 5
>
> **What you produce:** a task nobody has built, taken from an idea to a
> published environment with an honest write-up of what happened.

---

## The shape of it

Every previous unit handed you the task. This one doesn't, and that's the
difficulty — choosing something with the right amount of headroom turns out to be
most of the work.

Six stages:

```
  1. choose       what task, and why it's trainable
  2. build        environment + scoring + attack suite
  3. measure      pass@k, sample sizes, is it feasible
  4. train        on GPU, with a budget you set first
  5. publish      to the Environments Hub
  6. write up     what happened, including what didn't
```

Stage 3 exists to stop stage 4 wasting money, and stage 6 is where most of the
learning lands.

## Stage 1 — Choose a task

Three constraints, and the third is the one people get wrong.

**It must be verifiable.** A program checks the answer, with no model in the
loop. If you need a judge model to grade it, you've built a second thing that can
be gamed, and you now have two problems.

**You must be able to generate questions.** Twenty hand-written questions won't
survive contact with Unit 04's sample-size arithmetic. You need a generator with
a difficulty parameter, which is what Unit 06's infinite taskset was for.

**The model must succeed sometimes.** This is the hard one. Unit 03 measured what
happens at both extremes: a task the model always fails and a task it always
passes produce identical results, which is no learning at all. You are aiming for
the middle band, and you can't tell where that is by intuition.

Some directions that satisfy all three:

| task | verified by | difficulty knob |
|---|---|---|
| arithmetic with N-digit operands | exact comparison | number of digits |
| sort / dedupe / transform a list | recompute the answer | list length |
| a constraint puzzle (N-queens, sudoku) | check the constraints | board size |
| parse a format and answer about it | reference parser | nesting depth |
| a text game with a hidden state | game engine | search space size |
| unit-test a small function | run the tests | test strictness |

**A worked scoping example.** Suppose you pick multiplication. Two-digit ×
two-digit is where Unit 02 landed, and DeepSeek-V4-Pro scored 12/12 on it — that
task is saturated and useless for training. Push to four digits by four digits
and a mid-sized model may drop to near zero, which fails the other way. The
usable range is somewhere between, it depends on your model, and the only way to
find it is stage 3.

That search *is* the exercise. Budget real time for it.

## Stage 2 — Build it

Write the environment. Either API is fine; say which and why.

Two things that must exist before you go near a GPU:

**An attack suite.** At least ten adversarial replies, in the shape of Unit 02's
lab. Your scoring must survive all of them. Write them before you write the
scoring, because writing them afterwards means writing tests for the code you
already wrote rather than the behaviour you actually want.

**Separate scoring components.** Correctness and format at minimum, weighted so
that the format component can never outrank correctness. Unit 02 showed you where
that tips over.

## Stage 3 — Measure it, before spending anything

Run Unit 04's checks against your task:

- `pass@1` and `pass@k` at your intended group size.
- If `pass@k` is near zero, the model never produces a successful rollout and
  training cannot start. Make the task easier.
- If `pass@1` is near one, there's nothing left to teach. Make it harder.
- A confidence interval on your baseline score, with the sample size stated.

Write the numbers down now. They're your before-picture, and without them stage 6
has nothing to compare against.

**Gate:** do not proceed to stage 4 until `pass@k` sits meaningfully between the
extremes. This gate is the entire reason units 03 and 04 come before this one.

## Stage 4 — Train

Set a budget first, using Unit 05's arithmetic. Write down the number of steps
you can afford before you start, not after.

Run the config validator from Unit 05 lab 2. Then train.

While it runs, watch the four things from Unit 05 section 4: reward, completion
length, individual components, and group reward variance.

**Stop early if the reward climbs while completion length grows and correctness
stays flat.** That's reward hacking, it will not fix itself, and every further
step is money spent teaching the model to exploit you.

## Stage 5 — Publish

Package with `load_environment()` (Unit 04), push to the Hub.

Your README must state: what the task is, how it's scored, what the difficulty
range is, and what `pass@1`/`pass@k` a named model achieves. A published task
without those numbers is one nobody can decide whether to use.

## Stage 6 — Write up

Two to three pages in `WRITEUP.md`. The sections that matter:

**What you built, and why you expected it to be trainable.** Your stage 1
reasoning, including tasks you rejected.

**The feasibility numbers.** Stage 3's measurements, with sample sizes.

**What training did.** The curve, the cost, the wall-clock. Before and after
scores with confidence intervals.

**What went wrong.** The part with the most value in it. Attempts that failed,
reward functions you had to rewrite, a difficulty setting that turned out to be
saturated. If nothing went wrong, you either got lucky or you aren't looking hard
enough.

**What you'd do differently.** Concrete and specific.

---

## Rubric

100 points.

| | pts | what earns full marks |
|---|---|---|
| **Task choice** | 15 | Verifiable, generated, with a difficulty knob. Reasoning for why it should be trainable is stated up front, not reverse-engineered. |
| **Scoring design** | 20 | Survives a 10+ case attack suite. Components separated and sensibly weighted. You can name a hole that remains. |
| **Feasibility work** | 20 | `pass@1`/`pass@k` measured before training, sample sizes stated, difficulty tuned in response to the numbers rather than guessed. |
| **Training** | 15 | Budget set in advance and respected. Curve interrogated rather than reported. Rollouts actually inspected. |
| **Publishing** | 10 | Installs and runs from a clean checkout. README states the difficulty range and baseline numbers. |
| **Write-up** | 20 | Honest. Confidence intervals on every comparison. Failures described specifically enough that someone could avoid them. |

Deductions worth knowing about:

- **−10** for any model comparison reported without a sample size. Unit 04 was
  mostly about this.
- **−10** if training ran on a task whose `pass@k` was never measured.
- **−5** if the write-up contains no failures.

## What "done" looks like

A stranger can install your task, reproduce your baseline number, read your
write-up, and know both what it's good for and where it breaks.

That's a real contribution to the Hub rather than another saturated math
environment, and it's a fair test of everything in the previous seven units.
