# Unit 04 — Packaging, publishing, and honest evaluation

### Lecture notes

> **Time:** 2–3 hours · **Prerequisites:** Units 01–03 · **Needs:** an API key
> (optional); a Prime Intellect account for the publishing section
>
> **By the end of this unit you will be able to:**
> 1. Package a task as an installable module using the `load_environment()`
>    convention.
> 2. Publish it to the Environments Hub.
> 3. Put a confidence interval on an evaluation score, and say what sample size
>    a claim requires.
> 4. Recognise when a difference between two models is indistinguishable from
>    noise — and compute how many more samples you'd need to tell.
> 5. Compute `pass@k` correctly, and explain why it matters for RL specifically.
>
> **Deliverables:** 3 labs, autograder green, problem set in `NOTES.md`.

---

## 1. Two unrelated things, and why they're in the same unit

This unit covers packaging and statistics, which sound like an odd pairing. The
connection is that both are about **making your work mean something to someone
else**.

Packaging means another person can run your task. Statistics means the number
you hand them is defensible. Neither is glamorous, and skipping either one is how
you end up with results nobody can reproduce or believe.

The statistics half is the one that will change how you work. Most people's first
model comparison is noise, and they don't know it.

---

## 2. Packaging: the `load_environment` convention

A `verifiers` task is a normal, installable Python package with one requirement:
it exposes a function called `load_environment()` that returns a configured
environment.

```
my-task/
  pyproject.toml
  my_task.py          ← contains load_environment()
```

```python
# my_task.py
import verifiers as vf

def load_environment(num_problems: int = 100, **kwargs) -> vf.Environment:
    dataset = build_dataset(num_problems)
    return vf.SingleTurnEnv(dataset=dataset, rubric=build_rubric(), **kwargs)
```

That's the entire contract. Anyone can then do:

```python
env = vf.load_environment("my-task", num_problems=50)
```

**Why a function rather than a module-level variable?** Three reasons, and
they're worth understanding rather than memorising:

- **Arguments.** Callers can configure the task — dataset size, difficulty,
  which split — without editing your code.
- **Laziness.** Building a dataset can mean downloading gigabytes. A function
  means that only happens when someone actually wants the environment, not on
  import.
- **Freshness.** Each call builds a new object, so two training runs can't
  accidentally share mutable state. (Same reasoning as Unit 03's "don't put game
  state on `self`.")

Everything else is standard Python packaging. Your `pyproject.toml` declares
dependencies — and this matters more than usual, because someone else's training
run will install your package on a machine you've never seen.

## 3. Publishing to the Hub

```bash
uv tool install prime
prime login

prime env init my-task      # scaffolds the directory
# ... write your environment ...
prime env push              # publishes it
```

Others then install it with `prime env install <your-name>/my-task`.

Before you push, ask yourself the question that separates a useful published task
from noise on the Hub: **would this produce a useful gradient?** From Unit 03, a
task where the model always succeeds or always fails teaches nothing. Publishing
a task that's saturated is publishing an eval, not a training environment. Both
are fine — just label it honestly.

---

## 4. Now the part that will change how you work

You evaluate two models on your task, 20 questions each.

```
model A:  12 / 20 correct   =  60%
model B:  15 / 20 correct   =  75%
```

B is better, right? It's 15 points ahead.

**No. You cannot conclude that.** That difference is entirely consistent with the
two models being identical and the coin landing differently. Lab 2 has you
compute it, and the answer is uncomfortable: with 20 samples, you'd need a gap of
roughly 30 points before you could say much at all.

### Why this happens

Each question is a coin flip. A model with a true 60% success rate doesn't score
exactly 60% on 20 questions — it scores somewhere in a range, and that range is
wide:

```
true rate 60%, n = 20  →  95% of the time you'll measure between 36% and 81%
true rate 60%, n = 100 →  95% of the time you'll measure between 50% and 70%
true rate 60%, n = 500 →  95% of the time you'll measure between 56% and 64%
```

That spread is called a **confidence interval**, and the headline is that it
shrinks like `1/√n`. To halve your uncertainty you need **four times** the
samples. This is the tax on all empirical work and there is no way around it.

### The rule

> **A score without a sample size attached is not information.**

Report `62% ± 7% (n=200)`, never `62%`. If you take one habit from this course,
take this one — it's the difference between a result and a vibe.

---

## 5. `pass@k`, and why RL people care

Standard accuracy asks: *if the model gets one attempt, how often is it right?*
That's `pass@1`.

`pass@k` asks: *if the model gets `k` attempts, how often does at least one
succeed?*

Why does this matter here specifically? Think back to Unit 03's sparse reward
problem, where learning requires *spread within a group*.

Consider a model with `pass@1` of about 5% — it almost never gets a question
right on one try. Standard accuracy says it's hopeless. But sample it 8 times and
ask what `pass@8` is. If it's 35%, then in roughly a third of groups at least one
rollout succeeds while others fail — and that is exactly the spread that produces
a gradient.

So:

- `pass@1 = 5%`, `pass@8 = 35%` → **trainable.** Groups have spread; RL has
  something to reinforce.
- `pass@8 = 0%` → **not trainable.** The model never produces a single successful
  rollout, so every group is uniformly zero and there is nothing to learn from.

Same `pass@1`-looks-terrible model, completely different prognosis, and standard
accuracy can't tell them apart.

So `pass@k` works as a feasibility check: it tells you whether RL on this task
can get off the ground at all, before you spend anything finding out.

### Computing it correctly

The naive approach — generate `k` samples, check if any passed — works but is
noisy and wastes data. The standard estimator (from the Codex paper) is better:
generate `n` samples, count `c` successes, then

```
pass@k  =  1 − C(n−c, k) / C(n, k)
```

Read it as: "one minus the probability that all `k` of your draws come from the
`n−c` failures." Lab 3 has you implement it, and shows why the naive version is
biased.

---

## Labs

| file | what you build |
|---|---|
| `exercise_1_packaging.py` | package a task with `load_environment()` |
| `exercise_2_evaluation.py` | confidence intervals and required sample sizes |
| `exercise_3_pass_at_k.py` | `pass@k` done properly |

All three run offline. The Hub publishing walkthrough in section 3 needs an
account and is not autograded.

## How to work

```bash
uv run python modules/04-package-and-publish/exercise_2_evaluation.py
uv run python modules/04-package-and-publish/verify.py
```

## Checkpoint

You should be able to say:

> A task is an installable package exposing `load_environment()`. And a score
> without a sample size is not a result — differences smaller than the confidence
> intervals are noise, halving uncertainty costs four times the samples, and
> `pass@k` tells me whether a task is trainable at all before I rent a GPU.
