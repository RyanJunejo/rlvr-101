# Unit 04 — Packaging, publishing, and honest evaluation

### Lecture notes

> **Time:** 2–3 hours · **Prerequisites:** Units 01–03 · **Needs:** an API key
> (optional); a Prime Intellect account for the publishing section
>
> **By the end of this unit you will be able to:**
> 1. Package a taskset as an installable module, with its config as the
>    public API.
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

## 2. Packaging: the taskset contract

A publishable task is an installable Python package whose module exports exactly
one `Taskset` subclass through `__all__`:

```
mult-task/
  pyproject.toml
  mult_task/
    __init__.py       <- ends with  __all__ = ["MultTaskset"]
```

That single line is the whole registration. The loader — the same one behind
the eval CLI and the trainer — imports your module, reads `__all__`, and finds
the one class that subclasses `Taskset`. A Hub id like `org/mult-task@1.0` adds
only an install step first; a local module name skips even that.

The second half of the contract is the **config**. Your `TasksetConfig`
subclass is the public API of the task — every field on it becomes a knob a
runner can turn without touching your code:

```python
class MultConfig(vf.TasksetConfig):
    num_tasks: int = 64
    digits: int = 2          # the difficulty knob
```

```bash
uv run eval mult_task --env.taskset.digits 4      # CLI flag
```

```toml
[env.taskset]                                     # or a training TOML
id = "org/mult-task"
digits = 4
```

Three properties the lab insists on, each with a reason you've already met:

- **Every config field has a default.** Tooling constructs the config bare to
  inspect your task; a required field breaks that.
- **Generation is seeded.** Two people running "the same" taskset must see the
  same questions, or their numbers aren't comparable.
- **There is a difficulty knob.** Unit 03 measured what happens without one:
  the model saturates the task, every group scores identically, and the task
  teaches nothing forever after. `digits` is what lets a user find the band
  where their model succeeds *sometimes*.

One version note: verifiers 0.3.0's hook is `load()`; the lab-cookbook shows
`load_tasks()`, which tracks a newer release. The installed source wins —
another instance of the course's standing rule.

## 3. Publishing to the Hub

```bash
uv tool install prime
prime login

prime env init mult-task     # scaffolds the directory
# move your taskset module + pyproject in
prime env push               # publishes it
```

Others then run it by id — `uv run eval yourname/mult-task` — and the loader
installs it from the Hub on first use.

Before you push, ask the question that separates a useful published task from
noise on the Hub: **at which difficulty does a model people care about succeed
sometimes?** Measure it (section 5 gives you the tool), and put the number in
your README. A task without a known working difficulty band is one nobody can
decide whether to train on.

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
| `exercise_1_packaging.py` | a taskset package with a difficulty knob |
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

> A task is an installable package exporting one `Taskset` via `__all__`,
> with its config as the public API. And a score
> without a sample size is not a result — differences smaller than the confidence
> intervals are noise, halving uncertainty costs four times the samples, and
> `pass@k` tells me whether a task is trainable at all before I rent a GPU.
