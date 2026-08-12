# Unit 06 — The legacy API

### Lecture notes

> **Time:** 1–2 hours · **Prerequisites:** Unit 02 · **Needs:** nothing (runs
> offline)
>
> **By the end of this unit you will be able to:**
> 1. Read a v0 environment from the Hub and name what each piece corresponds to
>    in the API you know.
> 2. Write a v0 reward function and `load_environment()`, with the guard v0
>    makes your responsibility.
> 3. Port a v0 environment to v1 and prove the port changed nothing.
> 4. Say which two v0 bug classes the v1 design retired.
>
> **Deliverables:** 2 labs, autograder green, problem set in `NOTES.md`.

When you've finished the labs, [`THEORY.md`](THEORY.md) goes deeper: bug
classes, and why a silent failure costs more than a loud one.

---

## 1. Why a legacy unit exists

Everything you've built used `verifiers.v1`. The same installed package also
contains the original API — plain `import verifiers`, no `.v1` — and it is not
a museum piece: the Environments Hub carries environments written against it,
and the eval CLI runs them behind a flag whose name tells you Prime Intellect's
own view of the matter:

```
uv run eval --legacy.id <env-id> ...
```

So the realistic situation is the one this unit rehearses. You find a Hub
environment worth using; it's v0; you need to read it, maybe run it, maybe port
it. An hour of fluency here saves you a confused afternoon there.

## 2. The mapping

Every v0 concept is one you already have, in different furniture:

| v1 (what you know) | v0 (what you'll read) |
|---|---|
| `import verifiers.v1 as vf` | `import verifiers as vf` |
| `TaskData` subclass, typed fields | a HF `Dataset` row with columns `question` / `answer` |
| `Task` with `@vf.reward` methods | loose functions collected in a `Rubric` |
| weight on the decorator | a parallel `weights=[...]` list |
| `trace.last_reply` | `completion[-1]["content"]` |
| `Taskset` subclass + `__all__` | a `load_environment()` function |
| `@vf.metric` | a zero-weight rubric entry |
| `Env.run()` | `MultiTurnEnv` with `env_response()` and `@vf.stop` methods |

Argument injection works in both — v0 reward functions name `completion`,
`answer`, `prompt`, `state` and receive what they name.

## 3. Two differences that are lessons, not furniture

**The empty-completion guard is your job.** A v0 reward reads
`completion[-1]["content"]`, and an aborted rollout — dropped connection,
timeout — delivers `completion` as an *empty list*. Unguarded, your grader
raises `IndexError`, which the library reports as
`Error calling reward function`, hiding the real cause. So every v0 reward
function you will ever read starts with some form of:

```python
text = completion[-1]["content"] if completion else ""
```

v1's `trace.last_reply` returns `""` in the same situation. The API absorbed
the bug class; in v0 you carry it by hand, in every function, forever.

**The parallel weights list.** `Rubric(funcs=[a, b], weights=[1.0, 0.2])`
keeps two lists aligned by position. Insert a function without inserting its
weight and every later weight silently shifts onto the wrong function — no
error, just a rubric that means something different from what you wrote. v1's
decorator-borne weight retired this one too.

One more v0 quirk worth knowing before you go looking for your own reward
function: the library wraps your rubric in a `RubricGroup` alongside its
monitor rubrics, so `env.rubric.funcs` is empty and yours lives under
`env.rubric.rubrics`.

## 4. When you'd port, and what "done" means

Run a v0 environment as-is when you just want its numbers — `--legacy.id` is
one flag. Port it when you want to extend it, or want one stack across your
project.

The port is mechanical (lab 2 walks it), and it has an exact definition of
done: **every reply scores identically through both stacks.** A migration that
changes any score has changed the task, and numbers from before and after stop
being comparable. Lab 2 scores the same replies through both stacks and prints
the agreement.

## Labs

| file | what you build |
|---|---|
| `exercise_1_v0_env.py` | a v0 environment: reward function, `load_environment()` |
| `exercise_2_port.py` | the same environment on v1, scored through both stacks side by side |

Both run offline.

## How to work

```bash
uv run python modules/06-legacy-v0/exercise_1_v0_env.py
uv run python modules/06-legacy-v0/verify.py
```

## Checkpoint

You should be able to open an unfamiliar v0 environment from the Hub and
identify the dataset, the grader and its guard, the weights and the ratio they
encode, and what each piece becomes in v1 — then name the test that would prove
a port changed nothing.
