# RLVR 101 — Training Language Models with Verifiable Rewards

**A hands-on course.** Eight units, roughly 30–40 hours of work, taken at your
own pace.

---

## Course description

Most machine learning courses teach you to train a model by showing it correct
answers. This course is about the other situation: what do you do when you
*can't* write down the right answer, and all you have is a way to **check** an
answer once you see one?

That question turns out to have a short answer, followed by a long tail of
practical difficulty that has almost nothing to do with the algorithm and almost
everything to do with the checker you write.

You will build the training algorithm from scratch in plain numpy, write task
environments that survive contact with an optimizer trying to cheat them, train
a real model on rented GPUs, and publish your work.

The tools are Prime Intellect's open-source stack (`verifiers`, `prime-rl`, the
Environments Hub), which is the same stack used to train their INTELLECT model
series.

## Prerequisites

- **Python.** You should be comfortable writing functions and classes. Nothing
  exotic is used.
- **High-school algebra.** You need to be OK with the idea of a slope. No
  calculus is required — where derivatives show up, they're either given to you
  or verified numerically.
- **No prior reinforcement learning knowledge.** Every term is defined the first
  time it appears, and [`GLOSSARY.md`](GLOSSARY.md) collects them all.
- **No prior deep learning knowledge.** You never build a neural network here;
  you work with a deliberately tiny model where you can see every number.

## Learning outcomes

By the end of the course you should be able to:

1. **Explain** why a model can be trained from a score alone, and precisely where
   ordinary gradient-based training breaks down.
2. **Derive and implement** policy gradient training, including baselines and
   group-relative advantages (GRPO), from scratch.
3. **Design** a reward function for a new task, and **anticipate** how an
   optimizer will try to exploit it.
4. **Build** single-turn, multi-turn, and tool-using task environments with
   `verifiers`.
5. **Evaluate** a model honestly — with appropriate sample sizes and an
   understanding of what your numbers can and cannot support.
6. **Run** a real RL training job on GPU hardware and read its output critically.
7. **Package and publish** a task environment for others to use.

## How the course works

Each unit is a folder under `modules/`, containing:

| file | what it is | analogous to |
|---|---|---|
| `README.md` | the lecture notes — read this first | lecture |
| `WORKED_EXAMPLES.md` | arithmetic done out in full, with real numbers | section / recitation |
| `exercise_*.py` | starter code with `TODO`s | lab assignment |
| `verify.py` | automatic grading of your code | autograder |
| `NOTES.md` | written questions, with point values | problem set |

Fully worked solutions are in `solutions/`, mirroring the same structure. They
explain *why* each piece is the way it is, and are worth reading even after your
code runs.

**The intended loop:** read the lecture notes → do the lab → run the autograder →
answer the problem set → compare against the solutions and argue with them.

## Schedule

| unit | title | time | needs |
|---|---|---|---|
| **00** | Setup and orientation | 30 min | API key |
| **01** | Training on a score | 3–4 h | — |
| **02** | Writing the scoring function | 3–4 h | API key |
| **03** | Conversations and tools | 3–4 h | API key |
| **04** | Packaging, publishing, and honest evaluation | 2–3 h | API key |
| **05** | Training for real | 4–6 h | **GPU** |
| **06** | The legacy API (v0) | 1–2 h | — |
| **07** | Capstone project | 8–12 h | **GPU** |

Units 00–04 and 06 run on a laptop and cost only API tokens. Units 05 and 07
need an NVIDIA GPU — the training library does not run on Apple silicon — so
they use rented compute.

**Unit 05 is deliberately late.** Designing good tasks is the skill that
transfers; renting a GPU before you have it is an expensive way to learn nothing.

## Assessment

There are no grades, but there is a standard. Each unit's problem set is worth
100 points, allocated per question. You should be able to:

- **90+** — you could teach this unit to someone else.
- **70–89** — solid. Move on; the gaps will close as you use it.
- **below 70** — reread the lecture notes before continuing. The units build
  directly on each other, and Unit 01 in particular is load-bearing for
  everything after it.

The capstone (Unit 07) has its own rubric covering task design, reward
robustness, evaluation methodology, training results, and written analysis.

## On using the solutions

The `solutions/` folder is a teaching resource, not an answer bank. The intended
use is: attempt the problem, get stuck, attempt it again, *then* read the
solution and compare reasoning.

**A practical warning.** If you're working in an AI-assisted editor, the
solutions are sitting right there in the workspace and autocomplete will happily
write your GRPO implementation for you. That defeats the point of Unit 01
entirely — the value is in the twenty minutes of confusion before the code works,
not in the code.

If you want to keep yourself honest, add `solutions/` to `.cursorignore` (or your
editor's equivalent) while working through a unit.

## Materials

Everything needed is in this repository. [`READINGS.md`](READINGS.md) lists
optional primary sources, ordered by approachability, with notes on what to
actually read and what to skip. No reading is required to complete any unit.

There's also an interactive web version of the lecture material, diagrams, and
self-check questions: [`workbook.html`](workbook.html).

**On the two APIs.** The course teaches `verifiers.v1` — the API the training
stack's own configs are written in — from Unit 02 onward. The original API
survives as Unit 06, because the Environments Hub still carries environments
written against it and you will meet them.

**On Unit 05.** Everything in this course was run on the machine that wrote it,
with one exception: the GPU runbook in Unit 05 needs NVIDIA hardware that wasn't
available. Its config and commands come from the prime-rl repository and are
quoted verbatim; the runbook itself is verified by your own training curve. That
unit says so at the top.

## Getting started

```bash
cd ~/RL+prime
uv sync
cp .env.example .env        # add your API key
uv run python modules/00-setup/check_setup.py
```

Then open [`modules/00-setup/README.md`](modules/00-setup/README.md).
