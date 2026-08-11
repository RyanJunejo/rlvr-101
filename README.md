# RL + Prime Intellect: a hands-on workbook

## What this is

A self-paced course for learning how models are actually trained with
reinforcement learning — by building the pieces yourself, in small runnable
files, starting from nothing.

**You don't need prior RL knowledge.** If you've watched a video or two and can
write Python, you're in the right place. Every term gets defined the first time
it shows up, and there's a [`GLOSSARY.md`](GLOSSARY.md) with plain-language
definitions and examples for everything.

## The idea in one paragraph

Normally you train a model by showing it the right answer: "when you see this,
produce that." But for a lot of what we want models to do, you don't *have* the
right answer written out — you only have a way to **check** an answer once you
see one. Did the code run? Is the number 391?

So the question becomes: how do you improve a model when all you can do is score
its attempts? That question has an answer, the answer is simpler than you'd
expect, and this workbook builds it from scratch.

## How it works

Each module is a folder under `modules/`:

| file | what it is |
|---|---|
| `README.md` | the lesson. Read this first. |
| `exercise_*.py` | starter code with `TODO`s. This is where you work. |
| `verify.py` | run it to grade yourself. It says which specific thing broke. |
| `WORKED_EXAMPLES.md` | the arithmetic done by hand, with real numbers |
| `NOTES.md` | questions to answer in your own words |

Fully explained answers live in `solutions/`, mirroring the same structure. They
explain *why* each piece is the way it is, so they're worth reading even after
your code works.
They're in a separate folder so you can't stumble onto one by accident.

The loop: read the lesson → fill in the `TODO`s → run `verify.py` → compare with
the answer key.

## Setup

```bash
cd ~/RL+prime
uv sync
cp .env.example .env
```

Add a model API key to `.env`, then:

```bash
uv run python modules/00-setup/check_setup.py
```

That checks your install and makes one real API call, so you know everything
works before any concepts arrive.

There's also an interactive web version of the lessons, diagrams, and quizzes:
[`workbook.html`](workbook.html), published at
<https://claude.ai/code/artifact/f06dbf50-19cd-4f8c-9458-ac3c3cf6307d>.

## The modules

**00 — Setup.** What Prime Intellect's tools are and how they fit together.

**01 — Training on a score instead of an answer.** The core of everything. Plain
numpy, no GPU, runs in seconds. You build up from slot machines to **GRPO**, the
algorithm that trained most open reasoning models of the last two years. By the
end it should feel obvious rather than impressive.

**02 — Writing the scoring function.** Now that you can train on a score, the
hard part is writing a score that means what you think it means. You'll write a
reward function with a hole in it, watch a good model drive a truck through the
hole, then fix it. This is the most useful exercise here.

**03 — Conversations and tools.** Multi-turn tasks, where the model's move
changes what it sees next. You build a game, then measure the **sparse reward
problem** — why a model that's bad at a task can get *zero* learning signal — and
fix it with partial credit.

**04 — Packaging, publishing, and honest evaluation.** Making your work
installable and sharing it. Plus the statistics half: confidence intervals,
required sample sizes, and `pass@k` as a check on whether a task is trainable at
all before you rent a GPU.

**05 — Training for real.** Read a `prime-rl` config and recognise every number
in it from earlier units. Work out what a run costs before starting it. Then rent
a GPU, train, and read the curve critically.

**06 — The redesigned API.** Port your Unit 02 task to the v1
taskset/harness/runtime split, and find out what it buys you: a task definition <!-- prose-ok: "harness" is the verifiers v1 API term -->
that no longer assumes how the model will attempt it.

**07 — Capstone.** Choose and build a task nobody has built, take it from idea to
published environment, and write up what happened. Six stages and a rubric.

## Hardware

Modules 00–04 and 06 run on your Mac and cost nothing but API tokens.

Module 05 needs an NVIDIA GPU — the training library doesn't run on Apple
silicon at all — so it uses rented compute. It's deliberately near the end:
designing good tasks is the skill that transfers, and it's worth having that
before you start paying by the hour.
