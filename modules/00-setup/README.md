# Unit 00 — Setup and orientation

### Lecture notes

> **Time:** 30 minutes · **Prerequisites:** none · **Needs:** an API key
>
> **By the end of this unit you will be able to:**
> 1. Name the pieces of the Prime Intellect stack and say what each one does.
> 2. Explain how they chain together, and which single piece you'll actually be
>    writing.
> 3. Run a working environment end to end and get a real score back.
>
> **Deliverables:** `check_setup.py` passing against a live model.

---

## 1. The 90-second version

Prime Intellect builds open-source versions of the tools that big AI labs use
internally to train models with reinforcement learning. Four pieces, chained in
one direction:

```
   verifiers          Environments Hub          prime-rl            compute
   ─────────          ────────────────          ────────            ───────
   write the    →     share and install   →     train against  →    on rented
   task               other people's            it                  GPUs
```

They also sell the same loop as a hosted product called **Lab** — you point it
at a task and it runs the training and evaluation on their infrastructure. This
course uses the open-source pieces, because seeing the machinery is the point;
Unit 08 covers where Lab fits and when it's the better choice.

Each piece is smaller than it sounds. Taking them in turn:

### 1. `verifiers` — where you'll spend most of your time

This is the library for writing the task and its scoring rules. In their
vocabulary the thing you write is called an **environment**, which is a bigger
word than it deserves — there's no simulator involved. It's just:

- a set of questions (and their correct answers),
- optionally, a way to find the model's answer inside its reply,
- one or more functions that look at the reply and return a number.

That last piece is the whole game, and Unit 02 is about writing it well.

### 2. The Environments Hub — a shared library of tasks

Tasks are ordinary installable Python packages. The Hub is where people publish
them so others can use and modify them. You'll push one in Unit 04.

Why it exists: the hard part of open RL research isn't algorithms or compute,
it's *tasks*. Everybody re-implements the same math grading logic from scratch.
The Hub is a bet that making tasks shareable is what unblocks the field.

### 3. `prime-rl` — the trainer

Takes your task and a model and runs the actual training. It's three separate
programs running at once:

- one **generates** the model's answers,
- one **hands out work and scores it** using your rules,
- one **updates the model's weights**.

They're split up because generating text and updating weights want very
different things from the hardware, and keeping them separate means neither sits
around waiting for the other. Unit 05.

### 4. Compute

A marketplace for renting GPUs. Only relevant in Unit 05, and we'll be using
your GMI Cloud credits instead.

### The models

INTELLECT-1, -2 and -3 are models Prime Intellect trained with exactly this
stack. They're the evidence it works end to end.

---

## 2. What's coming, and why in this order

Notice something missing from that list: there's no menu of algorithms to
choose between. The field has largely settled on one family — you'll build the
canonical member of it in Unit 01 — and the interesting design work moved into
**what you choose to reward**.

That's why Unit 01 has you write the training algorithm by hand exactly once,
so it stops being a black box — and then we essentially never write one again.
Everything after that is about the scoring function.

---

## Do the setup

You need one model API endpoint. Almost anything works — OpenAI, GMI Cloud,
Together, OpenRouter, or a server you run yourself — because `verifiers` talks to
all of them the same way. That's also why the task you write today can score a
hosted model now and train a local one in Unit 05 without changes.

```bash
cd ~/RL+prime
uv sync
cp .env.example .env
```

Open `.env` and fill in `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL`.

Then run the check:

```bash
uv run python modules/00-setup/check_setup.py
```

It verifies your Python version, your install, and your key, then makes one real
API call and scores the response. When it prints a score, you're ready for
Unit 01.

### Optional: the Prime CLI

Not needed until Unit 04, install it whenever:

```bash
uv tool install prime
prime login
```
