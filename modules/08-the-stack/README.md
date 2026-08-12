# Unit 08 — The whole stack

### Lecture notes

> **Time:** 1–2 hours · **Prerequisites:** Units 01–07 · **Needs:** nothing
> (runs offline)
>
> **By the end of this unit you will be able to:**
> 1. Name every piece of the Prime Intellect stack and say which layer it sits
>    in.
> 2. Explain what Lab is and when hosted training beats renting a box.
> 3. Read the installed package as evidence of where the stack is going.
> 4. Say what an RLM is and why they think it's the shape of 2026.
> 5. Keep the picture current after this course goes stale.
>
> **Deliverables:** 1 lab, autograder green, problem set in `NOTES.md`.

**Currency note.** Written 2026-08-12 against verifiers 0.3.0. This company
ships weekly — their blog carried 11 posts between May and August 2026 — so
treat dated claims as dated. Section 6 is about staying current without me.

When you've finished the labs, [`THEORY.md`](THEORY.md) goes deeper: the
bottleneck argument for open environments, and where it might be wrong.

---

## 1. The shape of the whole thing

Seven units taught you one thin slice: write a task, score it, train on it.
Here's where that slice sits.

```
   LAB  (the platform: hosted training, hosted evals, adapters, inference)
   └──────────────────────────────────────────────────────────────────┐
                                                                      │
   verifiers ──── Environments Hub ──── prime-rl ──── compute          │
   write the      share and install     train        GPUs, sandboxes   │
   task           tasks                 against it                     │
   └──────────────────────────────────────────────────────────────────┘
        ↑
   you have been here the whole time

   models:  INTELLECT-1 (10B) → -2 (32B) → -3 (100B+ MoE)
   agents:  Prime Agent / RLM
```

The open-source pieces are what you've used. **Lab** is the same loop as a
product: rather than assembling verifiers + prime-rl + your own GPUs, you point
Lab at a task and it runs hosted training, hosted evaluation, and deploys the
resulting LoRA adapter to an inference endpoint. Priced per token rather than
per cluster-hour.

The commercial logic is worth seeing plainly. The open-source stack is real and
complete — you trained with it, or could have. Lab sells the operational
burden: no cluster to babysit, no idle GPU to forget about. That's a defensible
thing to sell to people who want models rather than infrastructure, and it's why
the same company maintains both.

## 2. The model line, and what it was for

| model | size | what it demonstrated |
|---|---|---|
| INTELLECT-1 | 10B | training across the public internet at all |
| INTELLECT-2 | 32B | globally distributed *RL*, a harder problem than pretraining |
| INTELLECT-3 | 100B+ MoE | large-scale RL producing a competitive frontier model |

Read the progression as an argument. Each model exists to prove that a thing
people assumed needed one datacenter can be done distributed and in the open. If
that argument holds, the Environments Hub matters enormously — because then the
bottleneck on open frontier models is *tasks*, and tasks are the thing a
community can produce in parallel.

That thesis is what makes your capstone a contribution: the corpus of tasks
is the bottleneck, and it grows one task at a time.

## 3. Prime Agent and the RLM idea

Their August 2026 release, and the most interesting thing to understand if you
want to know where this is heading.

An **RLM** — Recursive Language Model — treats context as a *variable* and
sub-agent calls as *function calls*, inside a live Python REPL. Instead of a
model with a fixed context window filling up until it degrades, the model writes
programs that manipulate its own context: store this transcript in a variable,
spawn a sub-agent to summarize it, keep the handle.

On top of that sits a **continual harness**: the agent can create, read, update <!-- prose-ok: harness is the verifiers v1 API term -->
and delete its own prompts, skills, memory and sub-agents from inside its own
trajectory. A `/refine` command reads what the agent just did and makes the
smallest edit to its own scaffolding that would have improved the outcome.

Reported result: **95.5% on ARC-AGI-3, against a 95.4% human-expert baseline.**
That is a benchmark number from the company that built the thing, and it should
be read the way you'd read any such number after Unit 04 — with a request for
the sample size.

The connection to this course is concrete, not thematic. `rlm` is one of the
fourteen harnesses in the package you installed. The abstraction that lets a <!-- prose-ok: harness is the verifiers v1 API term -->
taskset run under any harness is what lets them swap a self-modifying agent in <!-- prose-ok: harness is the verifiers v1 API term -->
where Unit 02 had a model answering once.

## 4. What the package tells you

Lab 1 has you inventory the installed library. The headline:

```
harnesses (14): bash, browser_use, claude_code, codex, hermes_agent, kimi_code,
                mini_swe_agent, node, null, openclaw, pi, pool, rlm, terminus_2
runtimes   (4): docker, modal, prime, subprocess
judges     (2): reference, rubric
```

You used `null` (and Unit 03's own Env), `subprocess`, and no judge at all.

Three things that roster says:

**Six of fourteen harnesses are coding agents, and four are other companies'** — <!-- prose-ok: harness is the verifiers v1 API term -->
Claude Code, Codex, Kimi, mini-SWE-agent. Shipping adapters for competitors'
agents is a costly signal that they mean the harness to be swappable. It also <!-- prose-ok: harness is the verifiers v1 API term -->
tells you what they think the valuable tasks are.

**Four runtimes exist because tasks that run code need isolation.** During RL the
model is searching for inputs that maximize a score, which makes it a fuzzer
pointed at wherever the rollout executes. `subprocess` is fine for arithmetic and
wrong for anything that shells out.

**Two judges exist, and you never used one.** A judge scores with a model rather
than a program. That buys tasks no program can grade, and costs you the property
this whole course was built on: after Unit 02, you know what happens to a grader
that can be gamed, and a judge is a grader you cannot read at all. Use one when
you must, and hold out a programmatic check when you do.

## 5. Where the research is going

Four threads worth knowing by name, from their 2026 posts:

**Scale of environments.** They report crossing 365,000 environments for
software engineering, terminal and search tasks. The Hub stopped being a
curated shelf and became a corpus.

**An algorithms layer in prime-rl.** GRPO — the thing you implemented in Unit
01 — is now the default option among several. Recent releases add
multi-agent RL, hierarchical GRPO, and APIs for bringing your own loss and
advantage functions. What you learned still holds: every one of these computes
an advantage and pushes probability toward what scored better. The group-mean
baseline is one choice within a family.

**Multi-agent and world modelling.** Several posts argue that agents need to
model the world they act in rather than react turn by turn. This is the frontier
end and the least settled.

**Reward hacking, taken seriously.** They published their own writeup on
systematic reward hacking. Unit 02's lab is not a teaching toy — it's a
compressed version of a problem the people building this stack write papers
about.

## 6. Staying current after this course

This unit will go stale. The method that doesn't:

1. **Read the installed package first.** Three times in building this course,
   the docs and the code disagreed and the code was right — including once
   where the vendor's own tutorial used a method (`load_tasks`) that doesn't
   exist in the pinned version (`load`). `site-packages` is a primary source.
2. **Read `configs/` before source.** In `prime-rl`, the config files are the
   clearest available statement of what the system does.
3. **Watch the release notes for breaking changes.** They're frequent and
   honestly labelled.
4. **Check the harness and runtime rosters.** New entries are the cheapest <!-- prose-ok: harness is the verifiers v1 API term -->
   possible signal of direction — they appear before the blog post.

## Lab

| file | what you build |
|---|---|
| `exercise_1_read_the_package.py` | inventory the installed stack and read it as a map |

Runs offline.

```bash
uv run python modules/08-the-stack/exercise_1_read_the_package.py
uv run python modules/08-the-stack/verify.py
```

## Checkpoint

You should be able to sketch the stack from memory, say where your capstone sits
in it, name the tradeoff between renting a box and using Lab, and explain to
someone else why a company that sells a training platform also maintains the
open-source trainer underneath it.
