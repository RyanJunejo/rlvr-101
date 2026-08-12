# Unit 08 — Problem Set

**100 points.** This unit is mostly reading, so the questions are mostly
synthesis. Answer in your own words.

### The vocabulary these questions use

Here so you don't have to leave the page. Fuller entries, with examples, in
[`GLOSSARY.md`](../../GLOSSARY.md).

| term | in one line |
|---|---|
| **harness** | the SOLVER: how a model attempts a task (one turn, agent loop, coding agent) | <!-- prose-ok: harness is the verifiers v1 API term -->
| **runtime** | WHERE a rollout executes: this machine, a container, a sandbox |
| **judge** | scoring by a MODEL rather than by a program |
| **Lab** | Prime Intellect's hosted platform: training, evals, adapter deployment |
| **adapter** | the small set of trained weights (a LoRA) you deploy after training |
| **RLM** | Recursive Language Model: context as a variable, sub-agents as function calls |

---

## Part A — The stack (30 pts)

**A1. (10 pts)** Sketch the stack from memory: the open-source pieces, the
platform above them, and where the thing you built in Unit 07 sits.

> 

**A2. (10 pts)** Why does a company selling a hosted training platform also
maintain the open-source trainer underneath it? Give the commercial logic, not
a charitable one.

> 

**A3. (10 pts)** When would you rent two GPUs and drive `prime-rl` yourself, and
when would you use Lab? Frame it as what each one bills you for.

> 

---

## Part B — Reading the package (30 pts)

**B1. (10 pts)** Six of fourteen harnesses are coding agents and four are other <!-- prose-ok: harness is the verifiers v1 API term -->
companies'. What does that cost them, and what does it buy? What does it imply
about which tasks they think matter?

> 

**B2. (10 pts)** You used `subprocess` all course. Name a task where that would
be a genuine security problem, and say why RL specifically makes it worse than
the same code run normally.

> 

**B3. (10 pts)** The package ships two judges and you never used one. State the
trade a judge makes, in terms of the Unit 02 lesson.

> 

---

## Part C — Direction (25 pts)

**C1. (10 pts)** Explain the RLM idea to someone who has finished Unit 03: what
is a "context as a variable", and what does the continual harness let the agent <!-- prose-ok: harness is the verifiers v1 API term -->
change about itself?

> 

**C2. (5 pts)** Prime Agent reports 95.5% on ARC-AGI-3 against a 95.4% human
baseline. Apply Unit 04 to that sentence — what would you want before believing
it, and what would you conclude if the sample were 20 tasks?

> 

**C3. (10 pts)** `prime-rl` now has an algorithms layer: GRPO is the default,
not the only option. Does anything you learned in Unit 01 stop being true? Say
precisely what generalizes and what was specific to GRPO.

> 

---

## Part D — Staying current (15 pts)

**D1. (10 pts)** Three times while building this course, the documentation and
the installed code disagreed and the code was right. Write the rule you'd give
someone starting on this stack next month.

> 

**D2. (5 pts)** The autograder fails if the package ships a harness this unit <!-- prose-ok: harness is the verifiers v1 API term -->
doesn't classify. Why is that a feature?

> 

---

## Reflection (ungraded)

Having seen the whole map: which part of it do you actually want to work in?

> 
