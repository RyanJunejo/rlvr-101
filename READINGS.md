# Readings

**None of this is required.** The modules are written to stand on their own —
you shouldn't need to read a paper to finish an exercise. This is here for when
you finish something and want to go deeper.

They're ordered by how approachable they are, not by importance. The note under
each one matters more than the link: it tells you what to actually read and what
to skip.

---

## Start here (no background needed)

**Karpathy, "Deep Reinforcement Learning: Pong from Pixels" (2016)**
<http://karpathy.github.io/2016/05/31/rl/>

The best intuition-builder for this material ever written, and it assumes very
little. The key passage is where he points out that this is really just ordinary
supervised learning, where the training examples are the model's own samples,
weighted by how well each one did.

Skip the Atari-specific parts. Read it after Module 01 exercise 2 and it'll
click.

**"Concrete Problems in AI Safety" (2016)** — arXiv:1606.06565

Read **section 3 only**, on reward hacking. It's plain prose, no math, and it
predicts essentially every failure you'll hit in the Module 02 lab. The example
of a boat that learns to spin in circles collecting points instead of finishing
the race is the canonical one.

---

## When you want the actual algorithm

**The GRPO paper** (DeepSeekMath, 2024) — arXiv:2402.03300

Read **section 4.1 only**. Two pages. This is the algorithm you build in Module
01 exercise 4, and reading it *afterward* is satisfying: you'll
recognize everything, and you'll notice how short the justification for dropping
the value network actually is.

Ignore the rest of the paper unless you care about math benchmarks.

**Sutton & Barto, *Reinforcement Learning: An Introduction*** (free online)
<http://incompleteideas.net/book/the-book-2nd.html>

*The* RL textbook. Be selective — most of it is about a setting we don't use.

- **Chapter 2** is the slot machine chapter. Directly relevant to exercise 1.
- **Chapter 13** covers the method from exercises 2 and 3, more formally.
- **Chapters 3–12** are about the setting where your actions change what you face
  next (robots, games). Interesting, but not what we're doing. Don't start there
  and don't feel obliged to.

**PPO** (2017) — arXiv:1707.06347

The older, more complicated relative of GRPO. Skim it. You want the idea of
capping how far the model can move in a single update, because that safety
mechanism carries over and you'll see it named in training configs.

---

## Once you've trained something

**DeepSeek-R1 (2025)** — arXiv:2501.12948

Large-scale training that actually worked, described unusually honestly. The
striking part is R1-Zero: pure RL starting from a base model with no worked
examples at all, and long chains of reasoning emerging on their own.

**INTELLECT-3 technical report** — arXiv:2512.16144

Prime Intellect's own model, trained on the exact stack you're learning. The
closest thing to "here's what all of this looks like at full scale." Read it
after Module 05, when the vocabulary will mean something.

**`prime-rl`** — <https://github.com/PrimeIntellect-ai/prime-rl>

Read the `configs/` folder before the source. The config files are the clearest
statement of how the system is organized.

---

## Reference

**`verifiers`** — <https://github.com/PrimeIntellect-ai/verifiers>

Read the source over the docs; the docs lag the library. The files that matter
are `v1/task.py`, `v1/taskset.py`, and `v1/trace.py` — each readable in one
sitting. For legacy environments (Unit 06): `envs/environment.py` and
`rubrics/rubric.py`.

**lab-cookbook** — <https://github.com/PrimeIntellect-ai/lab-cookbook>

Prime Intellect's own guided walkthrough of the same stack: thirteen guides from
setup through custom solvers, with runnable configs. Guides 01–03 cover this
course's Units 02–05 from the vendor's angle; guide 13 is their authoring
contract and failure-mode list, worth reading before the capstone. Note their
code sometimes tracks a newer verifiers than the pinned one here — where they
disagree, your installed source wins.

**Environments Hub announcement** —
<https://www.primeintellect.ai/blog/environments>

The reasoning behind the product. Strategy, not mechanics.

---

## Recent developments (as of 2026-08-12)

This company ships weekly, so this section dates fastest. Unit 08 covers the
picture; these are the primary sources behind it.

**"Releasing Lab: the training platform for self-improving agents"** (May 2026)
<https://www.primeintellect.ai/blog/lab-is-open>
The hosted product: training, evaluation, adapter deployment and inference in
one place, priced per token. Read it for the shape of the commercial offering
and where the open-source stack sits underneath it.

**"Systematic Reward Hacking and Prime Sprints"** (May 2026)
Their own writeup of the failure mode Unit 02's lab reproduces. Read it right
after doing the lab.

**"verifiers v1: Decomposing Tasksets and Harnesses for Agentic RL"** (July 2026)
The design argument for the API the course teaches — why the task definition
stopped containing an assumption about how the model attempts it.

**"Scaling Agentic RL: 365,000+ Environments"** (July 2026)
The Hub at corpus scale. Relevant to the capstone: it changes what counts as a
useful contribution.

**"prime-rl gets an Algorithms layer"** (July 2026)
GRPO becomes one option among several, with hooks for your own advantage
function. Nothing in Unit 01 stops being true; it stops being the only thing
that's true.

**"Prime Agent: A self-improving RLM agent"** (August 2026)
<https://www.primeintellect.ai/blog/prime-agent>
The frontier end: context as a variable, sub-agents as function calls, and a
harness the agent edits from inside its own trajectory. Reported 95.5% on <!-- prose-ok: harness is the verifiers v1 API term -->
ARC-AGI-3 against a 95.4% human-expert baseline — read that number with Unit
04's questions in hand. Start with "Recursive Language Models: the paradigm of <!-- prose-ok: quoted post title -->
2026" (January 2026) if the RLM idea is new to you.

**"$130M Series A to Build the Open Superintelligence Stack"** (July 2026)
Company context, if you want to know who's funding the thing you're learning.

---

## What you can safely ignore

If you go looking for RL material online you'll find a lot about Atari, DQN,
replay buffers, target networks, robot control, and TD-lambda. It's all real RL
and it's almost entirely irrelevant here — it exists to solve the problem of
your actions changing what you face next, which mostly doesn't happen in our
setting.

If you find yourself three hours into a video about Q-learning, you've wandered
off the path.
