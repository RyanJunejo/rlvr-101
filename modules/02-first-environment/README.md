# Unit 02 — Writing the scoring function

### Lecture notes

> **Time:** 3–4 hours · **Prerequisites:** Unit 01 · **Needs:** nothing for the
> labs; an API key for the live run at the end
>
> **By the end of this unit you will be able to:**
> 1. Build a task with `verifiers.v1`: the data, the scoring, and a trace to
>    score.
> 2. Write reward methods with the argument-injection convention, and explain
>    why they must be `async`.
> 3. Combine components with weights, and explain why only the *ratio* matters.
> 4. Run your task against a live model with the eval CLI and read the output.
> 5. Break a scoring function the way an optimizer would, then fix it to
>    survive a 12-attack suite.
>
> **Deliverables:** 3 labs, autograder green, problem set in `NOTES.md`.

New terms are defined as they appear; [`GLOSSARY.md`](../../GLOSSARY.md) has them
all in one place.

---

## 1. Where we are

Unit 01 answered "how do you train a model on a score?" The answer turned out to
be short: sample answers, compare each one to the average of others from the
same question, make the better-than-average ones more likely.

Notice what that leaves wide open. Everything now depends on the score. The
training algorithm has no opinion about what's good — it faithfully maximizes
whatever number you hand it.

So the interesting question is no longer "which algorithm?" It's "what exactly am
I rewarding?" That's this unit, and honestly it's where the rest of the
difficulty in this field lives.

## 2. The three pieces of a task

The tool is `verifiers`, Prime Intellect's library, and we use its current API:

```python
import verifiers.v1 as vf      # note the .v1 — plain `import verifiers` is the
                               # legacy API (Unit 06), and the two don't mix
```

A scoring setup splits into three objects, each with one job:

```
  TaskData          Task                    Trace
  ────────          ────                    ─────
  one question:     how to score it:        one attempt:
  prompt, answer    reward methods          every message, scored
```

**`TaskData`** holds the ground truth for one question. You subclass it and add
what your task needs:

```python
class MathData(vf.TaskData):
    answer: str
```

It's frozen — immutable — because it gets serialized verbatim into the trace
file. What you read in `traces.jsonl` after a run is exactly what the task was
built from.

**`Task`** carries the scoring, as decorated methods:

```python
class MathTask(vf.Task[MathData, vf.State, vf.TaskConfig]):
    @vf.reward
    async def correct_answer(self, task: MathData, trace: vf.Trace) -> float:
        matches = ANSWER_RE.findall(trace.last_reply)
        if not matches:
            return 0.0
        return 1.0 if matches[-1].strip() == task.answer else 0.0
```

Three conventions to absorb, each of which is a lab-1 TODO:

- **Arguments are injected by name.** Ask for `task` and you get your
  `MathData`; ask for `trace` and you get the attempt. Name only what you use —
  a method that takes just `trace` is documenting that it never looks at the
  answer.
- **Weights ride on the decorator**: `@vf.reward(weight=0.2)`. There is no
  separate weights list to keep aligned by position, which retires a real class
  of bug.
- **Reward methods must be `async`.** Forget it and scoring raises
  `An asyncio.Future, a coroutine or an awaitable is required` — an error that
  nowhere mentions the word async. You will meet this once; better here than on
  a GPU bill.

**`Trace`** is the record of one attempt. The model's final text is
`trace.last_reply`; the full transcript is in `trace.nodes`. One flag matters
when you build traces by hand for testing: each message node carries `sampled`,
marking whether the *model produced it* or it *arrived with the prompt* (a
few-shot example, a conversation being continued). `last_reply` reads only
sampled messages. Forget `sampled=True` and every reward silently returns 0.0 —
no error, nothing. The flag exists because scoring prompt-supplied text as the
model's own work would inflate every number you report.

## 3. Running against a real model

Tasks become runnable through a **taskset** — a generator of tasks that the
`eval` CLI can find by module name. Lab 1 includes one, written for you (you
build your own in Unit 04). With it, the live run is one command:

```bash
export $(grep -vE '^#|^$' .env | xargs)
PYTHONPATH=modules/02-first-environment uv run eval exercise_1_first_task \
  -m "$MODEL" -n 4 -r 2 --rich False \
  --client.base-url "$OPENAI_BASE_URL" --client.api-key-var OPENAI_API_KEY \
  --env.agent.harness.id null -o outputs/first-task
```

Three details that will save you time:

- **The key must be exported.** Setting it in your shell is not enough:
  `--client.api-key-var` names an environment variable the eval process reads —
  your secret never appears in a config file, but it does have to be in the
  process's environment. Symptom of forgetting: every rollout fails with
  `upstream 401`.
- **`-r 2` is the group size.** The CLI's own help text calls it "the trainer's
  group size" — it is exactly the *G* from Unit 01.
- **`--env.agent.harness.id null`** means no agent loop: the model answers once.
  That's all a single-turn task needs.

The run directory holds `config.toml` (the resolved run — good for seeing what
your flags actually did), `eval.log` (one line per rollout with its reward), and
`traces.jsonl` (every attempt in full).

Rollouts fail in real life — a dropped connection, a timeout, a bad key — and a
failed rollout appears as `ok: false` with an `errors` list and no sampled
messages. Your rewards score it 0.0 without crashing, because `last_reply` is
just the empty string. Check `stop_condition` before believing a 0.0 meant
"wrong answer."

## Labs

| file | what you build |
|---|---|
| `exercise_1_first_task.py` | data, task, rewards, and a trace to score them on |
| `exercise_2_weights.py` | two components, and where their ratio tips over |
| `exercise_3_reward_hacking.py` | **the important one.** Break a grader, then fix it. |

All three run offline and deterministic. The live run in section 3 is the
optional coda to lab 1.

## 4. About the reward-hacking lab

Let me set this one up properly, because it's the thing I'd most want you to
take away from the whole course.

You'll be handed a scoring method that looks completely reasonable:

```python
@vf.reward
async def sloppy_correctness(self, task, trace) -> float:
    return 1.0 if task.answer in trace.last_reply else 0.0
```

Read it in English: *"did the right answer appear anywhere in the reply?"* That
seems like a fine definition of correct. Versions of this ship in real
codebases.

It is badly broken, and you'll watch it give full marks to:

- a reply that says the answer is **not** 391,
- a reply that guesses five numbers at once,
- a reply that just lists every number from 380 to 400,
- a reply saying `3912` when the answer was `391` (because `"391" in "3912"`).

Why this matters more than a list of bugs: a scoring function that's
merely *noisy* is survivable — random errors average out over thousands of
examples. A scoring function that's **exploitable** does not average out,
because training is a search process and its entire job is finding the highest
score available.

Any hole you leave, it will find and drive through — because "list every number
from 380 to 400" genuinely *does* score higher than "carefully multiply 17 by
23," and it's much easier. The model isn't cheating. It's doing precisely what
you asked.

Trace what happens. One answer in a group stumbles onto the enumeration trick
and scores 1.0, while the careful answers score 0.0 on an arithmetic slip. From
Unit 01: that gives enumeration a **positive advantage**, so it gets reinforced
and sampled more often, so it wins more groups. A few hundred steps later you
have a model that has completely abandoned arithmetic — and a training curve
that looks like a triumph the entire time.

You wouldn't notice from the score. That's the whole problem. **The score is
the thing being hacked.**

Then you'll fix it, and your fix gets graded against twelve adversarial
replies. The broken version scores 5/12.

## 5. How you catch this in real life

- **Read the actual model outputs.** `traces.jsonl` holds every attempt in
  full. Not the average score — the text. Regularly.
- **Watch how long the replies get.** Score climbing while answers get longer
  and stranger is the classic signature.
- **Keep a separate test your scoring function can't touch**, and check the two
  agree.
- **Assume every scoring function you write has a hole in it**, because it
  does. The only question is whether you find it before training does.

## Optional reading

- The installed source: `verifiers/v1/task.py`, `taskset.py`, and `trace.py`.
  Three files, each readable in a sitting, and more accurate than any docs.
- The [lab-cookbook](https://github.com/PrimeIntellect-ai/lab-cookbook) guides
  01–02 — Prime Intellect's own walkthrough of the same ground.
- **"Concrete Problems in AI Safety"** (arXiv:1606.06565), section 3 only — the
  reward hacking section. Written years before any of this tooling existed and
  it predicts every failure you're about to see.

## Running

```bash
uv run python modules/02-first-environment/exercise_1_first_task.py
uv run python modules/02-first-environment/verify.py
```
