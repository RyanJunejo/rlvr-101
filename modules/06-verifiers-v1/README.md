# Unit 06 — The redesigned API

### Lecture notes

> **Time:** 2–3 hours · **Prerequisites:** Unit 02 · **Needs:** nothing (runs
> offline)
>
> **By the end of this unit you will be able to:**
> 1. Say what problem the v1 redesign solves, in terms of a task you couldn't
>    express in v0.
> 2. Write a task with `TaskData`, `Task`, and `async` reward methods.
> 3. Build a `Taskset` that yields tasks lazily.
> 4. Read a `Trace` and know where the model's reply actually lives.
> 5. Decide which API to use for a given piece of work.
>
> **Deliverables:** 2 labs, autograder green, problem set in `NOTES.md`.

---

## 1. A task you can't write in v0

You built a math task in Unit 02. Now suppose you want to know how a *coding
agent* does on it. Not a chat model answering in one turn, but something that
runs for forty turns with a terminal, writes a Python script, executes it, and
reports back.

Try to express that with `SingleTurnEnv`. You can't. Your environment holds three
different things at once:

- the questions and answers,
- the scoring,
- the assumption that the model replies in one turn.

That third one is baked into the class you chose. Swapping in a forty-turn agent
means rewriting the environment, which means the "same" task scored two ways
isn't the same task anymore, and the numbers aren't comparable.

That's the problem v1 solves. It splits the environment into pieces that vary
independently:

```
   taskset          harness            runtime
   ───────          ───────            ───────
   the work:        the solver:        where it runs:
   data + scoring   how the model      local, sandbox,
                    attempts it        container
```

Any taskset can run under any compatible harness. <!-- prose-ok: "harness" is the v1 API's own term (verifiers/v1/harness.py) -->
You define "answer this math question correctly" once, then score a chat model, a
ReAct loop, and a CLI agent against the identical definition.

## 2. Both APIs ship in the same package

`verifiers` 0.3.0 contains v0 and v1 side by side:

```python
import verifiers as vf        # v0 — SingleTurnEnv, Rubric, Parser
import verifiers.v1 as vf     # v1 — Task, Taskset, Trace
```

Two CLIs, too: `vf-eval` for v0, `eval` for v1.

So this isn't a migration you're forced into. Units 02 through 04 used v0
because that's what `prime-rl` trains against today and what most of the
Environments Hub is written in. v1 is where the design is going, and knowing
both tells you which to reach for.

## 3. The three pieces, with code

### TaskData — the data for one task instance

A frozen pydantic model. Subclass it and add whatever your task needs:

```python
class MathData(vf.TaskData):
    answer: str
```

You inherit `prompt`, `system_prompt`, `name`, `idx`, and some container-related
fields. You add `answer`.

Frozen means immutable, which matters because this object gets serialized into
the trace file. What you see in `traces.jsonl` afterward is exactly what the task
was initialized with.

### Task — the behavior, including scoring

```python
class MathTask(vf.Task[MathData, vf.State, vf.TaskConfig]):
    @vf.reward
    async def correct_answer(self, task: MathData, trace: vf.Trace) -> float:
        matches = ANSWER_RE.findall(trace.last_reply)
        if not matches:
            return 0.0
        return 1.0 if matches[-1].strip() == task.answer else 0.0
```

Three things differ from the v0 reward functions you wrote in Unit 02:

**They're methods on the task, not loose functions.** So they can read
`self.config` and `task.answer` without anything being threaded through.

**They must be `async`.** A synchronous reward method raises
`TypeError: An asyncio.Future, a coroutine or an awaitable is required` at
scoring time, which is not an obvious error message for the mistake. This is the
single most likely thing to trip you up in lab 1.

**Weights live on the decorator**, rather than in a separate parallel list:

```python
@vf.reward(weight=0.2)
async def has_answer_line(self, trace: vf.Trace) -> float:
    ...
```

In v0 you wrote `Rubric(funcs=[a, b], weights=[1.0, 0.2])` and had to keep the
two lists aligned by position. Here the weight sits on the function it belongs
to.

Argument injection works as it did in v0 — ask for `task`, `trace`, or `runtime`
by name and you get them.

### Taskset — where tasks come from

```python
class MathTaskset(vf.Taskset[MathTask, vf.TasksetConfig]):
    def load(self):
        for i, (question, answer) in enumerate(PROBLEMS):
            yield MathTask(MathData(idx=i, prompt=question, answer=answer))
```

`load()` is a generator, so tasks are built as they're consumed. A taskset can be
infinite (set `INFINITE = True`), and `.head(n)` gives you a finite view of the
first `n`.

## 4. The Trace, and where the reply hides

The `Trace` is the record of one attempt: every message, every tool call, the
rewards, the timing. Rewards read from it.

Getting the model's text is `trace.last_reply`. There's one detail here that will
cost you twenty minutes if you meet it cold.

A trace's `nodes` list holds messages, and each node carries a `sampled` flag.
`sampled=True` means the model produced it; `sampled=False` means it came in with
the prompt. `last_reply` only looks at sampled nodes.

That distinction exists because a prompt can legitimately contain assistant
messages — few-shot examples, a partially completed conversation you're asking
the model to continue. Scoring those as if the model said them would be a grading
bug in exactly the direction that inflates your numbers.

When you construct a trace by hand for testing, you have to set the flag
yourself. Forget it and every reward silently returns 0.0, because `last_reply`
is the empty string.

Here is what scoring actually produced when I ran the lab 1 answer key:

```
reply                       correct_answer          has_answer_line
'Answer: 391'               score=1.0 weight=1.0    score=1.0 weight=0.2
'Answer: 3912'              score=0.0 weight=1.0    score=1.0 weight=0.2
'The product is 391.'       score=0.0 weight=1.0    score=0.0 weight=0.2
'no idea'                   score=0.0 weight=1.0    score=0.0 weight=0.2
```

Note the second row: wrong answer, correct format. Both components reported
separately, so you can see which one moved. Same debugging property as v0's
metrics, expressed differently.

## 5. Which one should you use?

Use **v0** when you want to train with `prime-rl` today, publish something most
people can install, or start from an existing Hub environment.

Use **v1** when the task involves an agent rather than a single reply, when you
want to score the same task under several different solvers, or when you're
writing something new and want to be on the current design.

The honest summary: v1 is better designed and less battle-tested. The
documentation lags the code by enough that I read the installed source rather
than the docs to write this unit, and found three things that don't match what's
published. If that bothers you, stay on v0 until it settles.

## Labs

| file | what you build |
|---|---|
| `exercise_1_port.py` | port the Unit 02 math task to v1 |
| `exercise_2_taskset.py` | a taskset, and reading traces |

Both run offline.

## How to work

```bash
uv run python modules/06-verifiers-v1/exercise_1_port.py
uv run python modules/06-verifiers-v1/verify.py
```

Real output with real numbers is in [`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md).

## Checkpoint

You should be able to say what v1 buys you: the task definition no longer
contains an assumption about how the model attempts it, so one taskset can be
scored across many different solvers and the numbers stay comparable.
