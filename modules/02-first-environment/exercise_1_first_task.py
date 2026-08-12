"""
Lab 1 — your first task.

You're going to build the three pieces of a scoring setup: the DATA for one
question, the TASK that scores an attempt, and a TRACE -- the record of one
attempt -- to score. All offline, no API key.

At the bottom of the file (written for you) the same pieces are bundled into a
TASKSET, which is what makes them runnable against a live model with one shell
command. You'll write your own taskset in Unit 04; today you just get to see it
work.

THE THREE GOTCHAS, up front, because each one costs twenty minutes cold:

  1. Reward methods MUST be `async def`. Forget the async and scoring raises
     "An asyncio.Future, a coroutine or an awaitable is required" -- which does
     not read like "you forgot async", but that's what it means.

  2. When you build a trace by hand, the model's message needs `sampled=True`.
     Without it, `trace.last_reply` is the empty string and every reward
     silently returns 0.0. No error. Nothing.

  3. `import verifiers.v1 as vf` -- note the `.v1`. Plain `import verifiers`
     gets you the legacy API (Unit 06), and the two don't mix.

Fill in the three TODOs, then run me:

    uv run python modules/02-first-environment/exercise_1_first_task.py
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.trace import TraceTask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# The answer slot, same discipline as it will be all course: the model commits
# on a designated line, and only that line counts.
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "You are a careful calculator. Think briefly, then end your reply with the "
    "final answer on its own last line, in exactly this format:\n"
    "Answer: <number>"
)


class MathData(vf.TaskData):
    """The data for ONE question.

    Subclass `vf.TaskData` and declare one field, pydantic-style:

        answer: str

    You inherit `prompt`, `system_prompt`, `name`, and `idx` from the base
    class -- don't redeclare those. The base is frozen (immutable), and that's
    deliberate: this object is serialized verbatim into the trace file, so what
    you read in traces.jsonl afterwards is exactly what the task was built from.
    """

    # TODO: declare the `answer` field.


class MathTask(vf.Task[MathData, vf.State, vf.TaskConfig]):
    """The behavior: how a finished attempt gets scored.

    Write two reward methods. BOTH must be `async def`.

    1. `correct_answer`, default weight (1.0):

           @vf.reward
           async def correct_answer(self, task: MathData, trace: vf.Trace) -> float:

       Find every ANSWER_RE match in `trace.last_reply`. No match -> 0.0.
       Otherwise compare the LAST match -- stripped of whitespace and a
       trailing period -- to `task.answer`, exactly. (Last, because models
       restate the format while thinking and correct themselves; the final
       commitment is the real one.)

    2. `has_answer_line`, weight 0.2:

           @vf.reward(weight=0.2)
           async def has_answer_line(self, trace: vf.Trace) -> float:

       1.0 if ANSWER_RE matches anything at all, else 0.0. Deliberately says
       nothing about correctness -- it measures instruction-following, so you
       can tell "can't multiply" apart from "ignored the output format". Lab 2
       is about why they must stay separate.

    HOW ARGUMENTS WORK: the library inspects each method's parameter names and
    passes what you asked for -- `task` (your MathData), `trace` (the attempt),
    or `runtime`. Ask for exactly what you use, nothing more.

    WHERE WEIGHTS LIVE: on the decorator, attached to the method they weight.
    There is no separate weights list to keep aligned by position.
    """

    # TODO: write the two async reward methods.


def make_trace(task: MathTask, reply: str) -> vf.Trace:
    """Build the record of one (pretend) attempt, so we can score offline.

    In a real run the eval CLI builds traces for you. Building one by hand is
    how you test scoring without an API key, and it's three steps:

        trace = vf.Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(), name="offline",
                               trainable=False),
        )
        trace.nodes.append(
            vf.MessageNode(message=vf.AssistantMessage(content=reply),
                           sampled=True)
        )
        return trace

    DO NOT DROP `sampled=True`. It marks the message as something the model
    produced, as opposed to something that arrived with the prompt (a few-shot
    example, a conversation being continued). `trace.last_reply` reads only
    sampled messages -- scoring prompt-supplied text as the model's own work
    would inflate every number you report, so the library refuses to.

    Args:
        task: the task being attempted.
        reply: what the model supposedly said.
    Returns:
        a Trace ready to score.
    """
    # TODO: build and return the trace.
    return todo("make_trace: Trace(task=..., agent=...), append a sampled node")


# ---------------------------------------------------------------------------
# Written for you: the taskset that makes this runnable against a live model.
# You'll build one of these yourself in Unit 04.
# ---------------------------------------------------------------------------


class MathTaskset(vf.Taskset[MathTask, vf.TasksetConfig]):
    """Eight two-digit multiplications, seeded so every run sees the same ones."""

    def load(self) -> Iterator[MathTask]:
        rng = random.Random(0)
        for i in range(8):
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            yield MathTask(
                MathData(
                    idx=i,
                    prompt=f"What is {a} * {b}?",
                    system_prompt=SYSTEM_PROMPT,
                    answer=str(a * b),
                ),
                self.config.task,
            )


# The packaging contract, one line: a module exports exactly one Taskset
# subclass via __all__, and the eval CLI can find it by module name.
__all__ = ["MathTaskset"]


REPLIES = [
    ("Answer: 391", "391"),
    ("17 * 23 = 391.\nAnswer: 391.", "391"),
    ("Answer: 3912", "391"),
    ("The product of 17 and 23 is 391.", "391"),
    ("no idea", "391"),
]


def score(task: MathTask, reply: str) -> dict:
    trace = make_trace(task, reply)
    asyncio.run(task.score(trace))
    return dict(trace.rewards)


def main() -> None:
    print(f"{BOLD}1. The pieces{RESET}\n")
    data = MathData(idx=0, prompt="What is 17 * 23?", system_prompt=SYSTEM_PROMPT, answer="391")
    task = MathTask(data)
    print(f"  data:   MathData(prompt={data.prompt!r}, answer={data.answer!r})")
    print(f"  frozen: {type(data).model_config.get('frozen')}")
    print(f"  task:   {type(task).__name__}")

    print(f"\n{BOLD}2. Scoring five replies{RESET}\n")
    print(f"  {'reply':<32} {'correct':>8} {'format':>8}")
    print("  " + "-" * 50)
    for reply, answer in REPLIES:
        t = MathTask(MathData(idx=0, prompt="q", answer=answer))
        r = score(t, reply)
        shown = reply.replace("\n", " / ")[:32]
        print(f"  {shown:<32} {r['correct_answer'].score:>8.1f} "
              f"{r['has_answer_line'].score:>8.1f}")
    print(
        f"\n{DIM}  Row 3 is the one to notice: wrong answer, right format. Because the\n"
        f"  components report separately, you can see WHICH one moved -- the\n"
        f"  difference between 'can't multiply' and 'ignored the format'.{RESET}"
    )

    print(f"\n{BOLD}3. What a trace holds{RESET}\n")
    t = MathTask(MathData(idx=0, prompt="q", answer="391"))
    tr = make_trace(t, "Answer: 391")
    asyncio.run(t.score(tr))
    print(f"  last_reply: {tr.last_reply!r}")
    print(f"  nodes:      {len(tr.nodes)}   sampled messages: {len(tr.assistant_messages)}")
    print(f"  rewards:    {{k: v.score for ...}} = "
          f"{ {k: v.score for k, v in tr.rewards.items()} }")

    print(
        f"\n{BOLD}4. Run it against a real model{RESET}\n"
        "\n"
        "  This file exports a taskset via __all__, which makes it runnable with\n"
        "  the eval CLI. From the repo root, with your .env filled in:\n"
        "\n"
        "    export $(grep -vE '^#|^$' .env | xargs)\n"
        "    PYTHONPATH=modules/02-first-environment uv run eval exercise_1_first_task \\\n"
        "      -m \"$MODEL\" -n 4 -r 2 --rich False \\\n"
        "      --client.base-url \"$OPENAI_BASE_URL\" --client.api-key-var OPENAI_API_KEY \\\n"
        "      --env.agent.harness.id null -o outputs/first-task\n"
        "\n"
        "  The export line matters: the key must be in the eval process's\n"
        "  ENVIRONMENT, not just your shell -- api-key-var names a variable, so\n"
        "  your secret never appears in a config file.\n"
        "\n"
        "  -r 2 asks each question twice. The CLI's own help calls that 'the\n"
        "  trainer's group size' -- it is exactly the G from Unit 01.\n"
        "\n"
        "  Results land in outputs/first-task/: config.toml (the resolved run),\n"
        "  eval.log (one line per rollout, with its reward), and traces.jsonl\n"
        "  (every attempt in full -- and reading these is a habit worth building\n"
        "  before Unit 05 gives you expensive reasons to have it)."
    )


if __name__ == "__main__":
    run_main(main)
