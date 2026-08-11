"""
Lab 1 — port the Unit 02 math task to v1.

Same task, same scoring rules, different API. Doing it as a port rather than
something new keeps the comparison honest: everything that changes is the API,
not the problem.

WHAT MAPS TO WHAT:

    v0                                  v1
    ------------------------------      ------------------------------
    Dataset with question/answer   ->   a TaskData subclass per row
    SingleTurnEnv(...)             ->   a Task subclass
    def reward(completion, answer) ->   async method with @vf.reward
    Rubric(funcs=[...],            ->   @vf.reward(weight=...) on each
           weights=[...])               method

THE ONE THAT WILL BITE YOU: reward methods must be `async`. A synchronous one
raises "An asyncio.Future, a coroutine or an awaitable is required" during
scoring, which does not obviously mean "add async".

No API key needed.

Fill in the three TODOs, then run me:

    uv run python modules/06-verifiers-v1/exercise_1_port.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers.v1 as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "You are a careful calculator. Think briefly, then end your reply with the "
    "final answer on its own last line, in exactly this format:\n"
    "Answer: <number>"
)

PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 144 / 12?", "12"),
    ("What is 89 + 156?", "245"),
    ("What is 1000 - 377?", "623"),
]


class MathData(vf.TaskData):
    """The data for one math question.

    Subclass `vf.TaskData` and add ONE field: `answer`, a string.

    You inherit `prompt`, `system_prompt`, `name` and `idx` from the base class,
    so you don't redeclare those. The base is a frozen pydantic model, which
    means instances are immutable and get serialized verbatim into the trace
    file. What you see in traces.jsonl afterwards is exactly what went in.

    Declare it the normal pydantic way:

        answer: str
    """

    # TODO: add the `answer` field.


class MathTask(vf.Task[MathData, vf.State, vf.TaskConfig]):
    """The task's behavior: how a finished attempt gets scored.

    You write two reward methods. BOTH MUST BE `async`.

    1. `correct_answer`, default weight (1.0):

           @vf.reward
           async def correct_answer(self, task: MathData, trace: vf.Trace) -> float:

       Find every match of ANSWER_RE in `trace.last_reply`. No match means 0.0.
       Otherwise compare the LAST match, stripped of whitespace and a trailing
       period, to `task.answer`. Exactly. Same rules as Unit 02.

    2. `has_answer_line`, weight 0.2:

           @vf.reward(weight=0.2)
           async def has_answer_line(self, trace: vf.Trace) -> float:

       1.0 if ANSWER_RE matches anything at all, 0.0 otherwise. Says nothing
       about correctness -- it's the Unit 02 format component, so you can tell
       "can't multiply" apart from "ignored the output spec".

    ON ARGUMENTS: injected by name, as in v0. Ask for `task` and you get the
    MathData; ask for `trace` and you get the Trace. Ask for neither and you get
    neither.

    ON WEIGHTS: in v0 you kept `funcs=[...]` and `weights=[...]` aligned by
    position. Here the weight rides on the method it belongs to, so they can't
    drift apart.
    """

    # TODO: write the two async reward methods described above.


def make_trace(task: MathTask, reply: str) -> vf.Trace:
    """Build a Trace by hand containing one model reply, so we can score offline.

    A real rollout builds this for you. We're faking one so the lab needs no API
    key and stays deterministic.

    Three pieces:

        tr = vf.Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(), name="offline",
                               trainable=False),
        )
        tr.nodes.append(
            vf.MessageNode(message=vf.AssistantMessage(content=reply),
                           sampled=True)
        )
        return tr

    `TraceTask` is imported for you below.

    DO NOT SKIP `sampled=True`. Nodes carry a flag saying whether the model
    produced the message or whether it arrived with the prompt. `last_reply`
    only reads sampled nodes -- so without the flag, `trace.last_reply` is the
    empty string and every reward silently returns 0.0.

    The flag exists because prompts can legitimately contain assistant messages
    (few-shot examples, a conversation you're asking the model to continue).
    Scoring those as the model's own work would inflate every number you report.

    Args:
        task: the task being attempted.
        reply: what the model supposedly said.
    Returns:
        a Trace ready to be scored.
    """
    # TODO: build and return the trace.
    return todo("make_trace: build a Trace, append a sampled AssistantMessage node")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------

from verifiers.v1.trace import TraceTask  # noqa: E402

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
    print(f"{BOLD}1. The task object{RESET}\n")
    data = MathData(idx=0, prompt=PROBLEMS[0][0], system_prompt=SYSTEM_PROMPT, answer="391")
    task = MathTask(data)
    print(f"  data:   {type(data).__name__}(answer={data.answer!r}, prompt={data.prompt!r})")
    print(f"  task:   {type(task).__name__}")
    print(f"  frozen: {type(data).model_config.get('frozen')}")

    print(f"\n{BOLD}2. Scoring some replies{RESET}\n")
    print(f"  {'reply':<30} {'correct':>9} {'format':>9} {'weighted':>10}")
    print("  " + "-" * 62)
    for reply, answer in REPLIES:
        t = MathTask(MathData(idx=0, prompt="q", answer=answer))
        r = score(t, reply)
        c = r["correct_answer"].score
        f = r["has_answer_line"].score
        total = c * r["correct_answer"].weight + f * r["has_answer_line"].weight
        shown = reply.replace("\n", " / ")
        print(f"  {shown[:30]:<30} {c:>9.1f} {f:>9.1f} {total:>10.2f}")

    print(
        f"\n{DIM}  Look at row 3: wrong answer, correct format. The two components are\n"
        f"  reported separately, so when a training run's score moves you can see\n"
        f"  which part moved. Same debugging property as v0's metrics.{RESET}"
    )

    print(f"\n{BOLD}3. What a Trace holds{RESET}\n")
    t = MathTask(MathData(idx=0, prompt="q", answer="391"))
    tr = make_trace(t, "Answer: 391")
    asyncio.run(t.score(tr))
    print(f"  last_reply:          {tr.last_reply!r}")
    print(f"  assistant_messages:  {len(tr.assistant_messages)}")
    print(f"  nodes:               {len(tr.nodes)}")
    print(f"  rewards:             {dict(tr.rewards)}")

    print(
        f"\n{DIM}  Try setting sampled=False in make_trace and re-running. Every reward\n"
        f"  drops to 0.0 and nothing errors. That silent failure is the reason the\n"
        f"  flag gets its own paragraph in the lecture notes.{RESET}"
    )


if __name__ == "__main__":
    run_main(main)
