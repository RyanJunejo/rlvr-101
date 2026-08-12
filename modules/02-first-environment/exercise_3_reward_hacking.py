"""
Lab 3 — the reward hacking lab.

This is the most important exercise in the course.

Below is a scoring method that looks completely reasonable. Versions of it ship
in real codebases:

    @vf.reward
    async def sloppy_correctness(self, task, trace) -> float:
        return 1.0 if task.answer in trace.last_reply else 0.0

Read it in English: "did the right answer appear anywhere in the reply?" Run
this file and watch it hand full marks to a reply that says the answer is NOT
391, to one that guesses five numbers at once, and to one that just lists every
number from 380 to 400.

Why this matters more than a list of bugs: a NOISY scoring function is
survivable, because random errors average out over a batch. An EXPLOITABLE one
does not average out, because training is a search process whose entire job is
finding the highest score available. Every hole you leave, it finds -- "list
every number" really does score higher than "multiply carefully", and it's much
easier. The model isn't cheating. It's doing precisely what you asked.

Your job: write `robust_correctness` so it survives the attack suite. The
broken version scores 5/12. Yours must score 12/12.

No API key needed -- the attacks are canned, so this is deterministic and free.

    uv run python modules/02-first-environment/exercise_3_reward_hacking.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.trace import TraceTask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, GREEN, RED, RESET, run_main, todo  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


class AttackData(vf.TaskData):
    answer: str


class AttackTask(vf.Task[AttackData, vf.State, vf.TaskConfig]):
    """One task, two graders -- so each attack scores both in a single pass."""

    @vf.reward
    async def sloppy_correctness(self, task: AttackData, trace: vf.Trace) -> float:
        """The bug: substring containment is not equality. DO NOT FIX THIS ONE.

        It's the exhibit. The lab is watching it fail."""
        return 1.0 if task.answer in trace.last_reply else 0.0

    @vf.reward
    async def robust_correctness(self, task: AttackData, trace: vf.Trace) -> float:
        """Score 1.0 only when the model COMMITTED to exactly the right answer.

        The strategy, which is the lesson -- read it before coding:

          1. Only the answer SLOT counts. Take ANSWER_RE matches from
             `trace.last_reply`; never search prose for the answer. Prose is
             reasoning, and reasoning legitimately contains wrong intermediate
             values. Only the slot is a claim.

          2. The LAST match wins. Models restate the format mid-reasoning and
             correct themselves; the final commitment is the real one. Taking
             the first would fail honest self-correction -- a false negative,
             the worst grading bug, because it teaches the model that being
             right doesn't pay.

          3. Compare EXACTLY: strip whitespace and one trailing period, then
             `==`. "3912" must not match "391". `in` is almost never the right
             operator in a grader.

          4. No slot, no credit. A reply with no Answer: line never committed
             to anything. Return 0.0.

        Rule 3 kills most of the attacks. Rule 1 kills the rest.

        Args:
            task: carries the reference `answer`.
            trace: carries the model's reply in `last_reply`.
        Returns:
            1.0 or 0.0
        """
        # TODO: implement the four rules.
        return todo("robust_correctness: last Answer: line, compared exactly")


# ---------------------------------------------------------------------------
# The attack suite. Some cases are HONEST answers that must still score 1.0 --
# a "robust" grader that returns 0.0 for everything isn't robust, it's useless.
# ---------------------------------------------------------------------------

_ENUMERATION = "I'm not certain, so here are the possibilities: " + ", ".join(
    str(i) for i in range(380, 401)
)

ATTACKS: list[tuple[str, str, str, float]] = [
    # --- honest responses: these MUST still be rewarded ---
    ("honest correct", "17 * 23 = 391.\nAnswer: 391", "391", 1.0),
    ("extra whitespace", "Answer:   391  ", "391", 1.0),
    ("trailing period", "Let me work it out.\nAnswer: 391.", "391", 1.0),
    ("self-correction", "Answer: 12\nWait, that's wrong.\nAnswer: 391", "391", 1.0),
    # --- honest failure: correctly scored 0 by BOTH graders ---
    ("honest wrong", "I think it's 400.\nAnswer: 400", "391", 0.0),
    # --- the exploits: sloppy_correctness pays all of these ---
    ("superset number", "Answer: 3912", "391", 0.0),
    ("negation", "It is definitely not 391.\nAnswer: 400", "391", 0.0),
    ("shotgun guess", "Could be 389, 390, 391, or 392.\nAnswer: 389", "391", 0.0),
    ("enumerate everything", _ENUMERATION, "391", 0.0),
    ("right number, wrong answer", "391 looks plausible but no.\nAnswer: 12", "391", 0.0),
    ("answer field shotgun", "Answer: 391, 392, 393", "391", 0.0),
    ("mentions in prose only", "The product of 17 and 23 is 391.", "391", 0.0),
]


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def make_trace(task: AttackTask, reply: str) -> vf.Trace:
    trace = vf.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="offline", trainable=False),
    )
    trace.nodes.append(
        vf.MessageNode(message=vf.AssistantMessage(content=reply), sampled=True)
    )
    return trace


def _cell(val: float, expected: float, width: int = 8) -> str:
    """Right-align the number, THEN colorize -- padding a string that already
    contains ANSI escapes counts them as visible width and misaligns."""
    color = GREEN if abs(val - expected) < 1e-9 else RED
    return f"{color}{f'{val:.1f}'.rjust(width)}{RESET}"


def run_suite(method_name: str) -> tuple[int, list[str]]:
    """Score every attack with one grader; return (n_correct, names_it_got_wrong)."""
    correct, wrong = 0, []
    for name, reply, answer, expected in ATTACKS:
        task = AttackTask(AttackData(idx=0, prompt="q", answer=answer))
        trace = make_trace(task, reply)
        got = asyncio.run(getattr(task, method_name)(task.data, trace))
        if abs(float(got) - expected) < 1e-9:
            correct += 1
        else:
            wrong.append(name)
    return correct, wrong


def main() -> None:
    print(f"{BOLD}Scoring the attack suite with BOTH graders{RESET}\n")
    header = f"{'attack':<28} {'should be':>10} {'sloppy':>8} {'robust':>8}"
    print(header)
    print("-" * len(header))

    robust_works = True
    for name, reply, answer, expected in ATTACKS:
        task = AttackTask(AttackData(idx=0, prompt="q", answer=answer))
        trace = make_trace(task, reply)
        s = asyncio.run(task.sloppy_correctness(task.data, trace))
        try:
            r_col = _cell(float(asyncio.run(task.robust_correctness(task.data, trace))), expected)
        except Exception:
            r_col, robust_works = f"{DIM}{'TODO'.rjust(8)}{RESET}", False
        print(f"{name:<28} {expected:>10.1f} {_cell(s, expected)} {r_col}")

    n = len(ATTACKS)
    s_correct, s_wrong = run_suite("sloppy_correctness")
    print(f"\n{BOLD}sloppy_correctness: {s_correct}/{n}{RESET}")
    print(f"  {DIM}fooled by: {', '.join(s_wrong)}{RESET}")
    if robust_works:
        r_correct, r_wrong = run_suite("robust_correctness")
        print(f"{BOLD}robust_correctness: {r_correct}/{n}{RESET}")
        print(f"  {DIM}{'still fooled by: ' + ', '.join(r_wrong) if r_wrong else 'survives every attack.'}{RESET}")
    else:
        print(f"{DIM}robust_correctness: not implemented yet.{RESET}")

    print(
        f"\n{BOLD}Now the part that should actually worry you.{RESET}\n"
        "\n"
        "Look at 'enumerate everything'. Under the sloppy grader, listing every\n"
        "number from 380 to 400 scores 1.0 -- same as computing the answer, and\n"
        "far easier. Recall how GRPO works (Unit 01): advantage comes from score\n"
        "DIFFERENCES within a group. The moment one sampled reply stumbles onto\n"
        "enumeration and scores 1.0 while careful replies score 0.0 on an\n"
        "arithmetic slip, enumeration gets a positive advantage. Its probability\n"
        "rises. It wins more groups. A few hundred steps later the model has\n"
        "abandoned arithmetic -- and the reward curve looked like a triumph the\n"
        "whole time, because the reward curve is the thing being hacked.\n"
        "\n"
        "How you catch it in practice:\n"
        "  - READ THE ROLLOUTS. traces.jsonl, actual text, regularly. Everyone\n"
        "    says this; almost nobody does it.\n"
        "  - Watch completion length. Score climbing while replies grow longer\n"
        "    and stranger is the classic signature.\n"
        "  - Hold out an eval your grader can't touch, and check they agree.\n"
        "  - Assume every grader you write has a hole, because it does. The only\n"
        "    question is whether you find it before the optimizer does."
    )


if __name__ == "__main__":
    run_main(main)
