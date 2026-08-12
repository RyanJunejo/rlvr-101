"""
ANSWER KEY -- Unit 02, Lab 2 — weights, and where the judgment lives.

Lab 1 gave the format reward weight 0.2 and correctness weight 1.0. This lab is
about why that ratio -- and only the ratio -- is the design decision.

Recall from Unit 01 that GRPO normalizes scores within each group: subtract the
group's mean, divide by its spread. Multiply every weight by 10 and the
advantages come out bit-identical. So the absolute numbers mean nothing.
`(1.0, 0.2)` is not "one point and a fifth of a point" -- it is the claim that
correctness matters five times more than formatting.

Get the ratio wrong and you train a model that produces beautifully formatted
nonsense. Part 2 shows you the exact tipping point.

Fill in the two TODOs, then run me:

    uv run python modules/02-first-environment/exercise_2_weights.py

No API key needed.

TERMS USED IN THIS FILE

  weight:    how much one scoring component counts toward the total
  advantage: from Unit 01: reward minus the group's average. Scale-free, which
             is why only the RATIO between weights matters.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.trace import TraceTask
from verifiers.v1.utils.decorators import discover_decorated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

# `(\S.*?)` -- the answer must start with a non-whitespace character, so a line
# of "Answer:   " (nothing after the colon) doesn't count as answering.
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


class QuizData(vf.TaskData):
    answer: str


class QuizTask(vf.Task[QuizData, vf.State, vf.TaskConfig]):
    """Two reward components with an explicit ratio between them.

    Write both methods, async as always:

    1. `correct_answer` -- weight 1.0 (bare `@vf.reward`):
       the last ANSWER_RE match in `trace.last_reply`, stripped of whitespace
       and a trailing period, compared exactly to `task.answer`. No match, 0.0.

    2. `format_reward` -- weight 0.2 (`@vf.reward(weight=0.2)`):
       1.0 if ANSWER_RE matches at all. Note it takes only `trace` -- it never
       looks at the answer, and that narrowness is the point. Each component
       measures ONE thing, so when a score moves you know what moved.
    """

    @vf.reward
    async def correct_answer(self, task: QuizData, trace: vf.Trace) -> float:
        matches = ANSWER_RE.findall(trace.last_reply)
        if not matches:
            return 0.0
        return 1.0 if matches[-1].strip().rstrip(".").strip() == task.answer else 0.0

    @vf.reward(weight=0.2)
    async def format_reward(self, trace: vf.Trace) -> float:
        return 1.0 if ANSWER_RE.findall(trace.last_reply) else 0.0


def describe_rewards(task: vf.Task) -> list[tuple[str, float]]:
    """List (name, weight) for every reward method on a task.

    The decorator stores metadata on the method; `discover_decorated(task,
    "reward")` returns the marked methods. Each carries its weight as the
    attribute `reward_weight`.

        return [(fn.__name__, fn.reward_weight) for fn in ...]

    Sort by name so the output is stable.

    Args:
        task: any v1 task instance.
    Returns:
        [(method_name, weight)], sorted by name.
    """
    # `_vf_weight` is what the decorator actually sets (mark("reward", ...,
    # _vf_weight=weight) in verifiers/v1/utils/decorators.py). The underscore
    # marks it internal, so pin your verifiers version if you rely on it.
    return sorted(
        (fn.__name__, fn._vf_weight) for fn in discover_decorated(task, "reward")
    )


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------

# Three hypothetical policies, described by how often each component pays out.
# These stand in for what a model might drift toward during training.
PERSONAS = {
    "careful reasoner": {"correct": 0.80, "format": 0.90},
    "sloppy genius": {"correct": 0.95, "format": 0.40},
    "empty formatter": {"correct": 0.02, "format": 1.00},  # emits `Answer: 0` every time
}


def expected_reward(persona: dict, w_correct: float, w_format: float) -> float:
    return persona["correct"] * w_correct + persona["format"] * w_format


def make_trace(task: QuizTask, reply: str) -> vf.Trace:
    trace = vf.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="offline", trainable=False),
    )
    trace.nodes.append(
        vf.MessageNode(message=vf.AssistantMessage(content=reply), sampled=True)
    )
    return trace


def main() -> None:
    print(f"{BOLD}PART 1 — the components and their ratio{RESET}\n")
    task = QuizTask(QuizData(idx=0, prompt="q", answer="391"))
    for name, weight in describe_rewards(task):
        print(f"  {name:<16} weight {weight}")

    samples = [
        ("perfect", "17*23 = 391.\nAnswer: 391", "391"),
        ("right, no format", "The answer is 391.", "391"),
        ("wrong, well formatted", "Hmm.\nAnswer: 400", "391"),
        ("wrong and unformatted", "no idea", "391"),
    ]
    print(f"\n  {'reply':<24} {'correct':>8} {'format':>8} {'weighted':>9}")
    print("  " + "-" * 52)
    for name, text, ans in samples:
        t = QuizTask(QuizData(idx=0, prompt="q", answer=ans))
        tr = make_trace(t, text)
        asyncio.run(t.score(tr))
        r = tr.rewards
        c, f = r["correct_answer"].score, r["format_reward"].score
        total = sum(v.score * v.weight for v in r.values())
        print(f"  {name:<24} {c:>8.1f} {f:>8.1f} {total:>9.2f}")

    print(
        f"\n{DIM}  'right, no format' scores 0.20 instead of 0.00 -- the components can\n"
        f"  tell a model that can't multiply from one that ignored the output\n"
        f"  spec. Those are different problems with different fixes.{RESET}"
    )

    print(f"\n\n{BOLD}PART 2 — where the ratio breaks{RESET}\n")
    print("  Expected reward for three policies, under different weightings.")
    print("  'empty formatter' always emits `Answer: 0` and computes nothing.\n")
    print(f"  {'weights (correct, format)':<28} " + " ".join(f"{n:>17}" for n in PERSONAS))
    print("  " + "-" * (28 + 18 * len(PERSONAS)))
    for wc, wf in [(1.0, 0.2), (1.0, 0.5), (1.0, 1.0), (1.0, 3.0)]:
        scores = {n: expected_reward(p, wc, wf) for n, p in PERSONAS.items()}
        best = max(scores, key=scores.get)
        cells = [f"{scores[n]:>8.2f}{' <-- best' if n == best else '':<9}" for n in PERSONAS]
        print(f"  {f'({wc}, {wf})':<28} " + " ".join(cells))

    print(
        f"\n{DIM}  At (1.0, 0.2) the careful reasoner wins, as intended. By (1.0, 3.0)\n"
        f"  the empty formatter -- a policy that computes NOTHING -- is the best\n"
        f"  policy in the room. No code changed. One number moved, and the task\n"
        f"  was redefined.{RESET}\n"
    )
    print(
        "The rule: an auxiliary reward is a TIEBREAKER, never a target. Format is\n"
        "dangerous specifically because it's cheap (any model can satisfy it\n"
        "immediately) and reliable (it pays out every time). Early in training,\n"
        "when the model is still bad at the real task, format may be the only\n"
        "component producing score differences within a group -- so it dominates\n"
        "the advantage exactly when the model is most impressionable.\n"
        "\n"
        "Anything a model can saturate without doing the task is a magnet for\n"
        "probability mass. Lab 3 is that sentence, weaponized."
    )


if __name__ == "__main__":
    main()
