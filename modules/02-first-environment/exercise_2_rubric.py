"""
Exercise 2 — rubrics, and why the weights are the whole ballgame.

Exercise 1 had one reward function. Real rubrics have several, and combining
them is where the judgment lives.

The motivating problem, straight from exercise 3's punchline: if correctness
requires a well-formed `Answer:` line, then a model that computes perfectly but
forgets the format scores identically to one that can't multiply. Your metric
has stopped distinguishing "can't do the task" from "won't follow instructions",
which are completely different problems with completely different fixes.

So: split them. One reward for being right, one for being well-formed, combined
with weights.

    rubric = vf.Rubric(funcs=[correct_answer, format_reward], weights=[1.0, 0.2])

Final reward is the weighted sum, and each component is *also* reported
separately as a metric under its function's `__name__` -- which is how you debug.
When reward moves you can see which part moved.

Now, recall from Module 01 that GRPO normalizes advantages within each group.
Multiply every weight by 10 and literally nothing changes. So the absolute
numbers are meaningless and the RATIO is everything. `[1.0, 0.2]` is a claim
that correctness matters five times more than formatting.

Get that ratio wrong and you build a machine for producing beautifully formatted
nonsense. Part 2 of this file shows you exactly where the tipping point is.

Fill in the TODOs, then run me:

    uv run python modules/02-first-environment/exercise_2_rubric.py

No API key required.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import verifiers as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# `(\S.*?)` -- the answer must START with a non-whitespace char. With `(.+?)`
# a line of "Answer:   " matches, capturing a single space as the "answer".
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def format_reward(completion, **kwargs) -> float:
    """1.0 if the reply contains a well-formed answer line, else 0.0.

    "Well-formed" here means: at least one line matching `Answer: <something>`,
    with something non-empty after the colon. Use ANSWER_RE above -- it's already
    written, anchored to line starts, case-insensitive.

    Note this says NOTHING about whether the answer is right. That's deliberate
    and it's the entire point of the exercise: this component measures
    instruction-following, and only that.

    Args:
        completion: list of chat messages.
    Returns:
        1.0 or 0.0
    """
    # TODO: check the reply for a well-formed Answer: line.
    return todo("format_reward: 1.0 if ANSWER_RE finds a non-empty answer line")


def correct_answer(completion, answer, **kwargs) -> float:
    """1.0 if the final `Answer:` line matches `answer` exactly.

    Same idea as exercise 1: take the LAST match from ANSWER_RE (models correct
    themselves mid-reply), strip whitespace and a single trailing period, and
    compare with `==`. No match at all means 0.0.

    Args:
        completion: list of chat messages.
        answer: reference answer string.
    Returns:
        1.0 or 0.0
    """
    # TODO: extract the last Answer: line and compare exactly to `answer`.
    return todo("correct_answer: exact match on the final Answer: line")


def build_rubric() -> vf.Rubric:
    """Combine the two reward functions.

    Use `vf.Rubric(funcs=[...], weights=[...])` with `correct_answer` first and
    `format_reward` second, weighted 1.0 and 0.2 respectively -- correctness
    matters five times more than formatting.

    Returns:
        a configured vf.Rubric
    """
    # TODO: build and return the Rubric.
    return todo("build_rubric: Rubric(funcs=[correct_answer, format_reward], weights=[1.0, 0.2])")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------

# Three hypothetical policies, described by how often they get each component
# right. These stand in for what a model might drift toward during training.
PERSONAS = {
    "careful reasoner": {"correct": 0.80, "format": 0.90},
    "sloppy genius": {"correct": 0.95, "format": 0.40},
    "empty formatter": {"correct": 0.02, "format": 1.00},  # emits `Answer: 0` every time
}


def expected_reward(persona: dict, w_correct: float, w_format: float) -> float:
    return persona["correct"] * w_correct + persona["format"] * w_format


def main() -> None:
    print(f"{BOLD}PART 1 — the rubric works{RESET}\n")
    rubric = build_rubric()
    print(f"  functions: {[f.__name__ for f in rubric.funcs]}")
    print(f"  weights:   {rubric.weights}")

    samples = [
        ("perfect", "17*23 = 391.\nAnswer: 391", "391"),
        ("right, no format", "The answer is 391.", "391"),
        ("wrong, well formatted", "Hmm.\nAnswer: 400", "391"),
        ("wrong and unformatted", "no idea", "391"),
    ]
    print(f"\n  {'reply':<24} {'correct':>8} {'format':>8} {'total':>8}")
    print("  " + "-" * 50)
    for name, text, ans in samples:
        c = correct_answer([{"role": "assistant", "content": text}], ans)
        f = format_reward([{"role": "assistant", "content": text}])
        print(f"  {name:<24} {c:>8.1f} {f:>8.1f} {c * 1.0 + f * 0.2:>8.2f}")

    print(
        f"\n  {DIM}Note 'right, no format' now scores 0.20 instead of 0.00. The rubric can\n"
        f"  tell the difference between a model that can't multiply and one that just\n"
        f"  ignored your output spec. Those need different fixes.{RESET}"
    )

    print(f"\n\n{BOLD}PART 2 — where the ratio breaks{RESET}\n")
    print("  Expected reward for three policies, under different weightings.")
    print("  'empty formatter' always emits `Answer: 0` and never computes anything.\n")

    settings = [(1.0, 0.2), (1.0, 0.5), (1.0, 1.0), (1.0, 3.0)]
    print(f"  {'weights (correct, format)':<28} " + " ".join(f"{n:>17}" for n in PERSONAS))
    print("  " + "-" * (28 + 18 * len(PERSONAS)))
    for wc, wf in settings:
        scores = {n: expected_reward(p, wc, wf) for n, p in PERSONAS.items()}
        best = max(scores, key=scores.get)
        cells = []
        for n in PERSONAS:
            tag = " <-- best" if n == best else ""
            cells.append(f"{scores[n]:>8.2f}{tag:<9}")
        print(f"  {f'({wc}, {wf})':<28} " + " ".join(cells))

    print(
        f"\n  {DIM}At (1.0, 0.2) the careful reasoner wins, as intended.\n"
        f"  By (1.0, 3.0) the empty formatter -- a policy that computes NOTHING --\n"
        f"  is the highest-reward policy in the room. Nothing about the code changed.\n"
        f"  You moved one number and redefined the task.{RESET}\n"
    )
    print(
        "The practical rule: an auxiliary reward should be a TIEBREAKER, not a\n"
        "target. If a component can be maxed out without doing the actual task, its\n"
        "weight has to stay small enough that it can never outrank the real thing.\n"
        "\n"
        "And notice what makes format so dangerous as a component: it's CHEAP and\n"
        "RELIABLE. Every policy can get 1.0 on it immediately. Anything the model can\n"
        "saturate without effort is a magnet for probability mass -- which is exactly\n"
        "what exercise 3 is about. Go do it now."
    )


if __name__ == "__main__":
    run_main(main)
