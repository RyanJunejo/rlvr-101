"""
ANSWER KEY — Exercise 2, rubrics and weights.

    uv run python solutions/02-first-environment/exercise_2_rubric.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import verifiers as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

# `(\S.*?)` -- the answer must START with a non-whitespace char. With `(.+?)`
# a line of "Answer:   " matches, capturing a single space as the "answer".
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def reply_text(completion) -> str:
    """The model's final message, or "" if the rollout produced nothing.

    Rollouts fail. An API connection drops, a request times out, the model hits
    a content filter -- and `completion` comes back as an empty list. Indexing
    `completion[-1]` then raises IndexError from inside your reward function,
    which verifiers reports as "Error calling reward function" and scores as a
    failure anyway. You lose the real reason.

    Returning "" instead means a failed rollout scores 0.0 cleanly, which is
    what you want: it didn't answer, so it doesn't get credit.
    """
    return completion[-1]["content"] if completion else ""


def format_reward(completion, **kwargs) -> float:
    """Instruction-following only. Says nothing about correctness.

    Note it doesn't ask for `answer` at all -- verifiers injects arguments by
    name, so a function that only needs the completion simply doesn't request
    the rest. Keeping components narrow like this is what makes the metrics
    readable when something moves.
    """
    text = reply_text(completion)
    return 1.0 if ANSWER_RE.findall(text) else 0.0


def correct_answer(completion, answer, **kwargs) -> float:
    """Exact match on the final answer slot."""
    matches = ANSWER_RE.findall(reply_text(completion))
    if not matches:
        return 0.0
    return 1.0 if matches[-1].strip().rstrip(".").strip() == str(answer).strip() else 0.0


def build_rubric() -> vf.Rubric:
    """Weighted sum. The ratio is the design decision; the levels are noise."""
    return vf.Rubric(funcs=[correct_answer, format_reward], weights=[1.0, 0.2])


# --- harness ----------------------------------------------------------------

PERSONAS = {
    "careful reasoner": {"correct": 0.80, "format": 0.90},
    "sloppy genius": {"correct": 0.95, "format": 0.40},
    "empty formatter": {"correct": 0.02, "format": 1.00},
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

    print(f"\n\n{BOLD}PART 2 — where the ratio breaks{RESET}\n")
    settings = [(1.0, 0.2), (1.0, 0.5), (1.0, 1.0), (1.0, 3.0)]
    print(f"  {'weights (correct, format)':<28} " + " ".join(f"{n:>17}" for n in PERSONAS))
    print("  " + "-" * (28 + 18 * len(PERSONAS)))
    for wc, wf in settings:
        scores = {n: expected_reward(p, wc, wf) for n, p in PERSONAS.items()}
        best = max(scores, key=scores.get)
        cells = [f"{scores[n]:>8.2f}{' <-- best' if n == best else '':<9}" for n in PERSONAS]
        print(f"  {f'({wc}, {wf})':<28} " + " ".join(cells))

    print(
        f"\n{BOLD}DISCUSSION{RESET}\n"
        "\n"
        "  WHY THE LEVELS DON'T MATTER AND THE RATIO DOES. GRPO z-scores rewards within\n"
        "  each group (Module 01, exercise 4), so scaling every weight by 10 leaves\n"
        "  advantages bit-identical. Weight tuning is ALWAYS relative. If you find\n"
        "  yourself asking 'should this be 1.0 or 100.0?', the question is malformed --\n"
        "  the only real question is 'how many format points is one correct answer\n"
        "  worth?'\n"
        "\n"
        "  THE TIPPING POINT. At (1.0, 1.0) 'empty formatter' scores 1.02 against the\n"
        "  careful reasoner's 1.70, so it still loses. By (1.0, 3.0) it scores 3.02\n"
        "  against 3.50 -- closing fast -- and any persona slightly worse at reasoning\n"
        "  gets overtaken outright. The exact crossover depends on the personas, but\n"
        "  the shape is general: once an auxiliary component's weight approaches the\n"
        "  primary one, a policy that maxes the auxiliary and ignores the task becomes\n"
        "  competitive.\n"
        "\n"
        "  WHY FORMAT IS THE DANGEROUS ONE. Two properties make a reward component a\n"
        "  magnet for probability mass:\n"
        "    - it's CHEAP: any policy can satisfy it immediately, no capability needed;\n"
        "    - it's RELIABLE: it pays out every single time, zero variance.\n"
        "  Correctness is neither. So early in training -- when the model is bad at the\n"
        "  task -- format is the only component reliably producing reward differences\n"
        "  within a group, which means it dominates the advantage signal precisely\n"
        "  when the model is most plastic. This is why format rewards are usually\n"
        "  weighted low, sometimes annealed to zero, and never left as the largest term.\n"
        "\n"
        "  THE RULE: an auxiliary reward is a TIEBREAKER, not a target. If a component\n"
        "  can be saturated without doing the task, its weight must stay low enough\n"
        "  that it can never outrank the task itself.\n"
        "\n"
        "  DEBUGGING NOTE: verifiers reports each function's score separately as a\n"
        "  metric keyed by its __name__. When mean reward climbs, check WHICH component\n"
        "  climbed. Reward up while correctness is flat means you're training a\n"
        "  formatter, and the aggregate number will never tell you."
    )


if __name__ == "__main__":
    main()
