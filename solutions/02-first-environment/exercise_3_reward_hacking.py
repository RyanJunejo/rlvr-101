"""
ANSWER KEY — Exercise 3, the reward hacking lab.

    uv run python solutions/02-first-environment/exercise_3_reward_hacking.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, GREEN, RED, RESET  # noqa: E402

# `(\S.*?)` -- the answer must START with a non-whitespace char. With `(.+?)`
# a line of "Answer:   " matches, capturing a single space as the "answer".
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def _msg(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def _cell(val: float, expected: float, width: int = 8) -> str:
    """Right-align the number, THEN colorize. Padding a string that already
    contains ANSI escapes counts the escapes as visible width and misaligns."""
    color = GREEN if abs(val - expected) < 1e-9 else RED
    return f"{color}{f'{val:.1f}'.rjust(width)}{RESET}"


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


def sloppy_correctness(completion, answer, **kwargs) -> float:
    text = reply_text(completion)
    return 1.0 if answer in text else 0.0


_ENUMERATION = "I'm not certain, so here are the possibilities: " + ", ".join(
    str(i) for i in range(380, 401)
)

ATTACKS: list[tuple[str, str, str, float]] = [
    ("honest correct", "17 * 23 = 391.\nAnswer: 391", "391", 1.0),
    ("extra whitespace", "Answer:   391  ", "391", 1.0),
    ("trailing period", "Let me work it out.\nAnswer: 391.", "391", 1.0),
    ("self-correction", "Answer: 12\nWait, that's wrong.\nAnswer: 391", "391", 1.0),
    ("honest wrong", "I think it's 400.\nAnswer: 400", "391", 0.0),
    ("superset number", "Answer: 3912", "391", 0.0),
    ("negation", "It is definitely not 391.\nAnswer: 400", "391", 0.0),
    ("shotgun guess", "Could be 389, 390, 391, or 392.\nAnswer: 389", "391", 0.0),
    ("enumerate everything", _ENUMERATION, "391", 0.0),
    ("right number, wrong answer", "391 looks plausible but no.\nAnswer: 12", "391", 0.0),
    ("answer field shotgun", "Answer: 391, 392, 393", "391", 0.0),
    ("mentions in prose only", "The product of 17 and 23 is 391.", "391", 0.0),
]


def robust_correctness(completion, answer, **kwargs) -> float:
    """Commit-then-compare.

    Four lines of actual logic. Every one of them is load-bearing:

      matches = ANSWER_RE.findall(text)   # rule 1: only the answer SLOT counts
      if not matches: return 0.0          # rule 4: no commitment, no credit
      claim = matches[-1]                 # rule 2: the LAST commitment wins
      return float(claim == answer)       # rule 3: equality, never containment

    WHICH RULE KILLS WHICH ATTACK -- worth tracing, because it shows that no
    single fix is sufficient:

      'superset number' (Answer: 3912)  -> rule 3. Containment said 391 is in
          3912. Equality says 3912 != 391. This is THE bug in sloppy_correctness.
      'negation', 'right number, wrong answer', 'mentions in prose only'
          -> rule 1. The digits appear in the reasoning, but reasoning is not a
          claim. Only the slot is.
      'shotgun guess'                   -> rules 1 + 3 together.
      'answer field shotgun'            -> rule 3. "391, 392, 393" != "391".
          Note this one gets PAST rule 1 -- it's in the slot. Only exact
          comparison stops it, which is why you can't rely on the slot alone.
      'enumerate everything'            -> rule 4. Never emitted a slot at all.
      'self-correction'                 -> rule 2, and this is the one that
          protects HONEST answers rather than blocking dishonest ones. Take the
          first match instead of the last and you'd fail a legitimately correct
          model for thinking out loud.
    """
    text = reply_text(completion)
    matches = ANSWER_RE.findall(text)
    if not matches:
        return 0.0
    claim = matches[-1].strip().rstrip(".").strip()
    return 1.0 if claim == str(answer).strip() else 0.0


# --- harness ----------------------------------------------------------------


def run_suite(fn) -> tuple[int, list[str]]:
    correct, wrong = 0, []
    for name, reply, answer, expected in ATTACKS:
        try:
            got = float(fn(_msg(reply), answer))
        except Exception:
            got = -1.0
        if abs(got - expected) < 1e-9:
            correct += 1
        else:
            wrong.append(name)
    return correct, wrong


def main() -> None:
    print(f"{BOLD}Scoring the attack suite with BOTH reward functions{RESET}\n")
    header = f"{'attack':<28} {'should be':>10} {'sloppy':>8} {'robust':>8}"
    print(header)
    print("-" * len(header))
    for name, reply, answer, expected in ATTACKS:
        s = sloppy_correctness(_msg(reply), answer)
        r = robust_correctness(_msg(reply), answer)
        print(f"{name:<28} {expected:>10.1f} {_cell(s, expected)} {_cell(r, expected)}")

    n = len(ATTACKS)
    s_correct, s_wrong = run_suite(sloppy_correctness)
    r_correct, r_wrong = run_suite(robust_correctness)
    print(f"\n{BOLD}sloppy_correctness: {s_correct}/{n}{RESET}")
    print(f"  {DIM}fooled by: {', '.join(s_wrong)}{RESET}")
    print(f"{BOLD}robust_correctness: {r_correct}/{n}{RESET}")
    print(f"  {DIM}{'still fooled by: ' + ', '.join(r_wrong) if r_wrong else 'survives every attack.'}{RESET}")

    print(
        f"\n{BOLD}DISCUSSION{RESET}\n"
        "\n"
        "  IS robust_correctness ACTUALLY SAFE? No. It is safe against THIS suite,\n"
        "  which I wrote, which means it's safe against the attacks I thought of. That\n"
        "  is a much weaker claim and it's the honest one. Holes that remain:\n"
        "\n"
        "    - `Answer: 391` reached by pure guessing still scores 1.0. We reward\n"
        "      outcomes, not reasoning, so a lucky guess is indistinguishable from\n"
        "      competence. On multiple-choice-shaped tasks this is a real problem, and\n"
        "      it's a large part of why process supervision (Let's Verify Step by Step)\n"
        "      exists.\n"
        "    - Answer leakage. If the prompt contains the answer anywhere -- in a\n"
        "      few-shot example, in retrieved context, in a tool result -- the model\n"
        "      can copy it. Always check what's actually in your prompts.\n"
        "    - Semantic equivalence. `Answer: 0.5` vs reference `1/2` scores 0.0.\n"
        "      Here that's fine (we asked for a number). On math datasets generally it\n"
        "      is NOT fine, and it's why `math_verify`-style symbolic comparison exists,\n"
        "      and why verifiers ships a MathRubric. Loosening comparison to fix false\n"
        "      negatives reopens the door to false positives; that tradeoff never fully\n"
        "      goes away.\n"
        "\n"
        "  THE GENERAL PATTERN, which transfers to every environment you'll write:\n"
        "\n"
        "    1. Make the model COMMIT to an answer in a designated slot. Never search\n"
        "       free-form prose for evidence of correctness -- prose is reasoning, and\n"
        "       reasoning legitimately contains wrong intermediate values.\n"
        "    2. Compare EXACTLY, or with a comparison you can defend. `in` is almost\n"
        "       never the right operator.\n"
        "    3. Reward the OUTCOME, and separately reward the FORM. Don't smuggle\n"
        "       format compliance into your correctness signal (exercise 2).\n"
        "    4. Write the attack suite BEFORE you train. Ten adversarial strings cost\n"
        "       twenty minutes and will save you a GPU-day.\n"
        "\n"
        "  And the meta-lesson: you cannot verify a reward function by reading it. It\n"
        "  looked fine. Everyone's looks fine. You verify it by trying to break it."
    )


if __name__ == "__main__":
    main()
