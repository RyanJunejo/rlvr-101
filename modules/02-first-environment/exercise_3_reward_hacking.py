"""
Exercise 3 — the reward hacking lab.

This is the most important exercise in the workbook.

Below is a reward function that looks fine. It ships in real codebases. It is
the single most common way people write "did the model get it right":

    def sloppy_correctness(completion, answer, **kwargs) -> float:
        return 1.0 if answer in completion[-1]["content"] else 0.0

"Did the correct answer appear somewhere in the response?" Reasonable, right?

It is catastrophically broken. Run this file and watch it hand out full marks to
a response that says the answer is *not* 391, to a response that guesses five
numbers at once, and to a response that just lists every integer up to 400.

Here's the part that matters. A merely NOISY reward function is survivable --
noise averages out over a batch. An EXPLOITABLE one does not average out,
because RL is a search procedure and its entire job is to find exploits. Every
hole you leave, gradient descent finds and drives through, because "list every
number" really does score higher than "reason carefully."

The optimizer is not misbehaving. It is doing exactly what you asked.

This is Goodhart's law with a training loop attached: your reward function isn't
a *measurement* of what you want, it's a *definition* of it. The moment you
optimize against it, the difference stops being academic.

Your job: write `robust_correctness` so it survives the attack suite.

No API key needed. The attacks are canned, so this is deterministic and free.

    uv run python modules/02-first-environment/exercise_3_reward_hacking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, GREEN, RED, RESET, run_main, todo  # noqa: E402


def _msg(text: str) -> list[dict]:
    """Wrap raw text as a completion (a list of chat messages)."""
    return [{"role": "assistant", "content": text}]


def _cell(val: float, expected: float, width: int = 8) -> str:
    """Right-align the number, THEN colorize (padding a string that already
    contains ANSI escapes counts them as visible width and misaligns)."""
    color = GREEN if abs(val - expected) < 1e-9 else RED
    return f"{color}{f'{val:.1f}'.rjust(width)}{RESET}"


# ---------------------------------------------------------------------------
# The broken reward function. Do not fix this one -- we want to watch it fail.
# ---------------------------------------------------------------------------


def sloppy_correctness(completion, answer, **kwargs) -> float:
    """The bug: substring containment is not equality."""
    text = completion[-1]["content"]
    return 1.0 if answer in text else 0.0


# ---------------------------------------------------------------------------
# The attack suite.
#
# Each case is (name, model_reply, reference_answer, what_a_correct_grader_says).
# Some of these are HONEST answers that must still score 1.0 -- a "robust" reward
# function that just returns 0.0 for everything is not robust, it's useless.
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
    # --- honest failures: correctly scored 0 by BOTH graders ---
    ("honest wrong", "I think it's 400.\nAnswer: 400", "391", 0.0),
    # --- the exploits: sloppy_correctness gives these full marks ---
    ("superset number", "Answer: 3912", "391", 0.0),
    ("negation", "It is definitely not 391.\nAnswer: 400", "391", 0.0),
    ("shotgun guess", "Could be 389, 390, 391, or 392.\nAnswer: 389", "391", 0.0),
    ("enumerate everything", _ENUMERATION, "391", 0.0),
    ("right number, wrong answer", "391 looks plausible but no.\nAnswer: 12", "391", 0.0),
    ("answer field shotgun", "Answer: 391, 392, 393", "391", 0.0),
    ("mentions in prose only", "The product of 17 and 23 is 391.", "391", 0.0),
]


def robust_correctness(completion, answer, **kwargs) -> float:
    """Score 1.0 only when the model committed to exactly the right answer.

    THE STRATEGY (this is the lesson, so read it before coding):

      1. Make the model COMMIT. Don't search the whole response for the answer --
         look only at the designated answer slot. Our system prompt asks for a
         final line `Answer: <number>`, so that line, and nothing else, is the
         model's claim. Prose is reasoning; only the slot is an assertion.

      2. Take the LAST such line, scanning bottom-up. Models restate the format
         mid-reasoning and correct themselves. The final commitment is the real one.

      3. Compare EXACTLY, not by containment. Strip whitespace and one trailing
         period, then use `==`. "3912" must not match "391".

      4. No slot, no credit. If the model never emitted an `Answer:` line, return
         0.0. It never committed to anything.

    Requirement 3 is what kills most of the attacks. Requirement 1 kills the rest.

    A NOTE ON RULE 4, because it's a real design decision and not obviously right:
    a model that replies with a bare "391" and no `Answer:` line gets 0.0 here,
    even though it was correct. You've folded format compliance into your
    correctness signal, which means your metric can no longer tell "can't do
    arithmetic" apart from "won't follow instructions". That's bad instrumentation.
    The fix isn't to loosen this function -- it's to add a SEPARATE format reward
    and let the rubric weight them independently. Which is exactly what exercise 2
    has you build.

    Args:
        completion: list of chat messages; model text is completion[-1]["content"].
        answer: the reference answer string.
    Returns:
        1.0 or 0.0
    """
    # TODO: implement the four rules above.
    return todo("robust_correctness: parse the last Answer: line, compare exactly")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def run_suite(fn) -> tuple[int, list[str]]:
    """Score every attack with `fn`; return (n_correct, names_it_got_wrong)."""
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

    robust_works = True
    for name, reply, answer, expected in ATTACKS:
        s = sloppy_correctness(_msg(reply), answer)
        try:
            r_col = _cell(float(robust_correctness(_msg(reply), answer)), expected)
        except Exception:
            r_col, robust_works = f"{DIM}{'TODO'.rjust(8)}{RESET}", False
        print(f"{name:<28} {expected:>10.1f} {_cell(s, expected)} {r_col}")

    n = len(ATTACKS)
    s_correct, s_wrong = run_suite(sloppy_correctness)
    print(f"\n{BOLD}sloppy_correctness: {s_correct}/{n} correct{RESET}")
    print(f"  {DIM}fooled by: {', '.join(s_wrong)}{RESET}")

    if robust_works:
        r_correct, r_wrong = run_suite(robust_correctness)
        print(f"{BOLD}robust_correctness: {r_correct}/{n} correct{RESET}")
        if r_wrong:
            print(f"  {DIM}still fooled by: {', '.join(r_wrong)}{RESET}")
        else:
            print(f"  {GREEN}survives every attack.{RESET}")
    else:
        print(f"{DIM}robust_correctness: not implemented yet.{RESET}")

    print(
        f"\n{BOLD}Now the part that should actually worry you.{RESET}\n"
        "\n"
        "Look at 'enumerate everything'. Under the sloppy reward, a model that lists\n"
        "every number from 380 to 400 scores 1.0 -- the same as a model that computed\n"
        "the answer correctly. And listing numbers is MUCH easier than multiplying.\n"
        "\n"
        "Now recall how GRPO works (Module 01): advantage comes from differences\n"
        "WITHIN a group of completions on the same prompt. So the moment one sampled\n"
        "completion stumbles onto enumeration and scores 1.0 while the careful ones\n"
        "score 0.0 because of an arithmetic slip, enumeration gets a positive\n"
        "advantage and its logits go up. Next step it's sampled more often. It wins\n"
        "more groups. Within a few hundred steps you have a model that has completely\n"
        "abandoned arithmetic in favour of listing numbers, and a training curve that\n"
        "looks like a triumph.\n"
        "\n"
        "You would not notice from the reward curve. That's the point. The reward\n"
        "curve is the thing being hacked.\n"
        "\n"
        "How you actually catch this in practice:\n"
        "  - READ THE ROLLOUTS. Not the mean reward -- the actual text. Regularly.\n"
        "    Every practitioner will tell you this and most people still don't do it.\n"
        "  - Watch completion length. Reward climbing while replies get longer and\n"
        "    weirder is the classic signature.\n"
        "  - Hold out an eval your rubric can't touch, and check the two agree.\n"
        "  - Assume every reward function you write has a hole, because it does.\n"
        "    The question is only whether you find it before the optimizer does."
    )


if __name__ == "__main__":
    run_main(main)
