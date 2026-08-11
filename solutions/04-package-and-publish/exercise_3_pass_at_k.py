"""
ANSWER KEY -- Unit 04, Lab 3 — pass@k, and why it decides whether a task is trainable.

pass@1 asks: given ONE attempt, how often is the model right?
pass@k asks: given k attempts, how often does AT LEAST ONE succeed?

WHY THIS MATTERS HERE SPECIFICALLY, and it's not the usual eval reason.

Think back to Unit 03. Learning needs SPREAD WITHIN A GROUP. If all 8 rollouts
score the same, every advantage is zero and nothing is learned.

Now take a model with pass@1 of about 5% -- by ordinary accuracy, hopeless. But
sample it 8 times per question and ask what pass@8 is. If it's 35%, then in
roughly a third of groups at least one rollout succeeds while others fail, and
that is exactly the spread that produces a gradient.

    pass@1 = 5%,  pass@8 = 35%   ->  TRAINABLE. Groups have spread.
    pass@8 = 0%                  ->  NOT TRAINABLE. Every group uniformly zero,
                                     nothing to reinforce, ever.

Same terrible-looking pass@1, completely different prognosis. So pass@k isn't
just a score here -- it's a FEASIBILITY CHECK you run before renting a GPU.

Fill in the two TODOs, then run me:

    uv run python modules/04-package-and-publish/exercise_3_pass_at_k.py
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402


def pass_at_k(n: int, c: int, k: int) -> float:
    """The standard unbiased pass@k estimator.

    You generated `n` samples for a question and `c` of them were correct.
    Estimate the probability that a fresh batch of `k` samples would contain at
    least one correct answer.

        pass@k = 1 - C(n-c, k) / C(n, k)

    Read it as: "one minus the probability that all k of your draws land among
    the (n-c) failures."

    Use `math.comb` (already imported).

    EDGE CASES, all of which show up in practice:
      - if k > n, you can't estimate it; return float("nan")
      - if n - c < k, then it's IMPOSSIBLE to draw k failures, so pass@k is
        exactly 1.0. (C(n-c, k) would be 0 here anyway, so the formula handles
        it -- but make sure you don't crash first.)
      - c == 0 gives 0.0, c == n gives 1.0. Both fall out of the formula.

    Args:
        n: total samples generated.
        c: how many were correct.
        k: the k in pass@k.
    Returns:
        the estimated pass@k, in [0, 1], or nan if k > n.
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def naive_pass_at_k(samples_correct: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """The obvious-but-worse approach: draw k samples and see if any passed.

    `samples_correct` is a 1-D array of 0/1 for one question's n samples. Draw
    `k` of them WITHOUT replacement and return 1.0 if any is correct, else 0.0.

    Hint: `rng.choice(samples_correct, size=k, replace=False)`.

    This is what most people write first. It's not wrong exactly -- it's an
    unbiased estimate of a single draw -- but it's a coin flip rather than a
    number, so it has enormous variance. Part 2 shows you the cost.

    Args:
        samples_correct: shape (n,), 1.0 for correct and 0.0 for incorrect.
        k: how many to draw.
        rng: numpy Generator.
    Returns:
        1.0 or 0.0
    """
    drawn = rng.choice(samples_correct, size=k, replace=False)
    return 1.0 if drawn.sum() > 0 else 0.0


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def simulate_question(true_rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < true_rate).astype(float)


def main() -> None:
    print(f"{BOLD}PART 1 — more attempts, more chances{RESET}\n")
    print("  A model that gets a question right 20% of the time, sampled n=100:\n")
    print(f"  {'k':>4} | {'pass@k':>8}")
    print("  " + "-" * 17)
    for k in (1, 2, 4, 8, 16, 32):
        print(f"  {k:>4} | {pass_at_k(100, 20, k) * 100:>7.1f}%")
    print(
        f"\n{DIM}  A 20% model succeeds at least once in 8 tries about 83% of the time.\n"
        f"  It looks far more capable when you let it try repeatedly -- which is\n"
        f"  exactly the situation during RL training, where you sample a group.{RESET}"
    )

    print(f"\n\n{BOLD}PART 2 — why the naive estimator is worse{RESET}\n")
    rng = np.random.default_rng(0)
    samples = simulate_question(0.20, 100, rng)
    c = int(samples.sum())
    exact = pass_at_k(100, c, 8)
    naive_runs = np.array([naive_pass_at_k(samples, 8, rng) for _ in range(2000)])
    print(f"  same 100 samples, {c} of them correct\n")
    print(f"    unbiased estimator:  {exact:.4f}     (one number, computed)")
    print(f"    naive, averaged 2000x: {naive_runs.mean():.4f}   (agrees -- it IS unbiased)")
    print(f"    naive, std deviation:  {naive_runs.std():.4f}   <- this is the problem")
    print(
        f"\n{DIM}  The naive version returns only 0.0 or 1.0, so a single run tells you\n"
        f"  almost nothing. Both are unbiased; one has a fraction of the variance\n"
        f"  and uses all the data you already paid to generate. This is the same\n"
        f"  bias-versus-variance distinction from Unit 01 -- being right on average\n"
        f"  is not the same as being useful.{RESET}"
    )

    print(f"\n\n{BOLD}PART 3 — the feasibility check{RESET}\n")
    print("  Is a task trainable? Look at whether groups will have any spread.\n")
    print(f"  {'true rate':>10} | {'pass@1':>8} | {'pass@8':>8} | {'verdict':<34}")
    print("  " + "-" * 70)
    for rate in (0.0, 0.005, 0.02, 0.05, 0.20, 0.60, 0.95, 1.0):
        rng2 = np.random.default_rng(7)
        n = 400
        s = simulate_question(rate, n, rng2)
        cc = int(s.sum())
        p1, p8 = pass_at_k(n, cc, 1), pass_at_k(n, cc, 8)
        if p8 < 0.02:
            verdict = "NOT trainable - never succeeds"
        elif p1 > 0.98:
            verdict = "NOT trainable - already saturated"
        elif p8 < 0.15:
            verdict = "marginal - needs reward shaping"
        else:
            verdict = "trainable"
        print(f"  {rate * 100:>9.1f}% | {p1 * 100:>7.1f}% | {p8 * 100:>7.1f}% | {verdict:<34}")

    print(
        f"\n{DIM}  Both ends fail, for the reasons Unit 03 measured: no spread within a\n"
        f"  group means no advantage means no gradient. pass@k lets you check this\n"
        f"  with a few hundred API calls instead of a wasted GPU day.{RESET}"
    )

    print(
        f"\n\n{BOLD}THE HABIT{RESET}\n"
        "\n"
        "  Before you train on a task, measure pass@1 and pass@k at your intended\n"
        "  group size. If pass@k is ~0, the model can never generate a successful\n"
        "  rollout to learn from -- fix the task or the starting model, because no\n"
        "  amount of training will rescue it. If pass@1 is ~1, the task is already\n"
        "  saturated and there's nothing left to teach.\n"
        "\n"
        "  You want the middle. That check costs a few dollars; skipping it costs\n"
        "  a GPU day."
    )


if __name__ == "__main__":
    main()
