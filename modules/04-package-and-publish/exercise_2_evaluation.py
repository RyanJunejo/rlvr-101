"""
Lab 2 — is that difference real?

This is the lab most likely to change how you work.

THE SCENARIO. You test two models on your task, 20 questions each.

    model A:  12 / 20  =  60%
    model B:  15 / 20  =  75%

B looks 15 points better. Is it?

You're going to compute the answer, and it's uncomfortable: that gap is entirely
consistent with the two models being IDENTICAL and the coin landing differently.
With 20 samples you need a gap of roughly 30 points before you can say much at
all.

WHY. Each question is a coin flip. A model with a true 60% success rate does not
score 60% on 20 questions -- it scores somewhere in a wide range. Quantifying
that range is what this lab is about.

No API key needed. Everything here is arithmetic on numbers you already have.

Fill in the three TODOs, then run me:

    uv run python modules/04-package-and-publish/exercise_2_evaluation.py

TERMS USED IN THIS FILE

  standard error:      how much a measurement would jump around if you re-ran it
  confidence interval: the range your true rate plausibly sits in, given n
  statistical power:   the chance of detecting a real difference if one exists
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# 1.96 standard errors covers 95% of a normal distribution. That's where the
# conventional "95% confidence" comes from -- it isn't a magic number, it's a
# choice about how often you're willing to be wrong.
Z95 = 1.96


def standard_error(successes: int, n: int) -> float:
    """How much would this measurement jump around if you ran it again?

    For a yes/no outcome measured n times, the standard error of the observed
    rate is:

        p  = successes / n
        se = sqrt( p * (1 - p) / n )

    Notice the `/ n` under the square root -- that's the whole story of this
    lab. Uncertainty shrinks like 1/sqrt(n), so FOUR TIMES the samples buys you
    HALF the uncertainty. There is no way around this and it is the tax on all
    empirical work.

    Edge case: if n is 0, return 0.0 rather than dividing by zero.

    Args:
        successes: how many questions the model got right.
        n: how many questions were asked.
    Returns:
        the standard error of the success rate.
    """
    # TODO: implement the formula above.
    return todo("standard_error: sqrt(p * (1 - p) / n)")


def confidence_interval(successes: int, n: int) -> tuple[float, float]:
    """A 95% confidence interval for the true success rate.

        p = successes / n
        margin = Z95 * standard_error(successes, n)
        return (p - margin, p + margin)

    Clamp the result to [0.0, 1.0] -- a success rate below 0% or above 100% is
    not a thing, and the formula will happily produce one near the extremes.

    HOW TO READ THE RESULT: it is NOT "there's a 95% chance the true rate is in
    here." It's "if I repeated this whole experiment many times, 95% of the
    intervals I built this way would contain the true rate." The distinction
    rarely matters in practice, but claiming the first one in a paper will get
    you an unpleasant review.

    A REAL LIMITATION, because you will hit it: this is the Wald interval, the
    standard textbook one, and it is BADLY WRONG near 0 and 1. Simulated over
    20,000 experiments at a true rate of 0.95 with n=20, it contains the truth
    only 63.6% of the time while claiming 95%. That is exactly the regime you
    care about when checking whether a task is saturated. Use it to learn what
    an interval is; use the Wilson interval for real work. THEORY.md section 2
    has the numbers and the better formula.

    Args:
        successes: number correct.
        n: number of questions.
    Returns:
        (low, high), both in [0, 1].
    """
    # TODO: implement, and clamp to [0, 1].
    return todo("confidence_interval: p +/- 1.96 * standard_error, clamped to [0,1]")


def samples_needed(p1: float, p2: float) -> int:
    """Roughly how many samples PER MODEL to reliably tell these two rates apart?

    The standard rule of thumb for comparing two proportions at 95% confidence
    with reasonable power:

        p_bar = (p1 + p2) / 2
        n = ceil( 16 * p_bar * (1 - p_bar) / (p1 - p2)^2 )

    (The 16 bundles up the z-values for 95% confidence and 80% power. Don't
    worry about deriving it -- what matters is the SHAPE.)

    LOOK AT THE DENOMINATOR. It's the difference SQUARED. So detecting a gap
    half as large costs you FOUR TIMES the samples. Small differences are
    brutally expensive to establish, which is why "we improved by 1%" claims
    deserve suspicion unless n is enormous.

    If p1 == p2, return a very large number (there's no sample size that
    reliably distinguishes identical things) -- use 10**9.

    Args:
        p1, p2: the two true rates you want to distinguish, in [0, 1].
    Returns:
        approximate samples needed per model.
    """
    # TODO: implement the formula above, handling p1 == p2.
    return todo("samples_needed: 16 * p_bar * (1 - p_bar) / (p1 - p2)^2, rounded up")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Do two confidence intervals overlap? If so, you can't call a winner.

    (This is a slightly conservative test -- non-overlapping intervals definitely
    means a real difference, but overlapping ones can still occasionally be
    distinguishable by a proper two-sample test. It's the right instinct to
    build, and it will never make you overclaim.)
    """
    return not (a[1] < b[0] or b[1] < a[0])


def simulate(true_rate: float, n: int, trials: int = 4000, seed: int = 0) -> np.ndarray:
    """Actually run the experiment `trials` times and see what scores come out."""
    rng = np.random.default_rng(seed)
    return rng.binomial(n, true_rate, size=trials) / n


def main() -> None:
    print(f"{BOLD}PART 1 — the scenario{RESET}\n")
    a_lo, a_hi = confidence_interval(12, 20)
    b_lo, b_hi = confidence_interval(15, 20)
    print(f"  model A: 12/20 = 60.0%   95% CI: [{a_lo * 100:5.1f}%, {a_hi * 100:5.1f}%]")
    print(f"  model B: 15/20 = 75.0%   95% CI: [{b_lo * 100:5.1f}%, {b_hi * 100:5.1f}%]")
    overlap = intervals_overlap((a_lo, a_hi), (b_lo, b_hi))
    print(f"\n  intervals overlap? {overlap}  ->  "
          f"{'CANNOT call a winner' if overlap else 'difference is real'}")
    print(
        f"\n{DIM}  Those intervals are enormous, and they overlap heavily. The honest\n"
        f"  statement is 'we could not distinguish these two models', which is a\n"
        f"  much less exciting sentence than '25% relative improvement'.{RESET}"
    )

    print(f"\n\n{BOLD}PART 2 — what a bigger n buys you{RESET}\n")
    print(f"  A model whose TRUE success rate is exactly 60%:\n")
    print(f"  {'n':>6} | {'95% interval':>22} | {'width':>7}")
    print("  " + "-" * 42)
    for n in (10, 20, 50, 100, 500, 2000):
        lo, hi = confidence_interval(int(round(0.6 * n)), n)
        print(f"  {n:>6} | [{lo * 100:6.1f}%, {hi * 100:6.1f}%]      | {(hi - lo) * 100:>6.1f}%")
    print(
        f"\n{DIM}  Going from n=50 to n=200 (4x the work) halves the width. That 1/sqrt(n)\n"
        f"  relationship is the tax on all empirical work.{RESET}"
    )

    print(f"\n\n{BOLD}PART 3 — how much work is your claim going to cost?{RESET}\n")
    print(f"  {'claim':<34} | {'samples needed each':>19}")
    print("  " + "-" * 56)
    for lo, hi, label in [
        (0.50, 0.70, "50% vs 70%  (huge)"),
        (0.60, 0.70, "60% vs 70%  (large)"),
        (0.65, 0.70, "65% vs 70%  (moderate)"),
        (0.68, 0.70, "68% vs 70%  (small)"),
        (0.695, 0.70, "69.5% vs 70%  (tiny)"),
    ]:
        print(f"  {label:<34} | {samples_needed(lo, hi):>19,}")
    print(
        f"\n{DIM}  Read the bottom row before you next claim a 0.5% improvement.{RESET}"
    )

    print(f"\n\n{BOLD}PART 4 — don't take my word for it{RESET}\n")
    print("  Two IDENTICAL models, both with a true rate of exactly 60%.")
    print("  We test each on 20 questions and declare a 'winner'. 4000 times.\n")
    a = simulate(0.60, 20, seed=1)
    b = simulate(0.60, 20, seed=2)
    gap = np.abs(a - b)
    print(f"    median observed gap between identical models: {np.median(gap) * 100:.1f} points")
    print(f"    gap of 15+ points occurred in:                {np.mean(gap >= 0.15) * 100:.1f}% of runs")
    print(f"    gap of 25+ points occurred in:                {np.mean(gap >= 0.25) * 100:.1f}% of runs")
    print(
        f"\n{DIM}  Two models that are genuinely the same produced a 15-point gap in about\n"
        f"  a third of experiments. That is exactly the 'result' from Part 1.{RESET}"
    )

    print(
        f"\n\n{BOLD}THE RULE{RESET}\n"
        "\n"
        "  A score without a sample size attached is not information.\n"
        "\n"
        "  Report `62% +/- 7% (n=200)`. Never `62%`. If you take one habit from this\n"
        "  entire course, take this one -- it's the difference between a result and\n"
        "  a vibe, and it will stop you burning a GPU day chasing noise."
    )


if __name__ == "__main__":
    run_main(main)
