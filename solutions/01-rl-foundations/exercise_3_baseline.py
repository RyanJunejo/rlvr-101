"""
ANSWER KEY — Exercise 3, baselines and variance.

    uv run python solutions/01-rl-foundations/exercise_3_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exercise_2_reinforce import grad_log_prob, sample_action, softmax  # noqa: E402

REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])


def gradient_sample(
    theta: np.ndarray,
    rng: np.random.Generator,
    baseline: float = 0.0,
    rewards: np.ndarray | None = None,
) -> np.ndarray:
    """One draw of the REINFORCE estimator, with an optional baseline."""
    rewards = REWARDS if rewards is None else rewards
    probs = softmax(theta)
    a = sample_action(probs, rng)
    r = float(rewards[a])
    return (r - baseline) * grad_log_prob(probs, a)


def total_variance(samples: np.ndarray) -> float:
    """Trace of the covariance: sum of per-component variances.

    WHY THIS SUMMARY: we need ONE number to compare two estimators. The trace of
    the covariance matrix is the standard choice (it's E||g - E[g]||^2, the
    expected squared distance from the mean gradient). Any monotone summary
    would rank the two setups the same way; this one has a clean meaning.
    """
    return float(samples.var(axis=0).sum())


# --- harness ----------------------------------------------------------------


def compare(theta: np.ndarray, rewards: np.ndarray, n: int = 20000, seed: int = 0) -> dict:
    probs = softmax(theta)
    b = float(probs @ rewards)
    rng_a = np.random.default_rng(seed)
    no_b = np.array([gradient_sample(theta, rng_a, 0.0, rewards) for _ in range(n)])
    rng_b = np.random.default_rng(seed)
    with_b = np.array([gradient_sample(theta, rng_b, b, rewards) for _ in range(n)])
    return {
        "baseline": b,
        "var_without": total_variance(no_b),
        "var_with": total_variance(with_b),
        "mean_without": no_b.mean(axis=0),
        "mean_with": with_b.mean(axis=0),
    }


def main() -> None:
    theta = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    print("=" * 68)
    print("PART 1 — variance, with and without a baseline")
    print("=" * 68)
    out = compare(theta, REWARDS)
    print(f"  baseline used (E[r] under current policy): {out['baseline']:.4f}")
    print(f"  estimator variance WITHOUT baseline: {out['var_without']:.5f}")
    print(f"  estimator variance WITH    baseline: {out['var_with']:.5f}")
    print(f"  -> reduction: {(1 - out['var_with'] / out['var_without']) * 100:.1f}%")
    print(f"\n    mean gradient without baseline: {np.round(out['mean_without'], 4)}")
    print(f"    mean gradient with    baseline: {np.round(out['mean_with'], 4)}")

    print("\n" + "=" * 68)
    print("PART 2 — rewards shifted by +10")
    print("=" * 68)
    shifted = REWARDS + 10.0
    out2 = compare(theta, shifted)
    print(f"  baseline used: {out2['baseline']:.4f}")
    print(f"  estimator variance WITHOUT baseline: {out2['var_without']:.5f}")
    print(f"  estimator variance WITH    baseline: {out2['var_with']:.5f}")
    print(f"  -> reduction: {(1 - out2['var_with'] / out2['var_without']) * 100:.1f}%")
    print(f"\n  variance blowup from the shift: {out2['var_without'] / out['var_without']:.0f}x")

    print(
        "\nDISCUSSION\n"
        "  WHY THE TWO MEAN GRADIENTS AREN'T BIT-IDENTICAL. Both estimators are unbiased,\n"
        "  so their TRUE means are exactly equal. What you see printed are two finite\n"
        "  20k-sample estimates of that mean, and the high-variance one is a worse\n"
        "  estimate. They agree to a couple of decimals and the baseline one is closer\n"
        "  to the truth. That gap IS the variance reduction, viewed from the other side.\n"
        "\n"
        "  WHY THE +10 SHIFT IS SO DESTRUCTIVE. Without a baseline the estimator is\n"
        "  (r) * grad_log_prob with r ~ 10. Every single update is a large push in the\n"
        "  direction of whatever token was sampled, and the actual signal -- the 0-to-1\n"
        "  spread between tokens -- is a 10% perturbation on top of a huge common-mode\n"
        "  term. Those large pushes cancel in EXPECTATION but not in any finite batch.\n"
        "  Variance scales roughly with E[r^2], so a shift that changes nothing about\n"
        "  the problem changes everything about how hard it is to learn.\n"
        "\n"
        "  WHY THIS MATTERS FOR REWARD DESIGN (the practical payoff): if you write a\n"
        "  rubric that hands out 10 points for 'valid JSON' and 1 point for 'correct\n"
        "  answer', you have built the +10 shift into your reward function. Almost every\n"
        "  completion gets the 10; the part you care about is noise on top. A baseline\n"
        "  rescues you, but you should not have needed rescuing. Keep reward components\n"
        "  on comparable scales, and prefer rewards that actually VARY across\n"
        "  completions. Come back and reread this after the Module 02 reward-hacking lab.\n"
        "\n"
        "  THE OPTIMAL BASELINE, footnote: E[r] is not quite variance-minimizing. The\n"
        "  true optimum is a gradient-magnitude-weighted average of the reward\n"
        "  (Sutton & Barto Ch.13, and Greensmith et al. 2004). Nobody uses it. E[r] gets\n"
        "  you most of the benefit and is trivially estimable -- which, as exercise 4\n"
        "  shows, is the whole opening GRPO walks through."
    )


if __name__ == "__main__":
    main()
