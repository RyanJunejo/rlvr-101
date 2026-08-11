"""
Exercise 3 — subtracting the average, and why it helps so much.

Exercise 2 had a problem. Every reward in our table is positive, so every token
we sampled got pushed UP -- including the bad ones, just less hard. We were
reinforcing bad answers.

The fix is one subtraction. Instead of multiplying by the reward:

    theta = theta + lr * r * direction

multiply by the reward MINUS THE AVERAGE:

    theta = theta + lr * (r - b) * direction

where `b` is the average reward. Now anything below average gets a NEGATIVE
multiplier, which pushes it down. Our reward table averages to 0.44, so:

    token 1 (reward 0.2):  0.2 - 0.44 = -0.24   -> pushed DOWN
    token 4 (reward 1.0):  1.0 - 0.44 = +0.56   -> pushed UP

That subtracted number is called a BASELINE. The result -- how much better than
typical this output was -- is called the ADVANTAGE.

--------------------------------------------------------------------------
THE CLAIM: subtracting a baseline costs you NOTHING.

It sounds like it should distort things. It doesn't, and here's the intuition.

The nudge direction always sums to zero (you checked this in exercise 2). So
subtracting a constant from every reward adds a term that's "the same constant
times a bunch of directions that sum to zero" -- and averaged over the tokens
you actually sample, that extra term cancels out exactly.

Put another way: subtracting a constant shifts every token equally, and
"shift everything equally" is precisely the direction softmax ignores.

So you get strictly less noise and pay nothing for it.
--------------------------------------------------------------------------

"Less noise" is a claim, and this workbook doesn't do claims. You're going to
MEASURE it, and then watch something genuinely alarming in Part 2.

Fill in the two TODOs, then run me:

    uv run python modules/01-rl-foundations/exercise_3_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for sibling exercises
from common.grading import run_main  # noqa: E402

# We reuse your exercise 2 work. If exercise 2 still has TODOs in it, this file
# will fail with NotImplementedYet -- that's expected, go finish exercise 2 first.
from exercise_2_reinforce import grad_log_prob, sample_action, softmax  # noqa: E402

REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])


def gradient_sample(
    theta: np.ndarray,
    rng: np.random.Generator,
    baseline: float = 0.0,
    rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Produce ONE nudge-direction, exactly as exercise 2 did.

    This is the thing whose noisiness we're about to measure. Each call samples
    a token and returns the nudge that would result -- a different one every
    time, because the sampling is random. That variability is the point.

        probs = softmax(theta)
        a     = sample_action(probs, rng)
        r     = rewards[a]
        return (r - baseline) * grad_log_prob(probs, a)

    Args:
        theta: shape (K,), current logits.
        rng: numpy Generator.
        baseline: constant subtracted from the reward.
        rewards: shape (K,) reward table; defaults to REWARDS.
    Returns:
        shape (K,), one sample of the gradient estimator.
    """
    rewards = REWARDS if rewards is None else rewards
    # TODO: implement the four steps in the docstring.
    probs = softmax(theta)
    a = sample_action(probs, rng)
    r = rewards[a]
    return (r - baseline) * grad_log_prob(probs, a)


def total_variance(samples: np.ndarray) -> float:
    """How noisy is this batch of nudges? Boil it down to one number.

    `samples` is shape (n, K): n separate nudge-directions, each K numbers long.
    We want a single "how much do these jump around" score so we can compare two
    setups.

    Recipe: for each of the K positions, compute the variance across the n
    samples (numpy: `.var(axis=0)`), then add those K numbers together.

    "Variance" here just means the usual statistical spread -- average squared
    distance from the mean. Bigger = noisier = you need more samples before the
    average settles down.

    Args:
        samples: shape (n, K).
    Returns:
        float: one number summarizing the noise.
    """
    # TODO: variance along axis 0, then sum.
    return samples.var(axis=0).sum()


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def compare(theta: np.ndarray, rewards: np.ndarray, n: int = 20000, seed: int = 0) -> dict:
    """Measure estimator variance with and without a mean baseline."""
    probs = softmax(theta)
    # The optimal constant baseline is roughly the expected reward under the
    # current policy. That's what a value function would learn to predict.
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
    theta = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # a mildly-trained policy

    print("=" * 68)
    print("PART 1 — variance, with and without a baseline")
    print("=" * 68)
    out = compare(theta, REWARDS)
    print(f"  baseline used (E[r] under current policy): {out['baseline']:.4f}")
    print(f"  estimator variance WITHOUT baseline: {out['var_without']:.5f}")
    print(f"  estimator variance WITH    baseline: {out['var_with']:.5f}")
    print(f"  -> reduction: {(1 - out['var_with'] / out['var_without']) * 100:.1f}%")

    print("\n  Sanity check that we didn't break anything. Averaged over many samples,")
    print("  both versions should point the SAME WAY -- one is just less jittery:")
    print(f"    average nudge without baseline: {np.round(out['mean_without'], 4)}")
    print(f"    average nudge with    baseline: {np.round(out['mean_with'], 4)}")

    print("\n" + "=" * 68)
    print("PART 2 — the demonstration that should bother you")
    print("=" * 68)
    print("  Same problem, but every reward shifted up by +10.")
    print("  The OPTIMAL POLICY IS IDENTICAL -- adding a constant to every reward")
    print("  changes nothing about which action is best.\n")

    shifted = REWARDS + 10.0
    out2 = compare(theta, shifted)
    print(f"  baseline used: {out2['baseline']:.4f}")
    print(f"  estimator variance WITHOUT baseline: {out2['var_without']:.5f}")
    print(f"  estimator variance WITH    baseline: {out2['var_with']:.5f}")
    print(f"  -> reduction: {(1 - out2['var_with'] / out2['var_without']) * 100:.1f}%")

    blowup = out2["var_without"] / out["var_without"]
    print(
        f"\n  Without a baseline, variance went up {blowup:.0f}x from a shift that\n"
        f"  is mathematically meaningless. With a baseline: unchanged.\n"
    )
    print(
        "What to take away:\n"
        "  - A baseline is not a heuristic. Without one, your gradient estimator is\n"
        "    sensitive to an arbitrary constant that has nothing to do with the task.\n"
        "  - 'Subtract the expected reward' is what a value network V(s) is FOR. Its\n"
        "    entire job is predicting the baseline.\n"
        "  - Which raises the question exercise 4 answers: if all you need is the\n"
        "    expected reward for this prompt... why train a whole network to guess it,\n"
        "    when you could just sample the prompt a few times and average?"
    )


if __name__ == "__main__":
    run_main(main)
