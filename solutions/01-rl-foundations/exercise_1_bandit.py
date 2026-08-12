"""
ANSWER KEY — Exercise 1, the k-armed bandit.

Run me directly to see the same output the exercise produces:
    uv run python solutions/01-rl-foundations/exercise_1_bandit.py

TERMS USED IN THIS FILE

  policy:       your rule for choosing what to do next
  exploration:  trying options you're unsure about, to find out if they're good
  exploitation: taking the option that has looked best so far
  estimate:     your current best guess at an arm's true payout, from noisy samples
"""

from __future__ import annotations

import numpy as np


def select_action(Q: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy.

    WHY IT'S WRITTEN THIS WAY:
    We draw the exploration coin FIRST and unconditionally. You might be tempted
    to only call rng when exploring, but then the random stream depends on the
    policy's history and two runs with different epsilon become incomparable.
    Keeping the draw unconditional makes the comparison in main() honest.
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    return int(np.argmax(Q))


def update_estimate(q: float, reward: float, n: int) -> float:
    """Incremental sample average: Q += (1/n)(r - Q).

    WHY NOT JUST STORE EVERYTHING AND AVERAGE?
    Algebraically identical, but this form is O(1) memory and -- more importantly
    -- it generalizes. Replace 1/n with a constant alpha and you get an
    exponentially-weighted average that tracks a NON-stationary problem, which
    is what you actually have in RL training (the "arms" change as the policy
    changes). The shape `estimate += step * (target - estimate)` is the
    skeleton of TD learning, Q-learning, and, if you squint, the policy gradient.
    """
    return q + (reward - q) / n


# --- harness (identical to the exercise) ------------------------------------


def run_bandit(
    true_means: np.ndarray, steps: int = 2000, epsilon: float = 0.1, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    k = len(true_means)
    Q = np.zeros(k)
    counts = np.zeros(k, dtype=int)
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    best_arm = int(np.argmax(true_means))

    for t in range(steps):
        a = select_action(Q, epsilon, rng)
        r = rng.normal(true_means[a], 1.0)
        counts[a] += 1
        Q[a] = update_estimate(Q[a], r, counts[a])
        rewards[t] = r
        optimal[t] = 1.0 if a == best_arm else 0.0

    return {
        "Q": Q,
        "counts": counts,
        "mean_reward": float(rewards.mean()),
        "final_optimal_rate": float(optimal[-500:].mean()),
        "best_arm": best_arm,
    }


def main() -> None:
    true_means = np.array([0.1, 0.3, 0.55, 0.5, 0.2])
    print("True arm means:", true_means)
    print(f"Best arm is #{np.argmax(true_means)} (mean {true_means.max()})\n")
    print(f"{'epsilon':>10} | {'mean reward':>12} | {'% optimal (last 500)':>21}")
    print("-" * 50)
    for eps in (0.0, 0.01, 0.1, 0.5):
        out = run_bandit(true_means, epsilon=eps)
        print(f"{eps:>10} | {out['mean_reward']:>12.4f} | {out['final_optimal_rate'] * 100:>20.1f}%")

    print(
        "\nDISCUSSION\n"
        "  epsilon=0 collapses immediately. With Q initialized to zeros, argmax picks\n"
        "  arm 0; if that pull returns a positive reward, Q[0] > 0 and no other arm is\n"
        "  ever tried again. The agent commits to a mediocre arm on the strength of a\n"
        "  single noisy sample.\n"
        "\n"
        "  Note the standard fix you may remember: OPTIMISTIC INITIALIZATION. Set Q to\n"
        "  +5 everywhere instead of 0. Every pull then disappoints, dropping that arm's\n"
        "  estimate below the untried ones, so the agent sweeps all arms before settling.\n"
        "  Free exploration with no epsilon at all. It's a lovely trick and it does not\n"
        "  transfer to LLM RL -- you can't 'optimistically initialize' a policy over a\n"
        "  combinatorial space of completions. In LLM RL, exploration comes from\n"
        "  SAMPLING TEMPERATURE and from the entropy of the policy, and keeping that\n"
        "  entropy from collapsing is a live practical problem (search: 'entropy\n"
        "  collapse' in RLVR).\n"
        "\n"
        "  arms 2 and 3 (means 0.55 vs 0.50, noise std 1.0) are the interesting case:\n"
        "  distinguishing them needs on the order of hundreds of samples. Your verifier\n"
        "  in Module 02 will have exactly this problem."
    )


if __name__ == "__main__":
    main()
