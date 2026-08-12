"""
Exercise 1 — slot machines.

Yes, slot machines. Bear with me; this is 40 lines and it's the cleanest place
to meet two ideas you'll need for everything else.

THE SETUP. You're in front of 5 slot machines. Each one pays out a random amount
every time you pull it, and some are better than others on average -- but you
don't know which. You get 2000 pulls. Make as much money as you can.

(The traditional name for this is a "multi-armed bandit," because a slot machine
used to be called a one-armed bandit. That's the whole story behind the term.)

THE TWO IDEAS:

  1. YOU HAVE TO TRY THINGS. If you only ever pull the machine that's done best
     so far, you'll lock onto whichever one got lucky first and never find out
     the others were better. But if you spend all your pulls experimenting, you
     waste money on machines you already know are bad. This tension -- usually
     called "exploration vs. exploitation" -- never fully goes away.

  2. YOU LEARN BY AVERAGING. There's nobody to tell you a machine's true payout.
     You pull it, you see one noisy number, and you average over time.

WHY THIS MATTERS FOR LANGUAGE MODELS. Look at the shape of it: you take an
action, you get a score, and then the next round is completely unrelated to what
you just did. That is exactly what happens when you train a model on single
questions -- it answers, it gets scored, and the next question has nothing to do
with the last one.

That's why this workbook doesn't need the heavy machinery from most RL courses
(chess, robots, Atari). Those exist because your move changes what you face
next. Here, it doesn't.

Fill in the two TODOs, then run me:

    uv run python modules/01-rl-foundations/exercise_1_bandit.py

TERMS USED IN THIS FILE

  policy:       your rule for choosing what to do next
  exploration:  trying options you're unsure about, to find out if they're good
  exploitation: taking the option that has looked best so far
  estimate:     your current best guess at an arm's true payout, from noisy samples
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import run_main  # noqa: E402


def select_action(Q: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy action selection.

    With probability `epsilon`, return a uniformly random arm (explore).
    Otherwise return the arm with the highest current estimate in `Q` (exploit).

    Args:
        Q: shape (k,), current value estimate for each arm.
        epsilon: probability of exploring, in [0, 1].
        rng: use `rng.random()` for the coin flip and `rng.integers(k)` to pick
             a random arm. (Using the passed-in rng keeps this reproducible.)

    Returns:
        int: the index of the chosen arm.

    Note on ties: `np.argmax` returns the first maximum, which is fine here.
    At init every Q is 0.0, so a greedy policy would always pick arm 0 forever
    if epsilon were 0 -- that's the whole reason exploration exists.
    """
    # TODO: implement epsilon-greedy.
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    return int(np.argmax(Q))


def update_estimate(q: float, reward: float, n: int) -> float:
    """Incremental sample-average update.

    You've seen this: rather than storing every reward and re-averaging, keep a
    running estimate and nudge it toward each new observation.

        Q_new = Q_old + (1/n) * (reward - Q_old)

    where `n` is the number of times this arm has now been pulled (including
    this one). The `(reward - Q_old)` term is a prediction error, and this shape
    -- estimate += step_size * error -- is the skeleton of every learning rule
    in this module, including the policy gradient.

    Args:
        q: current estimate for this arm.
        reward: the reward just observed.
        n: how many times this arm has been pulled, including now (n >= 1).

    Returns:
        float: the updated estimate.
    """
    # TODO: implement the incremental mean.
    return q + (reward - q) / n 


# ---------------------------------------------------------------------------
# Everything below is written for you. Read it, don't edit it.
# ---------------------------------------------------------------------------


def run_bandit(
    true_means: np.ndarray,
    steps: int = 2000,
    epsilon: float = 0.1,
    seed: int = 0,
) -> dict:
    """Run one epsilon-greedy bandit experiment.

    Rewards are Gaussian around each arm's true mean, so a single pull tells you
    very little -- you have to average. That noise is the point.
    """
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
        "\nWhat to notice:\n"
        "  - epsilon=0 is a disaster. It pulls arm 0 once, gets a positive reward\n"
        "    by luck, and never tries anything else. Pure exploitation is a trap.\n"
        "  - epsilon=0.5 explores so much it spends half its life on known-bad arms.\n"
        "  - Somewhere in between wins. There is no clever trick here; this tension\n"
        "    never fully goes away.\n"
        "\n"
        "Now hold this thought: arms 2 and 3 have means 0.55 and 0.50, and the reward\n"
        "noise has std 1.0. Telling them apart takes a LOT of samples. That is exactly\n"
        "the situation you're in when two prompts differ slightly in difficulty and your\n"
        "verifier is noisy -- and it's why Module 04 spends time on sample sizes."
    )


if __name__ == "__main__":
    run_main(main)
