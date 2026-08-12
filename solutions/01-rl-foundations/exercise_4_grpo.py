"""
ANSWER KEY — Exercise 4, GRPO.

    uv run python solutions/01-rl-foundations/exercise_4_grpo.py

TERMS USED IN THIS FILE

  group:         several answers to the SAME question, usually written G
  advantage:     reward minus baseline: how much better than typical this answer was
  value network: a second neural network PPO trains just to predict the baseline
  PPO:           the older algorithm that needs that second network. GRPO doesn't.
  normalizing:   subtract the mean, divide by the spread -- centers on zero at a
                 consistent scale (also called a z-score)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exercise_2_reinforce import grad_log_prob, sample_action, softmax  # noqa: E402

REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
EPS = 1e-8


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    """(r - mean) / (std + eps), with exact zeros when there's no spread.

    THE EXPLICIT ZERO-STD BRANCH IS NOT OPTIONAL. If you rely on the epsilon
    alone you get (0)/(0 + 1e-8) = 0 for the all-identical case, which happens to
    be right here -- but only because the numerator is exactly 0 too. The moment
    floating-point error makes std something like 1e-17 instead of 0, you divide
    a tiny numerator by a tiny denominator and get amplified garbage pointing in
    an arbitrary direction. Branch explicitly. Groups with no reward spread must
    contribute exactly nothing.
    """
    rewards = np.asarray(rewards, dtype=float)
    mean = rewards.mean()
    std = rewards.std()
    if std < EPS:
        return np.zeros_like(rewards)
    return (rewards - mean) / (std + EPS)


def grpo_step(
    theta: np.ndarray,
    rng: np.random.Generator,
    group_size: int = 8,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Sample a group, normalize within it, average the gradient, step."""
    probs = softmax(theta)
    actions = [sample_action(probs, rng) for _ in range(group_size)]
    rewards = np.array([REWARDS[a] for a in actions], dtype=float)
    adv = group_advantages(rewards)

    grad = np.zeros_like(theta)
    for a, A in zip(actions, adv):
        grad += A * grad_log_prob(probs, a)
    grad /= group_size

    return theta + lr * grad, float(rewards.mean())


# --- harness ----------------------------------------------------------------


def train(steps: int = 600, group_size: int = 8, lr: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(REWARDS))
    history = []
    for _ in range(steps):
        theta, mean_r = grpo_step(theta, rng, group_size=group_size, lr=lr)
        history.append(mean_r)
    return {"theta": theta, "probs": softmax(theta), "history": np.array(history)}


def main() -> None:
    print("=" * 70)
    print("PART 1 — does it learn?")
    print("=" * 70)
    out = train(group_size=8)
    for i, p in enumerate(out["probs"]):
        print(f"  token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {'#' * int(p * 40)}")
    h = out["history"]
    print(f"\n  mean reward, first 50: {h[:50].mean():.4f}")
    print(f"  mean reward, last  50: {h[-50:].mean():.4f}")

    print("\n" + "=" * 70)
    print("PART 2 — group_size=1")
    print("=" * 70)
    out1 = train(group_size=1)
    for i, p in enumerate(out1["probs"]):
        print(f"  token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {'#' * int(p * 40)}")
    print(f"\n  mean reward, first 50: {out1['history'][:50].mean():.4f}")
    print(f"  mean reward, last  50: {out1['history'][-50:].mean():.4f}")

    print("\n" + "=" * 70)
    print("PART 3 — group size sweep")
    print("=" * 70)
    print(f"{'G':>4} | {'final mean reward':>18} | {'P(best token)':>14}")
    print("-" * 44)
    for g in (1, 2, 4, 8, 16, 32):
        o = train(group_size=g)
        print(f"{g:>4} | {o['history'][-50:].mean():>18.4f} | {o['probs'][4]:>14.3f}")

    print(
        "\nDISCUSSION\n"
        "  THE G=1 RESULT IS THE WHOLE LESSON. Advantage is (r - mean(r)), and with one\n"
        "  sample the mean IS that sample, so every advantage is exactly 0.0 and theta\n"
        "  never moves. Not 'slow', not 'noisy' -- literally zero gradient, forever.\n"
        "\n"
        "  The practical consequence: GRPO's signal is entirely about RANKING\n"
        "  completions within a prompt. Absolute reward values are discarded by the\n"
        "  normalization. Two direct implications for your reward functions:\n"
        "\n"
        "    1. A prompt where all G completions score identically is wasted compute.\n"
        "       Too easy (everyone gets 1.0) and too hard (everyone gets 0.0) are the\n"
        "       SAME failure. Real pipelines filter for prompts with reward variance,\n"
        "       and curriculum design is largely the art of keeping prompts in the band\n"
        "       where the model succeeds sometimes.\n"
        "\n"
        "    2. Since advantages are scale-free, multiplying your whole reward function\n"
        "       by 10 does nothing. But changing the RELATIVE spacing between components\n"
        "       -- say, making a format bonus worth as much as correctness -- changes\n"
        "       everything. When you tune a rubric, you are tuning ratios, not levels.\n"
        "\n"
        "  WHY DIVIDE BY STD AT ALL? Subtracting the mean is what makes it unbiased-with-\n"
        "  lower-variance (exercise 3). Dividing by std is a separate, more pragmatic\n"
        "  choice: it puts every prompt's advantages on the same scale so a single\n"
        "  learning rate works across a heterogeneous batch. It's normalization, not\n"
        "  theory. Some variants (e.g. Dr. GRPO) drop it, arguing it biases the\n"
        "  objective toward low-variance prompts. Both are defensible; know that the\n"
        "  denominator is the negotiable part and the numerator isn't.\n"
        "\n"
        "  DIMINISHING RETURNS IN G. Advantage estimates converge at the usual 1/sqrt(G)\n"
        "  rate, while cost is linear in G. That's why configs land at 8-16: past that\n"
        "  you're buying accuracy in the baseline that the clipping and KL terms will\n"
        "  wash out anyway. Spend the compute on more PROMPTS instead."
    )


if __name__ == "__main__":
    main()
