"""
ANSWER KEY — Exercise 2, REINFORCE.

    uv run python solutions/01-rl-foundations/exercise_2_reinforce.py

TERMS USED IN THIS FILE

  token:           a chunk of text; models read and write these, not letters
  logits:          the model's raw scores, one per token, before they're probabilities
  softmax:         turns logits into probabilities: exponentiate each, divide by the total
  sampling:        picking a token by rolling weighted dice against those probabilities
  policy:          the model plus how it samples -- here, five logits and a dice roll
  policy gradient: estimating the update direction by sampling, since you can't
                   compute it directly through the dice roll
  REINFORCE:       the simplest policy gradient method, and what you build here
"""

from __future__ import annotations

import numpy as np

REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    WHY SUBTRACT THE MAX:
    softmax(x) == softmax(x - c) for any constant c -- the c cancels between
    numerator and denominator. So we pick c = max(x), which makes the largest
    exponent exactly exp(0) = 1 and every other one smaller. Without this,
    exp(1000) overflows to inf and you get nan. This is not paranoia: RL training
    pushes logits up without bound (see the discussion in main()), so unstable
    softmax WILL bite you eventually.
    """
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def grad_log_prob(probs: np.ndarray, action: int) -> np.ndarray:
    """onehot(action) - probs.

    THE DERIVATION, once more, because it's worth being able to reproduce:

        log pi(a) = theta_a - logsumexp(theta)

        d/d(theta_i) of theta_a          = 1[i == a]
        d/d(theta_i) of logsumexp(theta) = exp(theta_i)/sum_j exp(theta_j) = pi_i

        => d/d(theta_i) log pi(a) = 1[i == a] - pi_i

    SANITY CHECK YOU SHOULD DO IN YOUR HEAD: this vector always sums to zero
    (one-hot sums to 1, probs sums to 1). That's not a coincidence -- it's the
    statement that softmax is shift-invariant, so there's no gradient in the
    direction of "add a constant to every logit". The policy lives on a
    (K-1)-dimensional surface.
    """
    onehot = np.zeros_like(probs)
    onehot[action] = 1.0
    return onehot - probs


def reinforce_step(
    theta: np.ndarray, rng: np.random.Generator, lr: float = 0.1
) -> tuple[np.ndarray, float]:
    """Sample, score, push.

    NOTE `theta + ...` rather than `theta += ...`: we return a new array instead
    of mutating the caller's. Mutating in place here would silently corrupt the
    comparison loops in later exercises that reuse a starting theta.
    """
    probs = softmax(theta)
    a = sample_action(probs, rng)
    r = float(REWARDS[a])
    return theta + lr * r * grad_log_prob(probs, a), r


# --- harness ----------------------------------------------------------------


def train(steps: int = 3000, lr: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(REWARDS))
    history = []
    for _ in range(steps):
        theta, r = reinforce_step(theta, rng, lr=lr)
        history.append(r)
    return {"theta": theta, "probs": softmax(theta), "history": np.array(history)}


def main() -> None:
    print("Reward table:", REWARDS, f"(best token is #{np.argmax(REWARDS)})\n")
    out = train()
    probs, hist = out["probs"], out["history"]
    print("Final policy:")
    for i, p in enumerate(probs):
        print(f"  token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {'#' * int(p * 40)}")
    first, last = hist[:200].mean(), hist[-200:].mean()
    print(f"\nMean reward, first 200 steps: {first:.4f}")
    print(f"Mean reward, last  200 steps: {last:.4f}")
    print(f"Improvement: {last - first:+.4f}")

    print(
        "\nDISCUSSION\n"
        "  THE ALL-POSITIVE-REWARD PROBLEM. Every reward here is >= 0, so every update\n"
        "  is `theta += lr * (nonneg) * (onehot - probs)`, which always raises the\n"
        "  sampled token's logit. Token 1 (r=0.2) gets reinforced every time it's drawn.\n"
        "  Learning still happens -- token 4 is reinforced five times harder -- but the\n"
        "  useful signal is the DIFFERENCE between reward magnitudes, and we're making\n"
        "  the optimizer recover it from a pile of noise it didn't need to see.\n"
        "\n"
        "  Since the gradient vector sums to zero, raising one logit does implicitly\n"
        "  lower the others. So it's not that learning is impossible without a baseline;\n"
        "  it's that the estimator is needlessly noisy. Exercise 3 quantifies exactly\n"
        "  how needlessly.\n"
        "\n"
        "  WHY THE POLICY NEVER REACHES p=1.0: softmax is asymptotic. Driving p(token 4)\n"
        "  to 1 requires theta_4 -> infinity. In practice you stop long before that, and\n"
        "  in real LLM RL you actively DON'T want to get close -- a policy at p=1.0 has\n"
        "  zero entropy, samples nothing new, and stops learning entirely. Preventing\n"
        "  that collapse is why KL penalties and entropy bonuses exist.\n"
        "\n"
        "  TRY THIS: bump lr to 1.0 and re-run. The policy slams onto whichever token it\n"
        "  happens to sample early and never recovers. That is entropy collapse in\n"
        "  miniature, and it is a real failure mode of real training runs."
    )


if __name__ == "__main__":
    main()
