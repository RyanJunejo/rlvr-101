"""
Exercise 2 — training a model on a score instead of an answer.

This is the core of the whole workbook. Everything else is variations on it.

Our model is as small as a model can be while still being real:

  - a vocabulary of 5 tokens,
  - nothing to read (there's no prompt; it just produces a token),
  - 5 numbers inside it called `theta` -- one score per token, the LOGITS,
  - to produce output: turn the logits into probabilities, roll weighted dice.

That's a language model. It has a vocabulary of 5 instead of 100,000 and writes
1 token instead of 2,000, but nothing important is missing.

--------------------------------------------------------------------------
THE LOOP YOU'RE BUILDING

    1. turn logits into probabilities        (softmax)
    2. roll the dice, get a token            (sample_action)
    3. look up its reward
    4. nudge the logits to make that token more likely,
       by an amount proportional to the reward

Step 4 is the only surprising one, and the surprise is how simple it is. The
direction that makes token `a` more likely is:

    onehot(a) - probs

which is: "+1 on the token you drew, minus the current probabilities." Read as
a sentence, it says RAISE THE ONE YOU SAMPLED, LOWER EVERYONE ELSE. It always
sums to zero, because you're moving probability around, not creating it.

You do not have to take this on faith. WORKED_EXAMPLES.md section 3 checks it
numerically -- it wiggles each logit by a hair, measures what actually happens
to the probability, and compares. The formula matches to ten decimal places.

If you want the calculus: it's the derivative of log(softmax(theta)[a]) with
respect to theta. But you can finish this exercise without it.
--------------------------------------------------------------------------

Fill in the four TODOs, then run me:

    uv run python modules/01-rl-foundations/exercise_2_reinforce.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import run_main, todo  # noqa: E402

# Our reward function. Token 4 is "correct", token 3 is close, token 0 is wrong.
# In a real setup this is your verifier; here it's a lookup table so we can see
# everything clearly.
REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to a probability distribution.

    Subtract the max before exponentiating. This changes nothing mathematically
    (softmax is shift-invariant) but stops exp() from overflowing on large
    logits -- which will happen, because RL training drives logits up.

    Args:
        logits: shape (K,).
    Returns:
        shape (K,), non-negative, sums to 1.
    """
    # TODO: implement a numerically stable softmax.
    max_logits = np.max(logits)
    return np.exp(logits - max_logits) / np.sum(np.exp(logits - max_logits))


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one action from the categorical distribution `probs`.

    Hint: `rng.choice(len(probs), p=probs)`.

    Args:
        probs: shape (K,), sums to 1.
        rng: numpy Generator, for reproducibility.
    Returns:
        int in [0, K).
    """
    # TODO: sample an index according to probs. 
    return int(rng.choice(len(probs), p=probs))


def grad_log_prob(probs: np.ndarray, action: int) -> np.ndarray:
    """The direction that makes `action` more likely.

    Build a vector that is 1.0 at `action` and 0.0 everywhere else (a "one-hot"
    vector), then subtract `probs`.

        onehot(action) - probs

    Worked example, with probs = [0.2, 0.2, 0.2, 0.2, 0.2] and action = 3:

        onehot:  [0.0, 0.0, 0.0, 1.0, 0.0]
        probs:   [0.2, 0.2, 0.2, 0.2, 0.2]
        result: [-0.2, -0.2, -0.2, +0.8, -0.2]

    The sampled token gets a positive number, everything else negative. Sanity
    check for yourself once it's written: the result should always sum to zero.

    Args:
        probs: shape (K,), current probabilities (already softmaxed).
        action: which token was sampled.
    Returns:
        shape (K,), the direction to nudge the logits.
    """
    # TODO: build onehot(action) - probs.
    onehot = np.zeros_like(probs)
    onehot[action] = 1.0
    return onehot - probs


def reinforce_step(
    theta: np.ndarray,
    rng: np.random.Generator,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """One REINFORCE update: sample, score, push.

    The recipe:
        1. probs = softmax(theta)
        2. a = sample_action(probs, rng)
        3. r = REWARDS[a]
        4. theta = theta + lr * r * grad_log_prob(probs, a)

    Note step 4 is gradient *ascent* (we're maximizing reward), hence `+`.

    Notice what this update literally does: it increases the logit of whatever
    token you happened to sample, scaled by how much reward you got. Good
    samples get reinforced harder than bad ones. That's the name.

    Args:
        theta: shape (K,), current logits. Do not modify in place.
        rng: numpy Generator.
        lr: step size.
    Returns:
        (new_theta, reward_received)
    """
    # TODO: implement the four steps above.
    try:
        probs = softmax(theta)
        action = sample_action(probs, rng)
        r = REWARDS[action]
        theta = theta + lr * r * grad_log_prob(probs, action)
        return theta, r
    except Exception as e:
        print(f"Error in reinforce_step: {e}")
        return theta, 0.0


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def train(steps: int = 3000, lr: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(REWARDS))
    history = []
    for t in range(steps):
        theta, r = reinforce_step(theta, rng, lr=lr)
        history.append(r)
    return {
        "theta": theta,
        "probs": softmax(theta),
        "history": np.array(history),
    }


def main() -> None:
    print("Reward table:", REWARDS, f"(best token is #{np.argmax(REWARDS)})\n")

    out = train()
    probs = out["probs"]
    hist = out["history"]

    print("Final policy:")
    for i, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {bar}")

    first, last = hist[:200].mean(), hist[-200:].mean()
    print(f"\nMean reward, first 200 steps: {first:.4f}")
    print(f"Mean reward, last  200 steps: {last:.4f}")
    print(f"Improvement: {last - first:+.4f}")

    print(
        "\nWhat to notice:\n"
        "  - The policy concentrates on token 4, but never fully. It can't: the\n"
        "    softmax only reaches probability 1 in the limit of infinite logits.\n"
        "  - It also doesn't collapse instantly, because every sampled token gets\n"
        "    its logit pushed UP (rewards here are all >= 0). Token 1 with r=0.2 is\n"
        "    still being reinforced, just less than token 4. That is wasteful, and\n"
        "    fixing it is exactly what exercise 3 is about.\n"
        "\n"
        "Sit with that last point for a second -- it's the setup for the baseline."
    )


if __name__ == "__main__":
    run_main(main)
