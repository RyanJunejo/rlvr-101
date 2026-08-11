"""
Exercise 4 — GRPO from scratch.

This is the algorithm `prime-rl` actually runs. It is the algorithm behind
DeepSeek-R1, INTELLECT-3, and most open reasoning models of the last two years.

It is also, and I want to be blunt about this, *exercise 3 with one idea added*.

---------------------------------------------------------------------------
THE IDEA

Exercise 3 left us needing a number: the average reward for this prompt. Where
does it come from?

The old answer is to train a SECOND neural network whose entire job is to look
at a prompt and predict "you'll probably score about 0.44 here." It's called a
value network, and it's what the older algorithm PPO uses. It costs you a whole
extra model, it has to be trained alongside the first one, and early on its
guesses are bad.

GRPO's answer, which is much better and slightly embarrassing in hindsight:

    Don't predict the average. Just sample the prompt several times and
    MEASURE it.

Ask the same question G times. Score all G answers. Their mean IS the average
reward for that prompt. You didn't predict anything -- you looked.

A batch of answers to the same prompt is called a GROUP. Each answer's advantage
is then computed relative to its own group:

    advantage = (reward - group mean) / group spread

Subtracting the mean is the important part (that's the baseline from exercise 3).
Dividing by the spread is housekeeping: it puts every question on the same scale,
so an easy question where everything scores ~0.9 and a hard one ranging 0.0-1.0
contribute comparably instead of the hard one dominating.

That's the whole algorithm. "Group Relative Policy Optimization" is a complete
description of it: advantage, relative to the group.
---------------------------------------------------------------------------

Fill in the two TODOs, then run me:

    uv run python modules/01-rl-foundations/exercise_4_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for sibling exercises
from common.grading import run_main, todo  # noqa: E402

from exercise_2_reinforce import grad_log_prob, sample_action, softmax  # noqa: E402

REWARDS = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
EPS = 1e-8


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    """Normalize rewards within a group. This is the entire GRPO insight.

        A_i = (r_i - mean(r)) / (std(r) + eps)

    Two properties worth checking yourself once you've written it:
      - the advantages sum to ~0 (the mean IS the baseline, by construction),
      - they're scale-free (multiply every reward by 10, get the same advantages).

    IMPORTANT EDGE CASE: if every reward in the group is identical, std is 0 and
    every advantage must be 0.0. That's not a bug to paper over -- it's correct
    and meaningful. If all G completions scored the same, this prompt taught you
    nothing about which completion was better, so it should produce no gradient.
    In production this happens constantly (a prompt everyone gets right, or one
    everyone fails) and those groups are wasted compute. Real systems filter for
    prompts with reward variance for exactly this reason.

    Use `eps = EPS` in the denominator for numerical safety, but make sure the
    all-identical case returns exact zeros, not tiny noise.

    Args:
        rewards: shape (G,), the rewards for one group of completions.
    Returns:
        shape (G,), the advantages.
    """
    # TODO: implement group-relative advantages, handling the zero-std case.
    return todo("group_advantages: (r - mean) / (std + eps), all-zeros if std == 0")


def grpo_step(
    theta: np.ndarray,
    rng: np.random.Generator,
    group_size: int = 8,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """One GRPO update.

    The recipe:
        1. probs = softmax(theta)
        2. sample `group_size` actions from probs (this is the GROUP)
        3. look up their rewards
        4. adv = group_advantages(rewards)
        5. estimate the gradient by AVERAGING over the group:
               grad = (1/G) * sum_i  adv[i] * grad_log_prob(probs, actions[i])
        6. theta = theta + lr * grad

    Compare step 5 to exercise 2's update. Same shape. The only differences are
    that we average over a group instead of using one sample, and that `r` has
    been replaced by `adv`.

    Args:
        theta: shape (K,), current logits. Do not modify in place.
        rng: numpy Generator.
        group_size: G, how many completions to sample per update.
        lr: step size.
    Returns:
        (new_theta, mean_reward_of_the_group)
    """
    # TODO: implement the six steps above.
    return todo("grpo_step: sample a group, normalize rewards within it, then update")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


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
    print("Final policy:")
    for i, p in enumerate(out["probs"]):
        print(f"  token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {'#' * int(p * 40)}")
    h = out["history"]
    print(f"\n  mean reward, first 50 steps: {h[:50].mean():.4f}")
    print(f"  mean reward, last  50 steps: {h[-50:].mean():.4f}")

    print("\n" + "=" * 70)
    print("PART 2 — the one that should click")
    print("=" * 70)
    print("  Same code, but with group_size=1.\n")
    out1 = train(group_size=1)
    h1 = out1["history"]
    print("  Final policy:")
    for i, p in enumerate(out1["probs"]):
        print(f"    token {i} (r={REWARDS[i]:.1f}): {p:6.3f} {'#' * int(p * 40)}")
    print(f"\n  mean reward, first 50: {h1[:50].mean():.4f}")
    print(f"  mean reward, last  50: {h1[-50:].mean():.4f}")
    print(
        "\n  It learned NOTHING. The policy is still uniform.\n"
        "\n"
        "  Why: with G=1 the group mean equals the single reward, so the advantage is\n"
        "  always exactly (r - r) = 0. Every gradient is zero. GRPO with a group of one\n"
        "  is not 'GRPO with high variance' -- it is a no-op.\n"
        "\n"
        "  This is worth internalizing because it tells you where GRPO's signal comes\n"
        "  from: NOT from absolute reward, only from DIFFERENCES WITHIN A GROUP. If your\n"
        "  verifier returns the same score for every completion of a prompt, that prompt\n"
        "  contributes nothing, no matter how carefully you designed the score."
    )

    print("\n" + "=" * 70)
    print("PART 3 — how much does group size buy you?")
    print("=" * 70)
    print(f"{'G':>4} | {'final mean reward':>18} | {'P(best token)':>14}")
    print("-" * 44)
    for g in (1, 2, 4, 8, 16, 32):
        o = train(group_size=g)
        print(f"{g:>4} | {o['history'][-50:].mean():>18.4f} | {o['probs'][4]:>14.3f}")
    print(
        "\n  Diminishing returns, and every sample costs a full generation. Picking G\n"
        "  is a real budget decision in prime-rl configs -- typically 8 to 16.\n"
    )

    print("=" * 70)
    print("What we left out (and where you'll meet it)")
    print("=" * 70)
    print(
        "  Real GRPO adds two things on top of what you just wrote:\n"
        "\n"
        "  1. A CLIPPED IMPORTANCE RATIO, inherited from PPO. Rollouts are generated by\n"
        "     a slightly stale policy, so you reweight by pi_new/pi_old and clip that\n"
        "     ratio to stop any single update from moving the policy too far. This\n"
        "     matters enormously in async training, which is exactly what prime-rl does\n"
        "     -- the trainer and the inference server drift apart on purpose.\n"
        "\n"
        "  2. A KL PENALTY toward a frozen reference model, to stop the policy wandering\n"
        "     off into degenerate text that happens to score well. You'll see `kl_coef`\n"
        "     in the configs.\n"
        "\n"
        "  Both are guardrails on the update you just implemented. Neither changes the\n"
        "  core idea, which is the four characters you wrote in group_advantages().\n"
        "\n"
        "  You are now done with algorithms. Everything from here is reward design."
    )


if __name__ == "__main__":
    run_main(main)
