"""
Lab 1 — what will this run cost?

Rented GPUs bill by the hour whether or not you're using them well. This lab is
the arithmetic that turns "let's try a training run" into a number you decided on
in advance.

It runs offline and costs nothing, which is the point: every question you can
answer on a laptop is a question you don't answer at $4.80/hour.

Fill in the three TODOs, then run me:

    uv run python modules/05-training-on-gpu/exercise_1_budget.py

TERMS USED IN THIS FILE

  batch size: how many questions per training step
  group size: how many answers sampled per question. The G from Unit 01.
  step:       one full cycle: generate answers, score them, update the weights
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# GMI Cloud, H100 SXM, on demand. Check the current rate before you rely on it.
RATE_PER_GPU_HOUR = 2.40
CREDIT = 50.0


def hourly_rate(num_gpus: int, rate_per_gpu: float = RATE_PER_GPU_HOUR) -> float:
    """What one hour of the whole box costs.

    The starter configs use two GPUs (one generating, one training), so the
    per-GPU rate is not the number you budget against.

        return num_gpus * rate_per_gpu

    Args:
        num_gpus: total GPUs rented.
        rate_per_gpu: dollars per GPU-hour.
    Returns:
        dollars per hour for the whole box.
    """
    # TODO: implement.
    return todo("hourly_rate: num_gpus * rate_per_gpu")


def completions_per_step(batch_size: int, group_size: int) -> int:
    """How many model answers get generated in one training step.

    Each step takes `batch_size` questions and samples `group_size` answers for
    each of them, because GRPO needs a group per question to compute a baseline
    (Unit 01, exercise 4).

        return batch_size * group_size

    For the reverse-text config that's 128 * 16 = 2048 completions per step,
    which is why generation gets its own GPU.

    Args:
        batch_size: questions per step.
        group_size: answers sampled per question.
    Returns:
        total completions generated per step.
    """
    # TODO: implement.
    return todo("completions_per_step: batch_size * group_size")


def affordable_steps(
    budget: float,
    seconds_per_step: float,
    num_gpus: int = 2,
    rate_per_gpu: float = RATE_PER_GPU_HOUR,
) -> int:
    """How many training steps fit in a budget.

        cost_per_second = hourly_rate(num_gpus, rate_per_gpu) / 3600
        cost_per_step   = cost_per_second * seconds_per_step
        return int(budget / cost_per_step + 1e-9)

    Round DOWN -- a step you can only half afford is a step you can't afford.
    If `seconds_per_step` is zero or negative, return 0 rather than dividing by
    zero.

    WHY THE 1e-9, because it looks like a fudge and isn't. At 2 GPUs and 30s a
    step, cost_per_step is exactly $0.04 in decimal -- but 0.04 has no exact
    binary representation, and the stored value sits a hair ABOVE it. So
    `50 // 0.04` is 1249, not 1250, even though `50 / 0.04` prints as 1250.0.
    Flooring a quantity that should land on a whole number is where this bites.
    The epsilon absorbs the representation error.

    Args:
        budget: dollars available.
        seconds_per_step: wall-clock seconds for one step.
        num_gpus: GPUs rented.
        rate_per_gpu: dollars per GPU-hour.
    Returns:
        whole steps affordable.
    """
    # TODO: implement, guarding against seconds_per_step <= 0.
    return todo("affordable_steps: budget // (cost per second * seconds per step)")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"{BOLD}1. What the box costs{RESET}\n")
    print(f"  {'GPUs':>5} | {'$/hour':>8} | {'hours from $50':>15}")
    print("  " + "-" * 34)
    for n in (1, 2, 4, 8):
        rate = hourly_rate(n)
        print(f"  {n:>5} | {rate:>8.2f} | {CREDIT / rate:>15.1f}")
    print(
        f"\n{DIM}  The starter configs need 2 GPUs: one generating answers, one training.\n"
        f"  So your $50 is about {CREDIT / hourly_rate(2):.0f} hours, not {CREDIT / hourly_rate(1):.0f}.{RESET}"
    )

    print(f"\n\n{BOLD}2. How much generation a step actually does{RESET}\n")
    print(f"  {'batch':>6} | {'group':>6} | {'completions/step':>17}")
    print("  " + "-" * 36)
    for b, gsz in [(128, 16), (128, 8), (64, 16), (32, 8), (256, 16)]:
        print(f"  {b:>6} | {gsz:>6} | {completions_per_step(b, gsz):>17,}")
    print(
        f"\n{DIM}  reverse-text is the first row: 2,048 completions per step. Halving the\n"
        f"  group size halves generation cost -- and costs you accuracy in the\n"
        f"  baseline, which is the tradeoff you measured in Unit 01, part 3.{RESET}"
    )

    print(f"\n\n{BOLD}3. What fits in the budget{RESET}\n")
    print(f"  {'sec/step':>9} | {'$/step':>8} | {'steps for $50':>14} | {'cost of 500':>12}")
    print("  " + "-" * 52)
    for sps in (5, 15, 30, 60, 180):
        per_step = hourly_rate(2) / 3600 * sps
        print(f"  {sps:>9} | {per_step:>8.3f} | {affordable_steps(CREDIT, sps):>14,} | "
              f"{per_step * 500:>11.2f}")

    print(
        f"\n{DIM}  Time one step on your actual run, then come back to this table. It's\n"
        f"  the difference between planning a capstone and hoping.{RESET}"
    )

    print(
        f"\n\n{BOLD}4. The three ways credits actually disappear{RESET}\n"
        "\n"
        f"  Idle box, forgotten overnight (14h):        ${hourly_rate(2) * 14:>7.2f}\n"
        f"  Two hours debugging a config typo:          ${hourly_rate(2) * 2:>7.2f}\n"
        f"  500 steps on a task with no reward spread:  ${hourly_rate(2) / 3600 * 30 * 500:>7.2f}\n"
        "\n"
        "  The first is the most common and the most avoidable. The third is the\n"
        "  one Units 03 and 04 exist to prevent: run the pass@k check before you\n"
        "  rent anything, because a task where every rollout scores the same\n"
        "  produces zero gradient no matter how long you train it."
    )


if __name__ == "__main__":
    run_main(main)
