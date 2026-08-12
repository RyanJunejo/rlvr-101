"""
ANSWER KEY -- Unit 05, Lab 2 — reading a training config.

The config file is the clearest description of what a training run will do. It's
also where the mistakes live, and every one you catch here is one you don't pay
$4.80/hour to discover.

The TOML below is the real `configs/basic/reverse-text/rl.toml` from the
prime-rl repository, quoted verbatim.

You'll write a validator. The most important check it performs is the one you
proved in Unit 01: a group size of 1 makes every advantage exactly zero, so the
run cannot learn anything. That's a config a person will absolutely write by
accident, and nothing about it looks wrong until you've spent the money.

Fill in the two TODOs, then run me:

    uv run python modules/05-training-on-gpu/exercise_2_config.py

TERMS USED IN THIS FILE

  orchestrator:  the process that hands out questions and runs your scoring
  trainer:       the process that computes gradients and updates weights
  learning rate: how big each weight update is
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

REVERSE_TEXT_TOML = """
max_steps = 20
seq_len = 2048

[deployment]
num_train_gpus = 1
num_infer_gpus = 1

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[orchestrator]
batch_size = 128
group_size = 16

[orchestrator.train.sampling]
max_completion_tokens = 128

[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text-v1"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[trainer.optim]
lr = 3e-6
"""


def summarize(cfg: dict) -> dict:
    """Pull the numbers that decide what a run does and what it costs.

    Return a dict with exactly these keys:

        model                str    cfg["model"]["name"]
        max_steps            int    cfg["max_steps"]
        batch_size           int    cfg["orchestrator"]["batch_size"]
        group_size           int    cfg["orchestrator"]["group_size"]
        total_gpus           int    num_train_gpus + num_infer_gpus, from [deployment]
        completions_per_step int    batch_size * group_size
        taskset_id           str    the taskset id (see below)
        learning_rate        float  cfg["trainer"]["optim"]["lr"]

    Getting the taskset id means walking the nested table. `[[...source]]` is an
    ARRAY of tables, so it parses to a list:

        cfg["orchestrator"]["train"]["source"][0]["env"]["taskset"]["id"]

    That nesting is not decoration -- a run can pull from several sources at
    once, which is how you'd train on a mix of tasks.

    Args:
        cfg: the parsed TOML.
    Returns:
        the summary dict described above.
    """
    orch = cfg["orchestrator"]
    dep = cfg.get("deployment", {})
    batch = orch["batch_size"]
    group = orch["group_size"]
    return {
        "model": cfg["model"]["name"],
        "max_steps": cfg["max_steps"],
        "batch_size": batch,
        "group_size": group,
        "total_gpus": dep.get("num_train_gpus", 0) + dep.get("num_infer_gpus", 0),
        "completions_per_step": batch * group,
        # `[[...source]]` is an array of tables, so this is a list -- a run can
        # draw from several task sources at once.
        "taskset_id": orch["train"]["source"][0]["env"]["taskset"]["id"],
        "learning_rate": cfg["trainer"]["optim"]["lr"],
    }


def validate(cfg: dict) -> list[str]:
    """Return a list of problems with this config. Empty list means it looks fine.

    Check for these, in this order, with these exact message prefixes:

      1. "FATAL: group_size is 1"
         If group_size < 2. This is the Unit 01 result: with one sample per
         question the group mean IS that sample, so every advantage is zero and
         the model never moves. The run will complete, report a flat reward
         curve, and teach nothing.

      2. "FATAL: needs at least 2 GPUs"
         If total GPUs (train + infer) is under 2. Generation and training run as
         separate processes and each needs a device.

      3. "WARNING: learning rate looks high"
         If lr > 1e-4. RL post-training nudges an already-trained model; rates
         from pretraining will wreck it.

      4. "WARNING: group_size is large"
         If group_size > 32. Generation cost is linear in group size while the
         baseline's accuracy improves like sqrt -- past ~16 you're mostly buying
         wall-clock. (Unit 01, part 3 measured the diminishing returns.)

    Return the messages as a list of strings. Order matters: fatals first, in
    the order above, then warnings.

    Args:
        cfg: the parsed TOML.
    Returns:
        list of problem descriptions, empty if none.
    """
    problems: list[str] = []
    orch = cfg.get("orchestrator", {})
    dep = cfg.get("deployment", {})
    group = orch.get("group_size", 1)
    gpus = dep.get("num_train_gpus", 0) + dep.get("num_infer_gpus", 0)
    lr = cfg.get("trainer", {}).get("optim", {}).get("lr", 0.0)

    if group < 2:
        problems.append(
            f"FATAL: group_size is {group}. Every advantage will be exactly zero "
            f"(Unit 01) and the run will train nothing."
        )
    if gpus < 2:
        problems.append(
            f"FATAL: needs at least 2 GPUs, config has {gpus}. Generation and "
            f"training are separate processes."
        )
    if lr > 1e-4:
        problems.append(
            f"WARNING: learning rate looks high ({lr}). RL post-training nudges an "
            f"already-trained model."
        )
    if group > 32:
        problems.append(
            f"WARNING: group_size is large ({group}). Generation cost is linear in "
            f"group size; baseline accuracy improves only like sqrt."
        )
    return problems


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def load(text: str) -> dict:
    return tomllib.loads(text)


def with_group_size(text: str, n: int) -> dict:
    cfg = load(text)
    cfg["orchestrator"]["group_size"] = n
    return cfg


def main() -> None:
    cfg = load(REVERSE_TEXT_TOML)

    print(f"{BOLD}1. What this run will do{RESET}\n")
    s = summarize(cfg)
    for k, v in s.items():
        shown = f"{v:,}" if isinstance(v, int) else v
        print(f"  {k:<22} {shown}")

    print(
        f"\n{DIM}  group_size is the G from Unit 01. Sixteen answers per question,\n"
        f"  their mean used as the baseline. batch_size x group_size = "
        f"{s['completions_per_step']:,}\n"
        f"  completions every single step, which is why generation gets its own GPU.{RESET}"
    )

    print(f"\n\n{BOLD}2. The validator on the real config{RESET}\n")
    problems = validate(cfg)
    print(f"  problems found: {problems if problems else 'none'}")

    print(f"\n\n{BOLD}3. The mistake that costs the most{RESET}\n")
    broken = with_group_size(REVERSE_TEXT_TOML, 1)
    for p in validate(broken):
        print(f"  {p}")
    print(
        f"\n{DIM}  Nothing about `group_size = 1` looks wrong. The run starts, completes,\n"
        f"  writes a checkpoint, and produces a perfectly flat reward curve -- because\n"
        f"  every advantage was exactly zero, as you measured in Unit 01. You'd pay\n"
        f"  full price to train nothing.{RESET}"
    )

    print(f"\n\n{BOLD}4. Other configs it catches{RESET}\n")
    cases = {
        "group_size = 64": with_group_size(REVERSE_TEXT_TOML, 64),
        "lr = 1e-3": {**load(REVERSE_TEXT_TOML), "trainer": {"optim": {"lr": 1e-3}}},
    }
    single = load(REVERSE_TEXT_TOML)
    single["deployment"] = {"num_train_gpus": 1, "num_infer_gpus": 0}
    cases["1 GPU total"] = single

    for label, c in cases.items():
        print(f"  {label:<18} -> {validate(c) or 'ok'}")

    print(
        f"\n\n{BOLD}Before you rent anything{RESET}\n"
        "\n"
        "  Run this validator on your config, then run the Unit 04 pass@k check on\n"
        "  your task. Between them they catch the two failures that produce a\n"
        "  completed run and no learning: a config that can't compute advantages,\n"
        "  and a task where every rollout scores the same."
    )


if __name__ == "__main__":
    main()
