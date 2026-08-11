"""
Unit 05 autograder.

    uv run python modules/05-training-on-gpu/verify.py

Runs offline. The GPU runbook in section 5 of the lecture notes is verified by
your own training curve, not by this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from common.grading import DIM, RESET, Grader, NotImplementedYet, report  # noqa: E402


def _skip(g: Grader, names: list[str], why: str) -> None:
    for n in names:
        g.check(n, False, why)


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: budget --{RESET}")
    try:
        import exercise_1_budget as l1
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        g.close("hourly_rate for 2 GPUs", l1.hourly_rate(2), 4.80, tol=1e-9)
        g.close("hourly_rate for 8 GPUs", l1.hourly_rate(8), 19.20, tol=1e-9)
        g.close("hourly_rate honours a custom rate", l1.hourly_rate(2, 1.50), 3.00, tol=1e-9)
    except NotImplementedYet as e:
        _skip(g, ["hourly_rate is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["hourly_rate is correct"], f"{type(e).__name__}: {e}")

    try:
        g.check("completions_per_step for reverse-text (128 x 16)",
                l1.completions_per_step(128, 16) == 2048,
                f"got {l1.completions_per_step(128, 16)}, expected 2048")
        g.check("halving the group size halves generation",
                l1.completions_per_step(128, 8) == 1024,
                f"got {l1.completions_per_step(128, 8)}")
    except NotImplementedYet as e:
        _skip(g, ["completions_per_step is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["completions_per_step is correct"], f"{type(e).__name__}: {e}")

    try:
        # $4.80/h = $0.001333/s; 30s/step = $0.04/step; $50 / 0.04 = 1250
        got = l1.affordable_steps(50.0, 30.0)
        g.check("affordable_steps: $50 at 30s/step on 2 GPUs", got == 1250,
                f"got {got}, expected 1250")
        g.check("affordable_steps rounds down",
                l1.affordable_steps(1.0, 30.0) == 25,
                f"got {l1.affordable_steps(1.0, 30.0)}, expected 25 (not 25.x)")
        g.check("affordable_steps handles a zero step time",
                l1.affordable_steps(50.0, 0.0) == 0,
                "guard against dividing by zero rather than raising")
        g.check("more GPUs means fewer affordable steps",
                l1.affordable_steps(50.0, 30.0, num_gpus=8)
                < l1.affordable_steps(50.0, 30.0, num_gpus=2),
                "an 8-GPU box burns budget four times faster than a 2-GPU one")
    except NotImplementedYet as e:
        _skip(g, ["affordable_steps is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["affordable_steps is correct"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: reading a config --{RESET}")
    try:
        import exercise_2_config as l2
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    cfg = l2.load(l2.REVERSE_TEXT_TOML)

    try:
        s = l2.summarize(cfg)
        expected = {
            "model": "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
            "max_steps": 20,
            "batch_size": 128,
            "group_size": 16,
            "total_gpus": 2,
            "completions_per_step": 2048,
            "taskset_id": "reverse-text-v1",
            "learning_rate": 3e-6,
        }
        g.check("summarize returns every required key",
                set(s) == set(expected), f"got keys {sorted(s)}")
        for k, want in expected.items():
            if k in s:
                g.check(f"summarize: {k}", s[k] == want, f"got {s[k]!r}, expected {want!r}")
    except NotImplementedYet as e:
        _skip(g, ["summarize is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["summarize is correct"], f"{type(e).__name__}: {e}")

    try:
        g.check("the real config validates clean", l2.validate(cfg) == [],
                f"got {l2.validate(cfg)}")

        broken = l2.with_group_size(l2.REVERSE_TEXT_TOML, 1)
        problems = l2.validate(broken)
        g.check("group_size = 1 is caught",
                any(p.startswith("FATAL: group_size is 1") for p in problems),
                f"got {problems}. This is the Unit 01 result and the most expensive "
                f"config mistake there is -- the run completes and teaches nothing.")

        single = l2.load(l2.REVERSE_TEXT_TOML)
        single["deployment"] = {"num_train_gpus": 1, "num_infer_gpus": 0}
        g.check("a 1-GPU config is caught",
                any(p.startswith("FATAL: needs at least 2 GPUs") for p in l2.validate(single)),
                f"got {l2.validate(single)}")

        hot = l2.load(l2.REVERSE_TEXT_TOML)
        hot["trainer"]["optim"]["lr"] = 1e-3
        g.check("a high learning rate is warned about",
                any(p.startswith("WARNING: learning rate looks high") for p in l2.validate(hot)),
                f"got {l2.validate(hot)}")

        big = l2.with_group_size(l2.REVERSE_TEXT_TOML, 64)
        g.check("an oversized group is warned about",
                any(p.startswith("WARNING: group_size is large") for p in l2.validate(big)),
                f"got {l2.validate(big)}")

        both = l2.with_group_size(l2.REVERSE_TEXT_TOML, 1)
        both["deployment"] = {"num_train_gpus": 1, "num_infer_gpus": 0}
        ps = l2.validate(both)
        g.check("fatals are reported before warnings, in order",
                len(ps) >= 2 and ps[0].startswith("FATAL: group_size")
                and ps[1].startswith("FATAL: needs at least 2 GPUs"),
                f"got {ps}")
    except NotImplementedYet as e:
        _skip(g, ["validate is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["validate is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 05 — training for real")
    check_lab1(g)
    check_lab2(g)
    report(g)
    print(
        f"\n{DIM}The GPU runbook (lecture notes, section 5) isn't autograded. Its\n"
        f"checkpoints are: two GPUs visible, `uv run rl --help` works, and a reward\n"
        f"curve that rises over 20 steps.{RESET}"
    )


if __name__ == "__main__":
    main()
