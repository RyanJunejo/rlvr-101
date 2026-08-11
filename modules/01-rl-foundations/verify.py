"""
Module 01 self-check.

    uv run python modules/01-rl-foundations/verify.py

Checks properties, not exact implementations -- there's more than one right way
to write most of these. When something fails it tells you which property broke
and what it saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from common.grading import DIM, RESET, Grader, NotImplementedYet, report  # noqa: E402


def _skip(g: Grader, names: list[str], why: str) -> None:
    for n in names:
        g.check(n, False, why)


def check_ex1(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 1: bandit --{RESET}")
    try:
        import exercise_1_bandit as ex1
    except Exception as e:
        _skip(g, ["ex1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex1 imports", True)

    rng = np.random.default_rng(0)
    Q = np.array([0.1, 0.9, 0.3])

    try:
        greedy = [ex1.select_action(Q, 0.0, rng) for _ in range(20)]
        g.check(
            "select_action with epsilon=0 always exploits",
            all(a == 1 for a in greedy),
            f"argmax(Q) is 1 for Q={Q}, but got choices {sorted(set(greedy))}",
        )
        explore = [ex1.select_action(Q, 1.0, rng) for _ in range(300)]
        g.check(
            "select_action with epsilon=1 explores all arms",
            len(set(explore)) == 3,
            f"expected all 3 arms to appear over 300 draws, saw {sorted(set(explore))}",
        )
        mixed = [ex1.select_action(Q, 0.5, rng) for _ in range(4000)]
        frac_best = sum(1 for a in mixed if a == 1) / len(mixed)
        # exploit half the time + 1/3 of the exploring half ~= 0.667
        g.between("select_action with epsilon=0.5 picks best ~2/3 of the time", frac_best, 0.60, 0.73)
    except NotImplementedYet as e:
        _skip(g, ["select_action behaves correctly"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["select_action behaves correctly"], f"{type(e).__name__}: {e}")

    try:
        g.close("update_estimate: first pull adopts the reward", ex1.update_estimate(0.0, 1.0, 1), 1.0)
        g.close("update_estimate: second pull averages", ex1.update_estimate(1.0, 0.0, 2), 0.5)
        g.close("update_estimate: converges toward reward", ex1.update_estimate(0.5, 1.0, 4), 0.625)
    except NotImplementedYet as e:
        _skip(g, ["update_estimate is the incremental mean"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["update_estimate is the incremental mean"], f"{type(e).__name__}: {e}")


def check_ex2(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 2: REINFORCE --{RESET}")
    try:
        import exercise_2_reinforce as ex2
    except Exception as e:
        _skip(g, ["ex2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex2 imports", True)

    try:
        p = ex2.softmax(np.array([1.0, 2.0, 3.0]))
        g.close("softmax sums to 1", float(p.sum()), 1.0, tol=1e-9)
        want = np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()
        g.close("softmax matches the reference values", float(np.abs(p - want).max()), 0.0, tol=1e-9)
        big = ex2.softmax(np.array([1000.0, 1001.0, 1002.0]))
        g.check(
            "softmax is numerically stable on large logits",
            bool(np.all(np.isfinite(big))) and abs(float(big.sum()) - 1.0) < 1e-9,
            f"got {big} -- subtract max(logits) before exponentiating",
        )
    except NotImplementedYet as e:
        _skip(g, ["softmax is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["softmax is correct"], f"{type(e).__name__}: {e}")

    try:
        probs = np.array([0.1, 0.6, 0.3])
        gl = ex2.grad_log_prob(probs, 1)
        g.shape("grad_log_prob returns a K-vector", np.asarray(gl), (3,))
        g.close("grad_log_prob sums to zero", float(np.sum(gl)), 0.0, tol=1e-9)
        expect = np.array([-0.1, 0.4, -0.3])
        g.close(
            "grad_log_prob equals onehot(a) - probs",
            float(np.abs(np.asarray(gl) - expect).max()),
            0.0,
            tol=1e-9,
        )
    except NotImplementedYet as e:
        _skip(g, ["grad_log_prob is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["grad_log_prob is correct"], f"{type(e).__name__}: {e}")

    try:
        rng = np.random.default_rng(1)
        draws = [ex2.sample_action(np.array([0.0, 0.0, 1.0]), rng) for _ in range(50)]
        g.check(
            "sample_action respects the distribution",
            all(d == 2 for d in draws),
            f"with probs=[0,0,1] every draw must be index 2, got {sorted(set(draws))}",
        )
    except NotImplementedYet as e:
        _skip(g, ["sample_action is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["sample_action is correct"], f"{type(e).__name__}: {e}")

    try:
        out = ex2.train(steps=3000, seed=0)
        p_best = float(out["probs"][4])
        g.greater("REINFORCE learns to prefer the best token (p > 0.5)", p_best, 0.5)
        h = out["history"]
        g.greater(
            "REINFORCE improves mean reward over training",
            float(h[-200:].mean() - h[:200].mean()),
            0.05,
        )
    except NotImplementedYet as e:
        _skip(g, ["REINFORCE actually trains"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["REINFORCE actually trains"], f"{type(e).__name__}: {e}")


def check_ex3(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 3: baselines --{RESET}")
    try:
        import exercise_3_baseline as ex3
    except NotImplementedYet as e:
        _skip(g, ["ex3 imports"], f"depends on exercise 2, still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["ex3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex3 imports", True)

    try:
        s = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
        got = ex3.total_variance(s)
        want = float(s.var(axis=0).sum())
        g.close("total_variance sums per-component variances", float(got), want, tol=1e-9)
    except NotImplementedYet as e:
        _skip(g, ["total_variance is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["total_variance is correct"], f"{type(e).__name__}: {e}")

    try:
        theta = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        out = ex3.compare(theta, ex3.REWARDS, n=8000, seed=3)
        g.check(
            "a baseline reduces estimator variance",
            out["var_with"] < out["var_without"],
            f"variance with baseline ({out['var_with']:.5f}) should be LOWER than "
            f"without ({out['var_without']:.5f})",
        )
        # unbiasedness: both estimators must point the same way
        diff = float(np.abs(out["mean_without"] - out["mean_with"]).max())
        g.check(
            "the baseline does not bias the gradient",
            diff < 0.02,
            f"mean gradients differ by {diff:.4f}; both estimators are unbiased so "
            f"they should agree closely. A large gap means the baseline is being "
            f"applied to the wrong term.",
        )
        shifted = ex3.compare(theta, ex3.REWARDS + 10.0, n=8000, seed=3)
        g.greater(
            "shifting rewards by +10 inflates variance without a baseline",
            shifted["var_without"] / out["var_without"],
            5.0,
        )
        g.check(
            "with a baseline, the +10 shift is harmless",
            abs(shifted["var_with"] - out["var_with"]) < 0.05 * max(out["var_with"], 1e-9) + 1e-6,
            f"variance with baseline changed from {out['var_with']:.5f} to "
            f"{shifted['var_with']:.5f}; a constant shift should cancel exactly",
        )
    except NotImplementedYet as e:
        _skip(g, ["baseline behaviour is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["baseline behaviour is correct"], f"{type(e).__name__}: {e}")


def check_ex4(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 4: GRPO --{RESET}")
    try:
        import exercise_4_grpo as ex4
    except NotImplementedYet as e:
        _skip(g, ["ex4 imports"], f"depends on exercise 2, still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["ex4 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex4 imports", True)

    try:
        r = np.array([0.0, 0.5, 1.0, 0.25])
        A = np.asarray(ex4.group_advantages(r))
        g.shape("group_advantages preserves shape", A, (4,))
        g.close("advantages have mean 0", float(A.mean()), 0.0, tol=1e-6)
        g.close("advantages have unit std", float(A.std()), 1.0, tol=1e-4)
        g.check(
            "advantages preserve reward ordering",
            bool(np.all(np.argsort(A) == np.argsort(r))),
            f"rewards {r} rank as {np.argsort(r)}, advantages {np.round(A, 3)} "
            f"rank as {np.argsort(A)}",
        )

        A10 = np.asarray(ex4.group_advantages(r * 10.0))
        g.close(
            "advantages are scale-invariant (x10 rewards -> same advantages)",
            float(np.abs(A - A10).max()),
            0.0,
            tol=1e-5,
        )

        flat = np.asarray(ex4.group_advantages(np.array([0.7, 0.7, 0.7, 0.7])))
        g.check(
            "a group with no reward spread yields exactly zero advantage",
            bool(np.all(flat == 0.0)),
            f"expected all zeros for identical rewards, got {flat}. Branch explicitly "
            f"on std == 0 rather than relying on the epsilon.",
        )
        allzero = np.asarray(ex4.group_advantages(np.array([0.0, 0.0, 0.0])))
        g.check(
            "an all-zero-reward group yields zero advantage",
            bool(np.all(allzero == 0.0)),
            f"got {allzero}",
        )
    except NotImplementedYet as e:
        _skip(g, ["group_advantages is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["group_advantages is correct"], f"{type(e).__name__}: {e}")

    try:
        rng = np.random.default_rng(0)
        theta = np.zeros(5)
        new_theta, _ = ex4.grpo_step(theta.copy(), rng, group_size=1, lr=0.5)
        g.close(
            "GRPO with group_size=1 produces no update (advantage is always 0)",
            float(np.abs(np.asarray(new_theta) - theta).max()),
            0.0,
            tol=1e-12,
        )

        out = ex4.train(steps=600, group_size=8, seed=0)
        g.greater("GRPO learns to prefer the best token (p > 0.5)", float(out["probs"][4]), 0.5)
        h = out["history"]
        g.greater(
            "GRPO improves mean group reward over training",
            float(h[-50:].mean() - h[:50].mean()),
            0.05,
        )

        flat_run = ex4.train(steps=200, group_size=1, seed=0)
        g.close(
            "training with group_size=1 leaves the policy uniform",
            float(np.abs(np.asarray(flat_run["probs"]) - 0.2).max()),
            0.0,
            tol=1e-9,
        )
    except NotImplementedYet as e:
        _skip(g, ["grpo_step is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["grpo_step is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Module 01 — RL foundations")
    check_ex1(g)
    check_ex2(g)
    check_ex3(g)
    check_ex4(g)
    report(g)


if __name__ == "__main__":
    main()
