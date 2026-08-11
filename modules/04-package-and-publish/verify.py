"""
Unit 04 autograder.

    uv run python modules/04-package-and-publish/verify.py

Runs entirely offline. No API key needed.
"""

from __future__ import annotations

import math
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


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: packaging --{RESET}")
    try:
        import exercise_1_packaging as l1
        import verifiers as vf
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        env = l1.load_environment()
        g.check("load_environment() works with NO arguments", isinstance(env, vf.Environment),
                f"returned {type(env).__name__}; tooling calls it bare to inspect your "
                f"task, so every argument needs a default")
        ds = env.get_dataset()
        g.check("the default builds 100 problems", len(ds) == 100, f"got {len(ds)} rows")
        g.check("the dataset has the required columns",
                {"question", "answer"} <= set(ds.column_names), f"got {ds.column_names}")
        g.check("a prompt column was built by the library",
                "prompt" in ds.column_names, f"got {ds.column_names}")
    except NotImplementedYet as e:
        _skip(g, ["load_environment is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["load_environment is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        g.check("num_problems is respected",
                len(l1.load_environment(num_problems=7).get_dataset()) == 7,
                "load_environment(num_problems=7) should yield 7 rows")

        a = list(l1.load_environment(num_problems=5, seed=42).get_dataset()["question"])
        b = list(l1.load_environment(num_problems=5, seed=42).get_dataset()["question"])
        c = list(l1.load_environment(num_problems=5, seed=99).get_dataset()["question"])
        g.check("the same seed gives the same questions", a == b,
                "a published task must be reproducible, or two people 'evaluating the "
                "same environment' aren't")
        g.check("a different seed gives different questions", a != c, "expected variation")

        g.check("each call returns a FRESH object",
                l1.load_environment(num_problems=2) is not l1.load_environment(num_problems=2),
                "returning a shared module-level object means concurrent training runs "
                "share mutable state")

        env_kw = l1.load_environment(num_problems=3, max_workers=7)
        g.check("**kwargs are forwarded to the environment",
                getattr(env_kw, "max_workers", None) == 7,
                "callers (including prime-rl) pass settings you didn't anticipate; "
                "forward **kwargs rather than swallowing them")
    except NotImplementedYet as e:
        _skip(g, ["load_environment handles arguments"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["load_environment handles arguments"], f"{type(e).__name__}: {e}")

    try:
        txt = l1.make_pyproject("mult-task")
        g.check("pyproject declares a [project] section", "[project]" in txt, f"got:\n{txt[:200]}")
        g.check("pyproject sets the name", 'name = "mult-task"' in txt.replace("'", '"'),
                "expected name = \"mult-task\"")
        g.check("pyproject sets a version", "version" in txt, "expected a version field")
        g.check("pyproject declares verifiers as a dependency", "verifiers" in txt,
                "someone else installs this on a machine you've never seen; anything "
                "undeclared becomes an ImportError in their training run")
    except NotImplementedYet as e:
        _skip(g, ["make_pyproject is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["make_pyproject is correct"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: evaluation --{RESET}")
    try:
        import exercise_2_evaluation as l2
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    try:
        g.close("standard_error(12, 20)", l2.standard_error(12, 20),
                math.sqrt(0.6 * 0.4 / 20), tol=1e-9)
        g.close("standard_error is 0 for a perfect score", l2.standard_error(20, 20), 0.0, tol=1e-12)
        g.close("standard_error handles n=0 without dividing by zero",
                l2.standard_error(0, 0), 0.0, tol=1e-12)
        big, small = l2.standard_error(60, 100), l2.standard_error(240, 400)
        g.check("4x the samples halves the standard error",
                abs(small - big / 2) < 1e-9,
                f"n=100 gave {big:.6f}, n=400 gave {small:.6f}; expected exactly half")
    except NotImplementedYet as e:
        _skip(g, ["standard_error is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["standard_error is correct"], f"{type(e).__name__}: {e}")

    try:
        lo, hi = l2.confidence_interval(12, 20)
        g.check("the interval brackets the observed rate", lo < 0.6 < hi,
                f"observed 60%, interval was [{lo:.3f}, {hi:.3f}]")
        g.close("interval width matches 2 * 1.96 * se", hi - lo,
                2 * 1.96 * math.sqrt(0.6 * 0.4 / 20), tol=1e-9)
        lo2, hi2 = l2.confidence_interval(20, 20)
        g.check("the interval is clamped to [0, 1]", 0.0 <= lo2 and hi2 <= 1.0,
                f"a 20/20 score gave [{lo2:.3f}, {hi2:.3f}]; rates above 100% aren't a thing")
        w_small = l2.confidence_interval(60, 100)
        w_big = l2.confidence_interval(1200, 2000)
        g.check("more samples gives a narrower interval",
                (w_big[1] - w_big[0]) < (w_small[1] - w_small[0]),
                f"n=100 width {w_small[1] - w_small[0]:.4f}, n=2000 width {w_big[1] - w_big[0]:.4f}")

        # the headline result of the unit
        a, b = l2.confidence_interval(12, 20), l2.confidence_interval(15, 20)
        g.check(
            "60% vs 75% at n=20 is NOT a distinguishable difference",
            l2.intervals_overlap(a, b),
            f"A=[{a[0]:.2f},{a[1]:.2f}] B=[{b[0]:.2f},{b[1]:.2f}] -- these should overlap. "
            f"This is the whole point of the lab.",
        )
    except NotImplementedYet as e:
        _skip(g, ["confidence_interval is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["confidence_interval is correct"], f"{type(e).__name__}: {e}")

    try:
        n_large = l2.samples_needed(0.50, 0.70)
        n_small = l2.samples_needed(0.68, 0.70)
        g.check("a big difference needs few samples", 50 < n_large < 200,
                f"50% vs 70% needed {n_large}; expected roughly 100")
        g.check("a small difference needs far more", n_small > 5000,
                f"68% vs 70% needed {n_small}; expected many thousands")
        g.check("halving the gap roughly quadruples the samples needed",
                3.0 < l2.samples_needed(0.65, 0.70) / l2.samples_needed(0.60, 0.70) < 5.0,
                f"got a ratio of "
                f"{l2.samples_needed(0.65, 0.70) / l2.samples_needed(0.60, 0.70):.2f}; "
                f"the difference is squared in the denominator, so expect ~4x")
        g.check("identical rates need an unbounded sample size",
                l2.samples_needed(0.7, 0.7) >= 10**8,
                "no sample size reliably distinguishes two identical things")
    except NotImplementedYet as e:
        _skip(g, ["samples_needed is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["samples_needed is correct"], f"{type(e).__name__}: {e}")


def check_lab3(g: Grader) -> None:
    print(f"\n{DIM}-- lab 3: pass@k --{RESET}")
    try:
        import exercise_3_pass_at_k as l3
    except Exception as e:
        _skip(g, ["lab 3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 3 imports", True)

    try:
        g.close("pass@1 equals the plain success rate", l3.pass_at_k(100, 20, 1), 0.20, tol=1e-9)
        g.close("pass@k is 0 when nothing succeeded", l3.pass_at_k(100, 0, 8), 0.0, tol=1e-12)
        g.close("pass@k is 1 when everything succeeded", l3.pass_at_k(100, 100, 8), 1.0, tol=1e-12)
        g.close("pass@k is 1 when failures are fewer than k",
                l3.pass_at_k(10, 8, 5), 1.0, tol=1e-12)
        g.check("pass@k is nan when k > n", math.isnan(l3.pass_at_k(5, 2, 8)),
                f"got {l3.pass_at_k(5, 2, 8)}; you can't estimate pass@8 from 5 samples")

        vals = [l3.pass_at_k(100, 20, k) for k in (1, 2, 4, 8, 16)]
        g.check("pass@k increases with k", all(a < b for a, b in zip(vals, vals[1:])),
                f"got {[round(v, 3) for v in vals]}")
        g.close("pass@8 for a 20% model", l3.pass_at_k(100, 20, 8), 0.8440, tol=5e-3)
    except NotImplementedYet as e:
        _skip(g, ["pass_at_k is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["pass_at_k is correct"], f"{type(e).__name__}: {e}")

    try:
        rng = np.random.default_rng(0)
        s = np.array([1.0] * 20 + [0.0] * 80)
        runs = np.array([l3.naive_pass_at_k(s, 8, rng) for _ in range(3000)])
        g.check("naive_pass_at_k returns only 0.0 or 1.0", set(np.unique(runs)) <= {0.0, 1.0},
                f"got values {np.unique(runs)[:5]}")
        exact = l3.pass_at_k(100, 20, 8)
        g.check("naive_pass_at_k agrees with the exact estimator on average",
                abs(runs.mean() - exact) < 0.05,
                f"naive averaged {runs.mean():.3f}, exact is {exact:.3f}; both are "
                f"unbiased so they should agree")
        g.check("naive_pass_at_k has much higher variance",
                runs.std() > 0.2,
                f"std was {runs.std():.3f}; the point of this comparison is that a "
                f"single naive run is a coin flip")
        allzero = np.zeros(50)
        g.close("naive_pass_at_k is 0 when nothing succeeded",
                l3.naive_pass_at_k(allzero, 8, rng), 0.0, tol=1e-12)
    except NotImplementedYet as e:
        _skip(g, ["naive_pass_at_k is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["naive_pass_at_k is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 04 — packaging and evaluation")
    check_lab1(g)
    check_lab2(g)
    check_lab3(g)
    report(g)


if __name__ == "__main__":
    main()
