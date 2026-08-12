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
        import verifiers.v1 as vf

        import exercise_1_packaging as l1
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        cfg = l1.MultConfig()
        g.check("MultConfig() works bare (every field has a default)",
                True, "")
        fields = l1.MultConfig.model_fields
        g.check("the config declares num_tasks and digits",
                {"num_tasks", "digits"} <= set(fields), f"fields: {sorted(fields)}")
        g.check("defaults are num_tasks=64, digits=2",
                cfg.num_tasks == 64 and cfg.digits == 2,
                f"got num_tasks={getattr(cfg, 'num_tasks', None)}, "
                f"digits={getattr(cfg, 'digits', None)}")
    except NotImplementedYet as e:
        _skip(g, ["MultConfig is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["MultConfig is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        import inspect

        g.check("load() is a generator, not a list",
                inspect.isgeneratorfunction(l1.MultTaskset.load),
                "use `yield` so tasks are built as they're consumed")
        tasks = list(l1.MultTaskset(l1.MultConfig(num_tasks=6)))
        g.check("num_tasks is respected", len(tasks) == 6, f"got {len(tasks)}")
        g.check("generated answers are correct",
                all(str(eval(t.data.prompt.replace("What is ", "").rstrip("?")))  # noqa: S307
                    == t.data.answer for t in tasks),
                "the answer must match the question it was generated with")
        four = list(l1.MultTaskset(l1.MultConfig(num_tasks=3, digits=4)))
        g.check("digits controls difficulty",
                all(len(str(int(n))) == 4
                    for t in four
                    for n in t.data.prompt.replace("What is ", "").rstrip("?").split(" * ")),
                f"digits=4 should produce 4-digit operands; got "
                f"{four[0].data.prompt!r}")
        again = [t.data.prompt for t in l1.MultTaskset(l1.MultConfig(num_tasks=5))]
        first = [t.data.prompt for t in l1.MultTaskset(l1.MultConfig(num_tasks=5))]
        g.check("generation is seeded (reproducible)", again == first,
                "two people running 'the same' taskset must see the same questions")
    except NotImplementedYet as e:
        _skip(g, ["MultTaskset.load is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["MultTaskset.load is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        from verifiers.v1.utils.loaders import taskset_class

        cls = taskset_class("exercise_1_packaging")
        g.check("the module resolves through verifiers' own loader",
                cls is l1.MultTaskset,
                f"taskset_class() found {cls!r}; __all__ must export exactly the "
                f"one Taskset subclass")
    except Exception as e:
        _skip(g, ["the loader resolves the module"], f"{type(e).__name__}: {e}")

    try:
        txt = l1.make_pyproject("mult-task")
        g.check("pyproject declares a [project] section", "[project]" in txt, f"got:\n{txt[:200]}")
        g.check("pyproject sets the name", 'name = "mult-task"' in txt.replace("'", '"'),
                'expected name = "mult-task"')
        g.check("pyproject sets a version", "version" in txt, "expected a version field")
        g.check("pyproject pins a verifiers floor", "verifiers>=" in txt.replace(" ", ""),
                "declare verifiers with a version floor -- this course found three "
                "API behaviors that changed between releases, and your package "
                "inherits every one")
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
