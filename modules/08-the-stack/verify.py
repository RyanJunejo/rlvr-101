"""
Unit 08 autograder.

    uv run python modules/08-the-stack/verify.py

Runs offline. Checks against the installed package, so it stays honest as the
package changes.
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
    print(f"\n{DIM}-- lab 1: reading the package --{RESET}")
    try:
        import exercise_1_read_the_package as l1
        from verifiers.v1 import harnesses, judges
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        hs = l1.list_plugins(harnesses)
        g.check("list_plugins returns a sorted list", hs == sorted(hs), f"got {hs}")
        g.check("it finds the harnesses that must exist",
                {"null", "bash", "claude_code", "rlm"} <= set(hs),
                f"found {hs}; expected at least null, bash, claude_code, rlm")
        # cross-check against the library's OWN discovery function
        from verifiers.v1.utils.loaders import builtin_harness_ids

        g.check("it agrees with verifiers' own builtin_harness_ids()",
                hs == builtin_harness_ids(),
                f"yours: {hs}\n         library's: {builtin_harness_ids()}")
        js = l1.list_plugins(judges)
        g.check("it works on the judges package too",
                {"reference", "rubric"} <= set(js), f"found {js}")
    except NotImplementedYet as e:
        _skip(g, ["list_plugins is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["list_plugins is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        rs = l1.runtime_kinds()
        g.check("runtime_kinds is sorted and lowercase",
                rs == sorted(rs) and all(r.islower() for r in rs), f"got {rs}")
        g.check("it finds the four runtime kinds",
                {"docker", "modal", "prime", "subprocess"} <= set(rs), f"got {rs}")
        g.check("it excludes the abstract base and the network policy",
                "runtime" not in rs and "networkpolicy" not in rs,
                f"got {rs}; RuntimeConfig and NetworkPolicyConfig aren't runtime kinds")
    except NotImplementedYet as e:
        _skip(g, ["runtime_kinds is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["runtime_kinds is correct"], f"{type(e).__name__}: {e}")

    try:
        cases = [
            ("null", "trivial"),
            ("claude_code", "coding"),
            ("codex", "coding"),
            ("mini_swe_agent", "coding"),
            ("browser_use", "browser"),
            ("rlm", "research"),
            ("bash", "plumbing"),
            ("something_new", "unknown"),
        ]
        for harness, want in cases:
            got = l1.classify(harness)
            g.check(f"classify({harness!r}) -> {want!r}", got == want, f"got {got!r}")

        # every shipped harness must land somewhere real -- this is the check
        # that will fail the day they add a fifteenth, which is the point.
        unknown = [h for h in l1.list_plugins(harnesses) if l1.classify(h) == "unknown"]
        g.check(
            "every shipped harness is classified",
            not unknown,
            f"unclassified: {unknown}. If this fails, the package added a harness "
            f"since this unit was written (2026-08-12) -- go look at what it is. "
            f"That's the unit working, not breaking.",
        )
    except NotImplementedYet as e:
        _skip(g, ["classify is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["classify is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 08 — the whole stack")
    check_lab1(g)
    report(g)


if __name__ == "__main__":
    main()
