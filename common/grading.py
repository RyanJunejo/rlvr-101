"""
Tiny grading helpers. The whole point of this file is that when your code is
wrong, the failure message tells you *which property* broke and *what it saw* --
not just `AssertionError`.

Nothing clever in here. Read it once, then forget about it.
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Callable

# ANSI colors, degrade gracefully if piped to a file
_TTY = sys.stdout.isatty()
GREEN = "\033[32m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


class NotImplementedYet(Exception):
    """Raised by starter code where you still have a TODO."""


def todo(what: str) -> Any:
    """Call this in starter code where the learner has to fill something in.

    Using an exception (rather than returning None) means verify.py can tell
    the difference between "not attempted yet" and "attempted and wrong",
    and print a much friendlier message for the first case.
    """
    raise NotImplementedYet(what)


class Grader:
    """Collects pass/fail checks and prints a readable report."""

    def __init__(self, title: str):
        self.title = title
        self.checks: list[tuple[bool, str, str]] = []  # (ok, name, detail)
        self.skipped: list[str] = []
        print(f"\n{BOLD}{title}{RESET}")
        print("=" * len(title))

    # -- individual assertions -------------------------------------------------

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((bool(ok), name, detail))
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {name}")
        if not ok and detail:
            print(f"         {DIM}{detail}{RESET}")
        return bool(ok)

    def close(self, name: str, got: float, want: float, tol: float = 1e-6) -> bool:
        """Assert a scalar is near a target, and say by how much it missed."""
        if got is None or (isinstance(got, float) and math.isnan(got)):
            return self.check(name, False, f"got {got!r}, expected a number near {want}")
        ok = abs(got - want) <= tol
        return self.check(
            name, ok, f"expected {want} (+/- {tol}), got {got:.6g} -- off by {abs(got - want):.3g}"
        )

    def between(self, name: str, got: float, lo: float, hi: float) -> bool:
        ok = got is not None and lo <= got <= hi
        return self.check(name, ok, f"expected a value in [{lo}, {hi}], got {got!r}")

    def greater(self, name: str, got: float, than: float) -> bool:
        ok = got is not None and got > than
        return self.check(name, ok, f"expected something greater than {than}, got {got!r}")

    def shape(self, name: str, arr: Any, want: tuple) -> bool:
        got = getattr(arr, "shape", None)
        ok = got == want
        return self.check(name, ok, f"expected array of shape {want}, got shape {got}")

    def run(self, name: str, fn: Callable[[], None]) -> bool:
        """Run a check that might raise. NotImplementedYet is reported as 'not done yet'."""
        try:
            fn()
            return self.check(name, True)
        except NotImplementedYet as e:
            self.skipped.append(name)
            print(f"  [{YELLOW}TODO{RESET}] {name}")
            print(f"         {DIM}still to implement: {e}{RESET}")
            self.checks.append((False, name, "not implemented"))
            return False
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc().strip().splitlines()
            # surface the last line of *your* code, not ours
            frame = [ln.strip() for ln in tb if "grading.py" not in ln and ln.strip().startswith("File")]
            if frame:
                detail += f"\n         {frame[-1]}"
            return self.check(name, False, detail)

    # -- report ----------------------------------------------------------------

    def summary(self) -> int:
        passed = sum(1 for ok, _, _ in self.checks if ok)
        total = len(self.checks)
        print()
        if passed == total:
            print(f"{GREEN}{BOLD}All {total} checks passed.{RESET} Nice.")
            return 0
        if self.skipped and passed + len(self.skipped) == total:
            print(
                f"{YELLOW}{passed}/{total} passed{RESET} -- the rest are still TODO. "
                f"Keep going, then re-run me."
            )
            return 1
        failed = [name for ok, name, _ in self.checks if not ok and name not in self.skipped]
        print(f"{RED}{BOLD}{passed}/{total} checks passed.{RESET}")
        if failed:
            print(f"{DIM}Still broken: {', '.join(failed)}{RESET}")
        return 1


def report(grader: Grader) -> None:
    """Exit with the right status code so you can chain these in a shell."""
    sys.exit(grader.summary())


def run_main(fn: Callable[[], None]) -> None:
    """Entry point for exercise files.

    Runs `fn`, but turns an unfinished TODO into a short friendly note instead of
    a wall of traceback. Any *other* exception is left alone -- if you've written
    a real bug, you want to see exactly where it is.
    """
    try:
        fn()
    except NotImplementedYet as e:
        print(
            f"\n{YELLOW}Stopped at a TODO:{RESET} {e}\n"
            f"{DIM}Fill it in and run this file again. "
            f"`verify.py` in the same directory will grade you.{RESET}"
        )
        sys.exit(1)
