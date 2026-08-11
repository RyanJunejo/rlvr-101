"""
Lab 2 — tasksets.

In v0 your data was a HuggingFace Dataset: a table, built up front, held in
memory. In v1 it's a Taskset, whose `load()` is a generator.

That difference buys two things.

LAZINESS. Nothing is built until something asks for it. A taskset backed by a
10GB download does the download when you iterate, not when you import.

INFINITE TASKSETS. A generator doesn't have to stop. If your questions are
procedurally generated -- random arithmetic, random game seeds -- there is no
reason to decide up front how many exist. Set INFINITE = True and yield forever;
callers take `.head(n)` when they need a finite slice.

That second one has no clean equivalent in v0, where you had to pick a dataset
size when you built the environment.

Fill in the two TODOs, then run me:

    uv run python modules/06-verifiers-v1/exercise_2_taskset.py
"""

from __future__ import annotations

import itertools
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exercise_1_port import SYSTEM_PROMPT, MathData, MathTask  # noqa: E402

PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 144 / 12?", "12"),
    ("What is 89 + 156?", "245"),
    ("What is 1000 - 377?", "623"),
    ("What is 13 * 13?", "169"),
]


class FixedMathTaskset(vf.Taskset[MathTask, vf.TasksetConfig]):
    """A taskset over the fixed list in PROBLEMS.

    Implement `load()`. It's a generator: for each entry in PROBLEMS, yield a
    MathTask wrapping a MathData.

        def load(self) -> Iterator[MathTask]:
            for i, (question, answer) in enumerate(PROBLEMS):
                yield MathTask(
                    MathData(
                        idx=i,
                        name=f"math#{i}",
                        prompt=question,
                        system_prompt=SYSTEM_PROMPT,
                        answer=answer,
                    )
                )

    `idx` and `name` aren't required, but set them. They're what identifies a
    task in the trace file when you're staring at 2000 rollouts trying to work
    out which question produced the weird one.

    Use `yield`, not a list. The whole point is that tasks are built as they're
    consumed.
    """

    def load(self) -> Iterator[MathTask]:
        # TODO: yield one MathTask per entry in PROBLEMS.
        return todo("FixedMathTaskset.load: yield a MathTask per PROBLEMS entry")


class InfiniteMathTaskset(vf.Taskset[MathTask, vf.TasksetConfig]):
    """A taskset that never runs out.

    Set the class attribute `INFINITE = True`, then implement `load()` to yield
    randomly generated multiplication questions forever.

        INFINITE = True

        def load(self) -> Iterator[MathTask]:
            rng = random.Random(0)
            for i in itertools.count():
                a, b = rng.randint(11, 99), rng.randint(11, 99)
                yield MathTask(
                    MathData(
                        idx=i,
                        name=f"gen#{i}",
                        prompt=f"What is {a} * {b}?",
                        system_prompt=SYSTEM_PROMPT,
                        answer=str(a * b),
                    )
                )

    `itertools.count()` counts up forever. The seeded `random.Random(0)` means
    two runs produce the same questions, which is what makes a published task
    reproducible.

    Note that nothing here decides how many questions exist. Whoever iterates
    decides, with `.head(n)`.
    """

    # TODO: set INFINITE and implement load().

    def load(self) -> Iterator[MathTask]:
        return todo("InfiniteMathTaskset: set INFINITE = True and yield forever")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"{BOLD}1. A fixed taskset{RESET}\n")
    ts = FixedMathTaskset(vf.TasksetConfig())
    tasks = list(ts)
    print(f"  INFINITE: {ts.INFINITE}")
    print(f"  tasks:    {len(tasks)}")
    for t in tasks[:3]:
        print(f"    [{t.data.idx}] {t.data.name:<8} {t.data.prompt:<24} -> {t.data.answer}")

    print(f"\n{BOLD}2. An infinite taskset{RESET}\n")
    inf = InfiniteMathTaskset(vf.TasksetConfig())
    print(f"  INFINITE: {inf.INFINITE}")
    first_five = list(inf.head(5))
    print(f"  .head(5) gave {len(first_five)} tasks:")
    for t in first_five:
        print(f"    [{t.data.idx}] {t.data.prompt:<24} -> {t.data.answer}")

    print(f"\n{BOLD}3. Reproducible?{RESET}\n")
    again = [t.data.prompt for t in InfiniteMathTaskset(vf.TasksetConfig()).head(5)]
    print(f"  same questions on a second construction: {again == [t.data.prompt for t in first_five]}")

    print(f"\n{BOLD}4. Laziness{RESET}\n")
    huge = InfiniteMathTaskset(vf.TasksetConfig())
    it = iter(huge)
    print(f"  took 3 from an infinite taskset: {[next(it).data.answer for _ in range(3)]}")
    print(
        f"\n{DIM}  That taskset has no end, and asking for three from it cost three\n"
        f"  objects. In v0 you'd have had to decide the dataset size when you\n"
        f"  built the environment, and materialise all of it.{RESET}"
    )

    print(
        f"\n{BOLD}Where this matters{RESET}\n"
        "\n"
        "  Unit 03 measured that a task teaches nothing once the model always wins\n"
        "  or always loses. Procedurally generated tasksets are how you respond:\n"
        "  make difficulty a config field, and generate harder questions as the\n"
        "  model improves rather than shipping a fixed file that goes stale.\n"
        "\n"
        "  You still have to choose the difficulty. The taskset just stops the\n"
        "  dataset itself from being the thing that limits you."
    )


if __name__ == "__main__":
    run_main(main)
