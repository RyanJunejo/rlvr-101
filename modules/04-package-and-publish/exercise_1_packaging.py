"""
Lab 1 — packaging a task so someone else can run it.

The contract is one sentence: a taskset is an installable module that exports
exactly one `Taskset` subclass through `__all__`. That's what the eval CLI, the
trainer, and the Environments Hub all resolve. There is no registration step,
no manifest -- the loader imports your module, reads `__all__`, and finds the
one class that subclasses `Taskset`.

The second half of the contract is the CONFIG. Your `TasksetConfig` subclass is
the public API of your task: every field on it becomes a knob a runner can turn
from a TOML file or a CLI flag, without touching your code. The one field this
lab insists on is `digits` -- a difficulty knob -- because Unit 03 measured what
happens to a task with no difficulty control: the model saturates it, every
group scores the same, and it teaches nothing forever after.

Fill in the three TODOs, then run me:

    uv run python modules/04-package-and-publish/exercise_1_packaging.py

TERMS USED IN THIS FILE

  taskset: the generator of tasks; a publishable one is a module exporting
           exactly one Taskset subclass via __all__
  config:  the public API of your task -- every field is a knob a runner can
           turn without editing your code
"""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "Solve the problem. Think briefly, then end your reply with exactly:\n"
    "Answer: <number>"
)


class MultConfig(vf.TasksetConfig):
    """The public API of the task. Declare two fields, pydantic-style:

        num_tasks: int = 64
        digits: int = 2

    Every field here is runner-tunable: `--env.taskset.digits 4` on the CLI, or
    `digits = 4` under `[env.taskset]` in a training TOML. Defaults matter --
    a bare `MultConfig()` must work, because tooling constructs it that way to
    inspect your task.

    `digits` is the difficulty knob. Two-digit multiplication is saturated for
    strong models (a hosted DeepSeek scored 12/12 on it in Unit 02's live run);
    four-digit is out of reach for small ones. The knob is what lets a user find
    the band in between -- and per Unit 03, outside that band a task teaches
    nothing.
    """

    # TODO: declare num_tasks and digits with defaults.


class MultData(vf.TaskData):
    answer: str


class MultTask(vf.Task[MultData, vf.State, vf.TaskConfig]):
    """Written for you -- Unit 02's grader, unchanged. Packaging is the lesson
    here, not scoring."""

    @vf.reward
    async def correct_answer(self, task: MultData, trace: vf.Trace) -> float:
        matches = ANSWER_RE.findall(trace.last_reply)
        if not matches:
            return 0.0
        return 1.0 if matches[-1].strip().rstrip(".").strip() == task.answer else 0.0


class MultTaskset(vf.Taskset[MultTask, MultConfig]):
    """The generator of tasks.

    Implement `load()` as a generator:

        rng = random.Random(0)
        lo, hi = 10 ** (self.config.digits - 1), 10 ** self.config.digits - 1
        for i in range(self.config.num_tasks):
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            yield MultTask(
                MultData(idx=i, prompt=f"What is {a} * {b}?",
                         system_prompt=SYSTEM_PROMPT, answer=str(a * b)),
                self.config.task,
            )

    Three things this snippet gets right that are easy to get wrong:

      - SEEDED. Two people running "the same" taskset must see the same
        questions, or their numbers aren't comparable. The seed makes a
        published task reproducible.
      - LAZY. `yield`, not a list. Tasks are built as they're consumed --
        nothing happens at import, and a `num_tasks=100000` config costs
        nothing until someone iterates.
      - CONFIG-DRIVEN. Both knobs come from `self.config`. The code has no
        opinions a runner can't override.
    """

    def load(self) -> Iterator[MultTask]:
        # TODO: yield seeded tasks at the configured difficulty.
        return todo("MultTaskset.load: seeded questions from num_tasks and digits")


# The whole packaging contract, in one line:
__all__ = ["MultTaskset"]


def make_pyproject(name: str = "mult-task") -> str:
    """The file that makes this directory installable.

    Return a string containing, at minimum:
      - a [project] section with `name = "<name>"` and a `version`
      - a `dependencies` list that includes "verifiers"

    A minimal one:

        [project]
        name = "mult-task"
        version = "0.1.0"
        dependencies = ["verifiers>=0.3.0"]

    Why dependencies matter more than usual here: someone else's training run
    installs your package on a machine you've never seen. Anything you forgot
    to declare -- but which happened to be installed on your laptop -- becomes
    an ImportError six hours into their GPU job. And PIN THE VERIFIERS FLOOR:
    this course found three API behaviors that changed between verifiers
    releases, and your package inherits every one of them.

    Args:
        name: the package name. Hyphens are fine -- the Hub id is the package
              name, and the importable module name swaps them for underscores.
    Returns:
        the pyproject.toml contents as a string.
    """
    # TODO: return a valid pyproject.toml string.
    return todo("make_pyproject: [project] with name, version, dependencies")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"{BOLD}1. The config is the public API{RESET}\n")
    for label, cfg in [("defaults", MultConfig()),
                       ("--env.taskset.digits 4", MultConfig(digits=4)),
                       ("small smoke set", MultConfig(num_tasks=4, digits=3))]:
        ts = MultTaskset(cfg)
        first = next(iter(ts))
        print(f"  {label:<26} -> {first.data.prompt:<26} answer={first.data.answer}")

    print(f"\n{BOLD}2. Reproducible?{RESET}\n")
    a = [t.data.prompt for t in MultTaskset(MultConfig(num_tasks=5))]
    b = [t.data.prompt for t in MultTaskset(MultConfig(num_tasks=5))]
    print(f"  two constructions, same questions: {a == b}")

    print(f"\n{BOLD}3. The loader finds it{RESET}\n")
    from verifiers.v1.utils.loaders import taskset_class

    cls = taskset_class("exercise_1_packaging")
    print(f"  taskset_class('exercise_1_packaging') -> {cls.__name__}")
    print(
        f"\n{DIM}  That call is exactly what the eval CLI and prime-rl do with a taskset\n"
        f"  id: import the module, read __all__, find the one Taskset subclass.\n"
        f"  A Hub id like org/mult-task@1.0 adds only an install step first.{RESET}"
    )

    print(f"\n{BOLD}4. The packaging file{RESET}\n")
    for line in make_pyproject().strip().splitlines():
        print(f"  {line}")

    print(
        f"\n\n{BOLD}To actually publish{RESET}\n"
        "\n"
        "  uv tool install prime\n"
        "  prime login\n"
        "  prime env init mult-task     # scaffolds the package directory\n"
        "  # move your taskset module + pyproject in\n"
        "  prime env push               # publishes to the Environments Hub\n"
        "\n"
        "  Others then use it by id -- `uv run eval yourname/mult-task` or\n"
        "  `taskset.id = \"yourname/mult-task\"` in a training config -- and the\n"
        "  loader installs it on first use.\n"
        "\n"
        "  Before you push, the Unit 03 question: at which `digits` does a model\n"
        "  you care about succeed SOMETIMES? Publish that number in your README.\n"
        "  A task without a known working difficulty band is one nobody can\n"
        "  decide whether to train on."
    )


if __name__ == "__main__":
    run_main(main)
