"""
Lab 1 — packaging a task so someone else can use it.

A `verifiers` task is an ordinary installable Python package with exactly one
requirement: it exposes a function called `load_environment()` that returns a
configured environment.

    my-task/
      pyproject.toml
      my_task.py        <- contains load_environment()

Then anyone can write:

    env = vf.load_environment("my-task", num_problems=50)

and `prime-rl` can train against it on a machine you've never seen.

WHY A FUNCTION AND NOT A MODULE-LEVEL VARIABLE. Worth understanding rather than
memorising, because the same three reasons keep coming up:

  - ARGUMENTS. Callers configure the task -- size, difficulty, split -- without
    editing your code.
  - LAZINESS. Building a dataset can mean downloading gigabytes. A function means
    that only happens when someone actually asks for the environment, not merely
    because they imported your module.
  - FRESHNESS. Each call builds a NEW object, so two concurrent training runs
    can't share mutable state. (Same reasoning as Unit 03's "don't put game state
    on `self`.")

Fill in the two TODOs, then run me:

    uv run python modules/04-package-and-publish/exercise_1_packaging.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "Solve the problem. Think briefly, then end your reply with exactly:\n"
    "Answer: <number>"
)


def reply_text(completion) -> str:
    """The model's final message, or "" if the rollout produced nothing.

    Rollouts fail. An API connection drops, a request times out, the model hits
    a content filter -- and `completion` comes back as an empty list. Indexing
    `completion[-1]` then raises IndexError from inside your reward function,
    which verifiers reports as "Error calling reward function" and scores as a
    failure anyway. You lose the real reason.

    Returning "" instead means a failed rollout scores 0.0 cleanly, which is
    what you want: it didn't answer, so it doesn't get credit.
    """
    return completion[-1]["content"] if completion else ""


def build_dataset(num_problems: int, seed: int = 0) -> Dataset:
    """Generate `num_problems` multiplication questions. Written for you.

    Note this is deterministic given the seed -- important for a published task,
    so that two people evaluating "the same" environment really are.
    """
    import random

    rng = random.Random(seed)
    qs, ans = [], []
    for _ in range(num_problems):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        qs.append(f"What is {a} * {b}?")
        ans.append(str(a * b))
    return Dataset.from_dict({"question": qs, "answer": ans})


def correct_answer(completion, answer, **kwargs) -> float:
    """Unit 02's pattern, unchanged. Written for you."""
    matches = ANSWER_RE.findall(reply_text(completion))
    if not matches:
        return 0.0
    return 1.0 if matches[-1].strip().rstrip(".").strip() == str(answer).strip() else 0.0


def load_environment(num_problems: int = 100, seed: int = 0, **kwargs) -> vf.Environment:
    """THE ENTRY POINT. This exact name is the whole convention.

    Build and return a `vf.SingleTurnEnv` with:
        dataset      = build_dataset(num_problems, seed)
        eval_dataset = the same
        system_prompt= SYSTEM_PROMPT
        rubric       = vf.Rubric(funcs=[correct_answer], weights=[1.0])
        **kwargs     = passed straight through

    THE `**kwargs` PASSTHROUGH MATTERS. Callers -- including `prime-rl` -- will
    hand you settings you didn't anticipate (concurrency limits, sampling
    arguments, sequence length caps). Swallowing them silently means your task
    quietly ignores a training config; forwarding them means it just works.

    NOTE THE DEFAULTS. Every argument has one, so `load_environment()` with no
    arguments must work. Tooling calls it bare to inspect your task, and a
    required argument will break that.

    Args:
        num_problems: how many questions to generate.
        seed: for reproducibility.
        **kwargs: forwarded to the environment.
    Returns:
        a configured vf.Environment
    """
    # TODO: build and return the environment.
    return todo("load_environment: return a SingleTurnEnv built from the arguments")


def make_pyproject(name: str = "mult-task") -> str:
    """Return the pyproject.toml text that would make this an installable package.

    Return a string containing, at minimum:
      - a [project] section with `name = "<name>"` and a `version`
      - a `dependencies` list that includes "verifiers"

    A minimal one looks like:

        [project]
        name = "mult-task"
        version = "0.1.0"
        dependencies = ["verifiers>=0.3.0"]

    WHY DEPENDENCIES MATTER MORE THAN USUAL HERE: someone else's training run
    will install your package on a machine you've never seen. Anything you forgot
    to declare -- but which happened to be installed on your laptop -- becomes an
    ImportError six hours into their GPU job.

    Args:
        name: the package name.
    Returns:
        the file contents as a string.
    """
    # TODO: return a valid pyproject.toml as a string.
    return todo("make_pyproject: return a [project] section with name, version, dependencies")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"{BOLD}1. Does it load with no arguments?{RESET}\n")
    env = load_environment()
    ds = env.get_dataset()
    print(f"  type:    {type(env).__name__}")
    print(f"  rows:    {len(ds)}")
    print(f"  columns: {ds.column_names}")
    print(f"  sample:  {ds[0]['question']!r}  ->  {ds[0]['answer']!r}")

    print(f"\n{BOLD}2. Do the arguments work?{RESET}\n")
    small = load_environment(num_problems=7)
    print(f"  load_environment(num_problems=7) -> {len(small.get_dataset())} rows")

    print(f"\n{BOLD}3. Is it reproducible?{RESET}\n")
    a = load_environment(num_problems=5, seed=42).get_dataset()["question"]
    b = load_environment(num_problems=5, seed=42).get_dataset()["question"]
    c = load_environment(num_problems=5, seed=99).get_dataset()["question"]
    print(f"  same seed  -> identical questions? {a == b}")
    print(f"  diff seed  -> different questions? {a != c}")

    print(f"\n{BOLD}4. Is each call a fresh object?{RESET}\n")
    print(f"  load_environment() is not load_environment(): "
          f"{load_environment(num_problems=2) is not load_environment(num_problems=2)}")
    print(
        f"\n{DIM}  That last one matters. If you'd built the environment once at module\n"
        f"  level and returned the same object every time, two concurrent training\n"
        f"  runs would share mutable state and interfere in ways that are\n"
        f"  genuinely miserable to debug.{RESET}"
    )

    print(f"\n{BOLD}5. The packaging file{RESET}\n")
    for line in make_pyproject().strip().splitlines():
        print(f"  {line}")

    print(
        f"\n\n{BOLD}To actually publish this{RESET}\n"
        "\n"
        "  uv tool install prime\n"
        "  prime login\n"
        "  prime env init mult-task      # scaffolds the directory for you\n"
        "  # ... move your load_environment() into it ...\n"
        "  prime env push                # publishes to the Environments Hub\n"
        "\n"
        "  Before you push, ask the Unit 03 question: would this produce a useful\n"
        "  gradient? A task where the model always succeeds or always fails teaches\n"
        "  nothing -- that's an eval, not a training environment. Both are worth\n"
        "  publishing. Just label it honestly."
    )


if __name__ == "__main__":
    run_main(main)
