"""
Lab 1 — a v0 environment, so you can read the Hub.

Everything you've built so far used `verifiers.v1`. The Environments Hub also
holds environments written against the ORIGINAL API -- plain `import verifiers`
-- and the eval CLI runs them behind a flag whose name says where things stand:

    uv run eval --legacy.id <env-id> ...

You'll meet v0 code whenever you open an existing Hub environment, so this lab
has you write a small one. The concepts are the ones you know; only the
furniture differs:

    v1 (what you know)                  v0 (this lab)
    ------------------                  -------------
    TaskData subclass                   a row in a HF Dataset
                                        (columns `question` and `answer`)
    Task with @vf.reward methods        loose functions in a `Rubric`
    weights on the decorator            a parallel `weights=[...]` list
    trace.last_reply                    completion[-1]["content"]
    Taskset exporting via __all__       a `load_environment()` function

Two differences deserve attention because they're LESSONS, not just furniture:

  1. v0 rewards receive `completion` -- a list of chat messages, so the text is
     `completion[-1]["content"]`, and an aborted rollout makes that list EMPTY.
     Index it unguarded and your grader crashes on the first failed rollout.
     (v1's `trace.last_reply` just returns "" -- the API absorbed this bug
     class.)

  2. v0 weights live in a list parallel to the functions list. Insert a
     function without inserting its weight and every later weight silently
     shifts onto the wrong function. (v1 put the weight on the decorator --
     same bug class, retired.)

Fill in the two TODOs, then run me:

    uv run python modules/06-legacy-v0/exercise_1_v0_env.py

TERMS USED IN THIS FILE

  completion: v0's name for the model's reply -- a LIST of chat messages
  rubric:     v0's bundle of reward functions with a parallel weights list
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers as vf  # v0: NO .v1 -- this is the whole point of the lab
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "Solve the problem. Think briefly, then end your reply with exactly:\n"
    "Answer: <number>"
)

PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 144 / 12?", "12"),
    ("What is 89 + 156?", "245"),
]


def correct_answer(completion, answer, **kwargs) -> float:
    """A v0 reward function: loose, with arguments injected by NAME.

    Same injection idea as v1, different names: v0 offers `prompt`,
    `completion`, `answer`, `state`, and more. Take `**kwargs` always -- the
    library passes what your signature requests plus nothing, and a rubric that
    offers more than you named must not crash you.

    Implement the Unit 02 grader in v0 clothes:

      1. text = completion[-1]["content"] if completion else ""
         THE GUARD IS NOT OPTIONAL. `completion` is a list of chat messages,
         and a failed rollout (dropped connection, timeout) delivers an EMPTY
         list. Unguarded, your grader raises IndexError and the library reports
         "Error calling reward function" -- hiding the real cause.
      2. Last ANSWER_RE match, stripped of whitespace and a trailing period,
         compared exactly to `answer`. No match, 0.0.

    Args:
        completion: list of chat messages; the model's text is in the last one.
        answer: the reference answer string.
    Returns:
        1.0 or 0.0
    """
    # TODO: implement with the empty-completion guard.
    return todo("correct_answer: guard the empty list, then commit-and-compare")


def load_environment(**kwargs) -> vf.Environment:
    """The v0 packaging convention: a function with THIS EXACT NAME.

    Where v1 resolves a Taskset class via __all__, v0 imports your module and
    calls `load_environment()`. Build and return:

        dataset = Dataset.from_dict({
            "question": [q for q, _ in PROBLEMS],
            "answer": [a for _, a in PROBLEMS],
        })
        return vf.SingleTurnEnv(
            dataset=dataset,
            eval_dataset=dataset,
            system_prompt=SYSTEM_PROMPT,
            rubric=vf.Rubric(funcs=[correct_answer], weights=[1.0]),
            **kwargs,
        )

    Notes while you type:
      - The dataset columns MUST be named `question` and `answer`; the library
        builds the actual `prompt` column itself. (v1 equivalent: TaskData's
        `prompt` field.)
      - `Rubric(funcs=..., weights=...)` is the parallel-list arrangement --
        keep them aligned by position, because nothing else will.
      - Forward **kwargs: callers (including the legacy trainer path) pass
        settings you didn't anticipate.

    Args:
        **kwargs: forwarded to the environment.
    Returns:
        a configured v0 environment.
    """
    # TODO: build the Dataset, the Rubric, and the SingleTurnEnv.
    return todo("load_environment: Dataset + Rubric + SingleTurnEnv")


# ---------------------------------------------------------------------------
# Written for you: v0's offline scoring path.
# ---------------------------------------------------------------------------


def score_reply(env: vf.Environment, reply: str, answer: str) -> dict:
    """v0 scores a `State` dict through the rubric -- the trace of its era."""
    state = vf.State(
        {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": reply}],
            "answer": answer,
            "task": "default",
            "info": {},
            "trajectory": [],  # the v0 monitor rubric counts turns from this
        }
    )
    asyncio.run(env.rubric.score_rollout(state))
    return {"reward": state["reward"], "metrics": state["metrics"]}


def main() -> None:
    env = load_environment()
    ds = env.get_dataset()

    print(f"{BOLD}1. The v0 environment{RESET}\n")
    print(f"  type:    {type(env).__name__}")
    print(f"  columns: {ds.column_names}")
    print(f"  rubric:  funcs={[f.__name__ for f in env.rubric.funcs]} "
          f"weights={env.rubric.weights}")
    print(
        f"\n{DIM}  Note the built `prompt` column -- the library made it from `question`\n"
        f"  plus the system prompt. And note the parallel weights list.{RESET}"
    )

    print(f"\n{BOLD}2. Scoring, v0 style{RESET}\n")
    for reply, ans in [("Answer: 391", "391"), ("Answer: 3912", "391"), ("no idea", "391")]:
        out = score_reply(env, reply, ans)
        print(f"  {reply!r:18} -> reward={out['reward']}  metrics={out['metrics']}")

    print(f"\n{BOLD}3. The failed-rollout test{RESET}\n")
    print(f"  correct_answer([], '391') -> {correct_answer([], '391')}")
    print(
        f"\n{DIM}  An empty completion scored 0.0 cleanly instead of raising. In v0 that\n"
        f"  guard is YOUR job, in every reward function, forever -- which is a good\n"
        f"  way to appreciate why v1 moved it into the API.{RESET}"
    )


if __name__ == "__main__":
    run_main(main)
