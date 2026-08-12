"""
Lab 2 — port a v0 environment to v1.

The realistic scenario: you find a Hub environment worth training on, it's
written in v0, and you want it on the current stack. The port is mechanical
once you can see both shapes side by side -- and you built the v1 side all
course, so this is a review lab wearing a migration costume.

The mapping, one row per move:

    v0 source                            v1 target
    ---------                            ---------
    dataset row (question / answer)      TaskData(prompt=..., answer=...)
    def reward(completion, answer)       @vf.reward async method (task, trace)
    completion[-1]["content"] + guard    trace.last_reply  (guard built in)
    Rubric(funcs=[...], weights=[...])   weights on each decorator
    load_environment()                   Taskset subclass + __all__

Fill in the two TODOs, then run me:

    uv run python modules/06-legacy-v0/exercise_2_port.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as v1
from verifiers.v1.trace import TraceTask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The v0 environment you're porting -- lab 1 must be done first.
from exercise_1_v0_env import ANSWER_RE, PROBLEMS, SYSTEM_PROMPT, load_environment  # noqa: E402


class MathData(v1.TaskData):
    answer: str


class MathTask(v1.Task[MathData, v1.State, v1.TaskConfig]):
    """Port lab 1's `correct_answer` to a v1 reward method.

    Same grader, new shape:

        @v1.reward
        async def correct_answer(self, task: MathData, trace: v1.Trace) -> float:

    The logic is identical except the empty-completion guard DISAPPEARS --
    `trace.last_reply` already returns "" for a rollout with no sampled
    messages. Deleting a guard because the API absorbed it is the pleasant
    kind of porting.
    """

    # TODO: the async reward method.


class MathTaskset(v1.Taskset[MathTask, v1.TasksetConfig]):
    """Port `load_environment()` to a taskset.

    Iterate the same PROBLEMS list lab 1 used:

        def load(self) -> Iterator[MathTask]:
            for i, (question, answer) in enumerate(PROBLEMS):
                yield MathTask(
                    MathData(idx=i, prompt=question,
                             system_prompt=SYSTEM_PROMPT, answer=answer),
                    self.config.task,
                )

    What was a Dataset with magic column names is now typed per-row data; what
    was a function with a magic name is now a class found via __all__.
    """

    def load(self) -> Iterator[MathTask]:
        # TODO: yield one MathTask per PROBLEMS entry.
        return todo("MathTaskset.load: one MathTask per PROBLEMS row")


__all__ = ["MathTaskset"]


# ---------------------------------------------------------------------------
# Written for you: score the SAME replies through BOTH stacks.
# ---------------------------------------------------------------------------


def score_v0(env, reply: str, answer: str) -> float:
    import verifiers as v0

    state = v0.State(
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
    return state["reward"]


def score_v1(task: MathTask, reply: str) -> float:
    trace = v1.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=v1.AgentInfo(config=v1.AgentConfig(), name="offline", trainable=False),
    )
    trace.nodes.append(
        v1.MessageNode(message=v1.AssistantMessage(content=reply), sampled=True)
    )
    asyncio.run(task.score(trace))
    return sum(r.score * r.weight for r in trace.rewards.values())


REPLIES = [
    ("Answer: 391", "391"),
    ("17 * 23 = 391.\nAnswer: 391.", "391"),
    ("Answer: 3912", "391"),
    ("The product is 391.", "391"),
    ("no idea", "391"),
]


def main() -> None:
    env = load_environment()

    print(f"{BOLD}The same replies through both stacks{RESET}\n")
    print(f"  {'reply':<32} {'v0':>6} {'v1':>6}")
    print("  " + "-" * 46)
    agree = True
    for reply, answer in REPLIES:
        r0 = score_v0(env, reply, answer)
        task = MathTask(MathData(idx=0, prompt="q", answer=answer))
        r1 = score_v1(task, reply)
        agree &= abs(r0 - r1) < 1e-9
        print(f"  {reply.replace(chr(10), ' / ')[:32]:<32} {r0:>6.1f} {r1:>6.1f}")

    print(f"\n  every reply scores identically: {agree}")
    print(
        f"\n{DIM}  That agreement is the port's definition of done. A migration that\n"
        f"  changes any score has changed the task, and numbers from before and\n"
        f"  after stop being comparable.{RESET}"
    )


if __name__ == "__main__":
    run_main(main)
