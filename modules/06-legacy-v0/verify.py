"""
Unit 06 autograder.

    uv run python modules/06-legacy-v0/verify.py

Runs offline. No API key needed.
"""

from __future__ import annotations

import inspect
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
    print(f"\n{DIM}-- lab 1: a v0 environment --{RESET}")
    try:
        import verifiers as v0

        import exercise_1_v0_env as l1
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        msg = [{"role": "assistant", "content": "Answer: 391"}]
        g.close("correct_answer accepts an exact match", l1.correct_answer(msg, "391"), 1.0)
        g.close("correct_answer rejects a superset number",
                l1.correct_answer([{"role": "assistant", "content": "Answer: 3912"}], "391"), 0.0)
        g.close("correct_answer ignores prose-only mentions",
                l1.correct_answer([{"role": "assistant", "content": "It is 391."}], "391"), 0.0)
        g.close(
            "correct_answer survives an EMPTY completion",
            l1.correct_answer([], "391"), 0.0,
        )
        g.check("the guard is a guard, not a crash", True,
                "an aborted rollout delivers completion=[] -- in v0 this guard is "
                "your job in every reward function")
    except NotImplementedYet as e:
        _skip(g, ["correct_answer is correct"], f"still TODO: {e}")
        return
    except IndexError:
        _skip(g, ["correct_answer survives an EMPTY completion"],
              "IndexError on completion=[] -- add the guard: "
              "completion[-1]['content'] if completion else ''")
        return
    except Exception as e:
        _skip(g, ["correct_answer is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        env = l1.load_environment()
        g.check("load_environment returns a v0 SingleTurnEnv",
                isinstance(env, v0.SingleTurnEnv), f"got {type(env).__name__}")
        ds = env.get_dataset()
        g.check("the dataset has question/answer columns and a built prompt",
                {"question", "answer", "prompt"} <= set(ds.column_names),
                f"got {ds.column_names}")
        # v0 wraps your rubric in a RubricGroup with its monitor rubrics, so the
        # top-level .funcs is empty and yours lives under .rubrics -- a classic
        # v0 where-did-my-reward-go moment.
        subs = getattr(env.rubric, "rubrics", None) or [env.rubric]
        pairs = [(f.__name__, w) for r in subs for f, w in zip(r.funcs, r.weights)]
        g.check("the rubric holds correct_answer at weight 1.0",
                ("correct_answer", 1.0) in pairs,
                f"found {pairs}; your function should be in there at weight 1.0")
        env2 = l1.load_environment(max_workers=7)
        g.check("**kwargs are forwarded", getattr(env2, "max_workers", None) == 7,
                "callers pass settings you didn't anticipate; forward them")
        out = l1.score_reply(env, "Answer: 391", "391")
        g.close("the v0 scoring path works end to end", out["reward"], 1.0)
    except NotImplementedYet as e:
        _skip(g, ["load_environment is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["load_environment is correct"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: the port --{RESET}")
    try:
        import exercise_2_port as l2
    except NotImplementedYet as e:
        _skip(g, ["lab 2 imports"], f"depends on lab 1, still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    try:
        from verifiers.v1.utils.decorators import discover_decorated

        task = l2.MathTask(l2.MathData(idx=0, prompt="q", answer="391"))
        rewards = discover_decorated(task, "reward")
        g.check("the ported reward is registered",
                [f.__name__ for f in rewards] == ["correct_answer"],
                f"found {[f.__name__ for f in rewards]}")
        g.check("the ported reward is async",
                all(inspect.iscoroutinefunction(f) for f in rewards),
                "v1 rewards must be `async def`")
    except NotImplementedYet as e:
        _skip(g, ["the ported reward is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["the ported reward is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        tasks = list(l2.MathTaskset(__import__("verifiers.v1", fromlist=["x"]).TasksetConfig()))
        g.check("the taskset yields one task per v0 dataset row",
                len(tasks) == len(l2.PROBLEMS), f"got {len(tasks)}")
        g.check("prompts and answers carried over",
                [(t.data.prompt, t.data.answer) for t in tasks] == l2.PROBLEMS,
                "the port must preserve the data exactly")
    except NotImplementedYet as e:
        _skip(g, ["the taskset port is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["the taskset port is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        env = l2.load_environment()
        for reply, answer in l2.REPLIES:
            r0 = l2.score_v0(env, reply, answer)
            t = l2.MathTask(l2.MathData(idx=0, prompt="q", answer=answer))
            r1 = l2.score_v1(t, reply)
            g.check(f"both stacks agree on {reply[:28]!r}",
                    abs(r0 - r1) < 1e-9,
                    f"v0 scored {r0}, v1 scored {r1}. A port that changes any score "
                    f"has changed the task -- numbers stop being comparable.")
    except NotImplementedYet as e:
        _skip(g, ["the stacks agree"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["the stacks agree"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 06 — the legacy API")
    check_lab1(g)
    check_lab2(g)
    report(g)


if __name__ == "__main__":
    main()
