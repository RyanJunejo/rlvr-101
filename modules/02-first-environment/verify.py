"""
Unit 02 autograder.

    uv run python modules/02-first-environment/verify.py

Runs offline. No API key needed.
"""

from __future__ import annotations

import asyncio
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


def _score(task, trace):
    asyncio.run(task.score(trace))
    return {k: v.score for k, v in trace.rewards.items()}


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: your first task --{RESET}")
    try:
        import verifiers.v1 as vf

        import exercise_1_first_task as l1
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        fields = l1.MathData.model_fields
        g.check("MathData declares `answer`", "answer" in fields, f"fields: {sorted(fields)}")
        g.check("MathData inherits the base fields",
                {"prompt", "system_prompt", "idx"} <= set(fields),
                f"expected the TaskData fields to be inherited; got {sorted(fields)}")
        g.check("MathData is frozen", l1.MathData.model_config.get("frozen") is True,
                "the base class sets frozen=True; don't override it")
    except NotImplementedYet as e:
        _skip(g, ["MathData is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["MathData is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        from verifiers.v1.utils.decorators import discover_decorated

        task = l1.MathTask(l1.MathData(idx=0, prompt="q", answer="391"))
        rewards = discover_decorated(task, "reward")
        names = sorted(fn.__name__ for fn in rewards)
        g.check("both reward methods are registered",
                names == ["correct_answer", "has_answer_line"], f"found {names}")
        for fn in rewards:
            g.check(f"`{fn.__name__}` is async", inspect.iscoroutinefunction(fn),
                    "v1 rewards must be `async def`; a sync one raises 'An "
                    "asyncio.Future, a coroutine or an awaitable is required' at "
                    "scoring time")
        weights = {fn.__name__: fn._vf_weight for fn in rewards}
        g.close("correct_answer has weight 1.0", weights.get("correct_answer", -1), 1.0)
        g.close("has_answer_line has weight 0.2", weights.get("has_answer_line", -1), 0.2)
    except NotImplementedYet as e:
        _skip(g, ["reward methods are correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["reward methods are correct"], f"{type(e).__name__}: {e}")
        return

    try:
        t = l1.MathTask(l1.MathData(idx=0, prompt="q", answer="391"))
        tr = l1.make_trace(t, "Answer: 391")
        g.check("make_trace returns a Trace", isinstance(tr, vf.Trace), f"got {type(tr).__name__}")
        g.check(
            "the node is marked sampled=True",
            tr.last_reply == "Answer: 391",
            f"trace.last_reply is {tr.last_reply!r}. If it's empty, the node is "
            f"missing sampled=True -- last_reply reads only sampled nodes, so every "
            f"reward would silently score 0.0.",
        )
    except NotImplementedYet as e:
        _skip(g, ["make_trace is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["make_trace is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        cases = [
            ("Answer: 391", "391", 1.0, 1.0, "exact match"),
            ("17 * 23 = 391.\nAnswer: 391.", "391", 1.0, 1.0, "trailing period"),
            ("Answer: 3912", "391", 0.0, 1.0, "superset number, right format"),
            ("The product is 391.", "391", 0.0, 0.0, "answer only in prose"),
            ("Answer: 12\nWait.\nAnswer: 391", "391", 1.0, 1.0, "last answer wins"),
            ("no idea", "391", 0.0, 0.0, "nothing at all"),
        ]
        for reply, answer, want_c, want_f, label in cases:
            t = l1.MathTask(l1.MathData(idx=0, prompt="q", answer=answer))
            r = _score(t, l1.make_trace(t, reply))
            g.check(f"scoring via task.score(): {label}",
                    r.get("correct_answer") == want_c and r.get("has_answer_line") == want_f,
                    f"reply {reply!r} with answer {answer!r} scored {r}; "
                    f"expected correct={want_c}, format={want_f}")
    except NotImplementedYet as e:
        _skip(g, ["scoring behaves correctly"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["scoring behaves correctly"], f"{type(e).__name__}: {e}")

    # The written-for-you taskset must resolve through verifiers' OWN loader --
    # the identical path the eval CLI takes. If this passes, the live command in
    # the lab will find the module.
    try:
        from verifiers.v1.utils.loaders import taskset_class

        cls = taskset_class("exercise_1_first_task")
        g.check("the lab file resolves as a taskset plugin",
                cls is l1.MathTaskset,
                f"taskset_class() found {cls!r}; expected MathTaskset via __all__")
        ts = cls(vf.TasksetConfig())
        tasks = list(ts)
        g.check("the taskset yields 8 seeded tasks", len(tasks) == 8, f"got {len(tasks)}")
        g.check("generated answers are correct",
                all(str(eval(t.data.prompt.replace("What is ", "").rstrip("?")))  # noqa: S307
                    == t.data.answer for t in tasks[:4]),
                "the answer must match the question it was generated with")
    except Exception as e:
        _skip(g, ["the taskset resolves via the real loader"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: weights --{RESET}")
    try:
        import exercise_2_weights as l2
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    try:
        task = l2.QuizTask(l2.QuizData(idx=0, prompt="q", answer="391"))
        desc = l2.describe_rewards(task)
        g.check("describe_rewards lists both components with their weights",
                desc == [("correct_answer", 1.0), ("format_reward", 0.2)],
                f"got {desc}")
    except NotImplementedYet as e:
        _skip(g, ["describe_rewards is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["describe_rewards is correct"], f"{type(e).__name__}: {e}")

    try:
        rows = [
            ("The answer is 391.", "391", 0.0, 0.0, "right but never committed"),
            ("Answer: 400", "391", 0.0, 1.0, "wrong, well formatted"),
            ("Answer: 391", "391", 1.0, 1.0, "right and formatted"),
        ]
        for reply, answer, want_c, want_f, label in rows:
            t = l2.QuizTask(l2.QuizData(idx=0, prompt="q", answer=answer))
            r = _score(t, l2.make_trace(t, reply))
            g.check(f"components separate: {label}",
                    r.get("correct_answer") == want_c and r.get("format_reward") == want_f,
                    f"got {r}; expected correct={want_c}, format={want_f}")
    except NotImplementedYet as e:
        _skip(g, ["the two components score correctly"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["the two components score correctly"], f"{type(e).__name__}: {e}")


def check_lab3(g: Grader) -> None:
    print(f"\n{DIM}-- lab 3: reward hacking --{RESET}")
    try:
        import exercise_3_reward_hacking as l3
    except Exception as e:
        _skip(g, ["lab 3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 3 imports", True)

    s_correct, _ = l3.run_suite("sloppy_correctness")
    g.check("sloppy_correctness is still broken (don't fix the exhibit)",
            s_correct < len(l3.ATTACKS),
            f"it scored {s_correct}/{len(l3.ATTACKS)}; the whole point is that it fails")

    try:
        for name, reply, answer, expected in l3.ATTACKS:
            task = l3.AttackTask(l3.AttackData(idx=0, prompt="q", answer=answer))
            trace = l3.make_trace(task, reply)
            got = float(asyncio.run(task.robust_correctness(task.data, trace)))
            g.check(f"robust_correctness survives: {name}",
                    abs(got - expected) < 1e-9,
                    f"reply {reply[:50]!r}... with answer {answer!r} should score "
                    f"{expected}, got {got}")

        # once robust works, the full score() path must report both graders
        task = l3.AttackTask(l3.AttackData(idx=0, prompt="q", answer="391"))
        trace = l3.make_trace(task, "Answer: 3912")
        r = _score(task, trace)
        g.check("task.score() reports both graders side by side",
                r == {"sloppy_correctness": 1.0, "robust_correctness": 0.0},
                f"got {r} -- on 'Answer: 3912' the sloppy grader pays and yours "
                f"must not; that daylight is the whole lab")
    except NotImplementedYet as e:
        _skip(g, ["robust_correctness survives the suite"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["robust_correctness survives the suite"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 02 — your first task (v1)")
    check_lab1(g)
    check_lab2(g)
    check_lab3(g)
    report(g)


if __name__ == "__main__":
    main()
