"""
Unit 06 autograder.

    uv run python modules/06-verifiers-v1/verify.py

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


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: porting to v1 --{RESET}")
    try:
        import exercise_1_port as l1
        import verifiers.v1 as vf
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        fields = l1.MathData.model_fields
        g.check("MathData declares an `answer` field", "answer" in fields,
                f"fields are {sorted(fields)}")
        g.check("MathData inherits the TaskData fields",
                {"prompt", "system_prompt", "idx"} <= set(fields),
                f"expected the base fields to be inherited; got {sorted(fields)}")
        d = l1.MathData(idx=0, prompt="q", answer="391")
        g.check("MathData is frozen (immutable)",
                l1.MathData.model_config.get("frozen") is True,
                "the base class sets frozen=True; don't override it")
        g.check("MathData carries the answer", d.answer == "391", f"got {d.answer!r}")
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
        names = sorted(f.__name__ for f in rewards)
        g.check("both reward methods are registered",
                names == ["correct_answer", "has_answer_line"], f"found {names}")

        for fn in rewards:
            g.check(f"`{fn.__name__}` is async",
                    inspect.iscoroutinefunction(fn),
                    "v1 reward methods must be `async def`. A sync one raises "
                    "'An asyncio.Future, a coroutine or an awaitable is required' "
                    "during scoring.")
    except NotImplementedYet as e:
        _skip(g, ["reward methods are correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["reward methods are correct"], f"{type(e).__name__}: {e}")
        return

    try:
        t = l1.MathTask(l1.MathData(idx=0, prompt="q", answer="391"))
        tr = l1.make_trace(t, "Answer: 391")
        g.check("make_trace returns a Trace", isinstance(tr, vf.Trace),
                f"got {type(tr).__name__}")
        g.check("the trace has one node", len(tr.nodes) == 1, f"got {len(tr.nodes)}")
        g.check(
            "the node is marked sampled=True",
            tr.last_reply == "Answer: 391",
            f"trace.last_reply is {tr.last_reply!r}, expected 'Answer: 391'. If it's "
            f"empty, the node is missing sampled=True -- last_reply only reads "
            f"sampled nodes, so every reward would silently score 0.0.",
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
            tr = l1.make_trace(t, reply)
            asyncio.run(t.score(tr))
            r = dict(tr.rewards)
            gc, gf = r["correct_answer"].score, r["has_answer_line"].score
            g.check(f"scoring: {label}", gc == want_c and gf == want_f,
                    f"reply {reply!r} with answer {answer!r} scored "
                    f"correct={gc}, format={gf}; expected {want_c}, {want_f}")

        t = l1.MathTask(l1.MathData(idx=0, prompt="q", answer="391"))
        tr = l1.make_trace(t, "Answer: 391")
        asyncio.run(t.score(tr))
        r = dict(tr.rewards)
        g.close("correct_answer has weight 1.0", r["correct_answer"].weight, 1.0)
        g.close("has_answer_line has weight 0.2", r["has_answer_line"].weight, 0.2)
    except NotImplementedYet as e:
        _skip(g, ["scoring behaves correctly"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["scoring behaves correctly"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: tasksets --{RESET}")
    try:
        import exercise_2_taskset as l2
        import verifiers.v1 as vf
    except NotImplementedYet as e:
        _skip(g, ["lab 2 imports"], f"depends on lab 1, still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    try:
        ts = l2.FixedMathTaskset(vf.TasksetConfig())
        tasks = list(ts)
        g.check("the fixed taskset yields one task per problem",
                len(tasks) == len(l2.PROBLEMS),
                f"expected {len(l2.PROBLEMS)}, got {len(tasks)}")
        g.check("tasks carry the right answers",
                [t.data.answer for t in tasks] == [a for _, a in l2.PROBLEMS],
                f"got {[t.data.answer for t in tasks]}")
        g.check("tasks carry an idx", all(t.data.idx == i for i, t in enumerate(tasks)),
                "set idx -- it's how you identify a task in the trace file")
        g.check("load() is a generator, not a list",
                inspect.isgeneratorfunction(l2.FixedMathTaskset.load),
                "use `yield`, so tasks are built as they're consumed")
        g.check("the fixed taskset is not marked infinite", ts.INFINITE is False,
                f"INFINITE is {ts.INFINITE}")
    except NotImplementedYet as e:
        _skip(g, ["FixedMathTaskset is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["FixedMathTaskset is correct"], f"{type(e).__name__}: {e}")

    try:
        inf = l2.InfiniteMathTaskset(vf.TasksetConfig())
        g.check("the generated taskset is marked INFINITE", inf.INFINITE is True,
                "set INFINITE = True on the class")

        five = list(inf.head(5))
        g.check(".head(5) yields exactly 5 tasks", len(five) == 5, f"got {len(five)}")

        # it must really be unbounded, not a long list
        it = iter(l2.InfiniteMathTaskset(vf.TasksetConfig()))
        many = [next(it) for _ in range(500)]
        g.check("it keeps producing tasks past any fixed size", len(many) == 500,
                "an infinite taskset must not run out")

        g.check("generated answers are correct",
                all(str(eval(t.data.prompt.replace("What is ", "").rstrip("?")))  # noqa: S307
                    == t.data.answer for t in five),
                "the answer must match the question it generated")

        again = [t.data.prompt for t in l2.InfiniteMathTaskset(vf.TasksetConfig()).head(5)]
        g.check("generation is reproducible across constructions",
                again == [t.data.prompt for t in five],
                "seed the RNG inside load() so a published task is reproducible")
    except NotImplementedYet as e:
        _skip(g, ["InfiniteMathTaskset is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["InfiniteMathTaskset is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 06 — the redesigned API")
    check_lab1(g)
    check_lab2(g)
    report(g)


if __name__ == "__main__":
    main()
