"""
Unit 03 autograder.

    uv run python modules/03-multiturn-and-tools/verify.py

Runs entirely offline. No API key needed.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from common.grading import DIM, RESET, Grader, NotImplementedYet, report  # noqa: E402


def _skip(g: Grader, names: list[str], why: str) -> None:
    for n in names:
        g.check(n, False, why)


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: the guessing game --{RESET}")
    try:
        import verifiers.v1 as vf

        import exercise_1_guessing_game as l1
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    try:
        g.check("parse_guess reads a plain guess", l1.parse_guess("Guess: 50") == 50,
                f"got {l1.parse_guess('Guess: 50')!r}")
        g.check("parse_guess takes the LAST guess in a message",
                l1.parse_guess("Guess: 5\nActually wait.\nGuess: 42") == 42,
                f"got {l1.parse_guess('Guess: 5' + chr(10) + 'Guess: 42')!r}; models "
                f"correct themselves, and the final commitment is the real one")
        g.check("parse_guess returns None when there's no guess",
                l1.parse_guess("I give up") is None,
                f"got {l1.parse_guess('I give up')!r}; None, not 0 -- the caller must "
                f"tell 'no guess' apart from a real number")
    except NotImplementedYet as e:
        _skip(g, ["parse_guess is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["parse_guess is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        g.check("respond: low", l1.respond(10, 42) == "Too low.", f"got {l1.respond(10, 42)!r}")
        g.check("respond: high", l1.respond(90, 42) == "Too high.", f"got {l1.respond(90, 42)!r}")
        g.check("respond: correct", l1.respond(42, 42) == "Correct!", f"got {l1.respond(42, 42)!r}")
    except NotImplementedYet as e:
        _skip(g, ["respond is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["respond is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        from verifiers.v1.utils.decorators import discover_decorated

        task = l1.GuessTask(l1.GuessData(idx=0, prompt="p", answer="42"))
        rewards = [f.__name__ for f in discover_decorated(task, "reward")]
        metrics = [f.__name__ for f in discover_decorated(task, "metric")]
        g.check("`solved` is registered as a reward", rewards == ["solved"], f"rewards: {rewards}")
        g.check("`num_guesses` is registered as a metric (weight-free)",
                metrics == ["num_guesses"],
                f"metrics: {metrics} -- instrumentation belongs in @vf.metric, not "
                f"a zero-weight reward")

        def play(guesses):
            t = l1.GuessTask(l1.GuessData(idx=0, prompt="p", answer="42"))
            tr = l1.make_game_trace(t, guesses)
            asyncio.run(t.score(tr))
            return ({k: v.score for k, v in tr.rewards.items()}, dict(tr.metrics))

        r, m = play(l1.binary_search_guesses(42))
        g.check("a winning game scores solved=1.0", r.get("solved") == 1.0, f"got {r}")
        g.close("num_guesses counts the guesses", m.get("num_guesses", -1), 7.0)
        r2, _ = play([1, 2, 3, 4, 5, 6, 7])
        g.check("a losing game scores solved=0.0", r2.get("solved") == 0.0, f"got {r2}")
        r3, m3 = play([])
        g.check("an empty game scores 0.0 without crashing",
                r3.get("solved") == 0.0 and m3.get("num_guesses") == 0.0,
                f"got rewards {r3}, metrics {m3} -- a rollout can fail before "
                f"guessing anything")
    except NotImplementedYet as e:
        _skip(g, ["the scoring replays the transcript"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["the scoring replays the transcript"], f"{type(e).__name__}: {e}")

    try:
        from verifiers.v1.utils.loaders import environment_class, taskset_class

        ts = taskset_class("exercise_1_guessing_game")
        g.check("the module resolves as a taskset plugin", ts is l1.GuessTaskset,
                f"taskset_class() found {ts!r}")
        env = environment_class("exercise_1_guessing_game")
        g.check("the module's own Env is found automatically", env is l1.GuessEnv,
                f"environment_class() found {env!r}; exporting the Env beside the "
                f"Taskset in __all__ is what wires the game up for the eval CLI")
    except Exception as e:
        _skip(g, ["the plugin exports resolve"], f"{type(e).__name__}: {e}")


def check_lab2(g: Grader) -> None:
    print(f"\n{DIM}-- lab 2: sparse rewards --{RESET}")
    try:
        import exercise_2_sparse_rewards as l2
    except Exception as e:
        _skip(g, ["lab 2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 2 imports", True)

    won = {"secret": 50, "guesses": [50], "solved": True}
    close = {"secret": 50, "guesses": [51], "solved": False}
    far = {"secret": 50, "guesses": [100], "solved": False}
    worst = {"secret": 100, "guesses": [1], "solved": False}
    none = {"secret": 50, "guesses": [], "solved": False}

    try:
        g.close("shaped_reward pays 1.0 for a win", l2.shaped_reward(won), 1.0)
        g.close("shaped_reward pays 0.0 when no guess was made", l2.shaped_reward(none), 0.0)
        g.close("shaped_reward pays ~0.0 for the worst possible miss",
                l2.shaped_reward(worst), 0.0, tol=1e-6)
        c, f = l2.shaped_reward(close), l2.shaped_reward(far)
        g.check("a closer guess scores higher than a distant one", c > f,
                f"off-by-1 scored {c:.4f}; off-by-50 scored {f:.4f}")
        g.check(
            "partial credit can never beat winning (the 0.5 cap)",
            c < 1.0 and c <= 0.5 + 1e-9,
            f"a near-miss scored {c:.4f}. It must stay strictly below the 1.0 paid "
            f"for winning -- otherwise the model can farm partial credit forever "
            f"without finishing. This is the whole point of the exercise.",
        )
        g.check(
            "shaped_reward uses the CLOSEST guess, not the last one",
            abs(l2.shaped_reward({"secret": 50, "guesses": [51, 100], "solved": False}) - c) < 1e-9,
            "getting within 1 then wandering away should score the same as getting "
            "within 1 and stopping",
        )
    except NotImplementedYet as e:
        _skip(g, ["shaped_reward is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["shaped_reward is correct"], f"{type(e).__name__}: {e}")

    try:
        g.check("group_has_signal is False when every score is identical",
                not l2.group_has_signal(np.array([0.0, 0.0, 0.0, 0.0])),
                "all-zeros produces no gradient, so this must be False")
        g.check("group_has_signal is False when every score is 1.0",
                not l2.group_has_signal(np.array([1.0, 1.0, 1.0])),
                "all-1.0 is the same failure as all-0.0 -- no spread, no learning")
        g.check("group_has_signal is True when scores differ",
                bool(l2.group_has_signal(np.array([0.0, 1.0, 0.0, 1.0]))))
        g.check("group_has_signal treats tiny float noise as no signal",
                not l2.group_has_signal(np.array([0.3, 0.3, 0.3 + 1e-15])),
                "use a tolerance, not exact equality")
    except NotImplementedYet as e:
        _skip(g, ["group_has_signal is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["group_has_signal is correct"], f"{type(e).__name__}: {e}")

    try:
        frac_expert, win_expert = l2.measure(1.0, l2.sparse_reward, n_groups=120)
        g.check(
            "a perfect player produces NO useful groups under a sparse reward",
            frac_expert < 0.02 and win_expert > 0.98,
            f"an expert wins {win_expert * 100:.0f}% of games and got signal from "
            f"{frac_expert * 100:.1f}% of groups; expected ~100% and ~0%",
        )
        frac_weak_sparse, _ = l2.measure(0.0, l2.sparse_reward, n_groups=120)
        frac_weak_shaped, _ = l2.measure(0.0, l2.shaped_reward, n_groups=120)
        g.check(
            "shaping rescues the weak player",
            frac_weak_shaped > frac_weak_sparse + 0.2,
            f"sparse gave signal in {frac_weak_sparse * 100:.1f}% of groups, shaped "
            f"in {frac_weak_shaped * 100:.1f}%",
        )
    except NotImplementedYet as e:
        _skip(g, ["the sparse-reward experiment runs"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["the sparse-reward experiment runs"], f"{type(e).__name__}: {e}")


def check_lab3(g: Grader) -> None:
    print(f"\n{DIM}-- lab 3: tools --{RESET}")
    try:
        import verifiers.v1 as vf

        import exercise_3_tools as l3
    except Exception as e:
        _skip(g, ["lab 3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 3 imports", True)

    try:
        toolset = l3.CalcToolset(vf.ToolsetConfig())
        g.check("calculator does arithmetic", toolset.calculator("17 * 23").strip() == "391",
                f"got {toolset.calculator('17 * 23')!r}")
        g.check("calculator handles precedence",
                toolset.calculator("144 * 12 - 500").strip() == "1228",
                f"got {toolset.calculator('144 * 12 - 500')!r}")
        g.check("calculator returns a string", isinstance(toolset.calculator("1 + 1"), str),
                "tool output is text -- it becomes the model's next turn")
        bad = toolset.calculator("hello")
        g.check("calculator returns an error message instead of raising",
                isinstance(bad, str) and "error" in bad.lower(),
                f"on bad input it returned {bad!r}; return an actionable string, "
                f"don't raise -- a raised exception is a dead rollout")
        danger = toolset.calculator("__import__('os')")
        g.check("calculator refuses non-arithmetic input",
                isinstance(danger, str) and "error" in danger.lower(),
                f"got {danger!r} -- never evaluate arbitrary model output")
        doc = inspect.getdoc(l3.CalcToolset.calculator) or ""
        g.check("the docstring is real (the model reads it!)",
                "TODO" not in doc and len(doc.strip()) > 40 and "example" in doc.lower(),
                "the docstring becomes the tool description sent to the model; "
                "include an example call")
    except NotImplementedYet as e:
        _skip(g, ["calculator is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["calculator is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        toolset = l3.CalcToolset(vf.ToolsetConfig())
        desc = l3.tool_descriptions(toolset)
        g.check("tool_descriptions finds the tool",
                len(desc) == 1 and desc[0][0] == "calculator",
                f"got {[(n, d[:20]) for n, d in desc]}")
        g.check("the description is the docstring",
                desc[0][1] == (inspect.getdoc(l3.CalcToolset.calculator) or "").strip()
                or desc[0][1] == (l3.CalcToolset.calculator.__doc__ or "").strip(),
                "return the stripped docstring, exactly what register() sends")
    except NotImplementedYet as e:
        _skip(g, ["tool_descriptions is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["tool_descriptions is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 03 — conversations and tools (v1)")
    check_lab1(g)
    check_lab2(g)
    check_lab3(g)
    report(g)


if __name__ == "__main__":
    main()
