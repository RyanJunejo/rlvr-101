"""
Unit 03 autograder.

    uv run python modules/03-multiturn-and-tools/verify.py

Runs entirely offline. No API key needed.
"""

from __future__ import annotations

import asyncio
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


def _run(coro):
    return asyncio.run(coro)


def check_lab1(g: Grader) -> None:
    print(f"\n{DIM}-- lab 1: the guessing game --{RESET}")
    try:
        import exercise_1_guessing_game as l1
        import verifiers as vf
    except Exception as e:
        _skip(g, ["lab 1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 1 imports", True)

    def fresh(secret=42):
        env = l1.build_env()
        st = vf.State({"answer": str(secret), "prompt": [], "trajectory": []})
        return env, _run(env.setup_state(st))

    try:
        env, st = fresh(42)
        g.check("setup_state stores the secret as an int", st.get("secret") == 42,
                f"expected state['secret'] == 42 (int), got {st.get('secret')!r}")
        g.check("setup_state starts an empty guess list", st.get("guesses") == [],
                f"expected [], got {st.get('guesses')!r}")
        g.check("setup_state starts unsolved", st.get("solved") is False,
                f"expected False, got {st.get('solved')!r}")
    except NotImplementedYet as e:
        _skip(g, ["setup_state is correct"], f"still TODO: {e}")
        return
    except Exception as e:
        _skip(g, ["setup_state is correct"], f"{type(e).__name__}: {e}")
        return

    try:
        env, st = fresh(42)

        def say(text):
            return _run(env.env_response([vf.AssistantMessage(content=text)], st))[-1]["content"]

        low = say("Guess: 10")
        g.check("a guess below the secret gets 'too low'", "low" in low.lower(),
                f"guessed 10 with secret 42, env replied {low!r}")
        high = say("Guess: 90")
        g.check("a guess above the secret gets 'too high'", "high" in high.lower(),
                f"guessed 90 with secret 42, env replied {high!r}")
        g.check("guesses are recorded in state", st["guesses"] == [10, 90],
                f"after guessing 10 then 90, state['guesses'] is {st['guesses']!r}")

        correct = say("Let me think.\nGuess: 42")
        g.check("the correct guess is recognised", "correct" in correct.lower(),
                f"guessed the secret, env replied {correct!r}")
        g.check("solving sets state['solved']", st.get("solved") is True,
                f"expected True after a correct guess, got {st.get('solved')!r}")

        env2, st2 = fresh(42)
        before = list(st2["guesses"])
        reply = _run(env2.env_response([vf.AssistantMessage(content="I have no idea")], st2))
        g.check("a message with no guess doesn't crash", isinstance(reply, list) and len(reply) > 0,
                f"expected a list of messages, got {reply!r}")
        g.check("a message with no guess records nothing", st2["guesses"] == before,
                f"guesses changed from {before!r} to {st2['guesses']!r}")

        env3, st3 = fresh(42)
        _run(env3.env_response([vf.AssistantMessage(content="Guess: 5\nActually wait.\nGuess: 42")], st3))
        g.check("the LAST guess in a message is the one that counts", st3.get("solved") is True,
                f"message ended with 'Guess: 42' but solved is {st3.get('solved')!r} "
                f"and guesses are {st3['guesses']!r}")
    except NotImplementedYet as e:
        _skip(g, ["env_response is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["env_response is correct"], f"{type(e).__name__}: {e}")

    try:
        env, st = fresh(42)
        g.check("game_solved is False before the game is won", _run(env.game_solved(st)) is False)
        st["solved"] = True
        g.check("game_solved is True after the game is won", _run(env.game_solved(st)) is True)
        g.check("game_solved tolerates a state with no 'solved' key",
                _run(env.game_solved({})) in (False, None),
                "use state.get('solved', False) rather than state['solved']")
    except NotImplementedYet as e:
        _skip(g, ["game_solved is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["game_solved is correct"], f"{type(e).__name__}: {e}")

    try:
        g.close("solved_reward pays for a win", l1.solved_reward({"solved": True}), 1.0)
        g.close("solved_reward pays nothing for a loss", l1.solved_reward({"solved": False}), 0.0)
    except NotImplementedYet as e:
        _skip(g, ["solved_reward is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["solved_reward is correct"], f"{type(e).__name__}: {e}")

    try:
        env = l1.build_env()
        g.check("the environment allows 7 turns", env.max_turns == 7,
                f"expected max_turns == 7 (binary search needs 7 for 1-100), got {env.max_turns}")
    except Exception as e:
        _skip(g, ["build_env is correct"], f"{type(e).__name__}: {e}")


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
                f"guessing 51 (off by 1) scored {c:.4f}; guessing 100 (off by 50) scored {f:.4f}")

        # THE key property of this lab
        g.check(
            "partial credit can never beat winning (the 0.5 cap)",
            c < 1.0 and c <= 0.5 + 1e-9,
            f"a near-miss scored {c:.4f}. It must stay strictly below the 1.0 paid for "
            f"winning -- otherwise the model can farm partial credit forever without "
            f"finishing. This is the whole point of the exercise.",
        )
        g.check(
            "shaped_reward uses the CLOSEST guess, not the last one",
            abs(l2.shaped_reward({"secret": 50, "guesses": [51, 100], "solved": False}) - c) < 1e-9,
            "a run that got within 1 and then wandered away should score the same as "
            "one that got within 1 and stopped",
        )
    except NotImplementedYet as e:
        _skip(g, ["shaped_reward is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["shaped_reward is correct"], f"{type(e).__name__}: {e}")

    try:
        g.check("group_has_signal is False when every score is identical",
                l2.group_has_signal(np.array([0.0, 0.0, 0.0, 0.0])) is False
                or l2.group_has_signal(np.array([0.0, 0.0, 0.0, 0.0])) == False,  # noqa: E712
                "a group of all-zeros produces no gradient, so this must be False")
        g.check("group_has_signal is False when every score is 1.0",
                not l2.group_has_signal(np.array([1.0, 1.0, 1.0])),
                "all-1.0 is the same failure as all-0.0 -- no spread, no learning")
        g.check("group_has_signal is True when scores differ",
                bool(l2.group_has_signal(np.array([0.0, 1.0, 0.0, 1.0]))),
                "these scores differ, so this group does produce a gradient")
        g.check("group_has_signal handles tiny float differences as no signal",
                not l2.group_has_signal(np.array([0.3, 0.3, 0.3 + 1e-15])),
                "use a tolerance, not exact equality -- floats from the shaped reward "
                "will differ in the last bit")
    except NotImplementedYet as e:
        _skip(g, ["group_has_signal is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["group_has_signal is correct"], f"{type(e).__name__}: {e}")

    # the headline result of the unit
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
            f"sparse gave signal in {frac_weak_sparse * 100:.1f}% of groups, shaped in "
            f"{frac_weak_shaped * 100:.1f}%; shaping should be dramatically better here",
        )
    except NotImplementedYet as e:
        _skip(g, ["the sparse-reward experiment runs"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["the sparse-reward experiment runs"], f"{type(e).__name__}: {e}")


def check_lab3(g: Grader) -> None:
    print(f"\n{DIM}-- lab 3: tools --{RESET}")
    try:
        import exercise_3_tools as l3
        import verifiers as vf
    except Exception as e:
        _skip(g, ["lab 3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("lab 3 imports", True)

    try:
        g.check("calculator does arithmetic", l3.calculator("17 * 23").strip() == "391",
                f"calculator('17 * 23') returned {l3.calculator('17 * 23')!r}, expected '391'")
        g.check("calculator handles precedence", l3.calculator("144 * 12 - 500").strip() == "1228",
                f"got {l3.calculator('144 * 12 - 500')!r}, expected '1228'")
        g.check("calculator returns a string", isinstance(l3.calculator("1 + 1"), str),
                "tools must return text -- it becomes the model's next turn")

        bad = l3.calculator("hello")
        g.check("calculator returns an error message instead of raising",
                isinstance(bad, str) and "error" in bad.lower(),
                f"on bad input it returned {bad!r}; return a helpful string so the "
                f"model can correct itself, don't raise")
        danger = l3.calculator("__import__('os')")
        g.check("calculator refuses non-arithmetic input",
                isinstance(danger, str) and "error" in danger.lower(),
                f"got {danger!r} -- never evaluate arbitrary model output")

        doc = (l3.calculator.__doc__ or "")
        g.check(
            "calculator has a real docstring (the model reads it!)",
            "TODO" not in doc and len(doc.strip()) > 40,
            "the docstring becomes the tool description sent to the model. Replace the "
            "TODO scaffolding with a real one-or-two-line description plus an example.",
        )
    except NotImplementedYet as e:
        _skip(g, ["calculator is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["calculator is correct"], f"{type(e).__name__}: {e}")

    try:
        g.close("correct_answer accepts an exact match",
                l3.correct_answer([{"role": "assistant", "content": "Answer: 391"}], "391"), 1.0)
        g.close("correct_answer rejects a superset number",
                l3.correct_answer([{"role": "assistant", "content": "Answer: 3912"}], "391"), 0.0)
        g.close("correct_answer rejects a reply with no Answer line",
                l3.correct_answer([{"role": "assistant", "content": "it's 391"}], "391"), 0.0)
    except NotImplementedYet as e:
        _skip(g, ["correct_answer is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["correct_answer is correct"], f"{type(e).__name__}: {e}")

    try:
        env = l3.build_tool_env([l3.calculator])
        g.check("build_tool_env returns a ToolEnv", isinstance(env, vf.ToolEnv))
        g.check("the tool is registered", "calculator" in env.tool_map,
                f"tool_map has {list(env.tool_map)}")
        names = [t["name"] for t in env.tool_defs]
        g.check("a tool definition was generated for the model", names == ["calculator"],
                f"got {names}")
        desc = env.tool_defs[0].get("description", "")
        g.check("the tool description came from your docstring", len(desc.strip()) > 40,
                f"the model will see: {desc!r}")
    except NotImplementedYet as e:
        _skip(g, ["build_tool_env is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["build_tool_env is correct"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Unit 03 — conversations and tools")
    check_lab1(g)
    check_lab2(g)
    check_lab3(g)
    report(g)


if __name__ == "__main__":
    main()
