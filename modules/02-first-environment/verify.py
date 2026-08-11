"""
Module 02 self-check.

    uv run python modules/02-first-environment/verify.py

Runs entirely offline -- no API key needed. Exercise 1's live eval is separate;
here we only check that the pieces are correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from common.grading import DIM, RESET, Grader, NotImplementedYet, report  # noqa: E402


def _msg(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def _skip(g: Grader, names: list[str], why: str) -> None:
    for n in names:
        g.check(n, False, why)


def check_ex1(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 1: first environment --{RESET}")
    try:
        import exercise_1_first_env as ex1
    except Exception as e:
        _skip(g, ["ex1 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex1 imports", True)

    try:
        ds = ex1.build_dataset()
        cols = set(ds.column_names)
        g.check(
            "dataset has `question` and `answer` columns",
            {"question", "answer"} <= cols,
            f"got columns {sorted(cols)}; verifiers looks for 'question' and 'answer'",
        )
        g.check(
            "dataset has one row per problem",
            len(ds) == len(ex1.PROBLEMS),
            f"expected {len(ex1.PROBLEMS)} rows, got {len(ds)}",
        )
        g.check(
            "dataset does NOT pre-create a `prompt` column",
            "prompt" not in cols,
            "verifiers builds `prompt` itself; supplying one makes it silently ignore "
            "your system_prompt",
        )
    except NotImplementedYet as e:
        _skip(g, ["build_dataset is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["build_dataset is correct"], f"{type(e).__name__}: {e}")

    try:
        cases = [
            ("Answer: 391", "391", "plain"),
            ("Answer:   391  ", "391", "extra whitespace"),
            ("Answer: 391.", "391", "trailing period"),
            ("Answer: 12\nActually no.\nAnswer: 391", "391", "takes the LAST answer line"),
            ("no answer line here", "", "returns '' when the format is absent"),
        ]
        for text, want, label in cases:
            got = ex1.extract_final_answer(text)
            g.check(
                f"extract_final_answer: {label}",
                got == want,
                f"input {text!r} -> expected {want!r}, got {got!r}",
            )
    except NotImplementedYet as e:
        _skip(g, ["extract_final_answer is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["extract_final_answer is correct"], f"{type(e).__name__}: {e}")

    try:
        g.close(
            "correct_answer rewards an exact match",
            ex1.correct_answer(_msg("17*23=391.\nAnswer: 391"), "391"),
            1.0,
        )
        g.close(
            "correct_answer rejects a superset number (391 vs 3912)",
            ex1.correct_answer(_msg("Answer: 3912"), "391"),
            0.0,
        )
        g.close(
            "correct_answer ignores the answer appearing only in prose",
            ex1.correct_answer(_msg("The product is 391."), "391"),
            0.0,
        )
    except NotImplementedYet as e:
        _skip(g, ["correct_answer is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["correct_answer is correct"], f"{type(e).__name__}: {e}")

    try:
        env = ex1.build_env()
        import verifiers as vf

        g.check("build_env returns a SingleTurnEnv", isinstance(env, vf.SingleTurnEnv))
        built = env.get_dataset()
        g.check(
            "the built dataset has a `prompt` column",
            "prompt" in built.column_names,
            f"got {built.column_names}",
        )
        first = built[0]["prompt"]
        g.check(
            "the system prompt is prepended to each prompt",
            isinstance(first, list) and first[0]["role"] == "system",
            f"expected the first message to be a system message, got {first[:1]}",
        )
    except NotImplementedYet as e:
        _skip(g, ["build_env is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["build_env is correct"], f"{type(e).__name__}: {e}")


def check_ex2(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 2: rubrics --{RESET}")
    try:
        import exercise_2_rubric as ex2
    except Exception as e:
        _skip(g, ["ex2 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex2 imports", True)

    try:
        g.close("format_reward accepts a well-formed reply", ex2.format_reward(_msg("Answer: 5")), 1.0)
        g.close(
            "format_reward accepts a WRONG but well-formed reply",
            ex2.format_reward(_msg("Answer: 999")),
            1.0,
        )
        g.close("format_reward rejects a reply with no answer line", ex2.format_reward(_msg("it's 5")), 0.0)
        g.close(
            "format_reward rejects an empty answer slot",
            ex2.format_reward(_msg("Answer:   ")),
            0.0,
        )
    except NotImplementedYet as e:
        _skip(g, ["format_reward is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["format_reward is correct"], f"{type(e).__name__}: {e}")

    try:
        g.close("ex2 correct_answer: exact match", ex2.correct_answer(_msg("Answer: 391"), "391"), 1.0)
        g.close("ex2 correct_answer: superset rejected", ex2.correct_answer(_msg("Answer: 3912"), "391"), 0.0)
        g.close("ex2 correct_answer: no slot means 0", ex2.correct_answer(_msg("391"), "391"), 0.0)
    except NotImplementedYet as e:
        _skip(g, ["ex2 correct_answer is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["ex2 correct_answer is correct"], f"{type(e).__name__}: {e}")

    try:
        r = ex2.build_rubric()
        names = [f.__name__ for f in r.funcs]
        g.check(
            "rubric contains both reward functions",
            set(names) == {"correct_answer", "format_reward"},
            f"got {names}",
        )
        g.check(
            "correctness is weighted above format",
            len(r.weights) == 2 and r.weights[names.index("correct_answer")]
            > r.weights[names.index("format_reward")],
            f"functions {names} with weights {r.weights}; correctness must outweigh format",
        )
    except NotImplementedYet as e:
        _skip(g, ["build_rubric is correct"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["build_rubric is correct"], f"{type(e).__name__}: {e}")


def check_ex3(g: Grader) -> None:
    print(f"\n{DIM}-- exercise 3: reward hacking lab --{RESET}")
    try:
        import exercise_3_reward_hacking as ex3
    except Exception as e:
        _skip(g, ["ex3 imports"], f"{type(e).__name__}: {e}")
        return
    g.check("ex3 imports", True)

    # The sloppy one is provided, but confirm the premise still holds --
    # if someone "fixes" it the lesson evaporates.
    s_correct, _ = ex3.run_suite(ex3.sloppy_correctness)
    g.check(
        "sloppy_correctness is still broken (don't fix it -- it's the exhibit)",
        s_correct < len(ex3.ATTACKS),
        f"it scored {s_correct}/{len(ex3.ATTACKS)}; the whole point is that it fails",
    )

    try:
        for name, reply, answer, expected in ex3.ATTACKS:
            got = float(ex3.robust_correctness(_msg(reply), answer))
            g.check(
                f"robust_correctness survives: {name}",
                abs(got - expected) < 1e-9,
                f"reply {reply[:60]!r}... with answer {answer!r} "
                f"should score {expected}, got {got}",
            )
    except NotImplementedYet as e:
        _skip(g, ["robust_correctness survives the attack suite"], f"still TODO: {e}")
    except Exception as e:
        _skip(g, ["robust_correctness survives the attack suite"], f"{type(e).__name__}: {e}")


def main() -> None:
    g = Grader("Module 02 — first environment")
    check_ex1(g)
    check_ex2(g)
    check_ex3(g)
    report(g)


if __name__ == "__main__":
    main()
