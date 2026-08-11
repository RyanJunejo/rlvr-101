"""
Lab 3 — giving the model tools.

A tool is just a Python function. `ToolEnv` handles the rest: it tells the model
what tools exist, notices when the model asks to call one, runs it, and feeds the
result back as another turn.

THE ONE THING TO INTERNALISE HERE:

    The library builds the tool description the model sees from your function's
    NAME, SIGNATURE and DOCSTRING.

Which means your docstring is not a comment. It is part of the prompt. A vague
docstring produces a model that misuses the tool, and you will spend an afternoon
wondering why. You'll print the generated definitions below and see exactly what
the model gets.

The second thing, which matters more than it looks: WHAT YOUR TOOL RAISES ON BAD
INPUT IS TRAINING DATA. The model sees the error text and gets another turn to
fix its mistake. "ValueError" teaches it nothing. "expression must contain only
digits and + - * / ( )" teaches it what to do differently.

Fill in the three TODOs, then run me:

    uv run python modules/03-multiturn-and-tools/exercise_3_tools.py

No API key needed for the main part.
"""

from __future__ import annotations

import ast
import operator
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# Safe arithmetic evaluation. `eval()` on model output would be a genuine
# security hole, so we walk the syntax tree and only allow arithmetic.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")


def calculator(expression: str) -> str:
    """TODO: write this docstring -- it becomes part of the model's prompt.

    Your job here is two things:

    1. WRITE THE DOCSTRING. It should tell the model, in one or two lines, what
       this tool does, what the argument looks like, and give an example. Write it
       the way you'd write it for a colleague who can't see the code. Something
       like: "Evaluate an arithmetic expression and return the result. Supports
       + - * / and **. Example: calculator('17 * 23') returns '391'."

    2. IMPLEMENT THE BODY:
         - parse `expression` with `ast.parse(expression, mode="eval")`
         - evaluate it with `_safe_eval(tree.body)`
         - return the result as a STRING (tools return text to the model)
         - on any exception, return a HELPFUL error string, not a raised
           exception. Something the model can act on, e.g.
           f"error: {e}. Provide a plain arithmetic expression like '2 * (3 + 4)'."

    Why return the error instead of raising: the model gets your return value as
    its next turn. A useful message lets it correct itself; a stack trace doesn't.

    Args:
        expression: an arithmetic expression, e.g. "17 * 23".
    Returns:
        the result as a string, or a helpful error message.
    """
    # TODO: implement, and replace the docstring above with a real one.
    return todo("calculator: evaluate the expression safely, return a string")


def build_tool_env(tools: list) -> vf.ToolEnv:
    """Wrap the tools in a ToolEnv.

    Use `vf.ToolEnv(...)` with:
        tools=tools
        max_turns=6                 -- generous; the tasks need 1-2 calls
        dataset / eval_dataset      -- use build_dataset() below for both
        system_prompt=SYSTEM_PROMPT
        rubric                      -- vf.Rubric(funcs=[correct_answer], weights=[1.0])

    Args:
        tools: list of plain Python functions.
    Returns:
        a configured vf.ToolEnv
    """
    # TODO: build and return the ToolEnv.
    return todo("build_tool_env: vf.ToolEnv(tools=..., max_turns=6, dataset=..., rubric=...)")


def correct_answer(completion, answer, **kwargs) -> float:
    """1.0 if the model's final message contains the right answer as its last line.

    Same discipline as Unit 02: look at the model's committed answer, compare
    exactly. Here the model's last message should end with "Answer: <number>".

    Steps:
      - text = reply_text(completion)
      - find all lines matching "Answer: <something>" (ANSWER_RE below)
      - no match -> 0.0
      - otherwise compare the LAST match to `answer`, stripped, exactly

    Args:
        completion: list of chat messages.
        answer: the reference answer string.
    Returns:
        1.0 or 0.0
    """
    # TODO: implement, reusing the Unit 02 pattern.
    return todo("correct_answer: exact match on the final 'Answer:' line")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------

import re  # noqa: E402

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "Answer the question. You have tools available -- use them for any arithmetic "
    "rather than doing it in your head.\n"
    "When you have the final result, end your message with exactly:\n"
    "Answer: <number>"
)

PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 4891 + 2778?", "7669"),
    ("What is 144 * 12 - 500?", "1228"),
    ("What is 2 ** 10?", "1024"),
]


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


def build_dataset() -> Dataset:
    return Dataset.from_dict(
        {"question": [q for q, _ in PROBLEMS], "answer": [a for _, a in PROBLEMS]}
    )


def describe_rubric(rubric) -> list[tuple[str, str, float]]:
    """List every scoring function, including ones nested in sub-rubrics.

    Adding a rubric to an environment wraps everything in a RubricGroup, so the
    top-level `.funcs` is empty and the real functions live in `.rubrics`. Worth
    knowing when you go looking for your own reward function and can't find it.
    """
    out = []
    subs = getattr(rubric, "rubrics", None)
    if subs:
        for sub in subs:
            for f, w in zip(sub.funcs, sub.weights):
                out.append((type(sub).__name__, f.__name__, w))
    else:
        for f, w in zip(rubric.funcs, rubric.weights):
            out.append((type(rubric).__name__, f.__name__, w))
    return out


def main() -> None:
    print(f"{BOLD}1. What your tool actually looks like to the model{RESET}\n")
    env = build_tool_env([calculator])
    for td in env.tool_defs:
        print(f"  name:        {td['name']}")
        print(f"  description: {td.get('description', '')!r}")
        print(f"  parameters:  {td.get('parameters')}")
    print(
        f"\n{DIM}  That description came straight from your docstring. If it's vague,\n"
        f"  the model's tool use will be vague. This is prompt engineering wearing\n"
        f"  a Python costume.{RESET}"
    )

    print(f"\n{BOLD}2. Does the tool work, including on bad input?{RESET}\n")
    cases = ["17 * 23", "4891 + 2778", "2 ** 10", "144 * 12 - 500", "__import__('os')", "hello"]
    for c in cases:
        print(f"  calculator({c!r:24}) -> {calculator(c)!r}")
    print(
        f"\n{DIM}  The last two are the interesting ones. A model WILL send you garbage,\n"
        f"  and what you return is its next turn. Note that '__import__' returns an\n"
        f"  error rather than executing -- never `eval()` model output.{RESET}"
    )

    print(f"\n{BOLD}3. Free monitoring{RESET}\n")
    print(f"  {'source rubric':<26} {'metric':<20} {'weight':>7}")
    print("  " + "-" * 55)
    for src, name, w in describe_rubric(env.rubric):
        print(f"  {src:<26} {name:<20} {w:>7.1f}")
    print(
        f"\n{DIM}  ToolEnv added tool-call counters automatically. They're weighted 0, so\n"
        f"  they don't affect the score -- they're diagnostics. This is how you'd\n"
        f"  notice a model that learned to call a tool 40 times per question because\n"
        f"  something in your scoring accidentally rewarded it.{RESET}"
    )

    print(
        f"\n{BOLD}The design question to sit with{RESET}\n"
        f"  You just gave a calculator to a model being scored on arithmetic. So what\n"
        f"  are you actually training now? Not arithmetic -- tool use. The model no\n"
        f"  longer needs to know that 17*23=391; it needs to know when to reach for\n"
        f"  the tool and how to call it.\n"
        f"\n"
        f"  That may be exactly what you want. Just make sure it IS what you want,\n"
        f"  because a tool that does the whole task isn't a tool, it's an answer key."
    )


if __name__ == "__main__":
    run_main(main)
