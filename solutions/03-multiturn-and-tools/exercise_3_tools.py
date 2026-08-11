"""
ANSWER KEY — Unit 03, Lab 3: tools.

    uv run python solutions/03-multiturn-and-tools/exercise_3_tools.py
"""

from __future__ import annotations

import ast
import operator
import re
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

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
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / ** and parentheses. Use this for any arithmetic instead of
    calculating in your head. Example: calculator("17 * 23") returns "391".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        return (
            f"error: {e}. Provide a plain arithmetic expression using only numbers "
            f"and + - * / ** and parentheses, for example '2 * (3 + 4)'."
        )


ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def correct_answer(completion, answer, **kwargs) -> float:
    """Unit 02's discipline, unchanged: commit to a slot, compare exactly."""
    matches = ANSWER_RE.findall(reply_text(completion))
    if not matches:
        return 0.0
    return 1.0 if matches[-1].strip().rstrip(".").strip() == str(answer).strip() else 0.0


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


def build_tool_env(tools: list) -> vf.ToolEnv:
    ds = build_dataset()
    return vf.ToolEnv(
        tools=tools,
        max_turns=6,
        dataset=ds,
        eval_dataset=ds,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.Rubric(funcs=[correct_answer], weights=[1.0]),
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

    print(f"\n{BOLD}2. Does the tool work, including on bad input?{RESET}\n")
    for c in ["17 * 23", "4891 + 2778", "2 ** 10", "144 * 12 - 500", "__import__('os')", "hello"]:
        print(f"  calculator({c!r:24}) -> {calculator(c)!r}")

    print(f"\n{BOLD}3. Free monitoring{RESET}\n")
    print(f"  {'source rubric':<26} {'metric':<20} {'weight':>7}")
    print("  " + "-" * 55)
    for src, name, w in describe_rubric(env.rubric):
        print(f"  {src:<26} {name:<20} {w:>7.1f}")

    print(
        f"\n{BOLD}DISCUSSION{RESET}\n"
        "\n"
        "  THE DOCSTRING IS PROMPT, NOT DOCUMENTATION. Look at what got printed in\n"
        "  section 1: the description the model receives is your docstring, verbatim.\n"
        "  The parameter schema came from your type hints. This means a habit that's\n"
        "  merely good practice in normal Python -- precise docstrings, real type\n"
        "  hints -- becomes load-bearing here. 'Does math' and the docstring above\n"
        "  produce measurably different tool-use behaviour.\n"
        "\n"
        "  Include an EXAMPLE CALL in every tool docstring. Models are much better at\n"
        "  imitating a concrete example than at interpreting a description.\n"
        "\n"
        "  RETURN ERRORS, DON'T RAISE THEM. Whatever this function returns becomes the\n"
        "  model's next turn. A returned error message is a chance to self-correct; a\n"
        "  raised exception is a dead rollout. Note the error text names what IS\n"
        "  allowed rather than only what went wrong -- 'invalid syntax' tells the\n"
        "  model nothing actionable.\n"
        "\n"
        "  NEVER eval() MODEL OUTPUT. `_safe_eval` walks the syntax tree and permits\n"
        "  only arithmetic nodes. Plain `eval(\"__import__('os').system('rm -rf ~')\")`\n"
        "  does exactly what it looks like. During RL this is worse than usual: the\n"
        "  model is actively searching for inputs that maximise a score, which makes\n"
        "  it an automated fuzzer pointed at your tool.\n"
        "\n"
        "  THE MONITORING METRICS ARE WEIGHTED ZERO. total_tool_calls and\n"
        "  calculator_calls don't change the score -- they're instrumentation. Watch\n"
        "  them during training. A model whose tool calls per question climbs steadily\n"
        "  has usually found something in your scoring that pays for calling tools,\n"
        "  and you want to know that before it's had 500 steps to perfect it.\n"
        "\n"
        "  THE DESIGN QUESTION. A calculator on an arithmetic task means you're no\n"
        "  longer training arithmetic -- you're training tool use. The model doesn't\n"
        "  need to know 17*23=391; it needs to know when to reach for the tool and how\n"
        "  to call it. That's a legitimate goal and often the one you actually want.\n"
        "  Just be deliberate about it: a tool that does the entire task isn't a tool,\n"
        "  it's an answer key, and your scores will look great while measuring nothing."
    )


if __name__ == "__main__":
    main()
