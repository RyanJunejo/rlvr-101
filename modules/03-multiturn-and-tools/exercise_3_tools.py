"""
Lab 3 — giving the model tools.

In v1, tools live on a `Toolset`: a class whose `@vf.tool` methods are exposed
to the model as callable tools. The library builds each tool's schema from the
method's SIGNATURE and its description from the DOCSTRING.

Which means the docstring is not a comment. It is part of the prompt. A vague
docstring produces a model that misuses the tool, and you will spend an
afternoon wondering why.

The second lesson, which matters more than it looks: WHAT YOUR TOOL RETURNS ON
BAD INPUT IS TRAINING DATA. The model sees the text and gets another turn.
"ValueError" teaches nothing; "only numbers and + - * / ** are allowed" tells
it what to do differently.

Fill in the two TODOs, then run me:

    uv run python modules/03-multiturn-and-tools/exercise_3_tools.py

No API key needed.

TERMS USED IN THIS FILE

  tool:    a function the model can call mid-conversation
  Toolset: a class whose @vf.tool methods are exposed to the model
  schema:  the machine-readable description of a tool's arguments, built from
           your type hints
"""

from __future__ import annotations

import ast
import operator
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.utils.decorators import discover_decorated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# Safe arithmetic evaluation. `eval()` on model output is a genuine security
# hole -- during RL the model is actively searching for high-scoring inputs,
# which makes it an automated fuzzer pointed at your tool. Walk the syntax
# tree; allow arithmetic; nothing else.
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


class CalcToolset(vf.Toolset[vf.ToolsetConfig, vf.State]):
    """A toolset with one tool. The model sees what the decorator exposes."""

    @vf.tool
    def calculator(self, expression: str) -> str:
        """TODO: replace this docstring -- it becomes the tool's description.

        Two jobs:

        1. THE DOCSTRING. One or two lines telling the model what the tool
           does, what the argument looks like, and an example call. Models
           imitate concrete examples far better than they interpret
           descriptions, so include one:
           'Evaluate an arithmetic expression and return the result. Supports
           + - * / ** and parentheses. Example: calculator("17 * 23") -> "391".'

        2. THE BODY:
             - parse with `ast.parse(expression, mode="eval")`
             - evaluate with `_safe_eval(tree.body)`
             - return the result AS A STRING (tool output is text to the model)
             - on any exception, RETURN a helpful error string -- don't raise.
               The model reads your return value as its next turn; a good error
               message is a chance to self-correct, a stack trace is a dead
               rollout. Name what IS allowed, not just what went wrong.

        Args:
            expression: an arithmetic expression, e.g. "17 * 23".
        Returns:
            the result as a string, or an actionable error message.
        """
        # TODO: implement, and replace the docstring above with a real one.
        return todo("calculator: safe-eval the expression, return a string")


def tool_descriptions(toolset: vf.Toolset) -> list[tuple[str, str]]:
    """What the model will actually see: (tool name, description) pairs.

    `discover_decorated(toolset, "tool")` returns the marked methods. The name
    is `tool_name` if the decorator overrode it, else the method's `__name__`;
    the description is the docstring, stripped -- exactly the fields the
    library's own `register()` sends to the MCP server.

    Args:
        toolset: any Toolset instance.
    Returns:
        [(name, description)], sorted by name.
    """
    # TODO: implement with discover_decorated.
    return todo("tool_descriptions: [(name, docstring)] from discover_decorated")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    toolset = CalcToolset(vf.ToolsetConfig())

    print(f"{BOLD}1. What the model sees{RESET}\n")
    for name, desc in tool_descriptions(toolset):
        print(f"  tool: {name}")
        for line in desc.splitlines()[:3]:
            print(f"        {line}")
    print(
        f"\n{DIM}  That text came straight from your docstring. If it's vague, the\n"
        f"  model's tool use will be vague. Prompt engineering in a Python costume.{RESET}"
    )

    print(f"\n{BOLD}2. Does it work -- including on garbage?{RESET}\n")
    cases = ["17 * 23", "4891 + 2778", "2 ** 10", "144 * 12 - 500",
             "__import__('os')", "hello"]
    for c in cases:
        print(f"  calculator({c!r:24}) -> {toolset.calculator(c)!r}")
    print(
        f"\n{DIM}  The last two rows are the lesson. A model WILL send garbage, and your\n"
        f"  return value is its next turn. Note '__import__' returns an error\n"
        f"  instead of executing -- never eval() model output.{RESET}"
    )

    print(
        f"\n{BOLD}3. The design question to sit with{RESET}\n"
        "\n"
        "  Give a calculator to a model scored on arithmetic and you are no longer\n"
        "  training arithmetic -- you are training tool use. The model doesn't\n"
        "  need to know 17*23=391; it needs to know when to reach for the tool\n"
        "  and how to call it. Often that IS what you want. Be sure it is,\n"
        "  because a tool that does the whole task isn't a tool, it's an answer\n"
        "  key, and your scores will look great while measuring nothing.\n"
        "\n"
        "  Two rules from the library worth knowing before the capstone:\n"
        "    - expensive, task-agnostic setup goes in the toolset's `setup`;\n"
        "      per-task inputs arrive in `setup_task`\n"
        "    - mutable per-rollout data lives on `self.state`, never on `self` --\n"
        "      a toolset instance is shared across concurrent rollouts, and\n"
        "      state on `self` is how two games corrupt each other."
    )


if __name__ == "__main__":
    run_main(main)
