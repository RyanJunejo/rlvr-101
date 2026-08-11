"""
Exercise 1 — a working environment, end to end.

Small and real: a handful of arithmetic problems, a reward function that checks
the answer, and a live model on the other end.

The thing to appreciate as you build it: this same object is an eval, a
synthetic-data generator, and an RL training environment. Nothing about it
changes between those uses. That's the design insight of the library.

Fill in the TODOs, then run me:

    uv run python modules/02-first-environment/exercise_1_first_env.py

You need `.env` configured (see Unit 00) -- this one makes real API calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import run_main, todo  # noqa: E402

SYSTEM_PROMPT = (
    "You are a careful calculator. Think briefly, then end your reply with the "
    "final answer on its own last line, in exactly this format:\n"
    "Answer: <number>"
)

PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 144 / 12?", "12"),
    ("What is 89 + 156?", "245"),
    ("What is 1000 - 377?", "623"),
    ("What is 13 * 13?", "169"),
    ("What is 7 * 8 + 4?", "60"),
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
    """A HF Dataset with the two columns verifiers expects.

    `question` and `answer` are the default column names. verifiers will add an
    `example_id` and build a `prompt` column (chat messages, with SYSTEM_PROMPT
    prepended) automatically -- you don't create `prompt` yourself.

    Returns:
        Dataset with columns "question" and "answer", one row per PROBLEMS entry.
    """
    # TODO: build the dataset from PROBLEMS.
    # Hint: Dataset.from_dict({"question": [...], "answer": [...]})
    return todo("build_dataset: turn PROBLEMS into a Dataset with question/answer columns")


def extract_final_answer(text: str) -> str:
    """Pull the number out of an 'Answer: <number>' line.

    Scan the lines of `text` from the BOTTOM up, find the first one starting with
    'Answer:' (case-insensitive), and return whatever follows the colon,
    stripped of whitespace and any trailing period.

    Bottom-up matters: models often restate the format mid-reasoning ("...so the
    Answer: format wants a number..."). The last one is the real one.

    If no such line exists, return "" -- an empty string, not None. Returning a
    sentinel string keeps every downstream comparison total; None would make you
    write a null check in every reward function that touches this.

    Args:
        text: the model's raw reply.
    Returns:
        the extracted answer string, or "" if the format wasn't followed.
    """
    # TODO: implement bottom-up scan for the Answer: line.
    return todo("extract_final_answer: find the last 'Answer:' line, return what follows")


def correct_answer(completion, answer, **kwargs) -> float:
    """Reward 1.0 for an exactly-correct final answer, else 0.0.

    Remember the shape of `completion`: it's a list of chat messages, so the
    model's text is `reply_text(completion)`.

    Use `extract_final_answer` to get the model's answer, then compare it to
    `answer` exactly (after stripping whitespace from both). Exact, not
    substring -- exercise 3 is entirely about why.

    Args:
        completion: list of chat messages; the last one is the model's reply.
        answer: the reference answer string from the dataset.
    Returns:
        1.0 or 0.0
    """
    # TODO: extract the model's answer and compare it exactly to `answer`.
    return todo("correct_answer: exact match between extracted answer and reference")


def build_env() -> vf.SingleTurnEnv:
    """Assemble the environment.

    Wire together:
      - dataset AND eval_dataset: both `build_dataset()` (tiny set, so we
        evaluate on the same rows we'd train on -- fine here, not fine in real life)
      - system_prompt: SYSTEM_PROMPT
      - rubric: a vf.Rubric with [correct_answer] and weights [1.0]

    Returns:
        a configured vf.SingleTurnEnv
    """
    # TODO: build and return the SingleTurnEnv.
    return todo("build_env: assemble SingleTurnEnv(dataset=..., eval_dataset=..., system_prompt=..., rubric=...)")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass

    env = build_env()
    ds = env.get_dataset()
    print(f"Dataset columns: {ds.column_names}")
    print(f"Rows: {len(ds)}")
    print("\nFirst prompt as the model will see it:")
    for msg in ds[0]["prompt"]:
        print(f"  [{msg['role']}] {msg['content'][:110]}")

    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()
    if not (key and model):
        print("\n(No API key/model in .env -- skipping the live eval.)")
        print("Set them up per Unit 00 to see this run for real.")
        return

    # verifiers does NOT take a raw `openai.OpenAI` object. It wants its own
    # ClientConfig, which describes HOW to reach the endpoint. Note `api_key_var`
    # is the NAME of the environment variable, not the key itself -- so your
    # secret never has to be passed around or serialised into a training config.
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="OPENAI_API_KEY",
        api_base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )

    print(f"\nEvaluating {model} on {len(ds)} problems, 2 rollouts each...")
    results = env.evaluate_sync(client=client, model=model, rollouts_per_example=2)

    # `results` is a dict; the per-rollout records live under "outputs".
    outputs = results["outputs"]
    rewards = [o["reward"] for o in outputs]
    print(f"\n  mean reward: {sum(rewards) / len(rewards):.3f}  (n={len(rewards)})")
    print(f"  per-rollout: {[round(r, 2) for r in rewards]}")
    print(f"  metrics:     {outputs[0]['metrics']}")

    # Rollouts can abort (dropped connection, timeout), leaving an empty
    # completion. Skip those rather than indexing into nothing, and say how many.
    ok = [o for o in outputs if o["completion"]]
    failed = len(outputs) - len(ok)
    if failed:
        print(f"\n  ({failed} of {len(outputs)} rollouts failed before answering: "
              f"{outputs[0].get('stop_condition')})")

    print("\nA couple of actual completions:")
    for o in ok[:2]:
        text = o["completion"][-1]["content"]
        print(f"\n  --- reward={o['reward']} ---")
        print("  " + text.strip().replace("\n", "\n  ")[:400])

    print(
        "\nWhat you just did: defined a task, scored a real model on it, and got a\n"
        "number. That number is an eval metric. It is also, unchanged, the reward\n"
        "signal you would train on. Same object, two readings."
    )


if __name__ == "__main__":
    run_main(main)
