"""
ANSWER KEY — Exercise 1, your first environment.

    uv run python solutions/02-first-environment/exercise_1_first_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """Two columns is genuinely all it takes.

    `question` and `answer` are the defaults; if your source data calls them
    something else, pass `question_key=` / `answer_key=` rather than renaming
    the dataset. Do NOT create a `prompt` column yourself -- verifiers builds it
    (wrapping each question in chat messages and prepending the system prompt),
    and if it finds one already there it assumes you know better and leaves it
    alone, silently ignoring your system_prompt and few_shot settings.
    """
    return Dataset.from_dict(
        {
            "question": [q for q, _ in PROBLEMS],
            "answer": [a for _, a in PROBLEMS],
        }
    )


def extract_final_answer(text: str) -> str:
    """Bottom-up scan for the answer slot.

    WHY BOTTOM-UP: models restate the output format while reasoning ("...the
    prompt wants Answer: <number>, so let me compute...") and they correct
    themselves. The last commitment is the real one. Taking the FIRST match
    fails legitimately-correct models, which is a false negative -- the worst
    kind of grading bug, because it silently teaches the model that being right
    doesn't pay.

    WHY "" AND NOT None: it keeps every downstream comparison total. With a
    sentinel string, `extract(...) == answer` is always a valid expression. With
    None you need a null check in every single reward function that touches
    this, and you will forget one.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.lower().startswith("answer:"):
            return stripped.split(":", 1)[1].strip().rstrip(".").strip()
    return ""


def correct_answer(completion, answer, **kwargs) -> float:
    """Exact match on the extracted answer.

    THE ARGUMENT INJECTION: verifiers inspects this signature and passes
    `completion` and `answer` by name. Available names include prompt,
    completion, answer, info, task, parser, and state. Always take **kwargs --
    without it you'll break the moment you use a rubric that supplies more.

    THE COMPLETION SHAPE: it's a list of chat messages, not a string. The
    model's text is reply_text(completion). Everyone gets this wrong once.
    """
    text = reply_text(completion)
    return 1.0 if extract_final_answer(text) == str(answer).strip() else 0.0


def build_env() -> vf.SingleTurnEnv:
    """Assemble it.

    We pass the same rows as both `dataset` and `eval_dataset` because this set
    is tiny and it's a teaching example. In real work these must be disjoint --
    otherwise your eval number is just a memorization check and will look great
    right up until it means nothing.
    """
    ds = build_dataset()
    return vf.SingleTurnEnv(
        dataset=ds,
        eval_dataset=ds,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.Rubric(funcs=[correct_answer], weights=[1.0]),
    )


# --- harness ----------------------------------------------------------------


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
        "\nDISCUSSION\n"
        "  WHY rollouts_per_example=2 AND NOT 1. Sampling is stochastic, so one rollout\n"
        "  per problem gives you a number with enormous variance -- and on 6 problems\n"
        "  you'd be reporting a mean over 6 Bernoulli draws. Unit 04 does this\n"
        "  properly. For now just internalize that a single eval number without a\n"
        "  sample size next to it is not information.\n"
        "\n"
        "  It also previews GRPO: multiple rollouts of the SAME prompt is exactly the\n"
        "  group structure from Unit 01. Here you're averaging them to reduce eval\n"
        "  noise; in training you'd subtract that same mean to get advantages. Same\n"
        "  samples, two different uses.\n"
        "\n"
        "  WHAT TO DO NEXT: print the actual completions and read them. Look at what\n"
        "  the failures look like. Almost every time I've done this on a new task I've\n"
        "  found that some 'wrong' answers were actually grading bugs -- and a grading\n"
        "  bug that produces false negatives is training signal pointing the wrong way."
    )


if __name__ == "__main__":
    main()
