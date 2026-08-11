"""
Module 00 exit check.

Proves the whole path works end-to-end before we introduce any concepts:
interpreter -> install -> credentials -> a real API call -> a real score.

There is nothing to implement here. Just run it:

    uv run python modules/00-setup/check_setup.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.grading import BOLD, DIM, GREEN, RESET, YELLOW, Grader  # noqa: E402


def main() -> int:
    g = Grader("Module 00 — setup check")

    # 1. Interpreter -----------------------------------------------------------
    major, minor = sys.version_info[:2]
    g.check(
        "Python is 3.12.x",
        (major, minor) == (3, 12),
        f"found {major}.{minor}. prime-rl requires 3.12, and we pin it everywhere so "
        f"the interpreter you use locally matches the one on the GPU box in Module 05. "
        f"Fix: `uv venv --python 3.12` from the repo root.",
    )

    # 2. The library -----------------------------------------------------------
    try:
        import verifiers as vf

        version = getattr(vf, "__version__", "unknown")
        g.check("`verifiers` imports", True)
        print(f"         {DIM}version {version}{RESET}")
        # v1 ships inside the same package -- we use both (Module 02 vs Module 06)
        has_v1 = (Path(vf.__file__).parent / "v1").is_dir()
        g.check(
            "`verifiers.v1` is present (needed in Module 06)",
            has_v1,
            "expected a v1/ submodule inside the installed package",
        )
    except ImportError as e:
        g.check("`verifiers` imports", False, f"{e}. Run `uv sync` from the repo root.")
        return g.summary()

    # 3. Credentials -----------------------------------------------------------
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass

    key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("MODEL", "").strip()

    have_key = g.check(
        "OPENAI_API_KEY is set",
        bool(key),
        "copy .env.example to .env and fill it in",
    )
    g.check("MODEL is set", bool(model), "set MODEL in .env, e.g. MODEL=gpt-4.1-mini")
    if key:
        print(f"         {DIM}endpoint: {base_url}{RESET}")
        print(f"         {DIM}model:    {model or '(unset)'}{RESET}")

    if not (have_key and model):
        print(f"\n{YELLOW}Stopping before the live call — fill in .env first.{RESET}")
        return g.summary()

    # 4. A real API call -------------------------------------------------------
    # Deliberately the dumbest possible verifiable task: we ask for an answer we
    # can check with `==`. This is a preview of the entire Module 02 idea --
    # a prompt, a reference answer, and a function returning a float.
    print(f"\n{BOLD}Making one real API call...{RESET}")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 17 * 23? Reply with only the number, nothing else.",
                }
            ],
            max_completion_tokens=4096,
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"         {DIM}model said: {text!r}{RESET}")

        # the world's smallest rubric
        reward = 1.0 if "391" in text else 0.0
        g.check(
            "model returned the correct answer (17*23 = 391)",
            reward == 1.0,
            f"got {text!r}. Not fatal for setup — your endpoint works either way — "
            f"but a model that can't do this will make Module 02 confusing. "
            f"Consider a stronger MODEL.",
        )
        print(f"\n         {GREEN}reward = {reward}{RESET}  <- you just ran a verifier")

    except Exception as e:
        g.check(
            "live API call succeeded",
            False,
            f"{type(e).__name__}: {e}\n"
            f"         Check OPENAI_BASE_URL ({base_url}) and that your key is valid "
            f"for that endpoint.",
        )

    print(
        f"\n{DIM}That last bit — prompt in, string out, float back — is the entire\n"
        f"shape of everything in this workbook. Module 02 just makes it composable.{RESET}"
    )
    return g.summary()


if __name__ == "__main__":
    sys.exit(main())
