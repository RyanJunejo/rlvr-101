"""
ANSWER KEY — Unit 03, Lab 1: the guessing game.

    uv run python solutions/03-multiturn-and-tools/exercise_1_guessing_game.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

GUESS_RE = re.compile(r"^\s*guess\s*:\s*(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "We're playing a guessing game. I'm thinking of a whole number from 1 to 100.\n"
    "Each turn, reason briefly, then end your message with exactly:\n"
    "Guess: <number>\n"
    "I'll reply 'Too low' or 'Too high'. You have 7 guesses."
)

MAX_TURNS = 7


class GuessingGameEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(max_turns=MAX_TURNS, **kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        """Runs once, before the model speaks.

        WHY `answer` IS A STRING: dataset columns are text, so the secret arrives
        as "42" and you convert it here. Doing the conversion once in setup
        rather than on every turn keeps env_response simple and means a malformed
        dataset row fails loudly at the start instead of mid-game.

        WHY WE RECORD `guesses`: the rubric runs AFTER the rollout is over and
        only sees the final state. Anything you want to score on -- how many
        turns it took, how close it got, whether it repeated itself -- has to be
        written down as it happens. This is the structural difference from Unit
        02, where the text alone was enough.
        """
        state["secret"] = int(state["answer"])
        state["guesses"] = []
        state["solved"] = False
        return state

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        """The environment's reply. Runs after every model turn.

        NOTE THE MALFORMED-INPUT BRANCH. It returns a reminder and records
        nothing. That's a deliberate choice with consequences:

          - We do NOT crash. A model that rambles without a guess is normal,
            especially early in training, and an exception would kill the rollout
            and lose all the data in it.
          - We do NOT count it as a guess. But it DOES consume a turn, because
            max_turns counts model messages. So rambling is implicitly penalised
            by costing the model a turn -- which is the right incentive and cost
            us no extra code.
        """
        text = messages[-1]["content"]
        matches = GUESS_RE.findall(text)
        if not matches:
            return [vf.UserMessage(content="I couldn't find a guess. End your message with 'Guess: <number>'.")]

        guess = int(matches[-1])
        state["guesses"].append(guess)

        if guess == state["secret"]:
            state["solved"] = True
            return [vf.UserMessage(content="Correct!")]
        if guess < state["secret"]:
            return [vf.UserMessage(content="Too low.")]
        return [vf.UserMessage(content="Too high.")]

    @vf.stop
    async def game_solved(self, state: vf.State) -> bool:
        """Our game's ending condition. The turn limit is handled by MultiTurnEnv.

        `.get()` rather than `[...]`: a rollout can end early (an API error, a
        prompt that was too long) before setup_state ever ran, and a KeyError
        inside a stop condition is an ugly way to find that out.
        """
        return state.get("solved", False)


def solved_reward(state, **kwargs) -> float:
    """Asks for `state`, not `completion`.

    In Unit 02 the text was the whole story. Here it isn't -- reconstructing
    "did it win" by re-parsing the transcript would be duplicating logic that
    env_response already did. Record it once, read it once.
    """
    return 1.0 if state.get("solved") else 0.0


# --- harness ----------------------------------------------------------------


def build_dataset(secrets: list[int]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": ["I'm thinking of a number from 1 to 100. Start guessing."] * len(secrets),
            "answer": [str(s) for s in secrets],
        }
    )


def build_env(secrets: list[int] | None = None) -> GuessingGameEnv:
    secrets = secrets or [42, 7, 99, 63, 15]
    ds = build_dataset(secrets)
    return GuessingGameEnv(
        dataset=ds,
        eval_dataset=ds,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.Rubric(funcs=[solved_reward], weights=[1.0]),
    )


async def play_offline(env: GuessingGameEnv, secret: int, strategy: str = "binary", verbose: bool = True):
    state = vf.State({"answer": str(secret), "prompt": [], "trajectory": []})
    state = await env.setup_state(state)
    lo, hi = 1, 100
    turns = 0
    for _ in range(MAX_TURNS):
        guess = (lo + hi) // 2 if strategy == "binary" else turns + 1
        turns += 1
        msgs = await env.env_response([vf.AssistantMessage(content=f"Guess: {guess}")], state)
        feedback = msgs[-1]["content"]
        if verbose:
            print(f"    turn {turns}: guessed {guess:>3}  ->  {feedback}")
        if await env.game_solved(state):
            break
        if "low" in feedback.lower():
            lo = guess + 1
        elif "high" in feedback.lower():
            hi = guess - 1
    return state, turns


def main() -> None:
    env = build_env()
    print(f"{BOLD}A perfect player (binary search){RESET}")
    for secret in (42, 7, 99):
        print(f"  secret = {secret}")
        state, turns = asyncio.run(play_offline(env, secret))
        print(f"    -> solved={state['solved']} in {turns} turns, reward={solved_reward(state)}\n")

    print(f"{BOLD}A bad player (guesses 1, 2, 3, ...){RESET}")
    print("  secret = 42")
    state, turns = asyncio.run(play_offline(env, 42, strategy="linear"))
    print(f"    -> solved={state['solved']} after {turns} turns, reward={solved_reward(state)}")

    print(
        f"\n{BOLD}DISCUSSION{RESET}\n"
        "\n"
        "  WHY BINARY SEARCH ALWAYS WINS IN 7. Each guess halves the remaining range:\n"
        "  100 -> 50 -> 25 -> 13 -> 7 -> 4 -> 2 -> 1. Seven halvings covers 128 > 100.\n"
        "  That's why max_turns is 7 and not 10: at 10 turns a mediocre player also\n"
        "  wins, the scores stop separating good play from bad, and -- per Unit 01 --\n"
        "  a group where everything scores 1.0 produces no gradient.\n"
        "\n"
        "  MAX_TURNS IS A REWARD DESIGN DECISION, not a technical detail. Set it too\n"
        "  high and everyone succeeds; too low and nobody does. Both give you groups\n"
        "  with no spread. You want the setting where the model wins SOMETIMES.\n"
        "\n"
        "  ON MUTATING STATE. env_response writes to state and the rubric reads it\n"
        "  afterwards. That's a side effect, and side effects are usually worth being\n"
        "  suspicious of -- but here it's the intended design: state is per-rollout,\n"
        "  created fresh by setup_state, and never shared between rollouts. Two games\n"
        "  running concurrently cannot interfere. What you must NOT do is keep game\n"
        "  state on `self` (the environment instance IS shared across rollouts) --\n"
        "  that's a real bug and it produces baffling results under concurrency.\n"
        "\n"
        "  WHAT THIS SCORING STILL CAN'T SEE. `solved_reward` gives 1.0 whether the\n"
        "  model won in 2 turns or 7, and 0.0 whether it got within 1 or was never\n"
        "  close. Both of those are real differences in skill that the score throws\n"
        "  away. Lab 2 is about what that costs you."
    )


if __name__ == "__main__":
    main()
