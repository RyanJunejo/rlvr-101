"""
Lab 1 — a multi-turn environment.

THE GAME. The environment picks a secret number from 1 to 100. The model guesses.
After each guess the environment says "too low" or "too high". The model gets 7
guesses.

Why 7? Because binary search on 100 numbers needs at most 7 guesses
(2^7 = 128 > 100). So a perfect player always wins, and a careless one doesn't.
That gap is what makes this worth scoring.

WHAT'S NEW COMPARED TO UNIT 02. In Unit 02 the model answered once and you scored
the text. Here the environment talks back, and the model's next move depends on
what you said. You need somewhere to remember the secret number and the guesses
so far -- that's what `state` is for.

THE THREE PIECES (see section 2 of the lecture notes):
    setup_state    runs ONCE at the start        -> set up the game
    env_response   runs after each model message -> reply to the model
    @vf.stop       runs after each model message -> is the game over?

You do not need an API key for this lab. There's a fake model at the bottom of
the file that plays the game, so you can watch your environment work immediately.

Fill in the four TODOs, then run me:

    uv run python modules/03-multiturn-and-tools/exercise_1_guessing_game.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

# The model is asked to end each message with a line like "Guess: 42".
GUESS_RE = re.compile(r"^\s*guess\s*:\s*(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "We're playing a guessing game. I'm thinking of a whole number from 1 to 100.\n"
    "Each turn, reason briefly, then end your message with exactly:\n"
    "Guess: <number>\n"
    "I'll reply 'Too low' or 'Too high'. You have 7 guesses."
)

MAX_TURNS = 7


class GuessingGameEnv(vf.MultiTurnEnv):
    """The environment. You implement three of its methods."""

    def __init__(self, **kwargs):
        super().__init__(max_turns=MAX_TURNS, **kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        """Set up one game. Runs ONCE, before the model says anything.

        `state` already contains "answer" -- the secret number, as a STRING,
        taken from your dataset's `answer` column.

        Add three things to it and return it:
            state["secret"]  -> the answer as an int
            state["guesses"] -> an empty list (we'll append to it each turn)
            state["solved"]  -> False

        Why store `guesses`: the rubric scores the game AFTER it's over, so
        anything you want to score on has to be recorded here as you go. This is
        the main structural difference from Unit 02.

        Args:
            state: the rollout's state dict. Already has "answer".
        Returns:
            the same dict, with your keys added.
        """
        # TODO: initialise secret / guesses / solved, then return state.
        return todo("setup_state: add secret (int), guesses (list), solved (False)")

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        """Reply to the model's latest message. Runs after every model turn.

        Steps:
          1. Get the model's text: `messages[-1]["content"]`
          2. Find its guess with GUESS_RE. Use `.findall(text)` and take the LAST
             match (same reasoning as Unit 02 -- models restate the format while
             thinking, and the final one is the real answer).
          3. If there's no guess at all, return a message reminding it of the
             format. Don't record anything. Don't crash.
          4. Otherwise: convert to int, append it to state["guesses"], and:
               - equal to the secret  -> set state["solved"] = True, reply "Correct!"
               - below the secret     -> reply "Too low."
               - above the secret     -> reply "Too high."

        Return a LIST of messages, built with `vf.UserMessage(content=...)`.
        (From the model's perspective the environment is the user.)

        Args:
            messages: the conversation; the last entry is what the model just said.
            state: your game state, as set up in setup_state.
        Returns:
            list of messages to send back to the model.
        """
        # TODO: parse the guess, update state, return the right feedback.
        return todo("env_response: parse the guess, update state, reply too low/high/correct")

    @vf.stop
    async def game_solved(self, state: vf.State) -> bool:
        """Stop the rollout early when the model has won.

        Return True if the game is solved, False otherwise.

        You do NOT need to check the turn limit here -- `MultiTurnEnv` already
        stops at `max_turns` on its own. This method is only for YOUR game's
        ending condition.

        Use `state.get("solved", False)` rather than `state["solved"]`, so that a
        rollout which somehow ends before setup finished doesn't raise.
        """
        # TODO: return whether the game has been solved.
        return todo("game_solved: return True once state['solved'] is set")


def solved_reward(state, **kwargs) -> float:
    """1.0 if the model won the game, else 0.0.

    Note the argument: this reward function asks for `state`, not `completion`.
    In Unit 02 you scored the text; here the interesting information is in the
    state you recorded during play. Both are available -- ask for what you need.

    Args:
        state: the finished rollout's state.
    Returns:
        1.0 or 0.0
    """
    # TODO: return 1.0 when the game was solved.
    return todo("solved_reward: 1.0 if state['solved'] else 0.0")


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------


def build_dataset(secrets: list[int]) -> Dataset:
    """One row per game. The 'question' is the same every time; only the secret differs."""
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


# -- a fake model, so you can test without an API key -----------------------


def binary_search_player(feedback: str, lo: int, hi: int) -> tuple[int, int, int]:
    """A perfect player. Narrows the range using the feedback."""
    if "low" in feedback.lower():
        lo = lo + 1 if lo == hi else lo
    guess = (lo + hi) // 2
    return guess, lo, hi


async def play_offline(env: GuessingGameEnv, secret: int, strategy: str = "binary", verbose: bool = True):
    """Drive the environment with a scripted player instead of a real model.

    This is exactly what a real rollout does -- setup_state, then alternate
    between the player speaking and env_response replying -- just with a Python
    function standing in for the model. Deterministic and free.
    """
    state = vf.State({"answer": str(secret), "prompt": [], "trajectory": []})
    state = await env.setup_state(state)

    lo, hi = 1, 100
    feedback = ""
    turns = 0
    for _ in range(MAX_TURNS):
        if strategy == "binary":
            guess = (lo + hi) // 2
        else:  # "linear" -- a bad player that counts upward
            guess = turns + 1

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
        print(f"    -> solved={state['solved']} in {turns} turns, "
              f"reward={solved_reward(state)}\n")

    print(f"{BOLD}A bad player (guesses 1, 2, 3, ...){RESET}")
    print("  secret = 42")
    state, turns = asyncio.run(play_offline(env, 42, strategy="linear"))
    print(f"    -> solved={state['solved']} after {turns} turns, "
          f"reward={solved_reward(state)}")

    print(
        f"\n{DIM}Notice what just happened. The good player won every game and the bad\n"
        f"player lost. That's a working environment -- your scoring can tell them\n"
        f"apart.\n"
        f"\n"
        f"Now ask yourself the Unit 01 question: if you trained a model that is\n"
        f"currently as bad as that second player, and every rollout in a group\n"
        f"scores 0.0... what is the advantage for each one? What does the model\n"
        f"learn from that group?\n"
        f"\n"
        f"That's lab 2.{RESET}"
    )


if __name__ == "__main__":
    run_main(main)
