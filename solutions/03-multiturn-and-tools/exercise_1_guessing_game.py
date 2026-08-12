"""
ANSWER KEY -- Unit 03, Lab 1 — a multi-turn environment.

THE GAME. The environment picks a secret number from 1 to 100. The model
guesses; after each guess the environment says "Too low." or "Too high." Seven
guesses -- because binary search on 100 numbers needs at most 7 (2^7 = 128), so
a perfect player always wins and a careless one doesn't. That gap is what makes
the game worth scoring.

WHAT'S NEW. In Unit 02 the model answered once and you scored the reply. Here
the environment TALKS BACK, and the model's next move depends on what it said.
In v1 that conversation is driven by an `Env` -- its `run()` method plays the
environment's side, turn by turn:

    async with agents.player.interaction(task) as interaction:
        segment = await interaction.turn()          # the model's opening move
        while ...:
            segment = await interaction.turn(feedback)   # reply, get next move

The Env for this game is written for you at the bottom -- it needs a live model
on the other end, so it can't run offline. What CAN run offline, and what you
write, is everything the Env is built from:

  - the game logic (parse a guess, respond to it), as plain functions
  - the scoring, which REPLAYS THE TRANSCRIPT rather than trusting any state

That last idea is the design lesson of this lab. The trace already contains
every guess the model made. A reward that re-derives "did it win" from the
transcript plus `task.answer` cannot disagree with what actually happened, and
you can test it on hand-built traces without a model anywhere in sight.

Fill in the three TODOs, then run me:

    uv run python modules/03-multiturn-and-tools/exercise_1_guessing_game.py

TERMS USED IN THIS FILE

  Env:    the code that plays the environment's side of a conversation
  trace:  the record of one attempt; `trace.assistant_messages` is everything
          the model said
  reward: scored after the rollout ends, and weighted into the total
  metric: recorded alongside rewards but NEVER weighted -- instrumentation
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.trace import TraceTask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

GUESS_RE = re.compile(r"^\s*guess\s*:\s*(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "We're playing a guessing game. I'm thinking of a whole number from 1 to 100.\n"
    "Each turn, reason briefly, then end your message with exactly:\n"
    "Guess: <number>\n"
    "I'll reply 'Too low.' or 'Too high.'. You have 7 guesses."
)

MAX_TURNS = 7


def parse_guess(text: str) -> int | None:
    """Extract the model's guess from one message, or None if it didn't make one.

    Find every GUESS_RE match and take the LAST -- the same rule as Unit 02's
    answer slot, for the same reason: models restate the format while thinking,
    and the final commitment is the real one.

    Return None (not 0, not -1) when there's no match. The caller needs to
    tell "no guess" apart from any actual number.

    Args:
        text: one model message.
    Returns:
        the guess as an int, or None.
    """
    matches = GUESS_RE.findall(text)
    return int(matches[-1]) if matches else None


def respond(guess: int, secret: int) -> str:
    """The environment's reply to one guess.

    Exactly one of three strings -- the model learns the game from these, so
    they must be consistent:

        guess < secret  ->  "Too low."
        guess > secret  ->  "Too high."
        guess == secret ->  "Correct!"

    Args:
        guess: what the model guessed.
        secret: the number it's trying to find.
    Returns:
        the feedback string.
    """
    if guess < secret:
        return "Too low."
    if guess > secret:
        return "Too high."
    return "Correct!"


class GuessData(vf.TaskData):
    answer: str  # the secret, as a string -- TaskData rides the wire as text


class GuessTask(vf.Task[GuessData, vf.State, vf.TaskConfig]):
    """Scoring that replays the transcript.

    Write two methods:

    1. `solved` -- `@vf.reward`, async:
       Parse every message in `trace.assistant_messages` with `parse_guess`,
       and return 1.0 if ANY guess equals `int(task.answer)`, else 0.0.

       Note what this does NOT do: it doesn't look for "Correct!" in the
       feedback, and it doesn't read any stored state. It re-derives the
       outcome from the model's own messages plus the ground truth. A grader
       built this way cannot drift out of sync with the game -- and if the Env
       had a bug in its feedback, this reward would still be right.

    2. `num_guesses` -- `@vf.metric`, async:
       The count of messages that contained a valid guess. A metric is
       recorded alongside rewards but has no weight -- it's instrumentation,
       the Unit 05 dashboards read it, and it costs nothing.

    Both methods ask for what they need by name, as in Unit 02.
    """

    @vf.reward
    async def solved(self, task: GuessData, trace: vf.Trace) -> float:
        """Replay the transcript against the ground truth.

        Deliberately does NOT look for "Correct!" in the feedback: that would
        make the grader trust the Env's output. If the Env had a feedback bug,
        a transcript-replaying grader still scores the truth -- when scoring
        and gameplay can disagree, one of them is a bug you won't notice.
        """
        secret = int(task.answer)
        guesses = (parse_guess(m.content or "") for m in trace.assistant_messages)
        return 1.0 if any(g == secret for g in guesses) else 0.0

    @vf.metric
    async def num_guesses(self, trace: vf.Trace) -> float:
        """Instrumentation, not reward: recorded, never weighted. A model
        averaging near 7 is scanning the range, not searching it."""
        return float(sum(
            parse_guess(m.content or "") is not None
            for m in trace.assistant_messages
        ))


# ---------------------------------------------------------------------------
# Written for you: the Env that plays the environment's side, live.
# ---------------------------------------------------------------------------


class GuessEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class GuessEnv(vf.Env[GuessEnvConfig]):
    """The conversation driver. Runs only against a live model.

    The shape to notice: `run()` is plain imperative code. Take the model's
    opening move with a bare `turn()` (the task's prompt speaks first), then
    loop -- parse, respond, send the feedback, get the next move. Your
    `parse_guess` and `respond` are the whole game; the Env is plumbing.
    """

    async def run(self, task: GuessTask, agents) -> None:
        secret = int(task.data.answer)
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()
            for _ in range(MAX_TURNS - 1):
                if segment.terminated:
                    return
                guess = parse_guess(segment.last_reply)
                if guess is None:
                    feedback = "I couldn't find a guess. End with 'Guess: <number>'."
                elif guess == secret:
                    return  # game over -- no further message; scoring replays the transcript
                else:
                    feedback = respond(guess, secret)
                segment = await interaction.turn(feedback)


class GuessTaskset(vf.Taskset[GuessTask, vf.TasksetConfig]):
    def load(self) -> Iterator[GuessTask]:
        rng = random.Random(0)
        for i in range(16):
            yield GuessTask(
                GuessData(
                    idx=i,
                    prompt="I'm thinking of a number from 1 to 100. Start guessing.",
                    system_prompt=SYSTEM_PROMPT,
                    answer=str(rng.randint(1, 100)),
                ),
                self.config.task,
            )


# The taskset AND its Env export together; the eval CLI finds the Env
# automatically because it lives in the taskset's own module.
__all__ = ["GuessTaskset", "GuessEnv"]


# ---------------------------------------------------------------------------
# Offline harness.
# ---------------------------------------------------------------------------


def make_game_trace(task: GuessTask, guesses: list[int]) -> vf.Trace:
    """Rebuild the trace a real game would leave: guess, feedback, guess, ..."""
    trace = vf.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="offline", trainable=False),
    )
    secret = int(task.data.answer)
    for guess in guesses:
        trace.nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content=f"Thinking...\nGuess: {guess}"),
                sampled=True,
            )
        )
        trace.nodes.append(
            vf.MessageNode(message=vf.UserMessage(content=respond(guess, secret)), sampled=False)
        )
    return trace


def binary_search_guesses(secret: int) -> list[int]:
    lo, hi, out = 1, 100, []
    while True:
        guess = (lo + hi) // 2
        out.append(guess)
        if guess == secret:
            return out
        lo, hi = (guess + 1, hi) if guess < secret else (lo, guess - 1)


def main() -> None:
    print(f"{BOLD}1. The game logic{RESET}\n")
    for text in ["Guess: 50", "Guess: 12\nno wait\nGuess: 42", "I give up"]:
        print(f"  parse_guess({text.replace(chr(10), ' / ')!r:38}) -> {parse_guess(text)}")
    for guess in (10, 90, 42):
        print(f"  respond({guess}, 42) -> {respond(guess, 42)!r}")

    print(f"\n{BOLD}2. A perfect game, replayed and scored{RESET}\n")
    task = GuessTask(GuessData(idx=0, prompt="p", answer="42"))
    guesses = binary_search_guesses(42)
    print(f"  binary search path to 42: {guesses} ({len(guesses)} guesses)")
    trace = make_game_trace(task, guesses)
    asyncio.run(task.score(trace))
    print(f"  rewards: { {k: v.score for k, v in trace.rewards.items()} }")
    print(f"  metrics: { {k: v for k, v in trace.metrics.items()} }")

    print(f"\n{BOLD}3. A losing game{RESET}\n")
    losing = GuessTask(GuessData(idx=0, prompt="p", answer="42"))
    ltrace = make_game_trace(losing, [1, 2, 3, 4, 5, 6, 7])
    asyncio.run(losing.score(ltrace))
    print(f"  guessed 1..7 with secret 42 -> rewards "
          f"{ {k: v.score for k, v in ltrace.rewards.items()} }")

    print(
        f"\n{DIM}  Both scored from nothing but the transcript and the answer. No stored\n"
        f"  game state, nothing the Env had to remember correctly. When scoring and\n"
        f"  gameplay can disagree, one of them is a bug you won't notice -- so make\n"
        f"  the grader re-derive.{RESET}"
    )

    print(
        f"\n{BOLD}4. Run it live{RESET}\n"
        "\n"
        "  This module exports the taskset AND its Env, so the eval CLI wires the\n"
        "  whole game up by module name:\n"
        "\n"
        "    export $(grep -vE '^#|^$' .env | xargs)\n"
        "    PYTHONPATH=modules/03-multiturn-and-tools uv run eval exercise_1_guessing_game \\\n"
        "      -m \"$MODEL\" -n 4 -r 1 --rich False \\\n"
        "      --client.base-url \"$OPENAI_BASE_URL\" --client.api-key-var OPENAI_API_KEY \\\n"
        "      -o outputs/guessing-game\n"
        "\n"
        "  (No --env.agent.harness.id this time: the Env drives the turns itself.)\n"
        "\n"
        "  Then read traces.jsonl and watch a real model play. num_guesses is in\n"
        "  each trace's metrics -- a model averaging near 7 is scanning, not\n"
        "  searching."
    )


if __name__ == "__main__":
    main()
