"""
ANSWER KEY — Unit 03, Lab 2: the sparse reward problem.

    uv run python solutions/03-multiturn-and-tools/exercise_2_sparse_rewards.py

TERMS USED IN THIS FILE

  sparse reward:  a score that's mostly 0.0 with an occasional 1.0, and nothing
                  in between
  reward shaping: giving partial credit for progress, so weak models get signal
  group:          several answers to the same question; no spread in a group means no
                  advantage and no learning (Unit 01)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

MAX_TURNS = 7


def sparse_reward(state: dict) -> float:
    return 1.0 if state["solved"] else 0.0


def shaped_reward(state: dict) -> float:
    """Partial credit, capped strictly below the win.

    THE CAP IS THE WHOLE DESIGN. Partial credit tops out just under 0.5; winning
    pays 1.0. Change that 0.5 to 1.0 and you have built a reward the model can
    farm forever: sit one away from the answer, collect ~1.0, never finish. It
    would look like it was learning -- the score would climb beautifully -- while
    the win rate went nowhere.

    That is the Unit 02 reward-hacking lesson arriving in a new costume. The only
    thing standing between you and it is this constant.

    ON THE CHOICE OF `best` (closest guess) rather than, say, the final guess or
    the average: we want the reward to be monotone in skill. A better player gets
    closer at some point, so "closest approach" rises with skill. Using the LAST
    guess would be noisier (a player can be close then jump away), and averaging
    would punish early exploration -- which is exactly the behaviour we want on a
    search task.
    """
    if state["solved"]:
        return 1.0
    if not state["guesses"]:
        return 0.0
    best = min(abs(g - state["secret"]) for g in state["guesses"])
    return 0.5 * (1.0 - best / 99.0)


def group_has_signal(rewards: np.ndarray) -> bool:
    """Any spread at all?

    Tolerance rather than `len(set(rewards)) > 1` because these are floats and
    the shaped reward produces values like 0.4949494949494949. Two rollouts that
    got equally close will differ in the last bit or two, and counting that as
    "signal" would overstate how useful the shaped reward is.
    """
    return bool(np.asarray(rewards).std() > 1e-9)


# --- harness ----------------------------------------------------------------


def simulate_game(secret: int, skill: float, rng: np.random.Generator) -> dict:
    lo, hi = 1, 100
    guesses: list[int] = []
    solved = False
    for _ in range(MAX_TURNS):
        guess = (lo + hi) // 2 if rng.random() < skill else int(rng.integers(1, 101))
        guesses.append(guess)
        if guess == secret:
            solved = True
            break
        if guess < secret:
            lo = max(lo, guess + 1)
        else:
            hi = min(hi, guess - 1)
        if lo > hi:
            lo, hi = 1, 100
    return {"secret": secret, "guesses": guesses, "solved": solved}


def measure(skill: float, reward_fn, n_groups: int = 400, group_size: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    useful = 0
    win_rate = []
    for _ in range(n_groups):
        secret = int(rng.integers(1, 101))
        games = [simulate_game(secret, skill, rng) for _ in range(group_size)]
        rewards = np.array([reward_fn(g) for g in games])
        win_rate.append(np.mean([g["solved"] for g in games]))
        if group_has_signal(rewards):
            useful += 1
    return useful / n_groups, float(np.mean(win_rate))


def main() -> None:
    skills = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"{BOLD}How often does a group actually teach the model anything?{RESET}\n")
    print(f"  {'skill':>6} | {'win rate':>9} | {'useful (sparse)':>16} | {'useful (shaped)':>16}")
    print("  " + "-" * 58)
    rows = []
    for s in skills:
        fs, win = measure(s, sparse_reward)
        fsh, _ = measure(s, shaped_reward)
        rows.append((s, win, fs, fsh))
        print(f"  {s:>6.1f} | {win * 100:>8.1f}% | {fs * 100:>15.1f}% | {fsh * 100:>15.1f}%")

    weak, strong = rows[0], rows[-1]
    print(
        f"\n{BOLD}DISCUSSION{RESET}\n"
        "\n"
        f"  THE TWO ENDS OF THE TABLE ARE THE SAME FAILURE. A beginner (skill 0.0) wins\n"
        f"  {weak[1] * 100:.1f}% of games, and over half its groups ({100 - weak[2] * 100:.1f}%) are wasted -- no\n"
        f"  rollout succeeded, so every score is 0.0. An expert (skill 1.0) wins\n"
        f"  {strong[1] * 100:.1f}% of games and wastes {100 - strong[2] * 100:.1f}% of its groups -- every score is 1.0.\n"
        f"  One is starved because success is rare; the other because success is\n"
        f"  certain. Per Unit 01, both mean zero advantage and zero gradient.\n"
        "\n"
        "  NOTE WHAT SHAPING DOES AND DOESN'T FIX. At low skill it takes you from\n"
        f"  {weak[2] * 100:.1f}% to {weak[3] * 100:.1f}% -- it completely solves the too-hard end. At skill 1.0 it\n"
        f"  is still {strong[3] * 100:.1f}%, exactly as useless as the sparse reward. Once every rollout\n"
        "  wins, nothing you compute from the transcript has any spread left. Shaping\n"
        "  buys you the bottom of the curve, not the top; the top is fixed by giving\n"
        "  the model harder questions.\n"
        "\n"
        "  So a sparse reward only works in the narrow middle band where the model\n"
        "  succeeds SOMETIMES. That band is what curriculum design is chasing, and\n"
        "  it's why serious pipelines filter out prompts with no score variance\n"
        "  before spending gradient steps on them.\n"
        "\n"
        "  WHY THE SHAPED REWARD HELPS SO MUCH AT LOW SKILL: two random players will\n"
        "  essentially never both win, but they will almost always get DIFFERENT\n"
        "  distances from the answer. Distance is a continuous quantity, so it\n"
        "  produces spread even when success never happens. That's the general\n"
        "  principle -- if you need signal from a model that can't yet do the task,\n"
        "  find something continuous to measure.\n"
        "\n"
        "  WHAT SHAPING COSTS YOU. Three things worth knowing:\n"
        "\n"
        "    1. You've added a second gameable objective. Safe here only because of\n"
        "       the 0.5 cap.\n"
        "    2. You've encoded YOUR theory of good play. Rewarding 'get close'\n"
        "       assumes closeness is progress -- true for this game, false for many.\n"
        "       On a task where the right move looks temporarily worse (a sacrifice\n"
        "       in chess, refactoring before a feature), naive shaping actively\n"
        "       teaches the wrong thing.\n"
        "    3. Shaped rewards are often ANNEALED -- weighted heavily early when the\n"
        "       model needs help getting off the ground, then decayed toward zero so\n"
        "       that late in training only real success counts.\n"
        "\n"
        "  THE HONEST SUMMARY: shaping trades a signal problem for a specification\n"
        "  problem. That's usually a good trade, because a specification problem is\n"
        "  at least one you can see."
    )


if __name__ == "__main__":
    main()
