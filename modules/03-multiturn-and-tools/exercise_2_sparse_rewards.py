"""
Lab 2 — the sparse reward problem, measured.

This is the most important lab in the unit, and it's a direct consequence of
something you already proved in Unit 01.

RECALL THE UNIT 01 RESULT. Advantage is "your score minus the group's average."
So if every rollout in a group gets the SAME score, every advantage is zero, and
that group produces no learning at all. It doesn't matter whether the shared
score is 1.0 or 0.0 -- what matters is that there's no spread.

NOW APPLY IT HERE. Your guessing game's obvious score is "did the model win?"
That's a SPARSE reward: 1.0 or 0.0, nothing in between.

Imagine a model that's currently bad and wins maybe 1 game in 50. Ask a question
8 times:

    scores:      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    average:     0.0
    advantages:  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

Nothing. You just paid for 8 full game rollouts and learned nothing. And the
model can never bootstrap out of it, because it needs to already be good to get
any signal at all.

You're going to measure exactly how bad this is across skill levels, then fix it
with PARTIAL CREDIT (called "reward shaping"), and then measure the fix.

No API key needed -- we simulate players of known skill.

Fill in the two TODOs, then run me:

    uv run python modules/03-multiturn-and-tools/exercise_2_sparse_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET, run_main, todo  # noqa: E402

MAX_TURNS = 7


def sparse_reward(state: dict) -> float:
    """The obvious score: 1.0 for winning, 0.0 otherwise. Written for you."""
    return 1.0 if state["solved"] else 0.0


def shaped_reward(state: dict) -> float:
    """Partial credit for getting close, full credit only for winning.

    THE DESIGN, and the reasoning behind each rule:

      1. If the model solved it, return 1.0.

      2. If it never made a valid guess, return 0.0. Nothing to reward.

      3. Otherwise, give partial credit based on the CLOSEST guess it made:

             best = the smallest |guess - secret| over all its guesses
             return 0.5 * (1.0 - best / 99.0)

         The secret is in 1..100, so the worst possible miss is 99, which maps to
         0.0. A miss of 1 maps to almost 0.5.

    NOW THE IMPORTANT PART -- why the 0.5. Partial credit maxes out just under
    0.5, while winning pays 1.0. That gap is deliberate and non-negotiable.

    If partial credit could reach or exceed 1.0, you'd have built a reward the
    model can farm forever without ever finishing the game: hover one away from
    the answer, collect near-full marks, never actually win. You would have
    recreated the Unit 02 reward-hacking problem in a new costume.

    THE RULE: a shaped reward is a ladder to the goal, never a destination. It
    must always be worth strictly less than succeeding.

    Args:
        state: a finished game, with keys "solved", "guesses", "secret".
    Returns:
        1.0 for a win, else a value in [0.0, 0.5).
    """
    # TODO: implement the three rules above.
    return todo("shaped_reward: 1.0 if solved, else 0.5 * (1 - best_distance/99)")


def group_has_signal(rewards: np.ndarray) -> bool:
    """Does this group of rollouts produce any learning at all?

    A group teaches the model something only if its scores DIFFER. If every
    rollout scored the same, all the advantages are zero and the group is wasted.

    Return True if there is any spread in `rewards`, False if they're all equal.

    (Use a small tolerance rather than exact equality -- these are floats. If
    `rewards.std()` is below about 1e-9, treat it as no spread.)

    Args:
        rewards: shape (G,), the scores for one group of rollouts.
    Returns:
        True if this group produces a non-zero gradient.
    """
    # TODO: return whether there's any spread in the rewards.
    return todo("group_has_signal: True if the rewards aren't all identical")


# ---------------------------------------------------------------------------
# Written for you: a simulated player whose skill we can dial.
# ---------------------------------------------------------------------------


def simulate_game(secret: int, skill: float, rng: np.random.Generator) -> dict:
    """Play one game with a player of the given skill.

    skill = 1.0 -> always plays the perfect binary-search move (always wins)
    skill = 0.0 -> guesses uniformly at random (usually loses)
    in between  -> plays the good move with probability `skill`, else guesses

    This stands in for "a model part-way through training."
    """
    lo, hi = 1, 100
    guesses: list[int] = []
    solved = False
    for _ in range(MAX_TURNS):
        if rng.random() < skill:
            guess = (lo + hi) // 2
        else:
            guess = int(rng.integers(1, 101))
        guesses.append(guess)
        if guess == secret:
            solved = True
            break
        if guess < secret:
            lo = max(lo, guess + 1)
        else:
            hi = min(hi, guess - 1)
        if lo > hi:  # the random guessing painted us into a corner
            lo, hi = 1, 100
    return {"secret": secret, "guesses": guesses, "solved": solved}


def measure(skill: float, reward_fn, n_groups: int = 400, group_size: int = 8, seed: int = 0):
    """For a player of the given skill, what fraction of groups actually teach anything?"""
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

    print(f"{BOLD}How often does a group actually teach the model anything?{RESET}")
    print("A group is useful only if its rollouts got DIFFERENT scores.\n")
    print(f"  {'skill':>6} | {'win rate':>9} | {'useful groups':>14} | {'useful groups':>14}")
    print(f"  {'':>6} | {'':>9} | {'(sparse)':>14} | {'(shaped)':>14}")
    print("  " + "-" * 56)

    rows = []
    for s in skills:
        frac_sparse, win = measure(s, sparse_reward)
        frac_shaped, _ = measure(s, shaped_reward)
        rows.append((s, win, frac_sparse, frac_shaped))
        print(f"  {s:>6.1f} | {win * 100:>8.1f}% | {frac_sparse * 100:>13.1f}% | {frac_shaped * 100:>13.1f}%")

    weak = rows[0]
    strong = rows[-1]
    print(
        f"\n{BOLD}Read the two ends of that table.{RESET}\n"
        f"\n"
        f"  A BEGINNER model (skill 0.0) wins {weak[1] * 100:.1f}% of games. Under the sparse\n"
        f"  reward only {weak[2] * 100:.1f}% of its groups produce any gradient at all -- the rest\n"
        f"  are 8 rollouts of pure cost. Under the shaped reward, {weak[3] * 100:.1f}% are useful.\n"
        f"\n"
        f"  An EXPERT model (skill 1.0) wins {strong[1] * 100:.1f}% of games, and now the sparse\n"
        f"  reward gives signal in only {strong[2] * 100:.1f}% of groups -- for the opposite reason.\n"
        f"  Everything scores 1.0, so again there's no spread.\n"
    )
    print(
        f"{DIM}That symmetry is the lesson. Too hard and too easy are the SAME failure:\n"
        f"no spread, no advantage, no learning. Sparse rewards give you signal only\n"
        f"in the narrow band where the model succeeds sometimes -- which is exactly\n"
        f"why real training pipelines filter for questions with score variance, and\n"
        f"why 'curriculum design' mostly means keeping questions in that band.\n"
        f"\n"
        f"The shaped reward widens the band enormously. But re-read the docstring of\n"
        f"shaped_reward before you get comfortable: you have just added a second\n"
        f"thing that can be gamed, and the only reason it's safe is the 0.5 cap.{RESET}"
    )


if __name__ == "__main__":
    run_main(main)
