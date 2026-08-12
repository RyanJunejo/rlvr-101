"""
Theory demo — does reward shaping change the optimal policy?

Not an exercise. Nothing to fill in; run it and read the table. It's the
evidence behind THEORY.md section 3, and it takes about a second.

THE SETUP. A chain of 7 states. You start at 0, the goal is state 6, and each
turn you move left, right, or stay put. Reaching the goal pays 1.0 and ends the
episode. Everything else pays nothing -- a sparse reward, exactly the Unit 03
problem.

We solve this exactly (value iteration, no learning, no sampling) under three
reward schemes, and compare the optimal policies:

  sparse      only the +1 at the goal
  naive       +1 at the goal, plus a bonus for BEING close: beta * phi(s')
  potential   +1 at the goal, plus gamma * phi(s') - phi(s)

`phi(s)` is the same "how close am I" measure in both shaped versions. The only
difference is that `naive` pays you for your position, and `potential` pays you
for the CHANGE in your position.

That difference decides whether the shaping is safe.

    uv run python modules/03-multiturn-and-tools/theory_shaping_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402

N = 6  # states 0..N; N is the goal
GAMMA = 0.95  # discount: a reward one step later is worth 0.95x
BETA = 0.3  # how big the shaping bonus is
ACTS = {"left": -1, "right": +1, "stay": 0}


def step(s: int, a: str) -> int:
    return min(max(s + ACTS[a], 0), N)


def phi(s: int) -> float:
    """Potential: how close to the goal, as a number from 0 to 1."""
    return s / N


def solve(shaping) -> tuple[dict, np.ndarray]:
    """Exact optimal policy by value iteration.

    V[s] is the best total future reward obtainable from state s. Iterate
    V[s] <- max over actions of (immediate reward + GAMMA * V[next state])
    until it stops changing, then read off which action achieves the max.

    No sampling and no learning -- this is the ground truth the training in
    Unit 01 would converge toward.
    """
    V = np.zeros(N + 1)
    for _ in range(5000):
        Vn = V.copy()
        for s in range(N + 1):
            if s == N:
                Vn[s] = 0.0  # terminal: the episode is over, nothing follows
                continue
            Vn[s] = max(
                (1.0 if step(s, a) == N else 0.0) + shaping(s, step(s, a)) + GAMMA * V[step(s, a)]
                for a in ACTS
            )
        if np.max(np.abs(Vn - V)) < 1e-12:
            break
        V = Vn
    policy = {
        s: max(
            ACTS,
            key=lambda a: (1.0 if step(s, a) == N else 0.0)
            + shaping(s, step(s, a))
            + GAMMA * V[step(s, a)],
        )
        for s in range(N)
    }
    return policy, V


def main() -> None:
    sparse, _ = solve(lambda s, s2: 0.0)
    naive, _ = solve(lambda s, s2: BETA * phi(s2))
    potential, _ = solve(lambda s, s2: GAMMA * phi(s2) - phi(s))

    print(f"{BOLD}Optimal action in each state (0 = start, {N} = goal){RESET}\n")
    print(f"  {'scheme':<12}" + "".join(f"{s:>8}" for s in range(N)))
    print("  " + "-" * (12 + 8 * N))
    for name, pol in [("sparse", sparse), ("naive", naive), ("potential", potential)]:
        print(f"  {name:<12}" + "".join(f"{pol[s]:>8}" for s in range(N)))

    print(f"\n{BOLD}Did the shaping change what's optimal?{RESET}\n")
    print(f"  naive     matches the sparse optimum: {naive == sparse}")
    print(f"  potential matches the sparse optimum: {potential == sparse}")

    hover = BETA * phi(N - 1) / (1 - GAMMA)
    finish = 1.0 + BETA * phi(N)
    print(
        f"\n{BOLD}Why naive breaks{RESET}\n"
        f"\n"
        f"  Look at state {N - 1}, one step from the goal. Under naive shaping the\n"
        f"  optimal action is {naive[N - 1]!r}.\n"
        f"\n"
        f"    sit at state {N - 1} forever:  {BETA} x {phi(N - 1):.3f} every turn,\n"
        f"                            discounted forever = {hover:.2f}\n"
        f"    walk into the goal:     {finish:.2f}, and the episode ends\n"
        f"\n"
        f"  Hovering pays {hover / finish:.1f}x more than finishing. The agent has found a\n"
        f"  way to farm the bonus, and it is behaving optimally -- for the reward\n"
        f"  function you actually wrote."
    )

    print(
        f"\n{BOLD}Why potential-based shaping is safe{RESET}\n"
        f"\n"
        f"  It pays the CHANGE in potential, not the potential. Any loop returns to\n"
        f"  where it started, so the changes cancel and the loop is worth zero. There\n"
        f"  is nothing to farm.\n"
        f"\n"
        f"  Ng, Harada & Russell (1999) proved this in general: shaping of the form\n"
        f"  F(s, s') = gamma * phi(s') - phi(s) leaves the optimal policy of ANY\n"
        f"  MDP unchanged, for any function phi. The table above is one instance.\n"
        f"\n"
        f"{DIM}  Unit 03's lab capped partial credit at 0.5 so it could never outrank the\n"
        f"  1.0 for winning. That's the same instinct, enforced by hand. This is the\n"
        f"  version with a proof -- see THEORY.md section 3 for what it would look\n"
        f"  like for the guessing game, and why the crude cap is often the right\n"
        f"  engineering call anyway.{RESET}"
    )


if __name__ == "__main__":
    main()
