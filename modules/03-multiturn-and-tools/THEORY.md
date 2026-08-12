# Unit 03 — Theory

> **Read this after the labs.** The lecture gave you a rule — cap partial credit
> below the win — and called it a practical instinct. It turns out there's a
> theorem underneath, and the theorem tells you exactly when the instinct is
> right and when it isn't.
>
> Run `theory_shaping_demo.py` alongside section 3; it computes the claims.

---

## 1. The setting Unit 01 postponed

Unit 01 said LLM training on single questions is a bandit problem: you act, you
get a score, and the next question is unrelated. Multi-turn breaks that, so we
finally need the general object.

A **Markov Decision Process** is five things:

| piece | says | in the guessing game |
|---|---|---|
| `S` | the set of states | what's known: the range still possible |
| `A` | the actions | the guesses |
| `P(s' \| s, a)` | where you land | deterministic: the range narrows |
| `R(s, a)` | the reward | 1.0 on the winning guess, else 0 |
| `γ` | discount, in `[0, 1]` | how much a later reward is worth now |

"Markov" means `P` depends only on the current state, not the history. It's an
assumption about how you defined `S`: if your state doesn't carry enough
information, the property fails and everything below gets shakier. In the
guessing game, "the remaining range" is a sufficient state; "the last guess"
alone would not be.

The agent's goal is the **return** — total discounted future reward:

```
G_t = R_t + γ·R_{t+1} + γ²·R_{t+2} + ...
```

γ < 1 does two jobs. It keeps the sum finite when episodes can run forever, and
it expresses impatience: a win now beats the same win later.

### Where the difficulty enters

Two things are true in an MDP that weren't true in a bandit, and each costs you
something.

**Credit assignment.** You played six turns and lost. The reward arrives once, at
the end. Which turn was the mistake? Nothing in the signal says. With `T` turns
and one scalar at the end, you're solving for `T` contributions from one
equation.

**Exploration.** In a bandit, every pull returns information. Here, a wrong
sequence returns *the same zero* as every other wrong sequence — so a policy that
can't yet succeed receives no gradient distinguishing near-misses from disasters.

The second one is what Unit 03's lab measured.

## 2. Sparse reward, in Unit 01's terms

Take a policy with success probability `p` on some question, and sample a group
of `G` under a 0/1 reward. The group is useless when every member scores the
same, which happens with probability

```
P(no spread) = pᴳ + (1 − p)ᴳ
```

Both terms matter, and they're the two ends of the lab's table:

| p | G | P(no spread) | why |
|---|---|---|---|
| 0.01 | 8 | 0.923 | almost nothing succeeds |
| 0.5 | 8 | 0.008 | the good regime |
| 0.99 | 8 | 0.923 | almost everything succeeds |

Now connect it to Unit 01 §3. A group with no spread has all advantages exactly
zero, so it contributes *nothing* — not a small gradient, no gradient. The
expected useful signal per group scales with `1 − pᴳ − (1−p)ᴳ`, which is
symmetric around `p = 0.5` and collapses at both ends.

That symmetry is why "too easy" and "too hard" are the same failure. They aren't
analogous; they're the same term in the same expression.

**Increasing `G` helps at the bottom and not at the top.** At small `p`,
`P(no spread) ≈ (1−p)ᴳ` falls with more samples — you can buy your way into
signal. At `p → 1`, the `pᴳ` term dominates and grows toward 1 no matter how many
you draw. This is exactly what Unit 04's `pass@k` measures, and why it works as a
feasibility check in only one direction.

## 3. Reward shaping, and the theorem

The lab's fix was partial credit. Formally you replace `R` with `R + F`, where
`F` is a shaping term. The question the lab left open: **does adding `F` change
what the optimal policy is?**

Usually yes, and catastrophically.

### The failure, computed

`theory_shaping_demo.py` sets up a 7-state chain: start at 0, goal at 6, +1 for
reaching it, and a shaping bonus for being close. Solved exactly by value
iteration — no sampling, no learning, just the ground truth.

```
  scheme             0       1       2       3       4       5
  ------------------------------------------------------------
  sparse         right   right   right   right   right   right
  naive          right   right   right   right   right    stay
  potential      right   right   right   right   right   right
```

Under **naive** shaping — a bonus for *being* near the goal, `F = β·Φ(s')` — the
optimal action one step from the goal becomes **stay**. The arithmetic:

```
sit at state 5 forever:  0.3 × 0.833 per turn, discounted = 5.00
walk into the goal:      1.30, and the episode ends
```

Hovering pays 3.8× more than finishing. The agent isn't malfunctioning; it is
behaving optimally for the reward function that was actually written. This is
"hover close forever" — the failure the lecture warned about — made exact.

### The fix, and why it works

Ng, Harada and Russell (1999) proved that shaping of one particular form is
always safe:

```
F(s, s') = γ·Φ(s') − Φ(s)
```

for any function `Φ` over states. **Potential-based shaping leaves the optimal
policy of any MDP unchanged.** The demo's third row confirms it on this MDP; the
theorem says it holds for all of them.

The intuition is a telescoping sum. Along a trajectory `s₀ → s₁ → ... → s_T`,
the shaping contributions collapse:

```
Σ γᵗ·F(sₜ, sₜ₊₁)  =  γᵀ·Φ(s_T) − Φ(s₀)
```

Every intermediate term cancels against its neighbour. So the total shaping
depends only on where you started and where you finished — and any **loop**
returns to its origin and is therefore worth exactly zero.

That's the whole result. Naive shaping pays you for *position*, so a loop
collects forever. Potential-based shaping pays you for *change in position*, so a
loop nets nothing. There is no farming strategy because there is nothing to farm.

A useful way to hold it: potential-based shaping can change *how fast* a learner
finds the optimum, never *what the optimum is*.

### What this means for your lab

The lab capped partial credit at 0.5 so it could never outrank the 1.0 for
winning. That is not potential-based shaping — it's a cruder thing that solves
the same problem by making the exploit unprofitable rather than impossible.

A properly potential-based version for the guessing game would use something like
`Φ(s) = −log₂(remaining range)` — the number of bits still unknown — and reward
the *reduction* in that quantity each turn. Halving the range would pay a
constant, and a guess that learns nothing would pay zero.

Is the crude version wrong? No, and it's usually what you should write:

- The cap is one line and obviously correct on inspection.
- Potential-based shaping requires a `Φ` over states, which needs a state
  representation you may not have.
- The guarantee is about the *optimal* policy. Real training stops early, and
  the theorem says nothing about the path taken to get there.

The reason to know the theorem isn't to always use it. It's to recognize which
kind of shaping you've written — and if it isn't potential-based, to know that
you have changed the objective and should go looking for the loop.

## 4. The credit assignment problem, unsolved

Sparse reward has a principled fix. Credit assignment mostly doesn't.

The options, honestly:

**Discounting.** `γ < 1` attributes more credit to recent actions. Cheap, crude,
and a bad model of most tasks — the losing move in chess is often move 12.

**Learned value functions.** Estimate `V(s)` and use the temporal difference
`R + γV(s') − V(s)` as a per-step signal. This is what actor-critic methods do
and it works, at the cost of the second model GRPO was designed to avoid.

**Uniform assignment.** Give every token in the completion the same advantage.
This is what nearly all LLM RL actually does. It is plainly wrong — the tokens in
a 2000-token chain of thought did not contribute equally — and it works better
than it has any right to.

Why it works is not well understood. The usual hand-wave is that averaged over
many samples the useful tokens correlate with success often enough for the signal
to survive the noise. That's an observation, not an explanation.

If you want a live research question from this course: this is one.

## Sources

- **Ng, Harada & Russell (1999)**, *Policy invariance under reward
  transformations* — the potential-based shaping theorem. Short and readable.
- **Sutton & Barto**, Chapters 3–6 — MDPs, returns, value functions. The
  machinery this unit borrows and Unit 01 skipped.
- **Randløv & Alstrøm (1998)** — the bicycle agent that learned to cycle in
  circles collecting shaping reward instead of reaching the goal. The canonical
  cautionary tale, and the same failure the demo reproduces.
