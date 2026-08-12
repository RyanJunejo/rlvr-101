# Unit 03 — Conversations and tools

### Lecture notes

> **Time:** 3–4 hours · **Prerequisites:** Units 01 and 02 · **Needs:** nothing
> for the labs; an API key for the live runs
>
> **By the end of this unit you will be able to:**
> 1. Build a multi-turn environment: an `Env` whose `run()` plays the
>    environment's side of a conversation.
> 2. Write scoring that replays the transcript instead of trusting stored
>    state, and say why that discipline matters.
> 3. Diagnose the **sparse reward problem** — and connect it directly to the
>    zero-advantage result from Unit 01.
> 4. Design a shaped reward that rescues learning, and identify the new way it
>    can be gamed.
> 5. Expose tools through a `Toolset`, and reason about what tool use does to
>    your scoring.
>
> **Deliverables:** 3 labs, autograder green, problem set in `NOTES.md`.

---

## 1. What changes when there's more than one turn

Everything so far has been one-shot: question in, answer out, score it, done.
That's why Unit 01 could get away with saying this is basically a bandit problem
— your answer never affected what you saw next.

Now it does.

Consider a guessing game. The model guesses a number; you say "too high"; it
guesses again. Its second guess *depends on* the feedback from the first, which
depended on what it guessed. There's a real chain of cause and effect now.

This is the setting most reinforcement learning courses actually teach, and it's
harder for two reasons:

- **Credit assignment.** The model played 6 turns and lost. Which turn was the
  mistake? A single score at the end doesn't say.
- **Sparse reward.** If the only score is "did you eventually win," then a model
  that can't yet win gets 0.0 every single time — and you'll see in section 4
  why that's fatal.

The good news is you don't need new machinery. The training algorithm from Unit
01 doesn't change at all. What changes is your environment, and your scoring.

## 2. The Env: playing the other side

In v1, the conversation is driven by an `Env`. Its `run()` method is plain
imperative code — take the model's move, reply, repeat:

```python
class GuessEnv(vf.Env[GuessEnvConfig]):
    async def run(self, task, agents) -> None:
        secret = int(task.data.answer)
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()        # the model's opening move
            for _ in range(MAX_TURNS - 1):
                guess = parse_guess(segment.last_reply)
                if guess == secret:
                    return                            # game over
                segment = await interaction.turn(respond(guess, secret))
```

Two conventions to notice:

- **A prompted task speaks first.** The task's `prompt` opens the exchange, so
  the first call is a *bare* `turn()` that collects the model's opening reply.
  Every later `turn(feedback)` sends your message and returns the model's next
  one.
- **Ending the exchange is just returning.** No terminal-state machinery — when
  `run()` returns, the rollout is over and scoring begins.

An `Env` needs a live model on the other end, so it can't run offline. That's
fine, because of the next idea.

## 3. Score the transcript, not your memory of the game

The lab's reward doesn't ask the Env whether the model won. It **replays the
transcript**: parse every guess out of `trace.assistant_messages`, compare
against `task.answer`, done.

```python
@vf.reward
async def solved(self, task, trace) -> float:
    secret = int(task.answer)
    guesses = (parse_guess(m.content or "") for m in trace.assistant_messages)
    return 1.0 if any(g == secret for g in guesses) else 0.0
```

This is a discipline worth adopting generally. The grader depends only on the
model's own messages plus the ground truth — so it cannot drift out of sync
with the gameplay. If the Env had a feedback bug, this reward would still score
the truth. When scoring and gameplay *can* disagree, one of them is a bug you
won't notice, because nothing ever compares them.

It's also what makes the labs testable offline: a hand-built trace with
alternating guess/feedback messages scores identically to a real game.

The library has a place for state a rollout does need —
`self.state` on an `Env` or `Toolset` — and one rule about it: mutable rollout
data goes on `self.state`, never on `self`. Instances are shared across
concurrent rollouts; state on `self` is how two games corrupt each other.

## 4. The sparse reward problem

Everything in this section follows from a result you already proved in Unit
01, and it decides whether multi-turn training works at all.

Suppose your only score is "did the model win the game." Now imagine a model
that's currently bad at the game — it wins maybe 1 time in 50.

Ask a question 8 times. What do you get?

```
group of 8 rollouts:  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
mean = 0.0
advantages:           0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```

Every advantage is zero. That group teaches the model nothing.

You proved this in Unit 01: advantage is score-minus-group-average, so a group
where every score is identical produces no gradient at all. It doesn't matter
that the scores are 0.0 rather than 1.0 — what matters is that they're *the
same*.

So with a sparse reward and a weak model, almost every group is wasted. You burn
generation compute to produce zero learning signal, and the model can never
bootstrap, because it needs to already be decent at the task to get any signal
at all.

### The fix, and its price

Give partial credit: reward *progress toward* winning, with the win itself
still worth the most. In the guessing game, score the closest approach:

```
group of 8 rollouts:  0.1  0.4  0.0  0.3  0.6  0.2  0.1  0.5
mean = 0.275
advantages:          -0.2 +0.5 -1.0 +0.1 +1.3 -0.3 -0.2 +0.9
```

Now there's spread, so there's signal, so the model can climb. This is called
**reward shaping**.

But you have just re-opened the door from Unit 02. A shaped reward is a second
scoring function, which means a second thing that can be gamed. If "getting
close" pays, and finishing doesn't pay much more, then hovering close forever is
a perfectly good strategy.

The rule that follows — the same rule as Unit 02 in a new costume:

> Shaped rewards must always be worth **strictly less** than actually winning.
> Partial credit is a ladder to the goal, never a destination.

Lab 2 has you measure the sparse-reward failure across skill levels, then build
the shaped version. It's pure numpy — the problem is about scores and groups,
not about any particular API.

## 5. Giving the model tools

Tools live on a `Toolset`. Each `@vf.tool` method becomes a tool the model can
call:

```python
class CalcToolset(vf.Toolset[vf.ToolsetConfig, vf.State]):
    @vf.tool
    def calculator(self, expression: str) -> str:
        """Evaluate an arithmetic expression. Example: calculator("17 * 23") -> "391"."""
        ...
```

The library builds the tool's schema from the method's **signature** and its
description from the **docstring**. The docstring is prompt, not
documentation: a vague one produces a model that misuses the tool.

Two things to think about, which lab 3 covers:

- **Tool errors are training data.** Whatever your tool returns becomes the
  model's next turn. Return an actionable error string ("only numbers and
  + - * / ** are allowed") rather than raising — a good message is a chance to
  self-correct, a stack trace is a dead rollout.
- **A tool that does the whole task is not a tool, it's an answer key.** A
  calculator on an arithmetic task means you're training tool use, not
  arithmetic. That might be what you want. Be sure it is.

## Labs

| file | what you build |
|---|---|
| `exercise_1_guessing_game.py` | game logic + transcript-replaying rewards; the Env is written for you |
| `exercise_2_sparse_rewards.py` | measure the sparse-reward failure, then fix it (pure numpy) |
| `exercise_3_tools.py` | a `Toolset` with a calculator, and what the model actually sees |

All three run offline and deterministic. Labs 1 and 3 print the eval-CLI
command for their live versions.

## How to work

```bash
uv run python modules/03-multiturn-and-tools/exercise_1_guessing_game.py
uv run python modules/03-multiturn-and-tools/verify.py
```

Worked numbers are in [`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md).

## Checkpoint

You should be able to say:

> Multi-turn is the same training algorithm with a harder scoring problem. The
> danger is that a sparse "did you win" reward gives every rollout the same
> score when the model is weak — identical scores mean zero advantage and zero
> learning. Partial credit fixes that, at the cost of introducing something new
> to game, so partial credit must always be worth less than winning. And my
> graders replay the transcript, because a grader that trusts the game's own
> bookkeeping can silently disagree with what happened.
