# Unit 03 — Conversations and tools

### Lecture notes

> **Time:** 3–4 hours · **Prerequisites:** Units 01 and 02 · **Needs:** an API
> key (lab 3 only; labs 1–2 run offline)
>
> **By the end of this unit you will be able to:**
> 1. Build a multi-turn environment where the model's action changes what it
>    sees next.
> 2. Explain what `setup_state`, `env_response` and `@vf.stop` each do, and when
>    each one runs.
> 3. Diagnose the **sparse reward problem** — and connect it directly to the
>    zero-advantage result from Unit 01.
> 4. Design a shaped reward that rescues learning, and identify the new way it
>    can be gamed.
> 5. Give a model tools, and reason about what tool use does to your scoring.
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

## 2. The three methods you implement

A multi-turn environment is a subclass of `vf.MultiTurnEnv` with three pieces.
What each one is for, and when it runs:

```
   rollout starts
        │
        ├─ setup_state(state)        ← ONCE, before anything.
        │                              Put your game's starting position here.
        │
        ├─ model produces a message
        ├─ @vf.stop checks           ← after each turn: are we done?
        ├─ env_response(msgs, state) ← your reply back to the model
        ├─ model produces a message
        ├─ @vf.stop checks
        │   ... repeat ...
        │
   rollout ends → rubric scores it
```

**`setup_state(state)`** runs once at the start. `state` is a dict that persists
for the whole rollout, and it already contains `answer` (from your dataset). Add
whatever your game needs and return it:

```python
async def setup_state(self, state):
    state["secret"] = int(state["answer"])
    state["guesses"] = []
    return state
```

**`env_response(messages, state)`** is the environment talking back. `messages`
ends with what the model just said; you return a list of new messages:

```python
async def env_response(self, messages, state, **kwargs):
    text = messages[-1]["content"]
    ...
    return [vf.UserMessage(content="Too high.")]
```

Note it returns a **list**, and note the messages are `vf.UserMessage` — from the
model's point of view, the environment is the user.

**`@vf.stop`** marks a method that decides whether the rollout is over:

```python
@vf.stop
async def solved(self, state) -> bool:
    return state.get("solved", False)
```

You get `max_turns` for free from the constructor — `MultiTurnEnv` already has a
built-in stop condition for it. You only write `@vf.stop` methods for *your*
game's ending conditions.

> **A note on the docs.** Older `verifiers` documentation describes an
> `is_completed()` method. That's not how version 0.3.0 works — it uses these
> `@vf.stop` decorated methods instead. When library docs and library source
> disagree, the source wins. This is worth internalizing generally.

## 3. Mutating state is the whole point

In Unit 02 your reward functions were pure: look at the text, return a number.
Here, `env_response` **writes to `state`**, and the rubric reads it afterward.

```python
state["guesses"].append(g)
if g == state["secret"]:
    state["solved"] = True
```

That's how the score at the end knows what happened during the game. Your reward
functions can now ask for `state` and inspect the whole history:

```python
def solved_reward(state, **kwargs) -> float:
    return 1.0 if state.get("solved") else 0.0
```

## 4. The sparse reward problem

This section matters more than the rest of the unit, and it follows directly from
something you already proved in Unit 01.

Suppose your only score is "did the model win the game." Now imagine a model
that's currently bad at the game — it wins maybe 1 time in 50.

Ask a question 8 times. What do you get?

```
group of 8 rollouts:  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
mean = 0.0
advantages:           0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```

**Every advantage is zero. That group teaches the model nothing.**

You proved this in Unit 01: advantage is score-minus-group-average, so a group
where every score is identical produces no gradient at all. It doesn't matter
that the scores are 0.0 rather than 1.0 — what matters is that they're *the
same*.

So with a sparse reward and a weak model, almost every group is wasted. You burn
enormous amounts of generation compute to produce exactly zero learning signal.
And the model can never bootstrap, because it needs to already be decent at the
task to get any signal at all.

This is *the* central difficulty of multi-turn RL, and it has a name in the
literature: the exploration problem.

### The fix, and its price

Give partial credit. Don't just reward winning — reward *progress toward*
winning. In the guessing game, reward narrowing the range:

```
group of 8 rollouts:  0.1  0.4  0.0  0.3  0.6  0.2  0.1  0.5
mean = 0.275
advantages:          -0.2 +0.5 -1.0 +0.1 +1.3 -0.3 -0.2 +0.9
```

Now there's spread, so there's signal, so the model can climb. This is called
**reward shaping**.

**But you have just re-opened the door from Unit 02.** A shaped reward is a
second scoring function, which means it's a second thing that can be gamed. A
model can learn to farm partial credit indefinitely without ever finishing — if
"narrowing the range" pays, and finishing doesn't pay much more, then narrowing
the range very slowly forever is a perfectly good strategy.

The rule that follows, and it's the same rule as Unit 02 in a new costume:

> Shaped rewards must always be worth **strictly less** than actually winning.
> Partial credit is a ladder to the goal, never a destination.

Lab 2 has you measure the sparse-reward failure and then build the shaped
version.

## 5. Giving the model tools

A tool is just a Python function. `ToolEnv` handles everything else:

```python
def calculator(expression: str) -> str:
    """Evaluate a arithmetic expression. Example: calculator("17 * 23")"""
    ...

env = vf.ToolEnv(tools=[calculator], max_turns=6, dataset=ds, rubric=rubric)
```

The library reads your function's **name, signature, and docstring** to tell the
model what the tool does. Your docstring goes into the prompt verbatim. A vague
docstring produces a model that misuses the tool.

`ToolEnv` also adds monitoring metrics for free: `total_tool_calls`, and a
per-tool count. These aren't rewards (they're weighted 0), they're diagnostics —
and they're how you notice a model that has learned to call a tool 40 times per
question because something in your scoring rewards it.

Two things to think about, which lab 3 covers:

- **Tool errors are training data.** If your tool raises on bad input, the model
  sees the error message and can correct. So the error text matters — it's a
  teaching signal, and `"ValueError"` teaches less than `"expression must
  contain only digits and + - * / ( )"`.
- **A tool that does the whole task is not a tool, it's an answer key.** If you
  give a calculator to a model on an arithmetic task, you're no longer training
  arithmetic — you're training tool-calling. That might be what you want. Be
  sure it is.

## Labs

| file | what you build |
|---|---|
| `exercise_1_guessing_game.py` | a complete multi-turn environment, offline |
| `exercise_2_sparse_rewards.py` | measure the sparse-reward failure, then fix it |
| `exercise_3_tools.py` | give a model a calculator and a lookup tool |

Labs 1 and 2 run entirely offline and deterministically. Lab 3 needs an API key
for the optional live section.

## How to work

```bash
uv run python modules/03-multiturn-and-tools/exercise_1_guessing_game.py
uv run python modules/03-multiturn-and-tools/verify.py
```

Worked numbers are in [`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md).

## Checkpoint

You should be able to say:

> Multi-turn is the same training algorithm with a harder scoring problem. The
> danger is that a sparse "did you win" reward gives every rollout the same score
> when the model is weak, and identical scores mean zero advantage and zero
> learning. Partial credit fixes that, at the cost of introducing something new
> to game — so partial credit must always be worth less than winning.
