# Worked examples — Unit 03

Real numbers from running the code. Reproduce any of them with the solution
files.

---

## 1. One game, turn by turn

Secret is 42, player uses binary search. Watch the range shrink:

```
                                    range      feedback
  turn 1: guess 50   (1+100)/2       1..100     Too high.
  turn 2: guess 25   (1+49)/2        1..49      Too low.
  turn 3: guess 37   (26+49)/2      26..49      Too low.
  turn 4: guess 43   (38+49)/2      38..49      Too high.
  turn 5: guess 40   (38+42)/2      38..42      Too low.
  turn 6: guess 41   (41+42)/2      41..42      Too low.
  turn 7: guess 42   (42+42)/2      42..42      Correct!
```

Seven turns, exactly the budget. That's why `max_turns = 7`.

Each guess halves the remaining possibilities: 100 → 50 → 25 → 13 → 7 → 4 → 2 →
1. Seven halvings covers 2⁷ = 128 > 100. A perfect player *always* wins within
7; anything less than perfect sometimes doesn't.

**`max_turns` is a reward design decision, not a technical detail.** Set it to 10
and a mediocre player also wins every time — scores stop separating good play
from bad, every group scores 1.0, and (per Unit 01) you get no gradient. Set it
to 4 and nobody ever wins, which fails the same way from the other side.

You want the setting where the model wins *sometimes*.

---

## 2. What the environment does with a message

```
model says:  "Hmm, let me split the range.\nGuess: 50"
             │
             ├─ GUESS_RE finds ["50"]  → take the last → 50
             ├─ state["guesses"].append(50)      → [50]
             ├─ 50 > 42, so:
             └─ return [UserMessage("Too high.")]
```

Two edge cases that matter:

**No guess in the message.**

```
model says:  "I'm not sure where to start."
             → GUESS_RE finds []
             → record nothing, reply with a format reminder
```

We don't crash and we don't record a guess — but the turn is still consumed,
because `max_turns` counts model messages. So rambling costs the model a turn.
That's the right incentive, and it came for free.

**Two guesses in one message.**

```
model says:  "Guess: 5\nActually, wait.\nGuess: 42"
             → GUESS_RE finds ["5", "42"] → take the LAST → 42
```

Same rule as Unit 02: models restate and correct themselves, and the final
commitment is the real one. Taking the *first* match would fail a model that
reasoned its way to the right answer — a false negative, which is the worst kind
of scoring bug because it teaches the model that being right doesn't pay.

---

## 3. The shaped reward, computed

```
shaped_reward:
    solved            -> 1.0
    no guesses        -> 0.0
    otherwise         -> 0.5 * (1 - best_distance / 99)
```

Secret is 50:

| guesses made | closest miss | score | working |
|---|---|---|---|
| solved | — | **1.000** | win pays full |
| `[51]` | 1 | **0.4949** | `0.5 × (1 − 1/99)` |
| `[45]` | 5 | **0.4747** | `0.5 × (1 − 5/99)` |
| `[100]` | 50 | **0.2475** | `0.5 × (1 − 50/99)` |
| `[51, 100]` | 1 | **0.4949** | closest guess counts, not the last |
| `[]` | — | **0.000** | nothing to reward |

**The gap between 0.4949 and 1.0 is the entire safety property.** Change the
`0.5` to `1.0` and a near-miss scores 0.9899 — at which point the best strategy
is to hover one away from the answer forever, collecting almost full marks and
never finishing. The score would climb beautifully while the win rate went <!-- prose-ok: ironic -->
nowhere.

That's the Unit 02 reward-hacking lesson in a new costume. One constant is all
that stands between you and it.

---

## 4. The headline experiment

For a player of a given skill, we ask: **what fraction of groups actually teach
the model anything?** A group is useful only if its rollouts got *different*
scores — otherwise every advantage is zero (Unit 01).

400 groups of 8, per row:

```
  skill |  win rate |  useful (sparse) |  useful (shaped)
 ---------------------------------------------------------
    0.0 |      7.0% |            45.2% |           100.0%
    0.2 |     11.4% |            61.5% |           100.0%
    0.4 |     22.1% |            85.2% |           100.0%
    0.6 |     38.6% |            93.8% |            99.2%
    0.8 |     63.4% |            89.0% |            89.2%
    1.0 |    100.0% |             0.0% |             0.0%
```

Three things to read out of this table.

**The two ends are the same failure.** A beginner wastes 54.8% of its groups
because nothing succeeds. An expert wastes **100%** of its groups because
everything succeeds. Different causes, identical consequence: no spread, no
advantage, no gradient. Too hard and too easy are the same problem.

**Shaping fixes the bottom, not the top.** At skill 0.0 it takes you from 45.2%
to 100%. At skill 1.0 it's still 0.0% — exactly as useless as the sparse reward.
Once every rollout wins, nothing you compute from the transcript has any spread
left in it. The top of the curve is fixed by giving the model *harder questions*,
not by better scoring.

**Why shaping helps so much at low skill.** Two random players will essentially
never both win — but they will almost always land at *different distances* from
the answer. Distance is continuous, so it produces spread even when success never
happens.

That's the general principle worth carrying: **if you need signal from a model
that can't yet do the task, find something continuous to measure.**

---

## 5. What the model actually sees when you give it a tool

Your Python function:

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / ** and parentheses. Use this for any arithmetic instead of
    calculating in your head. Example: calculator("17 * 23") returns "391".
    """
```

What the library generates and sends to the model:

```
name:        calculator
description: 'Evaluate an arithmetic expression and return the result.

              Supports + - * / ** and parentheses. Use this for any arithmetic
              instead of calculating in your head. Example: calculator("17 * 23")
              returns "391".'
parameters:  {'properties': {'expression': {'title': 'Expression',
                                            'type': 'string'}},
              'required': ['expression'],
              'type': 'object',
              'additionalProperties': False}
```

Your docstring, verbatim. Your type hint, turned into a schema.

Your docstring is prompt text. "Does math" and the docstring above produce
measurably different tool-use behaviour from the same model.

### Error messages are training data

```
calculator('17 * 23')          -> '391'
calculator('hello')            -> "error: only numbers and + - * / ** are
                                   allowed. Provide a plain arithmetic
                                   expression using only numbers and
                                   + - * / ** and parentheses, for example
                                   '2 * (3 + 4)'."
calculator("__import__('os')") -> "error: only numbers and + - * / ** are
                                   allowed. ..."
```

Whatever you return becomes the model's next turn, so a good error message is a
chance for it to self-correct. Notice the message says what *is* allowed, not
just what went wrong — `"invalid syntax"` gives the model nothing to act on.

And note the third case returns an error rather than executing. **Never
`eval()` model output.** During RL this is worse than usual: the model is
actively searching for inputs that maximise a score, which makes it an automated
fuzzer pointed at your tool.

---

## 6. Free instrumentation

Building a `ToolEnv` gives you monitoring you didn't ask for:

```
  source rubric              metric                weight
  -------------------------------------------------------
  Rubric                     correct_answer           1.0
  MultiTurnMonitorRubric     num_turns                0.0
  ToolMonitorRubric          total_tool_calls         0.0
  ToolMonitorRubric          calculator_calls         0.0
```

Weight `0.0` means these don't affect the score — they're diagnostics. Watch them
during training. A model whose `calculator_calls` climbs steadily has usually
found something in your scoring that pays for calling tools, and you want to know
that before it has had 500 steps to perfect the habit.

**A structural note.** Your rubric got wrapped in a `RubricGroup`, so
`env.rubric.funcs` is empty and the real functions live in `env.rubric.rubrics`.
Worth knowing before you go looking for your own reward function and conclude it
vanished.
