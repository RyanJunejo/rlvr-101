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

## 2. One turn through the Env

```
model says:  "Hmm, let me split the range.\nGuess: 50"
             |
             |  segment = await interaction.turn(...)   <- run() has the reply
             |  parse_guess(segment.last_reply)  -> 50
             |  50 > 42, so respond(50, 42)      -> "Too high."
             |
             v
             segment = await interaction.turn("Too high.")   <- next move arrives
```

Two edge cases that matter:

**No guess in the message.**

```
model says:  "I'm not sure where to start."
             -> parse_guess returns None
             -> feedback: a format reminder; nothing recorded
```

We don't crash and nothing counts as a guess — but the exchange still consumed a
turn, so rambling costs the model one of its seven. The right incentive, for
free.

**Two guesses in one message.**

```
model says:  "Guess: 5\nActually, wait.\nGuess: 42"
             -> parse_guess returns 42 (the LAST match)
```

Same rule as Unit 02's answer slot: models restate and correct themselves, and
the final commitment is the real one. Taking the *first* match would fail a
model that reasoned its way to the right answer — a false negative, the worst
kind of grading bug.

### And the scoring replays it all

```
rewards: {'solved': 1.0}      metrics: {'num_guesses': 7.0}
```

`solved` never reads the Env's feedback. It re-parses every guess from
`trace.assistant_messages` and compares against `task.answer` — so a feedback
bug in the Env couldn't corrupt the score, and the whole thing tests offline on
hand-built traces.

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

Your Python method:

```python
class CalcToolset(vf.Toolset[vf.ToolsetConfig, vf.State]):
    @vf.tool
    def calculator(self, expression: str) -> str:
        """Evaluate an arithmetic expression and return the result.

        Supports + - * / ** and parentheses. Use this for any arithmetic
        instead of calculating in your head. Example: calculator("17 * 23")
        returns "391".
        """
```

What the library registers, straight from the source of `Toolset.register()`:

```python
mcp.add_tool(
    self._with_state(fn),
    name=getattr(fn, "tool_name", None) or fn.__name__,
    description=(fn.__doc__ or "").strip() or None,
)
```

Your docstring, verbatim, is the description. Your signature becomes the schema.
"Does math" and the docstring above produce measurably different tool use from
the same model.

### Error messages are training data

```
calculator('17 * 23')          -> '391'
calculator('hello')            -> "error: only numbers and + - * / ** are
                                   allowed. Provide a plain arithmetic
                                   expression using only numbers and
                                   + - * / ** and parentheses, e.g. '2 * (3 + 4)'."
calculator("__import__('os')") -> "error: only numbers and + - * / ** are
                                   allowed. ..."
```

Whatever you return becomes the model's next turn, so a good error message is a
chance to self-correct. The message names what IS allowed — `"invalid syntax"` alone gives the
model nothing to act on.

And the third case returns an error rather than executing. Never `eval()` model
output: during RL the model is actively searching for inputs that maximize a
score, which makes it an automated fuzzer pointed at your tool.

## 6. Free instrumentation

The game task records a metric beside its reward:

```python
@vf.metric
async def num_guesses(self, trace) -> float:
    ...
```

```
rewards: {'solved': Reward(score=1.0, weight=1.0)}
metrics: {'num_guesses': 7.0}
```

A metric is recorded, never weighted — it can't affect training, so it's free
instrumentation. Watch it anyway. In Unit 05, a model whose `num_guesses` sits
near 7 is scanning the range instead of searching it, and a model whose tool
calls climb steadily has usually found something in your scoring that pays for
calling tools. You want to know either fact before it has had 500 steps to
perfect the habit.
