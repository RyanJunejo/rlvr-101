# Worked examples — Unit 06

Real output from the answer keys.

---

## 1. The v0 environment, inspected

```
type:    SingleTurnEnv
columns: ['question', 'answer', 'example_id', 'prompt']
rubric:  RubricGroup
  sub: Rubric                  ['correct_answer']  [1.0]
  sub: MultiTurnMonitorRubric  ['num_turns']       [0.0]
```

Three v0 fingerprints in one listing:

- The `prompt` column was **built by the library** from `question` plus the
  system prompt. You supply the magic column names; it does the assembly.
- Your rubric got wrapped in a `RubricGroup` beside a monitor rubric, so the
  top-level `funcs` is empty and yours is one level down. Worth knowing before
  you conclude your reward function vanished.
- The monitor's `num_turns` carries weight `0.0` — v0's spelling of what v1
  writes as `@vf.metric`.

## 2. Scoring, v0 style

```
'Answer: 391'      -> reward=1.0  metrics={'correct_answer': 1.0, 'num_turns': 0.0}
'Answer: 3912'     -> reward=0.0  metrics={'correct_answer': 0.0, 'num_turns': 0.0}
'no idea'          -> reward=0.0  metrics={'correct_answer': 0.0, 'num_turns': 0.0}
```

Same grader behavior as Unit 02 — commit-then-compare survives the superset
number — reached through `rubric.score_rollout(state)` on a hand-built `State`
dict, the trace of its era.

## 3. The failed-rollout test

```
correct_answer([], '391') -> 0.0
```

An empty completion scored cleanly because the function guards it:

```python
text = completion[-1]["content"] if completion else ""
```

Delete the guard and this exact call raises `IndexError`, which surfaces as
`Error calling reward function` with the real cause hidden. In v0 the guard is
your job in every reward function; v1's `trace.last_reply` absorbed it.

## 4. The port, proven

The same five replies scored through both stacks:

```
reply                                v0     v1
----------------------------------------------
Answer: 391                         1.0    1.0
17 * 23 = 391. / Answer: 391.       1.0    1.0
Answer: 3912                        0.0    0.0
The product is 391.                 0.0    0.0
no idea                             0.0    0.0

every reply scores identically: True
```

That bottom line is the port's definition of done. Anything less means the
migration changed the task, and numbers from before and after stop being
comparable.
