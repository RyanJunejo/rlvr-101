# Worked examples — Unit 06

Output from running the answer keys.

---

## 1. The same task, both APIs

**v0**, from Unit 02:

```python
def correct_answer(completion, answer, **kwargs) -> float:
    matches = ANSWER_RE.findall(completion[-1]["content"])
    return 1.0 if matches and matches[-1].strip() == answer else 0.0

env = vf.SingleTurnEnv(
    dataset=ds,
    rubric=vf.Rubric(funcs=[correct_answer, format_reward], weights=[1.0, 0.2]),
)
```

**v1**, same rules:

```python
class MathData(vf.TaskData):
    answer: str

class MathTask(vf.Task[MathData, vf.State, vf.TaskConfig]):
    @vf.reward
    async def correct_answer(self, task: MathData, trace: vf.Trace) -> float:
        matches = ANSWER_RE.findall(trace.last_reply)
        return 1.0 if matches and matches[-1].strip() == task.answer else 0.0

    @vf.reward(weight=0.2)
    async def has_answer_line(self, trace: vf.Trace) -> float:
        return 1.0 if ANSWER_RE.findall(trace.last_reply) else 0.0
```

Four differences:

| | v0 | v1 |
|---|---|---|
| where the text is | `completion[-1]["content"]` | `trace.last_reply` |
| where the answer is | injected `answer` arg | `task.answer` |
| sync or async | either | **must** be `async` |
| weights | parallel list, aligned by position | on the decorator |

The weight change is the one that prevents a real bug. In v0, inserting a
function into `funcs` without inserting its weight into `weights` silently
shifts every subsequent weight onto the wrong function.

---

## 2. Scoring, with numbers

```
  reply                            correct    format   weighted
  --------------------------------------------------------------
  Answer: 391                          1.0       1.0       1.20
  17 * 23 = 391. / Answer: 391.        1.0       1.0       1.20
  Answer: 3912                         0.0       1.0       0.20
  The product of 17 and 23 is 39       0.0       0.0       0.00
  no idea                              0.0       0.0       0.00
```

Row 3 is the one to read: wrong answer, correct format, so the two components
disagree. When a training run's score moves you can see which part moved.

The weighted column is `1.0 × correct + 0.2 × format`.

---

## 3. The `sampled` flag

Building a trace by hand:

```python
trace.nodes.append(
    vf.MessageNode(message=vf.AssistantMessage(content=reply), sampled=True)
)
```

With the flag:

```
last_reply:          'Answer: 391'
assistant_messages:  1
rewards:             {'correct_answer': Reward(score=1.0, weight=1.0),
                      'has_answer_line': Reward(score=1.0, weight=0.2)}
```

Without it — `sampled=False`, or omitted:

```
last_reply:          ''
assistant_messages:  0
rewards:             {'correct_answer': Reward(score=0.0, weight=1.0),
                      'has_answer_line': Reward(score=0.0, weight=0.2)}
```

No error. Every reward is zero. The relevant source is one line:

```python
return [n.message for n in self.nodes
        if n.sampled and isinstance(n.message, AssistantMessage)]
```

The flag distinguishes messages the model produced from messages that arrived
with the prompt — few-shot examples, or a conversation the model is continuing.
Scoring those as the model's own work would inflate every number you report.

---

## 4. Tasksets

A fixed one:

```
INFINITE: False
tasks:    5
  [0] math#0   What is 17 * 23?         -> 391
  [1] math#1   What is 144 / 12?        -> 12
  [2] math#2   What is 89 + 156?        -> 245
```

An infinite one, `.head(5)`:

```
INFINITE: True
  [0] What is 60 * 64?         -> 3840
  [1] What is 16 * 44?         -> 704
  [2] What is 76 * 73?         -> 5548
  [3] What is 62 * 49?         -> 3038
  [4] What is 72 * 56?         -> 4032

same questions on a second construction: True
took 3 from an infinite taskset: ['3840', '704', '5548']
```

The second taskset has no end. Asking for three cost three objects, and building
it cost nothing. The autograder pulls 500 from it to check it really is unbounded
rather than a long list.

Reproducibility comes from seeding inside `load()`. Without it, two people
"evaluating the same environment" would be evaluating different questions.

### Why this connects back to Unit 03

Unit 03 measured that a task teaches nothing once the model always wins or always
loses — no spread in a group, no advantage, no gradient. A fixed dataset walks
into that as the model improves.

A generated taskset lets difficulty be a config field, so you can produce harder
questions as the model gets better. It doesn't choose the difficulty for you. It
removes the dataset as the thing that limits you.
