# Worked examples — Unit 02

Real output from running the answer keys and, for the live sections, from an
actual eval run against a hosted model.

---

## 1. Scoring five replies

The lab 1 task, scored on hand-built traces:

```
  reply                            correct   format
  --------------------------------------------------
  Answer: 391                          1.0      1.0
  17 * 23 = 391. / Answer: 391.        1.0      1.0
  Answer: 3912                         0.0      1.0
  The product of 17 and 23 is 391.     0.0      0.0
  no idea                              0.0      0.0
```

Row 3 is the one to read: wrong answer, correct format, so the two components
disagree — and because they report separately, you can see which one moved.
That's the difference between "can't multiply" and "ignored the output spec,"
which need different fixes.

---

## 2. The `sampled` flag

Building a test trace:

```python
trace.nodes.append(
    vf.MessageNode(message=vf.AssistantMessage(content=reply), sampled=True)
)
```

With the flag:

```
last_reply: 'Answer: 391'
rewards:    {'correct_answer': 1.0, 'has_answer_line': 1.0}
```

Without it — `sampled=False`, or omitted:

```
last_reply: ''
rewards:    {'correct_answer': 0.0, 'has_answer_line': 0.0}
```

No error. Every reward is zero. The library source explains why in one line:

```python
return [n.message for n in self.nodes
        if n.sampled and isinstance(n.message, AssistantMessage)]
```

`last_reply` reads only messages the model produced. Prompts can legitimately
contain assistant messages — few-shot examples, a conversation being continued —
and scoring those as the model's own work would inflate every number you report.

---

## 3. A real eval run

The lab's live command, run against a hosted DeepSeek model. From `eval.log`:

```
INFO rollout done: id=22597a8807b8... task=2 reward=1.000 turns=1 stop=agent_completed
INFO rollout done: id=d64d2026799d... task=0 reward=1.000 turns=1 stop=agent_completed
```

`reward=1.000` is `correct_answer` × 1.0 + `has_answer_line` × 0.2, normalized
per the run's weighting; `stop=agent_completed` means the rollout ended because
the model finished, which is the healthy stop.

The run directory:

```
outputs/first-task/
  config.toml     ← the resolved run: model, base_url, taskset, harness
  eval.log        ← one line per rollout, with its reward
  traces.jsonl    ← every attempt in full, one episode per line
```

Each `traces.jsonl` line is an episode; the traces sit under its `traces` key:

```python
ep = json.loads(line)
for t in ep["traces"]:
    t["task"]["data"]["prompt"]     # the question, exactly as configured
    t["rewards"]                    # {'correct_answer': {'score': 1.0, ...}, ...}
    t["ok"], t["stop_condition"]    # did this rollout actually finish?
```

---

## 4. When rollouts fail

The same command was first run with the API key set in the shell but **not
exported**. Every rollout failed identically:

```
WARNING model call failed: ProviderError: upstream 401: {"error":"Invalid token, failed to decode"}
INFO rollout done: id=4d40ab2dd964... task=2 reward=0.000 turns=0 stop=ProviderError
```

And in `traces.jsonl`:

```
ok=False   rewards={}   stop_condition='error'
errors=[{'type': 'ProviderError', 'message': 'upstream 401: ...'}]
```

Two lessons, both cheap here and expensive later:

- `--client.api-key-var` names an environment variable the *eval process*
  reads. A shell variable that isn't exported never reaches it.
- **A 0.0 is not always a wrong answer.** These rollouts scored zero because
  they never ran. Check `ok` and `stop_condition` before reading a score as a
  verdict — in Unit 05, misreading failed rollouts as "the model got worse" is
  a classic way to waste a debugging afternoon.

---

## 5. The attack suite

```
sloppy_correctness: 5/12
  fooled by: superset number, negation, shotgun guess, enumerate everything,
             right number, wrong answer, answer field shotgun,
             mentions in prose only

robust_correctness: 12/12
  survives every attack.
```

Which rule defeats which attack — worth tracing, because no single fix covers
them all:

| attack | defeated by |
|---|---|
| `Answer: 3912` (superset) | rule 3: `==`, never `in` |
| negation, prose-only mention | rule 1: only the slot counts |
| shotgun guess | rules 1 + 3 together |
| `Answer: 391, 392, 393` | rule 3 — it got *past* rule 1, into the slot |
| enumerate everything | rule 4: no slot, no credit |
| self-correction (honest!) | rule 2 — protects a correct model, rather than blocking a dishonest one |

And the honest caveat: 12/12 means the grader survives the attacks *someone
thought of*. A lucky guess still scores 1.0; an answer leaked into the prompt
can be copied; `0.5` versus `1/2` scores zero. Closing those holes costs you
elsewhere. The tradeoff never fully goes away, and the meta-lesson stands: you
cannot verify a scoring function by reading it — you verify it by trying to
break it.
