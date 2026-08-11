# Unit 02 — Writing the scoring function

### Lecture notes

> **Time:** 3–4 hours · **Prerequisites:** Unit 01 · **Needs:** an API key
> (labs 1–2 only; the main lab runs offline)
>
> **By the end of this unit you will be able to:**
> 1. Build a working task environment with `verifiers` and score a real model
>    against it.
> 2. Write reward functions using the library's argument-injection convention.
> 3. Combine several reward components with weights, and explain why only the
>    *ratio* between them matters.
> 4. Identify how a scoring function can be exploited, and rewrite it to
>    survive an adversarial suite.
> 5. Explain why an exploitable reward is categorically worse than a noisy one,
>    and describe how to catch reward hacking during a real training run.
>
> **Deliverables:** 3 labs (`exercise_*.py`), autograder green, problem set in
> `NOTES.md`.

New terms are defined as they appear; [`GLOSSARY.md`](../../GLOSSARY.md) has them
all in one place.

---

## 1. Where we are

Unit 01 answered "how do you train a model on a score?" The answer turned out
to be short: sample answers, compare each one to the average of others from the
same question, make the better-than-average ones more likely.

Notice what that leaves wide open. **Everything now depends on the score.** The
training algorithm has no opinion about what's good — it just faithfully
maximizes whatever number you hand it.

So the interesting question is no longer "which algorithm?" It's "what exactly am
I rewarding?" That's this unit, and honestly it's where the rest of the
difficulty in this field lives.

## 2. What an environment actually is

`verifiers` is Prime Intellect's library for writing these scoring setups. The
thing you build is called an **environment**, which is a grander word than it
deserves. An environment is three things in a bundle:

```
  a dataset          a parser              a rubric
  of questions   →   pulls the answer  →   scores it   →   a number
  (and answers)      out of the reply
```

That's it. No simulator, no physics. Questions, a way to find the answer in the
model's rambling, and a function that scores it.

In code:

```python
env = vf.SingleTurnEnv(
    dataset=ds,                 # questions and reference answers
    system_prompt="...",        # instructions given to the model
    parser=parser,              # optional; finds the answer in the reply
    rubric=rubric,              # the scoring functions
)
```

Your dataset needs two columns, `question` and `answer`. The library builds the
actual prompt for you.

**The design idea worth pausing on:** that same object can be used three
ways without changing anything: as a **test** (run a model against it, see how it
scores), as a way to **generate training data** (keep the good answers), or as a
**training task** (feed the scores to the trainer). A test you can train against
*is* a training task. There's no difference, which is why the library doesn't
distinguish them.

## 3. Writing a scoring function

It's a normal Python function that returns a number:

```python
def correct_answer(completion, answer, **kwargs) -> float:
    ...
```

The library looks at your function's parameter names and passes in whatever you
asked for. Ask for `completion` and `answer` and you get those. Ask for nothing
extra and you get nothing extra. Always include `**kwargs` so you don't break
when the library offers more than you wanted.

**One thing that trips up everyone exactly once:** `completion` is not a string.
It's a list of chat messages. The model's text is `completion[-1]["content"]`.

You can combine several scoring functions with weights:

```python
rubric = vf.Rubric(
    funcs=[correct_answer, follows_format],
    weights=[1.0, 0.2],
)
```

The final score is the weighted sum, and each function's score is also reported
separately — which is how you debug. When the score moves, you can see *which
part* moved.

**The weights are where the judgment lives.** Remember from Unit 01 that GRPO
subtracts the group average and divides by the spread, so multiplying every
weight by 10 changes literally nothing. The absolute numbers are meaningless.
Only the **ratio** matters. `[1.0, 0.2]` says correctness is worth five times as
much as formatting.

Get that ratio wrong and you'll train a model that produces beautifully formatted <!-- prose-ok: ironic -->
nonsense. Exercise 2 shows you exactly where it tips over.

### Running against a real model

One gotcha worth knowing before you hit it. `verifiers` does **not** accept a raw
`openai.OpenAI` object — it wants its own `ClientConfig`, which describes how to
reach an endpoint:

```python
client = vf.ClientConfig(
    client_type="openai_chat_completions",
    api_key_var="OPENAI_API_KEY",          # the NAME of the env var, not the key
    api_base_url=os.getenv("OPENAI_BASE_URL"),
)
results = env.evaluate_sync(client=client, model=model, rollouts_per_example=2)
```

Note `api_key_var` takes the *name* of the environment variable rather than the
secret itself. That's deliberate: your key never gets passed around or written
into a training config.

The result is a **dict**, not an object. Per-rollout records live under
`results["outputs"]`, each with `reward`, `completion`, `metrics`, and more:

```python
rewards = [o["reward"] for o in results["outputs"]]
```

### Rollouts fail, and your reward function has to survive it

A dropped connection, a timeout, a content filter — and `completion` comes back
as an empty list. Writing `completion[-1]["content"]` then raises `IndexError`
from inside your reward function, and `verifiers` reports it as
`Error calling reward function`, which tells you nothing about the real cause.

Guard it:

```python
def reply_text(completion) -> str:
    return completion[-1]["content"] if completion else ""
```

A failed rollout now scores 0.0 cleanly, which is right: it didn't answer, so it
gets no credit.

This isn't hypothetical. While this course was being written the endpoint threw
SSL errors mid-run and every rollout came back empty. The reward function crashed
on the first one.

## Labs

| file | what you build |
|---|---|
| `exercise_1_first_env.py` | a working environment on arithmetic, scored against a real model |
| `exercise_2_rubric.py` | two scoring functions combined, and where the weights break |
| `exercise_3_reward_hacking.py` | **the important one.** Break a scoring function, then fix it. |

Exercises 1 and 2 need your `.env` set up. Exercise 3 doesn't — it runs offline.

## 4. About the reward-hacking lab

Let me set this one up properly, because it's the thing I'd most want you to
take away from the whole workbook.

You'll be handed a scoring function that looks completely reasonable:

```python
def sloppy_correctness(completion, answer, **kwargs) -> float:
    return 1.0 if answer in completion[-1]["content"] else 0.0
```

Read it in English: *"did the right answer appear anywhere in the model's
reply?"* That seems like a fine definition of correct. Versions of this ship in
real codebases.

It is badly broken, and you'll watch it give full marks to:

- a reply that says the answer is **not** 391,
- a reply that guesses five numbers at once,
- a reply that just lists every number from 380 to 400,
- a reply saying `3912` when the answer was `391` (because "391" *is* inside
  "3912").

**This matters more than a list of bugs.**

A scoring function that's merely *noisy* is survivable — random errors average
out over thousands of examples. A scoring function that's **exploitable** does
not average out, because training is a search process and its entire job is
finding the highest score available.

Any hole you leave, it will find and drive through — because "list every number
from 380 to 400" *does* score higher than "carefully multiply 17 by
23," and it's much easier. The model isn't cheating. It's doing precisely what
you asked.

Trace what happens. One answer in a group stumbles onto the enumeration trick and
scores 1.0, while the careful answers score 0.0 on an arithmetic slip. From
Unit 01: that gives enumeration a **positive advantage**, so it gets reinforced
and sampled more often, so it wins more groups. A few hundred steps later you
have a model that has completely abandoned arithmetic — and a training curve that
looks like a triumph the entire time.

You wouldn't notice from the score. That's the whole problem. **The score is the
thing being hacked.**

Then you'll fix it, and your fix gets graded against twelve adversarial replies.
The broken version scores 5/12.

## 5. How you catch this in real life

- **Read the actual model outputs.** Not the average score — the text. Regularly.
  Everyone says this and almost nobody does it.
- **Watch how long the replies get.** Score climbing while answers get longer and
  stranger is the classic signature.
- **Keep a separate test your scoring function can't touch**, and check the two
  agree.
- **Assume every scoring function you write has a hole in it**, because it does.
  The only question is whether you find it before training does.

## Optional reading

- The `verifiers` source: `envs/environment.py` and `rubrics/rubric.py`. Both
  readable in one sitting, and more accurate than the docs.
- **"Concrete Problems in AI Safety"** (arXiv:1606.06565), section 3 only — the
  reward hacking section. Written years before any of this tooling existed and it
  predicts every failure you're about to see.

## Running

```bash
uv run python modules/02-first-environment/exercise_3_reward_hacking.py
uv run python modules/02-first-environment/verify.py
```
