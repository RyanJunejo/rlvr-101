# Unit 05 — Training for real

### Lecture notes

> **Time:** 4–6 hours · **Prerequisites:** Units 01–04 · **Needs:** an NVIDIA
> GPU box (rented) and about $15–30 of credit
>
> **By the end of this unit you will be able to:**
> 1. Read a `prime-rl` config and say what every section controls.
> 2. Connect the config's numbers back to the algorithm you wrote in Unit 01.
> 3. Estimate what a training run will cost before starting it.
> 4. Run `reverse-text` end to end and read the output critically.
> 5. Train on an environment you wrote.
>
> **Deliverables:** 2 offline labs (autograded), plus the runbook in section 5,
> which is verified by your own training curve rather than by a script.

When you've finished the labs, [`THEORY.md`](THEORY.md) goes deeper: why async
training is off-policy, and where clipping and the KL term come from.

---

## A note on what's verified here

Everything in Units 01–04 was run on the machine that wrote it. This unit
can't be. `prime-rl` requires an NVIDIA GPU and there isn't one here, so:

- The **config in section 3** is the real file from the repository, quoted
  verbatim.
- The **commands in section 5** come from the project's own documentation.
- The **two labs** run offline and are autograded like every other unit.
- The **runbook has not been executed end to end**. Its checkpoints are how you
  confirm it worked, and if a command has drifted since this was written, the
  repository is the authority.

That's a real gap. It's flagged rather than papered over.

---

## 1. What actually runs

Three processes, which you met in Unit 00:

```
  inference          orchestrator            trainer
  ─────────          ────────────            ───────
  vLLM generates     hands out questions,    computes gradients,
  the model's        collects answers,       updates weights
  answers            runs YOUR scoring
                            │
                     your reward function
                     plugs in here
```

They're split because generating text and computing gradients want opposite
things from a GPU. Generation is memory-bandwidth-bound and wants big batches;
training is compute-bound and wants the weights resident. Running them in one
process means each waits on the other.

This is why the starter configs need **two** GPUs, not one: one for inference,
one for training.

## 2. What it costs

Two H100s at GMI's SXM rate is about $4.80/hour. Your $50 of credit is therefore
around **10 hours of two-GPU time**, which is enough for everything in this unit
several times over, provided you turn the box off.

The three ways people burn credits, in order of how often they happen:

1. **Leaving the box running.** By far the most common. An idle H100 costs the
   same as a busy one.
2. **Debugging on rented hardware.** Every config typo you find at $4.80/hour
   could have been found locally at zero.
3. **Training on a task that can't teach anything.** Units 03 and 04 gave you
   two checks for this. Run them before you rent anything.

Lab 1 has you build the arithmetic so the numbers are yours rather than mine.

### You may not have to rent anything

Renting two GPUs and driving `prime-rl` yourself is one path. Prime Intellect's
own platform, **Lab**, is the other: hosted training and hosted evaluation on
their infrastructure, priced per token rather than per cluster-hour, with the
trained LoRA adapter deployable straight to their inference endpoint.

Token pricing changes the shape of the decision. Renting bills you for wall
clock — including the hours the box sits idle while you read logs, and the
overnight you forgot to shut it down. Token pricing bills you for work done.
For a first training run, where most of your time goes to reading output and
thinking, that difference is most of the bill.

This unit teaches the rented-box path because it makes the machinery visible:
you see the three processes, you read the config, you own the failure modes.
That understanding transfers to Lab, and is hard to acquire from a dashboard.
Once you have it, use whichever is cheaper for the job.

## 3. Reading the config

Here is `configs/basic/reverse-text/rl.toml` from the repository, complete and
verbatim. Almost every number in it is something you already understand.

```toml
# Reverse-text on 2 GPUs (1 trainer + 1 inference) — the smallest end-to-end RL
# loop (single-turn, 0.6B). Dev-sized mirror of examples/basic/reverse-text.

max_steps = 20
seq_len = 2048

[deployment]
num_train_gpus = 1
num_infer_gpus = 1

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[wandb]
project = "reverse-text"
name = "reverse-text"

[orchestrator]
batch_size = 128
group_size = 16

[orchestrator.train.sampling]
max_completion_tokens = 128

[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text-v1"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[trainer.optim]
lr = 3e-6

[ckpt] # Checkpoint at the end of training

[inference]

[orchestrator.renderer]
name = "prime-qwen3"
```

**`group_size = 16`** is the `G` from Unit 01, exercise 4. Sixteen answers to the
same question, their mean used as the baseline, advantages computed by
subtracting it. The whole reason that number can't be 1 is the result you
measured: with a group of one, every advantage is exactly zero and nothing moves.

**`batch_size = 128`** is how many questions per step. So each step generates
128 × 16 = 2,048 completions. Generation dominates the wall clock, which is why
it gets its own GPU.

**`lr = 3e-6`** is small. RL post-training nudges an already-trained model; the
large learning rates you'd use for pretraining would destroy it.

**`[ckpt]` and `[inference]`, both empty.** An empty table means "on, with
defaults" — `[ckpt]` saves a checkpoint at the end, and without it a finished
run leaves you nothing. Cheap insurance on rented hardware.

**`[orchestrator.renderer]`** picks how text becomes tokens. Chat templates are
usually applied server-side by the inference engine, which is fine until you
need to know exactly which tokens the model was trained on — and in RL you do,
because the trainer computes gradients per token. Renderers move templating
client-side so generation and training agree. The comment in the real file
explains why this run names one explicitly: the `-SFT` suffix on the model name
misses the automatic lookup table.

**`taskset` / `harness` / `runtime`** is the vocabulary you've used since Unit
02. `harness.id = "null"` is the same flag from your first live eval: no agent
loop, the model answers once. Training against the environment you packaged in
Unit 04 is pointing `taskset.id` at it.

**The model is 0.6B parameters.** Small enough that a full run finishes in
minutes. That's the point: you want the loop to be boring before you scale it.

The other starter configs, all 2-GPU:

| config | task |
|---|---|
| `reverse-text` | reverse a string. Start here. |
| `alphabet-sort` | sort words alphabetically |
| `wordle` | multi-turn word guessing |
| `wiki-search` | tool use over Wikipedia |
| `hendrycks-sanity` | a math benchmark |

## 4. What a healthy run looks like

Reward should rise. That's the easy part. The things worth watching are the ones
that tell you the rise is real:

**Completion length.** Reward climbing while answers get steadily longer and
stranger is the Unit 02 failure mode arriving in production. Length is the
cheapest early warning you have.

**The individual reward components.** Unit 02's reward methods report
separately for exactly this reason. Total reward up while correctness is flat
means you're training a formatter.

**Sampled rollouts.** Read the actual text. Not the average — the text. Everyone
says this and almost nobody does it, and it is still the single highest-value
habit in the list.

**Group reward variance.** If it collapses toward zero, your questions have
stopped teaching anything, in either direction. Unit 03 measured both.

## 5. The runbook

Untested end to end, as flagged above. Each step has a checkpoint; if a
checkpoint fails, stop there rather than continuing.

### 5.1 Provision

Rent two H100s. Note the hourly rate before you start.

> **Checkpoint:** `nvidia-smi` lists two GPUs.

### 5.2 Install

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

> **Checkpoint:** `uv run rl --help` prints usage.

### 5.3 First run

```bash
uv run rl @ configs/basic/reverse-text/rl.toml
```

Twenty steps on a 0.6B model. Watch the reward curve.

> **Checkpoint:** reward is higher at step 20 than at step 1, and a checkpoint
> file was written.

If reward is flat, check group reward variance before anything else. A flat
curve with zero variance is the Unit 03 failure, not a bug in your setup.

### 5.4 Read the numbers

Before running anything else, answer these from your own run:

- How long did one step take?
- What did those 20 steps cost, at your hourly rate?
- What would 500 steps cost?

That third number is the one that determines whether your capstone is feasible.

### 5.5 A second task

```bash
uv run rl @ configs/basic/alphabet-sort/rl.toml
```

> **Checkpoint:** you can explain how this config differs from the first, and why.

### 5.6 Your own environment

Push the taskset package you built in Unit 04 to the Hub, and point a config's
`taskset.id` at it.

**Run the Unit 04 feasibility check first.** Measure `pass@1` and `pass@8` at
`group_size = 16`. If `pass@8` is near zero or `pass@1` is near one, fix the task
before you rent anything — no amount of training rescues a task with no spread.

Worth noting: when this course tested its own Unit 02 arithmetic environment
against DeepSeek-V4-Pro, the model scored 12/12. That task is saturated. It's
fine as a teaching example and useless as a training environment, which is
exactly the check this step exists to catch.

> **Checkpoint:** reward on your own task rises, and reading sampled rollouts
> shows the model doing the task rather than exploiting the scoring.

### 5.7 Shut the box down

Then check the billing page and confirm it stopped. Do this even when you're sure.

## Labs

| file | what you build |
|---|---|
| `exercise_1_budget.py` | cost arithmetic for a training run |
| `exercise_2_config.py` | read and modify a `prime-rl` config |

Both run offline and cost nothing.

## How to work

```bash
uv run python modules/05-training-on-gpu/exercise_1_budget.py
uv run python modules/05-training-on-gpu/verify.py
```

## Checkpoint

You should be able to look at a `prime-rl` config and say what every number does,
which of them you'd change first for a task of your own, and roughly what the run
will cost before you start it.
