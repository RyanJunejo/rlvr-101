# Glossary

Every term this workbook uses, in plain language, with a small example.

Look things up as you go. Nothing here needs to be memorized — the terms will
stick once you've written the code that uses them.

Terms are grouped by when you'll first meet them, not alphabetically, because
the later ones are built out of the earlier ones.

---

## The basic loop

**Model**
The thing you're training. It takes text in and produces text out. That's all we
need from it for now.

**Token**
Models don't read letters or words — they read *tokens*, which are chunks of
text. "Reinforcement" might be two tokens, `Rein` + `forcement`. A model produces
text one token at a time.

**Logits**
Before a model picks its next token, it produces one raw score per possible
token. Those scores are logits. They can be any number — negative, positive,
huge. They are *not* probabilities yet.

> Example: with a tiny vocabulary of 3 tokens, the model might output
> logits `[2.0, 1.0, -1.0]`. Higher means "more inclined toward this one."

**Softmax**
The function that turns logits into probabilities. It exponentiates each logit
and divides by the total, so everything comes out positive and sums to 1.

> `[2.0, 1.0, -1.0]` → `[0.705, 0.259, 0.035]`.
> Notice the ordering is preserved and the gaps get stretched.

**Sampling**
Picking a token by rolling weighted dice against those probabilities. With the
numbers above you'd get token 0 about 70% of the time — but not always. This
randomness is what lets a model discover outputs it wasn't already producing.

**Temperature**
A knob on sampling. Divide the logits by the temperature before the softmax.
Temperature above 1 flattens the distribution (more random, more exploration);
below 1 sharpens it (more predictable). Temperature 0 means always take the
highest-scoring token.

**Policy**
The model's rule for choosing what to do. In our case it's just "the model,
together with how it samples." When you read "the policy improved," it means
"the model got better at producing high-scoring output."

Written as `π` (Greek letter pi) in papers. `π(a)` means "the probability the
policy assigns to action `a`."

**Rollout** (also **completion**, **sample**, **generation**)
One run of the model on one prompt, producing one output. If you ask the same
question 8 times and get 8 different answers, that's 8 rollouts.

**Reward**
A number saying how good one rollout was. You write the function that produces
it. Higher is better. That's the entire definition — there's no deeper truth to
it, which is exactly why designing it well is hard.

**Verifier**
The function that computes the reward by *checking* the output, as opposed to
guessing at it. "Does this equal 391?" is a verifier. It's the thing that makes
this whole approach work: the score is grounded in something real.

**RLVR**
Just an acronym for "Reinforcement Learning from Verifiable Rewards" — the
approach where reward comes from a checkable fact (did the code run? is the math
right?) rather than from human opinion or another model's judgment. This whole
workbook is about RLVR.

---

## Making the model change

**Gradient**
For a function with many inputs, the gradient is the direction you'd nudge those
inputs to make the output go *up* fastest. It's a list of numbers, one per input.

> If moving knob A up increases your score a lot, and moving knob B up decreases
> it slightly, the gradient might be `[+0.9, -0.1]`.

**Gradient ascent / descent**
Repeatedly nudging the inputs along the gradient. *Ascent* to maximize
something (we want more reward), *descent* to minimize it (usually loss).
Same operation, opposite sign.

**Learning rate**
How big each nudge is. Too small and training crawls; too large and it
overshoots and destroys itself. Written `lr` in the code.

**The problem RL solves for us**
You cannot directly take the gradient of "reward" with respect to the model's
weights, because getting from weights to reward goes through *sampling a token*,
and sampling is a discrete jump — there's no smooth slope to follow through it.

**Policy gradient**
The workaround. Instead of differentiating through the sampling step, you
*estimate* the gradient by sampling: generate outputs, see which scored well, and
make those more likely. It's a way of getting a usable gradient out of a process
you can't differentiate.

**REINFORCE**
The simplest policy gradient method, and the first thing you'll implement.
Sample an output, get its reward, then push the model's parameters in the
direction that makes that output more likely — scaled by the reward.

> Good output (reward 1.0) → big push toward it.
> Mediocre output (reward 0.2) → small push toward it.

**Estimator**
Any recipe for guessing a quantity from a limited sample. "Average these 8
rewards" is an estimator of the true average reward. Estimators are wrong on any
given sample; the question is *how* wrong, and in what way.

**Unbiased**
An estimator is unbiased if it's correct *on average*, over infinitely many
samples. It can still be wildly wrong on any single sample. This distinction
matters a lot in Module 01.

**Variance**
How much an estimator jumps around from sample to sample. High variance means
you need many samples before the average settles down. An estimator can be
perfectly unbiased and still useless because its variance is enormous.

> Two unbiased estimators of the same number: one returns 4.9, 5.1, 5.0. The
> other returns -40, 90, -25. Both average to 5. Only one is usable.

---

## Making it actually work

**Baseline**
A number you subtract from the reward before using it to update the model.
Without one, if all your rewards are positive, every sampled output gets pushed
*up* — including the bad ones, just less. Subtracting the average means
above-average outputs go up and below-average outputs go **down**.

> Rewards `[1.0, 0.0]`, average 0.5. Subtract it: `[+0.5, -0.5]`.
> Now the good one is reinforced and the bad one is actively discouraged.

The remarkable part, which you'll prove in Module 01: subtracting a baseline
doesn't distort the answer at all. It's free.

**Advantage**
The reward *relative to the baseline* — how much better or worse this output was
than typical. Positive advantage means "more of this," negative means "less."
This is the number that actually drives learning.

**Value network** (also **critic**)
A second neural network whose only job is to predict the baseline — "given this
prompt, what reward should I expect?" It's how PPO gets its baseline. It costs
you a whole extra model to train, and it's often wrong early on.

**Group**
Several rollouts from the **same** prompt. Ask one question 8 times, get 8
answers — that's a group of size 8, usually written `G`.

**GRPO** — Group Relative Policy Optimization
The algorithm this whole stack runs on, and the punchline of Module 01. Instead
of training a value network to *predict* the baseline, sample the prompt several
times and just *average* the rewards you get. The group's mean is the baseline.

The name is a complete description: advantage is computed **relative** to the
**group**.

> Group of 4 with rewards `[1.0, 0.0, 1.0, 0.0]`.
> Mean 0.5, so advantages are `[+1, -1, +1, -1]` (after scaling).
> The two good answers get reinforced, the two bad ones get suppressed.

**Normalizing / z-score**
Subtracting the mean and dividing by the spread, so the numbers come out
centered on zero at a consistent scale. GRPO does this to each group so that an
easy prompt and a hard prompt contribute comparably.

**PPO** — Proximal Policy Optimization
The older, more complicated relative of GRPO. Uses a value network for the
baseline. Worth knowing the name because GRPO borrows its "don't move too far in
one step" safety mechanism.

**Clipping**
That safety mechanism. It caps how much a single update is allowed to change the
policy, so one weird batch can't wreck the model.

**KL penalty**
A term that pulls the model back toward its original, pre-training self. `KL`
measures how far two probability distributions have drifted apart. Without this,
a model optimizing hard for reward can wander into degenerate text that scores
well and is otherwise useless.

**Entropy**
How spread-out the model's probabilities are. High entropy means it's still
considering many options; low entropy means it's become deterministic.

**Entropy collapse**
The failure where a model becomes so confident it stops varying its output. Once
that happens it can't discover anything new, and learning stops. A real problem
in practice.

**On-policy / off-policy**
On-policy means you train on outputs generated by the *current* model.
Off-policy means the outputs came from an older version. Large-scale training is
somewhat off-policy on purpose — waiting for perfectly fresh samples wastes
enormous amounts of hardware time.

---

## Reward design

**Reward hacking**
When the model finds a way to score highly without doing the task you actually
wanted. Not a bug in the model — a bug in your reward function. The optimizer is
doing exactly what you asked.

> Reward function: "does the correct answer 391 appear in the response?"
> Model's discovery: reply with *every number from 380 to 400*. Scores 1.0
> every time, and it's far easier than multiplying.

**Rubric**
A bundle of reward functions with weights, combined into one score. Lets you say
"correctness is worth 5× as much as formatting."

**Parser**
The piece that extracts the model's actual answer out of its rambling, so you can
check it. Turns `"Let me think... so 17 × 23 = 391.\nAnswer: 391"` into `"391"`.

**Environment**
In this stack, an environment is just three things bundled together: a dataset of
prompts, an optional parser, and a rubric. That's it. Much humbler than the word
suggests — no simulator, no physics.

**Eval**
Running a model against an environment to measure how good it is, without
training. Same object, you just don't apply the updates.

**pass@k**
"If the model gets `k` attempts, how often does at least one succeed?"
`pass@1` is plain accuracy. Useful because a model that succeeds 1 time in 8 is
in a completely different position from one that never succeeds: only the first
is trainable.

---

## Where classical RL terms come from

You'll see these in papers and videos. What they mean, and why this workbook
mostly doesn't need them:

**Bandit** (as in "multi-armed bandit")
The simplest RL setting. Several slot machine levers, each paying out randomly
with a different average. Pull one, get a payout, repeat. Crucially: **your
choice doesn't change anything about the next round.**

**Exploration vs. exploitation**
The core bandit tension. Do you pull the lever that's paid best so far
(exploit), or try an untested one that might be better (explore)? Too much
exploiting and you lock onto something mediocre forever. Too much exploring and
you waste pulls on known-bad options.

**MDP** — Markov Decision Process
The more complicated setting most RL courses teach. You're in a *state*, you take
an action, and you land in a **new state**. Think a robot walking, or a chess
game. Your action changes what you face next, so you have to reason about long
chains of consequences.

**Credit assignment**
The hard problem in an MDP: you won the chess game — which of your 40 moves
deserves the credit? Most classical RL machinery exists to answer this.

**Why this matters here:** training a language model on single-turn tasks looks
much more like the bandit than the MDP. A prompt arrives, the model answers, it
gets scored, and the next prompt is unrelated. Your answer doesn't change which
question comes next. That's why you can skip most of the MDP machinery —
there's no long chain of consequences to reason about.

(Two honest caveats. Multi-turn and tool-using settings *do* bring real
sequential structure back, which is Module 03. And within a single answer the
tokens are sequential, so there *is* a credit assignment question across them —
the common answer today is to give every token in the answer the same advantage,
which works better than it probably should.)

**Value function / Q-learning / TD learning / replay buffer / target network**
Classical RL machinery for propagating credit backwards through time in an MDP.
You'll see these in any RL course. You don't need them here, and if you find
yourself deep in TD-lambda you've wandered off this path.

---

## Infrastructure (Module 05 and beyond)

**SFT** — Supervised Fine-Tuning
Ordinary training on example answers: "here's a question, here's the right <!-- prose-ok: quoted illustration -->
response, copy this." Usually done *before* RL to get the model into reasonable
shape.

**LoRA** — Low-Rank Adaptation
Training a small number of extra parameters instead of updating the whole model.
Much cheaper, much faster, and it usually reaches slightly worse final quality.
Good for experiments.

**vLLM**
A fast serving engine for generating model output. In training it's the piece
that produces rollouts.

**FSDP** — Fully Sharded Data Parallel
A way of splitting one model across several GPUs when it won't fit on one.

**Trainer / orchestrator / inference server**
The three separate processes `prime-rl` runs. The inference server generates
rollouts, the orchestrator hands out work and scores it with your rubric, and
the trainer does the gradient updates. They're separate because generating and
training want very different things from the hardware.

**Checkpoint**
A saved snapshot of the model's weights mid-training, so a crash doesn't cost
you everything.

**Environments Hub**
Prime Intellect's public library of shareable environments. Environments are
installable Python packages; the Hub is where people publish them.
