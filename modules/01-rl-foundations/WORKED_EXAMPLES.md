# Worked examples — Unit 01

Every number here was computed by running the code, not by hand-waving. You can
reproduce any of them.

Keep this open next to the exercises. When a formula feels like something you're
being told to accept, come here and watch it happen with real numbers.

---

## 1. Softmax: logits → probabilities

Softmax exponentiates each logit and divides by the total. Exponentiating makes
everything positive; dividing makes it sum to 1.

```
logits:        [0.0, 0.0, 0.0, 0.0, 0.0]
exp of each:   [1.0, 1.0, 1.0, 1.0, 1.0]        (exp(0) = 1)
sum:            5.0
probabilities: [0.2, 0.2, 0.2, 0.2, 0.2]
```

All zeros means "no opinion" — a flat, uniform distribution. That's where every
exercise starts.

A less trivial one:

```
logits:        [2.0, 1.0, -1.0]
exp of each:   [7.389, 2.718, 0.368]
sum:            10.475
probabilities: [0.705, 0.259, 0.035]
```

Notice the gaps got **stretched**. A logit gap of 1.0 (between 2.0 and 1.0)
became a probability ratio of nearly 3×. That's the exponential at work, and
it's why logits can be modest numbers while probabilities end up lopsided.

### Why we subtract the max first

In code you'll write `exp(logits - max(logits))` instead of `exp(logits)`.

This changes nothing about the answer — subtracting a constant from every logit
cancels out between the top and bottom of the fraction. But it prevents
`exp()` from overflowing to infinity on large inputs, and RL training does drive
logits up over time.

```
softmax([1000, 1001, 1002])  without the trick → exp(1000) = inf → nan
                             with the trick    → [0.090, 0.245, 0.665]  ✓
```

---

## 2. One REINFORCE update, start to finish

Setup: 5 tokens, all logits at 0, reward table `[0.0, 0.2, 0.4, 0.6, 1.0]`,
learning rate `0.1`.

**Sample.** Probabilities are all 0.2. Roll the dice — we get **token 3**.

**Score.** `reward = 0.6`.

**Find the direction.** The direction that makes token 3 more likely is
`onehot(3) − probabilities`:

```
onehot(3):       [0.0, 0.0, 0.0, 1.0, 0.0]
probabilities:   [0.2, 0.2, 0.2, 0.2, 0.2]
direction:      [-0.2, -0.2, -0.2, +0.8, -0.2]
```

Two things to notice, both of which will keep being true:

- The sampled token gets a **positive** entry; everything else gets negative.
  "More of this, less of everything else."
- The vector **sums to zero**. You're redistributing probability, not
  manufacturing it. (Verified: the actual sum is `-5.6e-17`, which is zero plus
  floating-point dust.)

**Take the step.** New logits = old + `lr × reward × direction`:

```
0.1 × 0.6 = 0.06

new logits = [0,0,0,0,0] + 0.06 × [-0.2, -0.2, -0.2, +0.8, -0.2]
           = [-0.012, -0.012, -0.012, +0.048, -0.012]
```

**Check what happened.** Convert back to probabilities:

```
before:  [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]
after:   [0.1976, 0.1976, 0.1976, 0.2098, 0.1976]
                                   ^^^^^^
                          token 3: +0.0098
```

One nudge. Token 3 went from 20.00% to 20.98%; the other four each gave up
0.24%. Thousands of these, with the dice landing on different tokens and the
high-reward ones getting bigger nudges, is the entire training process.

---

## 3. Proving `onehot − probs` is right, without doing calculus

You're being asked to trust a formula. Don't — check it.

**The idea.** A derivative is just "if I wiggle this input a tiny bit, how much
does the output move?" So wiggle it and measure. Nudge one logit up by a hair,
nudge it down by a hair, see how much `log(probability of our token)` changed,
and divide by how far you wiggled. That's the slope. Do it for all 5 logits.

This is called a **finite-difference check**, and it's the honest way to verify
any gradient.

Setup: logits `[0.5, -0.2, 1.3, 0.0, 0.7]`, our sampled token is **2**, wiggle
size `1e-6`.

```
measured by wiggling:  [-0.18017824, -0.08947387,  0.59900594, -0.10928363, -0.22007020]
the formula says:      [-0.18017824, -0.08947387,  0.59900594, -0.10928363, -0.22007020]

largest disagreement:  0.000000000176
```

They agree to ten decimal places. The formula is right, and now you know it
rather than believing it.

(The remaining `1.76e-10` is floating-point error in the wiggling, not error in
the formula. Make the wiggle smaller and it changes character — that's a
property of finite differences, not of the math.)

---

## 4. Why a baseline helps — with numbers

Reward table `[0.0, 0.2, 0.4, 0.6, 1.0]`. The average across tokens is
`2.2 / 5 = 0.44`.

**Without a baseline**, the multiplier on each update is the raw reward:

```
token:            0      1      2      3      4
multiplier:      0.0    0.2    0.4    0.6    1.0
effect:         none    UP     UP     UP     UP
```

Every token that isn't a total failure gets pushed **up**. Token 1 is a bad
answer and we're reinforcing it.

**With a baseline** of 0.44, the multiplier is `reward − 0.44`:

```
token:            0      1      2      3      4
multiplier:    -0.44  -0.24  -0.04  +0.16  +0.56
effect:         DOWN   DOWN   DOWN    UP     UP
```

Now below-average tokens are actively suppressed. Same information, far more
useful signal.

### Why it costs nothing

The worry: "doesn't subtracting a number bias the result?" No, and this is worth
seeing.

The nudge direction always **sums to zero** (section 2). Now, if you subtract a
constant `b` from every reward, the extra term you've introduced is
`b × (the direction)`, averaged over all the tokens you might sample. Because
each direction sums to zero, and because you're averaging over the same
distribution you sampled from, that whole extra term cancels to exactly zero.

Formally it's two lines of algebra. Intuitively: subtracting a constant shifts
everything equally, and "everything equally" is precisely the direction the
softmax doesn't care about.

So: strictly less noise, zero cost. It's the best deal in the subject.

### How much it matters

Exercise 3 measures this. Run it and you'll see roughly:

```
noise without a baseline:   0.243
noise with a baseline:      0.070      ← 71% less
```

Then it does something sneakier. It adds **+10 to every reward**:

```
rewards:  [10.0, 10.2, 10.4, 10.6, 11.0]
```

Nothing meaningful changed — token 4 is still best by the same margin. The best
possible model is identical. But:

```
noise without a baseline:  87.196      ← 359× worse than before
noise with a baseline:      0.070      ← unchanged
```

A completely meaningless change to the reward function made learning **359×
noisier**, purely because there was no baseline.

**Why you should care.** Suppose you write a scoring function that gives 10
points for "answered in valid JSON" and 1 point for "the answer is correct." You
have just built that +10 shift into your rewards. Almost every answer collects
the 10; the thing you actually care about is a small ripple on top. Keep your
reward components on comparable scales.

---

## 5. GRPO: a group, scored

One question, asked 4 times. Four answers come back, two right and two wrong.

```
answer:      A      B      C      D
reward:     1.0    0.0    1.0    0.0

mean = 0.5      (this is the baseline — we measured it, we didn't predict it)
spread = 0.5    (the standard deviation)
```

**Subtract the mean:**

```
advantages:  +0.5   -0.5   +0.5   -0.5
```

**Divide by the spread** to put every question on the same scale:

```
advantages:  +1.0   -1.0   +1.0   -1.0
```

A and C get reinforced, B and D get suppressed. No value network, no prediction,
no extra training. You asked four times and looked at what came back.

### The dividing-by-spread part

Why bother? Because different questions have different reward ranges. An easy
question where answers score 0.9, 0.9, 0.8, 1.0 has a tiny spread; a hard one
scoring 0.0, 1.0, 0.0, 1.0 has a big one. Without normalizing, the hard question
dominates every update just because its numbers are bigger.

Dividing by the spread means every question contributes comparably, so a single
learning rate works across a mixed batch. It's a practical convenience, not deep
theory — some variants drop it.

### The case that teaches you the most

Now try a group where every answer scored the same:

```
rewards:     0.7    0.7    0.7    0.7
mean = 0.7
advantages:  0.0    0.0    0.0    0.0
```

**Every advantage is zero. This question produces no learning whatsoever.**

Not "a little learning." None. And note it doesn't matter *what* the shared score
was:

```
rewards:  [1.0, 1.0, 1.0, 1.0]  →  advantages all 0.0   (too easy)
rewards:  [0.0, 0.0, 0.0, 0.0]  →  advantages all 0.0   (too hard)
```

A question the model always gets right and one it always gets wrong are the
**same failure**. Both burn generation compute and teach nothing.

This is why real training pipelines filter for questions where the model
succeeds *sometimes*, and why "curriculum design" mostly means keeping questions
in that band.

### The extreme version

Push it to one answer per question — a group of size 1:

```
reward:      0.8
mean = 0.8               (the mean of one number is that number)
advantage:   0.0
```

**Always zero. Forever.** GRPO with a group of 1 isn't "noisy GRPO," it's a
machine that does nothing. Exercise 4 runs it for 600 steps and the model comes
out exactly as it went in — every probability still 0.200.

That result is the sharpest way to see where GRPO's signal comes from: **not from
scores being high, only from scores being different within a group.**

---

## Reproducing these

Every number above comes out of the solution files:

```bash
uv run python solutions/01-rl-foundations/exercise_3_baseline.py
uv run python solutions/01-rl-foundations/exercise_4_grpo.py
```

If your own implementation produces different numbers, that's a real signal —
compare against these before assuming your setup is fine.
