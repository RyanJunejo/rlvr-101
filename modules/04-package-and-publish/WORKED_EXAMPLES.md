# Worked examples — Unit 04

All numbers produced by running the solution files.

---

## 1. The scenario that starts the unit

```
model A:  12 / 20  =  60.0%    95% CI: [38.5%, 81.5%]
model B:  15 / 20  =  75.0%    95% CI: [56.0%, 94.0%]

intervals overlap? YES  ->  you cannot call a winner
```

B is "15 points better" and the honest conclusion is **we could not distinguish
these two models**. Look at how wide those intervals are: model A's true rate
could plausibly be anywhere from 38% to 82%.

### Where the interval comes from

```
p        = 12/20                      = 0.600
se       = sqrt(0.6 × 0.4 / 20)       = 0.1095
margin   = 1.96 × 0.1095              = 0.2147
interval = 0.600 ± 0.215              = [0.385, 0.815]
```

The `1.96` covers 95% of a normal distribution. It isn't magic — it's a choice
about how often you're willing to be wrong.

---

## 2. What sample size buys you

A model whose **true** rate is exactly 60%:

| n | 95% interval | width |
|---:|:---|---:|
| 10 | [29.6%, 90.4%] | 60.7% |
| 20 | [38.5%, 81.5%] | 42.9% |
| 50 | [46.4%, 73.6%] | 27.2% |
| 100 | [50.4%, 69.6%] | 19.2% |
| 500 | [55.7%, 64.3%] | 8.6% |
| 2000 | [57.9%, 62.1%] | 4.3% |

Compare n=50 (27.2% wide) with n=200 — four times the work — which lands at about
13.6%, exactly half.

**That's the `1/√n` rule: to halve your error bars, quadruple your samples.**
There is no way around this. It's the tax on all empirical work.

---

## 3. What your claim will cost you

Samples needed **per model** to reliably detect a difference:

| claim | samples each |
|:---|---:|
| 50% vs 70% (huge) | 97 |
| 60% vs 70% (large) | 365 |
| 65% vs 70% (moderate) | 1,405 |
| 68% vs 70% (small) | 8,557 |
| 69.5% vs 70% (tiny) | **135,036** |

The difference appears **squared** in the denominator, so halving the gap you
want to detect quadruples the cost. Going from a 20-point claim to a 0.5-point
claim costs about 1,400× more evaluation.

Read that bottom row before you next report a 0.5% improvement.

---

## 4. Two identical models "disagreeing"

Take two models with the *same* true rate of 60%. Test each on 20 questions.
Repeat the whole experiment 4,000 times and look at the gap between them:

```
median gap between IDENTICAL models:     10.0 points
gap of 15+ points occurred in:           34.9% of runs
gap of 25+ points occurred in:           12.8% of runs
```

Two models with identical true rates produced a 15-point gap in **more than a
third** of experiments — which is exactly the "result" from section 1.

> **The rule: a score without a sample size is not information.**
> Report `62% ± 7% (n=200)`, never `62%`.

---

## 5. pass@k

A model that gets a question right 20% of the time, measured from n=100 samples:

| k | pass@k |
|---:|---:|
| 1 | 20.0% |
| 2 | 36.2% |
| 4 | 59.7% |
| 8 | 84.4% |
| 16 | 98.0% |

The same model looks far more capable when allowed to try repeatedly — which is
exactly its situation during RL training, where you sample a whole group per
question.

### Why the closed form beats the obvious approach

Same 100 samples, 19 correct, estimating pass@8:

```
closed-form estimator:     0.8272     one number, computed from all the data
naive, averaged 2000×:     0.8340     agrees -- it IS unbiased
naive, standard deviation: 0.3721     <- the problem
```

The naive version (draw 8, check if any passed) returns only 0.0 or 1.0, so a
single run tells you almost nothing.

Both are unbiased. One is useful. **This is the same distinction as Unit 01's
baseline result: being right on average is not the same as being usable.**

---

## 6. The feasibility check

Before renting a GPU, ask whether groups will have any spread at all:

```
 true rate |   pass@1 |   pass@8 | verdict
------------------------------------------------------------
      0.0% |     0.0% |     0.0% | NOT trainable - never succeeds
      0.5% |     0.2% |     2.0% | marginal - needs reward shaping
      2.0% |     1.2% |     9.7% | marginal - needs reward shaping
      5.0% |     5.2% |    35.3% | trainable
     20.0% |    21.8% |    86.2% | trainable
     60.0% |    58.5% |    99.9% | trainable
     95.0% |    94.8% |   100.0% | trainable
    100.0% |   100.0% |   100.0% | NOT trainable - already saturated
```

Both ends fail for the reason Unit 03 measured: no spread within a group means no
advantage means no gradient.

The middle rows are where RL works. And notice the 5% row — a model that looks
hopeless by ordinary accuracy succeeds at least once in 8 tries a third of the
time, which is plenty of signal.

**This check costs a few hundred API calls. Skipping it costs a GPU day.**
