# Unit 04 — Theory

> **Read this after the labs.** The labs handed you three formulas. This derives
> them, and flags one place where the formula the lab teaches is the *wrong*
> one for the job Unit 04 exists to do.
>
> Every number below was computed, not quoted.

---

## 1. Where the standard error comes from

Each question is a coin flip: the model gets it right with some unknown true
probability `p`. One question is a **Bernoulli** trial — a random variable `X`
that's 1 with probability `p` and 0 otherwise.

Its mean is `p`, and its variance is:

```
Var(X) = E[X²] − (E[X])²
       = p − p²            because X² = X when X is 0 or 1
       = p(1 − p)
```

That expression peaks at `p = 0.5` (variance 0.25) and vanishes at 0 and 1. A
model that's right half the time is the noisiest thing to measure; one that's
always right or always wrong is measured exactly.

Now ask `n` questions and average. For independent variables, variances add, and
scaling by `1/n` scales variance by `1/n²`:

```
Var(p̂) = (1/n²) · n · p(1−p) = p(1−p)/n
```

Take the square root and you have the **standard error**:

```
SE = √( p(1−p) / n )
```

The `√n` in the denominator is the whole economics of evaluation. To halve your
uncertainty you need **four times** the questions. That's not a convention; it
falls out of variances adding while the averaging divides by `n²`.

## 2. The interval, and a defect in what the lab teaches

The lab builds `p̂ ± 1.96·SE`. That's the **Wald interval**, and the 1.96 comes
from the Central Limit Theorem: a sum of many independent things is approximately
normal, and 95% of a normal distribution sits within 1.96 standard deviations.

"Approximately" is doing real work there, and it fails in a specific place.

Here is measured coverage — how often the interval actually contains the true
`p`, over 20,000 simulated experiments per row:

| true p | n | Wald covers | Wilson covers |
|---|---|---|---|
| 0.60 | 20 | **92.8%** | 96.2% |
| 0.60 | 200 | 94.8% | 94.8% |
| 0.95 | 20 | **63.6%** | 92.5% |
| 0.05 | 20 | **63.0%** | 92.6% |

An interval labelled "95%" that contains the truth **63.6%** of the time is not
slightly off. It is wrong in a way that will mislead you.

And look at *where* it's wrong: at `p` near 0 or 1. That is precisely the regime
Unit 04 exists to detect — a saturated task (`pass@1 → 1`) or a hopeless one
(`pass@1 → 0`). **The formula the lab teaches is least trustworthy exactly where
the lab tells you to use it.**

Why it breaks: the Wald interval estimates `SE` using `p̂` in place of the
unknown `p`. When `p̂` lands at 1.0 — which happens often at `p = 0.95, n = 20` —
that plugged-in estimate gives `SE = 0`, so the interval collapses to a point and
covers nothing.

The **Wilson score interval** fixes it by solving for the values of `p` consistent
with the observation, rather than plugging in an estimate:

```
            p̂ + z²/2n           z        ⎛ p̂(1−p̂)     z²   ⎞
center =  ─────────────    ±  ─────── · √⎜ ─────── + ───── ⎟
            1 + z²/n          1 + z²/n   ⎝    n       4n²   ⎠
```

Ugly, and it never collapses: the `z²/2n` term pulls the center away from the
boundary, so even 20/20 yields a real interval.

**What to do.** The lab keeps Wald because it's the formula everyone writes and
because seeing `p̂ ± 1.96·SE` is how you learn what an interval *is*. For real
work, and especially for the feasibility check before you rent a GPU, use Wilson.
Ten lines, and it's honest at the extremes.

This is my error, not a subtlety: I taught the interval that fails in the regime
the unit is about, and only caught it by simulating coverage while writing this.

## 3. Where the 16 comes from

The lab's `samples_needed` uses a constant with no explanation:

```
n = 16 · p̄(1 − p̄) / (p₁ − p₂)²
```

It's the two-proportion sample size formula. Detecting a difference requires
choosing two error rates:

- **α**, the false positive rate — claiming a difference that isn't real.
  Convention: 0.05, two-sided, giving `z_{α/2} = 1.960`.
- **β**, the false negative rate — missing a difference that is real. Convention:
  0.20, i.e. 80% **power**, giving `z_β = 0.842`.

The requirement is that the two rates' sampling distributions separate enough
that you can tell them apart at those error rates, which gives:

```
n = (z_{α/2} + z_β)² · [ p₁(1−p₁) + p₂(1−p₂) ] / (p₁ − p₂)²
```

Computing the front factor:

```
(1.960 + 0.842)² = 7.849
```

and when `p₁ ≈ p₂ ≈ p̄`, the bracket is about `2·p̄(1−p̄)`, so:

```
2 × 7.849 = 15.70   →   rounded to 16
```

The lab's constant is `2(z_{α/2} + z_β)²`, rounded up. Using the exact 15.70
changes its headline number from 135,036 to 132,486 — the rounding costs about
2%, and nobody plans an evaluation to three significant figures.

**The shape matters more than the constant.** The difference is *squared* in the
denominator, so halving the gap you want to detect quadruples the cost. Going
from a 20-point claim to a 0.5-point claim is a factor of 1,600.

And note what "80% power" concedes: even at the recommended sample size, a real
difference is missed one time in five. The convention is an economic
compromise, not a guarantee.

## 4. The pass@k estimator

You have `n` samples for a question, `c` of them correct. What's the probability
that a fresh batch of `k` would contain at least one success?

Work with the complement — all `k` draws are failures. Drawing `k` from `n`
without replacement, the number of ways to draw only from the `n − c` failures,
over the number of ways to draw anything:

```
P(all k fail) = C(n−c, k) / C(n, k)
```

So:

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

Edge cases fall out rather than needing special handling. If `n − c < k` there
aren't enough failures to fill the draw, `C(n−c, k) = 0`, and pass@k is exactly
1. If `c = 0` the two binomials are equal and it's 0.

**Verified.** Against a known truth of `p = 0.3`, `k = 5` (true pass@5 = 0.8319),
the estimator over 40,000 simulated draws averaged **0.8322** — bias `+0.0003`.

### Why the naive version is worse despite being unbiased

The obvious alternative: draw `k` samples, return 1.0 if any passed. That is
*also* unbiased. But it returns only 0.0 or 1.0, so a single run carries almost
no information — measured standard deviation 0.372 against the closed form's zero
variance given the same data.

This is a specific instance of a general result. The closed form is the
conditional expectation of the naive estimator given the sufficient statistic
`c`, and the **Rao–Blackwell theorem** says that conditioning an unbiased
estimator on a sufficient statistic never increases variance and usually
decreases it.

The practical reading: **being right on average is cheap; being right on the
sample you actually have is what you're paying for.** That's the same distinction
Unit 01 drew between unbiasedness and variance, arriving from a different
direction.

## 5. Why this unit sits before the GPU unit

Sections 1 and 3 combine into an uncomfortable fact. Evaluation cost scales as
`1/δ²` in the effect size you want to resolve, and training a model rarely moves
a benchmark by more than a few points.

So a real improvement of 2 points needs roughly `16 · 0.25 / 0.0004 ≈ 10,000`
samples per model to establish at conventional error rates. If your eval is 200
questions, you cannot detect it — and no amount of care in the training run
changes that.

This is why the feasibility check comes before the spend, and why the capstone
rubric deducts for a comparison reported without a sample size. The statistics
aren't hygiene bolted onto the end. They determine which experiments are worth
running at all.

## Sources

- **Brown, Cai & DasGupta (2001)**, *Interval Estimation for a Binomial
  Proportion* — the paper that documents the Wald interval's coverage failure in
  detail. The simulation in section 2 reproduces its central finding.
- **Wilson (1927)** — the score interval.
- **Chen et al. (2021)**, arXiv:2107.03374 — the Codex paper; appendix A gives
  the pass@k estimator.
- **Casella & Berger**, *Statistical Inference* §7.3 — Rao–Blackwell.
