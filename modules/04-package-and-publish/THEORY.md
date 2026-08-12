# Unit 04 — Theory

> **Read this after the labs.** The labs handed you three formulas. This derives
> them, and flags one place where the formula the lab teaches is the *wrong*
> one for the job Unit 04 exists to do.
>
> Every number below was computed, not quoted.
>
> *Math renders on GitHub and in any editor with markdown math preview.*

---

## 1. Where the standard error comes from

Each question is a coin flip: the model gets it right with some unknown true
probability $p$. One question is a **Bernoulli** trial — a random variable $X$
that is $1$ with probability $p$ and $0$ otherwise.

Its mean is $p$, and its variance is:

$$
\begin{aligned}
\operatorname{Var}(X) &= \mathbb{E}[X^2] - \big(\mathbb{E}[X]\big)^2 \\
&= p - p^2 && \text{because } X^2 = X \text{ when } X \in \{0, 1\} \\
&= p(1-p)
\end{aligned}
$$

That expression peaks at $p = 0.5$ (variance $0.25$) and vanishes at $0$ and $1$. A
model that's right half the time is the noisiest thing to measure; one that's
always right or always wrong is measured exactly.

Now ask $n$ questions and average. Variances of independent terms add, and
scaling by $1/n$ scales variance by $1/n^2$:

$$
\operatorname{Var}(\hat{p}) \;=\; \frac{1}{n^2} \cdot n \cdot p(1-p) \;=\; \frac{p(1-p)}{n}
$$

Take the square root and you have the **standard error**:

$$
\mathrm{SE} \;=\; \sqrt{\frac{p(1-p)}{n}}
$$

The $\sqrt{n}$ in the denominator is the whole economics of evaluation. To halve your
uncertainty you need **four times** the questions. That's not a convention; it
falls out of variances adding while the averaging divides by $n^2$.

## 2. The interval, and a defect in what the lab teaches

The lab builds $\hat{p} \pm 1.96\,\mathrm{SE}$. That's the **Wald interval**, and the $1.96$ comes
from the Central Limit Theorem: a sum of many independent things is approximately
normal, and $95\%$ of a normal distribution sits within $1.96$ standard deviations.

"Approximately" is doing real work there, and it fails in a specific place.

Here is measured coverage — how often the interval actually contains the true
$p$, over 20,000 simulated experiments per row:

| true p | n | Wald covers | Wilson covers |
|---|---|---|---|
| 0.60 | 20 | **92.8%** | 96.2% |
| 0.60 | 200 | 94.8% | 94.8% |
| 0.95 | 20 | **63.6%** | 92.5% |
| 0.05 | 20 | **63.0%** | 92.6% |

An interval labelled "95%" that contains the truth **63.6%** of the time is not
slightly off. It is wrong in a way that will mislead you.

And look at *where* it's wrong: at $p$ near $0$ or $1$. That is precisely the
regime Unit 04 exists to detect — a saturated task ($\mathrm{pass@}1 \to 1$) or a
hopeless one ($\mathrm{pass@}1 \to 0$). **The formula the lab teaches is least trustworthy exactly where
the lab tells you to use it.**

Why it breaks: the Wald interval estimates $\mathrm{SE}$ using $\hat{p}$ in place
of the unknown $p$. When $\hat{p}$ lands at $1.0$ — which happens often at
$p = 0.95$, $n = 20$ — the plugged-in estimate gives $\mathrm{SE} = 0$, so the
interval collapses to a point and covers nothing.

The **Wilson score interval** fixes it by solving for the values of $p$ consistent
with the observation, rather than plugging in an estimate ($z = 1.96$):

$$
\frac{\hat{p} + \dfrac{z^2}{2n}}{1 + \dfrac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \dfrac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

Ugly, and it never collapses: the $z^2/2n$ term pulls the centre away from the
boundary, so even $20/20$ yields a real interval.

**What to do.** The lab keeps Wald because it's the formula everyone writes and
because seeing $\hat{p} \pm 1.96\,\mathrm{SE}$ is how you learn what an interval *is*. For real
work, and especially for the feasibility check before you rent a GPU, use Wilson.
Ten lines, and it's honest at the extremes.

This is my error, not a subtlety: I taught the interval that fails in the regime
the unit is about, and only caught it by simulating coverage while writing this.

## 3. Where the 16 comes from

The lab's `samples_needed` uses a constant with no explanation:

$$
n \;=\; \frac{16 \cdot \bar{p}\,(1 - \bar{p})}{(p_1 - p_2)^2}
$$

It's the two-proportion sample size formula. Detecting a difference requires
choosing two error rates:

- $\alpha$, the false positive rate — claiming a difference that isn't real.
  Convention $0.05$, two-sided, giving $z_{\alpha/2} = 1.960$.
- $\beta$, the false negative rate — missing a difference that is real.
  Convention $0.20$, i.e. $80\%$ **power**, giving $z_\beta = 0.842$.

The requirement is that the two rates' sampling distributions separate enough
that you can tell them apart at those error rates, which gives:

$$
n \;=\; \frac{\big(z_{\alpha/2} + z_{\beta}\big)^2 \Big[\, p_1(1-p_1) + p_2(1-p_2) \,\Big]}{(p_1 - p_2)^2}
$$

Computing the front factor:

$$
(1.960 + 0.842)^2 = 7.849
$$

and when $p_1 \approx p_2 \approx \bar{p}$ the bracket is about
$2\,\bar{p}(1-\bar{p})$, so:

$$
2 \times 7.849 = 15.6978 \;\longrightarrow\; \text{rounded to } 16
$$

The lab's constant is $2\big(z_{\alpha/2} + z_\beta\big)^2$, rounded up. Using the exact $15.6978$
changes its headline number from 135,036 to 132,486 — the rounding costs about
2%, and nobody plans an evaluation to three significant figures.

**The shape matters more than the constant.** The difference is *squared* in the
denominator, so halving the gap you want to detect quadruples the cost. Going
from a 20-point claim to a 0.5-point claim is a factor of 1,600.

And note what "80% power" concedes: even at the recommended sample size, a real
difference is missed one time in five. The convention is an economic
compromise, not a guarantee.

## 4. The pass@k estimator

You have $n$ samples for a question, $c$ of them correct. What's the probability
that a fresh batch of $k$ would contain at least one success?

Work with the complement — all $k$ draws are failures. Drawing $k$ from $n$
without replacement, the number of ways to draw only from the $n-c$ failures,
over the number of ways to draw anything:

$$
\Pr[\text{all } k \text{ fail}] \;=\; \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

So:

$$
\mathrm{pass@}k \;=\; 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Edge cases fall out rather than needing special handling. If $n - c < k$ there
aren't enough failures to fill the draw, $\binom{n-c}{k} = 0$, and $\mathrm{pass@}k$
is exactly $1$. If $c = 0$ the two binomials are equal and it's $0$.

**Verified.** Against a known truth of $p = 0.3$, $k = 5$ (true $\mathrm{pass@}5 = 0.8319$),
the estimator over 40,000 simulated draws averaged **0.8322** — bias $+0.0003$.

### Why the naive version is worse despite being unbiased

The obvious alternative: draw $k$ samples, return $1.0$ if any passed. That is
*also* unbiased. But it returns only $0.0$ or $1.0$, so a single run carries
almost no information — measured standard deviation $0.372$, against the closed
form's zero variance given the same data.

This is a specific instance of a general result. The closed form is the
conditional expectation of the naive estimator given the sufficient statistic
$c$, and the **Rao–Blackwell theorem** says that conditioning an unbiased
estimator on a sufficient statistic never increases variance and usually
decreases it.

The practical reading: **being right on average is cheap; being right on the
sample you actually have is what you're paying for.** That's the same distinction
Unit 01 drew between unbiasedness and variance, arriving from a different
direction.

## 5. Why this unit sits before the GPU unit

Sections 1 and 3 combine into an uncomfortable fact. Evaluation cost scales as
$1/\delta^2$ in the effect size $\delta$ you want to resolve, and training a model rarely moves
a benchmark by more than a few points.

So a real improvement of 2 points needs roughly

$$
n \;\approx\; \frac{16 \times 0.25}{(0.02)^2} \;=\; 10{,}000
$$

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
