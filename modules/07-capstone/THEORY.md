# Unit 07 — Theory

> Short. The capstone's theory is about what makes a result worth believing,
> which is mostly Unit 04 pointed at your own work.

---

## The claim you are making

A capstone produces a sentence of the form: *"training on this task improved
this model on this measure, from X to Y."*

Four things have to hold for that sentence to be true, and they fail in roughly
this order of frequency.

**1. The measure has to mean what you think.** This is Unit 02. If your grader is
exploitable, X and Y are measurements of your grader's holes, and training moved
the model along them. The reward curve cannot tell you this — it *is* the
measurement.

**2. The difference has to exceed the noise.** This is Unit 04. With small `n`,
two identical models produced a 15-point gap in 34.9% of experiments. Any before
/ after comparison without an interval is a coin flip you've labelled a finding.

**3. The improvement has to be attributable to training.** If you changed the
prompt, the difficulty, or the model between measurements, the comparison is
confounded and no amount of statistics rescues it. Change one thing.

**4. It has to survive out of sample.** A model evaluated on the questions it
trained on tells you it memorized them. Hold out a split before you start, and
don't look at it twice.

## Why the write-up asks for failures

The rubric deducts for a write-up with no failures. That isn't a stylistic
preference.

A result reported without its failures is unfalsifiable — the reader cannot tell
whether you tried one configuration and got lucky, or twenty and reported the
best. Those two histories produce identical-looking numbers and completely
different truths. The second is the **multiple comparisons** problem: run enough
variants and one will look significant by chance alone, at exactly the rate your
`α` allows.

Reporting what you tried is what makes the number you kept interpretable. It's
also the part of the write-up someone else can actually learn from, since your
failures are the map of where the task is hard.

## The honest form of a negative result

Training may not work. Your task may be too hard, too easy, or too noisy at the
budget you have.

The correct write-up of that is not an apology. It is:

> Trained for N steps at cost C. Score moved from X ± a to Y ± b (n = ...). The
> intervals overlap, so I cannot distinguish these. The feasibility check said
> pass@8 was Z, which in hindsight was marginal.

That paragraph is a contribution. It tells the next person the difficulty band,
the cost, and the check that would have predicted the outcome — which is more
than most published environments carry.

An experiment that ran correctly and found nothing is a result. An experiment
that found something you can't defend is not.
