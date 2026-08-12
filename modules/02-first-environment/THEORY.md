# Unit 02 — Theory

> **Read this after the reward-hacking lab.** The lab showed you a grader being
> exploited. This is why that happens, why it gets worse the harder you
> optimize, and what the literature calls each piece.
>
> *Math renders on GitHub and in any editor with markdown math preview.*

---

## 1. There are two objectives, and you only ever optimize one

Write down what you actually want:

$$
U(x) \quad \text{the true objective — how good output } x \text{ really is}
$$

$U$ lives in your head. Nobody can hand it to a computer. What you write instead
is a proxy:

$$
R(x) \quad \text{the proxy — what your scoring function returns}
$$

Training maximizes $R$. It has no access to $U$ at all. So the entire question of
reward design is: **over the region where training will actually push, does $R$
still track $U$?**

Note what that question is *not*. It isn't "is $R$ a good measurement of $U$ on
typical inputs" — your grader was fine on typical inputs; it scored 5/12 only
because seven of the twelve were atypical. It's whether $R$ tracks $U$ on the
inputs an optimizer will *seek out*, which are by construction the ones where
$R$ is highest.

## 2. Why optimization pressure is the villain

Consider what changes as training gets more effective.

Early on, the model samples near its prior. $R$ and $U$ agree there, because you
wrote $R$ by thinking about typical outputs. Training works, and the reward
curve climbs.

As training pushes further, the model reaches parts of the output space you
never considered when writing $R$. Those regions are exactly where $R$ and $U$
come apart — because you only checked the region you imagined.

And here is the part that makes it a trap rather than a nuisance: the optimizer
does not wander into those regions by accident. It **searches for them**, because
they are where $R$ is highest. The gap between proxy and truth is not a random
error the optimizer stumbles across; it is the thing the optimizer is
systematically hunting.

This is why the distinction in the lab matters so much:

| kind of flaw | what optimization does to it |
|---|---|
| **noise** — $R = U + \varepsilon$, symmetric error | averages out; more samples help |
| **exploitable** — $R > U$ somewhere | gets found, then amplified |

Noise shrinks with sampling. An exploit gets *selected for*. No amount of extra
data fixes it, because the extra data is drawn from a distribution that is
itself moving toward the exploit.

### Goodhart's law, made precise

The folk statement — "when a measure becomes a target, it ceases to be a good
measure" — is true but doesn't tell you what to do. Manheim and Garrabrant
(2018) split it into mechanisms, and three of them are visible in your lab:

**Regressional.** $R = U + \text{noise}$. Selecting the highest $R$ selects partly
for high $U$ and partly for lucky noise, so the winner's true quality is
systematically below its measured quality. This is the mildest form; it's why
your best-scoring rollout is usually a bit worse than it looks.

**Extremal.** $R$ tracks $U$ in the normal range and stops tracking it in the
tail. This is the enumeration attack: over normal replies, "does 391 appear?"
correlates with correctness. Over the reply *"here are the numbers 380 to 400"*
it correlates with nothing, and that reply is in the tail precisely because the
optimizer went looking there.

**Adversarial.** Some agent is deliberately steering toward the gap. In classic
Goodhart that's a person gaming a metric. In RL the optimizer plays that role
without any intent — gradient ascent on $R$ is a search for $R$'s maximum, and that
search is functionally adversarial toward $R$'s defects.

The lab's twelve attacks are mostly extremal, and the training dynamic that
would find them is adversarial.

## 3. Why verifiable rewards help, and where they stop

This course insists on programmatic graders. The theoretical reason:

A **learned** reward model is itself a function fit to data, so it has its own
error surface, its own tail, and its own exploitable regions — and it drifts as
the policy moves away from the distribution it was trained on. You now have two
proxies stacked, and optimization pressure attacks the composition.

A **programmatic verifier** has no learned parameters. `matches[-1] == answer`
does not drift, and its failure modes are enumerable rather than statistical.
That's a stronger position, and it's why RLVR became the dominant recipe for
math and code.

But it is not a solution to Goodhart, only a reduction in surface area. Your
fixed grader passes 12/12 and is still not "correct":

- **A lucky guess scores 1.0.** $R$ rewards outcomes; $U$ cares about reasoning.
  Distinct objectives, and the gap is unfixable within outcome supervision.
- **Answer leakage.** If the prompt contains the answer anywhere, copying it is a
  legitimate maximum of $R$ and a total failure of $U$.
- **Semantic equivalence.** `0.5` versus `1/2`. Tightening comparison creates
  false negatives; loosening it re-opens false positives. There is no setting
  that eliminates both.

That last one is worth stating as a general shape: **every dial you have trades
false positives against false negatives.** A grader that never gets fooled also
punishes correct-but-unusual answers, and false negatives are the more insidious
failure — they teach the model that being right doesn't pay.

## 4. Process versus outcome

Since outcome supervision can't distinguish luck from competence, why not score
the *reasoning*?

That's process supervision, and Lightman et al. (2023) found it beats outcome
supervision on math — a model trained on per-step feedback outperformed one
trained only on final answers.

The catch is that scoring reasoning steps needs a judge, usually a model, which
puts you back in section 3's first paragraph: a learned proxy with its own
exploitable tail. You traded a grader that can't see reasoning for one that can
be fooled about it.

Nobody has resolved this. It's the live tension behind the judges in the package
you installed and never used.

## 5. What the practice follows from

Every rule in the lab is a consequence of section 2:

**Make the model commit to a slot.** Free-form prose search means grading a
larger surface, and surface area is exactly what the optimizer explores. A
designated answer line shrinks the region $R$ has to be correct over.

**Compare exactly.** `in` is a strictly weaker predicate than `==`, so it accepts
a strictly larger set. Every extra accepted string is an extra place where $R$
can exceed $U$.

**Weight auxiliary components low.** A component that a policy can saturate
without doing the task is a region of high $R$ and low $U$ that is trivially
reachable. Keeping its weight small keeps that region below the honest maximum.

**Write the attacks first.** This is the only one that isn't a corollary — it's a
methodological point. Writing the grader first anchors you to the outputs you
imagined; writing attacks first forces you to imagine the tail before you have a
solution to defend.

**Read the rollouts.** The reward curve cannot show you Goodhart, because the
reward curve *is* $R$. Any evidence about $U$ has to come from outside — reading
the text, watching completion length, or a held-out check your grader doesn't
touch.

That last point deserves emphasis: **your monitoring must be measured in a
different currency than your training signal**, or it is monitoring the thing
being gamed.

## Sources

- **Amodei et al. (2016)**, arXiv:1606.06565 §3 — "Reward Hacking." Pre-dates
  all of this tooling and calls every shot.
- **Manheim & Garrabrant (2018)**, arXiv:1803.04585 — the four-way taxonomy of
  Goodhart's law used above.
- **Lightman et al. (2023)**, arXiv:2305.20050 — process versus outcome
  supervision.
- **Prime Intellect**, *Systematic Reward Hacking and Prime Sprints* (May 2026) —
  the same problem, from the people who maintain the stack you're using.
