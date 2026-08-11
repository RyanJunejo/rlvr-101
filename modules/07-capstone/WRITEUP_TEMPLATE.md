# Capstone write-up

Copy this to `WRITEUP.md` and fill it in as you go, not at the end.

---

## 1. The task

What it is, in two sentences. Include an example question and its answer.

>

**Why I expected it to be trainable.** Your stage 1 reasoning: what makes the
difficulty tunable, and where you guessed the usable band would be.

>

**Tasks I rejected, and why.**

>

---

## 2. Scoring

The components and their weights:

| component | weight | what it measures |
|---|---|---|
| | | |

**Attack suite.** How many cases, and the two that were hardest to defend.

>

**A hole that remains.** Every scoring function has one. Name yours.

>

---

## 3. Feasibility (before training)

| metric | value | n |
|---|---|---|
| pass@1 | | |
| pass@k (k = ) | | |
| baseline score | ± | |

**What the numbers made me change.** Difficulty settings you tried and rejected.

>

---

## 4. Training

| | |
|---|---|
| config | |
| group_size / batch_size | |
| steps planned / run | |
| budget set in advance | $ |
| actual cost | $ |
| wall-clock | |

**The curve.**

>

**Completion length over the run.** Did it grow?

>

**Five rollouts I read from the end of training.** What the model was actually
doing.

>

---

## 5. Results

| | before | after |
|---|---|---|
| score (± CI, n) | | |

Is the difference larger than the confidence intervals? If not, say so plainly.

>

---

## 6. What went wrong

The most useful section. Be specific.

>

---

## 7. What I'd do differently

>
