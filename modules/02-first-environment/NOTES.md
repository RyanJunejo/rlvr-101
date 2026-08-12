# Unit 02 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

Scoring guide: **90+** you could teach this unit. **70–89** solid, move on.
**Below 70** reread the lecture notes — Unit 03 assumes all of this.

### The vocabulary these questions use

Here so you don't have to leave the page. Fuller entries, with examples, in
[`GLOSSARY.md`](../../GLOSSARY.md).

| term | in one line |
|---|---|
| **task** | one question's ground truth plus how to score it |
| **TaskData** | the frozen data for one question: prompt, answer |
| **trace** | the record of one attempt — every message, and its scores |
| **sampled** | a flag on a trace message: did the MODEL produce it, or did it arrive with the prompt? |
| **rollout** | one run of the model on one question. Also called a completion. |
| **taskset** | the generator of tasks; what the eval CLI resolves by name |
| **weight** | how much one scoring component counts toward the total |
| **reward hacking** | scoring highly without doing the task. A bug in your scoring, not the model. |

---

## Part A — The task object (30 pts)

**A1. (10 pts)** Name the three pieces of a v1 scoring setup and what each one
owns. Which of the three do *you* write the most of?

> 

**A2. (5 pts)** `TaskData` is frozen. Why does that matter after a run, when
you're reading `traces.jsonl`?

> 

**A3. (10 pts)** You build a trace by hand, score it, and every reward comes
back 0.0 with no error raised. What did you forget, and why does the library
treat that as correct behavior rather than a bug?

> 

**A4. (5 pts)** What error message tells you a reward method isn't `async`, and
why is it unhelpful?

> 

---

## Part B — Rewards and weights (25 pts)

**B1. (10 pts)** How does a reward method get its arguments? Why is a method
that asks only for `trace` making a useful statement about itself?

> 

**B2. (10 pts)** Multiplying every weight by 10 changes nothing about training.
Explain why (think about what GRPO does with the numbers), and then say what
*does* change training when you edit weights.

> 

**B3. (5 pts)** In the legacy API, weights lived in a list parallel to a list of
functions. What bug does putting the weight on the decorator retire?

> 

---

## Part C — The reward-hacking lab (35 pts)

**C1. (15 pts)** Pick three attacks from the suite. For each, say in one line why
`sloppy_correctness` fell for it, and which of your four rules stops it.

> 

**C2. (10 pts)** The central argument of this unit: why is an *exploitable*
scoring function categorically worse than a merely *noisy* one? Your answer must
say something about what training is doing as a process.

> 

**C3. (10 pts)** Trace the failure end to end. One reply in a group stumbles
onto "list every number from 380 to 400" and scores 1.0 while careful replies
score 0.0 on an arithmetic slip. Walk through the next few hundred training
steps, and describe what the score curve looks like while it happens.

> 

---

## Part D — Judgment (10 pts)

**D1. (5 pts)** In the real eval run, some rollouts scored 0.0 because of a 401
error, not a wrong answer. What two trace fields distinguish those cases, and
why will this matter in Unit 05?

> 

**D2. (5 pts)** Your fixed grader passes 12/12. Name two ways it can still be
fooled, and what closing each hole would cost.

> 

---

## Reflection (ungraded)

Which attack did you not see coming? Why do you think you missed it?

> 
