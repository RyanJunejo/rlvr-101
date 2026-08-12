# Unit 06 — Problem Set

**100 points.** Answer in your own words before consulting `solutions/`.

### The vocabulary these questions use

Here so you don't have to leave the page. Fuller entries, with examples, in
[`GLOSSARY.md`](../../GLOSSARY.md).

| term | in one line |
|---|---|
| **v0** | the original verifiers API — plain `import verifiers`, no `.v1` |
| **completion** | v0's name for the model's reply: a LIST of chat messages |
| **rubric** | v0's bundle of reward functions with a parallel weights list |
| **load_environment** | v0's packaging convention — a function with that exact name |
| **RubricGroup** | the wrapper v0 puts around your rubric, which is why `.funcs` looks empty |

---

## Part A — Reading v0 (40 pts)

**A1. (15 pts)** Map each v0 concept to its v1 equivalent: the dataset row, a
reward function, the weights, `load_environment()`, a zero-weight rubric entry.

>

**A2. (15 pts)** Why must every v0 reward function guard against an empty
`completion`? What happens without the guard, what does the library report, and
where did this responsibility move in v1?

>

**A3. (10 pts)** You open a v0 environment and `env.rubric.funcs` is an empty
list. Where did the reward functions go, and why?

>

---

## Part B — The parallel-lists bug (20 pts)

**B1. (10 pts)** Describe the exact failure when someone inserts a function into
`funcs` without touching `weights`. Why is it silent?

>

**B2. (10 pts)** v1 retired that bug by moving the weight onto the decorator.
Name one thing the v0 arrangement could still do that mattered, or argue there
is none.

>

---

## Part C — Porting (40 pts)

**C1. (15 pts)** State the port's definition of done, and explain why "the
scores look about the same" is not it.

>

**C2. (15 pts)** In your port, one piece of v0 code disappeared entirely rather
than being translated. Which, and why is deleting it correct?

>

**C3. (10 pts)** When would you run a v0 environment behind `--legacy.id`
instead of porting it? Give the tradeoff.

>

---

## Reflection (ungraded)

Both APIs live in one installed package, and the docs for each describe the
other's world imperfectly. What's your personal rule now for finding out what
an API actually does?

>
