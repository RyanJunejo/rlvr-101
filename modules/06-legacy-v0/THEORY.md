# Unit 06 — Theory

> Short, because the material is short. This unit is about an API change, and
> API design isn't mathematics. But there is one idea in it worth naming.

---

## Bug classes, and where they can live

The two v0 problems this unit had you meet are not bugs. They're **bug
classes** — shapes of mistake that a design either permits or forbids.

**The empty-completion guard.** In v0, `completion` is a list, an aborted rollout
makes it empty, and `completion[-1]` raises. Every reward function needs the same
guard. In v1, `trace.last_reply` returns `""`.

**The parallel weights list.** In v0, `Rubric(funcs=[...], weights=[...])` keeps
two lists aligned by position. Insert a function without its weight and every
later weight silently shifts. In v1 the weight sits on the decorator.

Both fixes have the same structure: **move the invariant from something the
caller must remember into something the type system or the API enforces.**

The v0 versions are not careless code. They're perfectly good designs that place
an obligation on the caller. The v1 versions relocate that obligation. What
changed is who is responsible for remembering, and callers forget more often than
libraries do — because there are more of them, and they're each doing something
else at the time.

## The asymmetry worth remembering

Compare the two failure modes:

| | v0 empty completion | v0 misaligned weights |
|---|---|---|
| how it fails | raises `IndexError` | silently scores differently |
| what you see | "Error calling reward function" | a training run that works, wrongly |
| cost | an afternoon | a GPU day and a wrong conclusion |

The first is a **loud** failure — it stops and you go fix it. The second is
**silent**, and silence is enormously more expensive, because nothing tells you
to look.

This is the same distinction Unit 02 drew between a noisy grader and an
exploitable one, and the same one Unit 04 drew about the Wald interval — a
formula that's *wrong at the extremes* is worse than one that refuses to answer
there. A design that fails loudly is a design you can trust; one that fails
silently is one you have to be right about.

When you're choosing between two ways to write something, that's the question
worth asking. Not "which is more elegant" but "when this is wrong, will I find
out?"

## What this predicts about the future

v1 will grow its own bug classes. Two candidates already visible in this course:

- **`sampled=True`.** Forget it on a hand-built trace and every reward silently
  returns 0.0. Silent, and this course hit it while being written.
- **`async`.** Forget it and you get an error that never mentions the word
  `async`. Loud, but unhelpfully so.

Neither is a reason to prefer v0. They're a reason to expect that a v2 will
eventually absorb them, and to notice which of your own conventions rely on you
remembering something.
