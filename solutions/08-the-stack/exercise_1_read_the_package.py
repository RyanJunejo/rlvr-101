"""
ANSWER KEY -- Unit 08, Lab 1 — read the package as a map of the company.

An installed library is a primary source. Blog posts describe intentions;
`site-packages` contains what actually shipped. This lab has you enumerate what
`verifiers` brings with it and read the roster as evidence of where the stack is
going.

You already used the pieces this exposes:

    harness   -- the SOLVER. `null` in Unit 02 (model answers once); Unit 03's
                 Env drove turns itself. The other twelve are the interesting
                 part.
    runtime   -- WHERE a rollout executes. `subprocess` all course; the others
                 are sandboxes and remote boxes.
    judge     -- scoring by a MODEL rather than by a program. The course avoided
                 these deliberately; this is where they live.

No API key, no network. Everything comes from the installed package.

Fill in the three TODOs, then run me:

    uv run python modules/08-the-stack/exercise_1_read_the_package.py
"""

from __future__ import annotations

import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.grading import BOLD, DIM, RESET  # noqa: E402


def list_plugins(package) -> list[str]:
    """Every submodule name inside a package, sorted.

    `pkgutil.iter_modules(package.__path__)` yields module info objects with a
    `.name`. Collect the names and sort them.

    This is exactly how the library finds its own built-in harnesses -- see
    `builtin_harness_ids()` in `verifiers/v1/utils/loaders.py`. You're using the
    library's own discovery mechanism to inventory it.

    Args:
        package: an imported package (e.g. verifiers.v1.harnesses).
    Returns:
        sorted list of submodule names.
    """
    return sorted(m.name for m in pkgutil.iter_modules(package.__path__))


def runtime_kinds() -> list[str]:
    """The runtime kinds the package ships, as lowercase names.

    `verifiers.v1.runtimes` exports config classes named `<Kind>Config` --
    DockerConfig, ModalConfig, and so on. Collect the names that end in
    "Config", strip that suffix, lowercase the rest, and sort.

    Exclude the two that aren't runtime kinds: the abstract base `RuntimeConfig`
    (which would become "runtime") and `NetworkPolicyConfig` (a policy, not a
    place). Filtering by name is fine here; the point is the inventory, not
    elegance.

    Returns:
        e.g. ["docker", "modal", "prime", "subprocess"]
    """
    from verifiers.v1 import runtimes

    skip = {"RuntimeConfig", "NetworkPolicyConfig"}
    return sorted(
        n[: -len("Config")].lower()
        for n in dir(runtimes)
        if n.endswith("Config") and n not in skip
    )


def classify(harness: str) -> str:
    """Group a harness id by what kind of solver it is.

    Return exactly one of these strings:

        "trivial"   -> "null"  (no agent loop; the model just answers)
        "coding"    -> claude_code, codex, kimi_code, mini_swe_agent, openclaw,
                       terminus_2
        "browser"   -> browser_use
        "research"  -> rlm, pi, hermes_agent
        "plumbing"  -> bash, node, pool
        "unknown"   -> anything else

    The grouping is a judgement call, not an API fact -- which is the point.
    Look at the list and notice what it says: SIX of the fourteen are coding
    agents, and four of those are third-party ones (Claude Code, Codex, Kimi,
    mini-SWE-agent). A library that ships adapters for its competitors' agents
    is telling you it expects the harness to be swappable.

    Args:
        harness: a harness id.
    Returns:
        one of the group names above.
    """
    groups = {
        "trivial": {"null"},
        "coding": {"claude_code", "codex", "kimi_code", "mini_swe_agent",
                   "openclaw", "terminus_2"},
        "browser": {"browser_use"},
        "research": {"rlm", "pi", "hermes_agent"},
        "plumbing": {"bash", "node", "pool"},
    }
    for name, members in groups.items():
        if harness in members:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Written for you.
# ---------------------------------------------------------------------------

GROUP_NOTES = {
    "trivial": "the model answers once -- Unit 02's setting",
    "coding": "a terminal/editor agent solving the task",
    "browser": "drives a real browser",
    "research": "their own frontier agents",
    "plumbing": "shells and pooling other harnesses",
}


def main() -> None:
    from verifiers.v1 import harnesses, judges

    print(f"{BOLD}1. What ships in the box{RESET}\n")
    hs = list_plugins(harnesses)
    js = list_plugins(judges)
    rs = runtime_kinds()
    print(f"  harnesses ({len(hs)}): {', '.join(hs)}")
    print(f"  runtimes  ({len(rs)}): {', '.join(rs)}")
    print(f"  judges    ({len(js)}): {', '.join(js)}")

    print(f"\n{BOLD}2. The harnesses, grouped{RESET}\n")
    groups: dict[str, list[str]] = {}
    for h in hs:
        groups.setdefault(classify(h), []).append(h)
    for name in ("trivial", "coding", "browser", "research", "plumbing", "unknown"):
        if name in groups:
            print(f"  {name:<10} {len(groups[name]):>2}  {', '.join(groups[name])}")
            print(f"  {'':<10}     {DIM}{GROUP_NOTES.get(name, '')}{RESET}")

    coding = len(groups.get("coding", []))
    print(
        f"\n{DIM}  {coding} of {len(hs)} are coding agents, and most of those are other\n"
        f"  people's: Claude Code, Codex, Kimi, mini-SWE-agent. Shipping adapters\n"
        f"  for competitors' agents is a strong statement that the harness is\n"
        f"  meant to be swapped -- which is the whole reason the task definition\n"
        f"  stopped containing one (Unit 02).{RESET}"
    )

    print(f"\n{BOLD}3. Runtimes: where a rollout runs{RESET}\n")
    for r in rs:
        note = {
            "subprocess": "same machine, no isolation -- everything you've run",
            "docker": "a container per rollout",
            "modal": "someone else's serverless boxes",
            "prime": "their own sandboxes, the hosted path",
        }.get(r, "")
        print(f"  {r:<12} {DIM}{note}{RESET}")
    print(
        f"\n{DIM}  A task that runs code needs isolation, because during RL the model is\n"
        f"  searching for high-scoring inputs -- which makes it a fuzzer aimed at\n"
        f"  your machine (Unit 03 said this about eval(); it goes double here).{RESET}"
    )

    print(f"\n{BOLD}4. Judges: the door this course kept shut{RESET}\n")
    for j in js:
        note = {"reference": "compare against a reference answer",
                "rubric": "score against written criteria"}.get(j, "")
        print(f"  {j:<12} {DIM}{note}{RESET}")
    print(
        f"\n{DIM}  A judge scores with a MODEL instead of a program. That buys you tasks\n"
        f"  no program can grade -- and costs you the property the whole course is\n"
        f"  built on, because now the grader is a second thing that can be gamed.\n"
        f"  Unit 02's lab was about a grader you could read in four lines. A judge\n"
        f"  is a grader you cannot read at all.{RESET}"
    )

    print(
        f"\n{BOLD}What the inventory tells you{RESET}\n"
        "\n"
        "  Fourteen harnesses, four runtimes, two judges -- and you used exactly\n"
        "  one of each all course (null/Env, subprocess, none). That's not a gap\n"
        "  in the course; single-turn tasks on your laptop are the right place to\n"
        "  learn what a reward function does.\n"
        "\n"
        "  But read the roster as a statement of direction: long-horizon coding\n"
        "  agents, in sandboxes, scored on real work. Everything in the package\n"
        "  points the same way, and it's the same way the blog posts point."
    )


if __name__ == "__main__":
    main()
