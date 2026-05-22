"""Interactive CLI for the Soul-Driven Agent.

Slash commands:
  /soul              show the current Soul
  /soul history      list snapshot versions
  /soul diff A B     diff two snapshot versions (e.g. /soul diff 1 2)
  /soul revert N     restore version N from soul_history/
  /soul reload       re-read soul.md from disk (pick up manual edits)
  /insights          list current insights
  /episodes [N]      show last N episodes (default 5)
  /reflect           force a reflection pass over unreflected episodes
  /reindex           rebuild the docs RAG index
  /help              show this help
  /quit              exit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config
from .agent import Agent
from .soul import Soul
from .tools import doc_rag


HELP_TEXT = __doc__


def _print_soul(soul: Soul) -> None:
    print(soul.to_prompt())
    print(f"\n(version={soul.version}, last_updated={soul.last_updated})")


def _list_history() -> list[int]:
    if not config.SOUL_HISTORY_DIR.exists():
        return []
    versions: list[int] = []
    for p in config.SOUL_HISTORY_DIR.glob("v*.yaml"):
        stem = p.stem
        if stem.startswith("v") and stem[1:].isdigit():
            versions.append(int(stem[1:]))
    return sorted(versions)


def _diff_versions(a: int, b: int) -> None:
    pa = config.SOUL_HISTORY_DIR / f"v{a}.yaml"
    pb = config.SOUL_HISTORY_DIR / f"v{b}.yaml"
    if not pa.exists() or not pb.exists():
        print(f"missing snapshot(s): v{a}={pa.exists()} v{b}={pb.exists()}")
        return
    sa, sb = Soul.load(pa), Soul.load(pb)
    diff = sa.diff(sb)
    if not diff:
        print(f"v{a} and v{b} are identical (in list fields).")
        return
    for field, changes in diff.items():
        print(f"== {field} ==")
        for line in changes.get("added", []):
            print(f"  + {line}")
        for line in changes.get("removed", []):
            print(f"  - {line}")


def _reload_soul(agent: Agent) -> None:
    """Re-read soul.md (or soul.yaml — whichever is newer) into the live agent."""
    before = agent.soul.version
    new_soul = Soul.load()
    # If the user manually bumped version in soul.md we keep it; otherwise
    # we keep the running version so reflection's monotonic versioning holds.
    if new_soul.version <= before:
        new_soul.version = before
    agent.soul = new_soul
    agent.memory_tools.soul = new_soul
    agent.soul.save()  # rewrite both files so they're in lockstep
    print(
        f"reloaded Soul (now v{agent.soul.version}); "
        f"{len(agent.soul.identity)} identity / "
        f"{len(agent.soul.values)} values / "
        f"{len(agent.soul.knowledge_about_user)} user-facts / "
        f"{len(agent.soul.learned_patterns)} patterns / "
        f"{len(agent.soul.open_questions)} open-questions."
    )


def _revert(agent: Agent, target: int) -> None:
    src: Path = config.SOUL_HISTORY_DIR / f"v{target}.yaml"
    if not src.exists():
        print(f"no snapshot v{target}.")
        return
    agent.soul.snapshot()  # keep what we are about to overwrite
    restored = Soul.load(src)
    # Bump version so history stays strictly increasing.
    restored.version = max(agent.soul.version, restored.version) + 1
    restored.evolution_log.append(
        {
            "from_version": agent.soul.version,
            "to_version": restored.version,
            "change": f"manual revert to snapshot v{target}",
            "triggered_by": "user",
            "at": restored.last_updated,
        }
    )
    restored.save()
    agent.soul = restored
    agent.memory_tools.soul = restored
    print(f"reverted: now v{restored.version} (content from v{target}).")


def _handle_command(agent: Agent, raw: str) -> bool:
    """Returns True if the command was handled (no LLM call needed)."""
    parts = raw.strip().split()
    if not parts:
        return True
    head = parts[0].lower()

    if head in ("/quit", "/exit"):
        print("bye.")
        sys.exit(0)
    if head == "/help":
        print(HELP_TEXT)
        return True
    if head == "/soul":
        if len(parts) == 1:
            _print_soul(agent.soul)
            return True
        sub = parts[1].lower()
        if sub == "history":
            versions = _list_history()
            print(
                f"snapshots: {versions or '—'}; current live version: {agent.soul.version}"
            )
            return True
        if sub == "diff" and len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
            _diff_versions(int(parts[2]), int(parts[3]))
            return True
        if sub == "revert" and len(parts) == 3 and parts[2].isdigit():
            _revert(agent, int(parts[2]))
            return True
        if sub == "reload":
            _reload_soul(agent)
            return True
        print("usage: /soul | /soul history | /soul diff A B | /soul revert N | /soul reload")
        return True
    if head == "/insights":
        if not agent.insights.insights:
            print("(no insights yet)")
            return True
        for ins in agent.insights.insights:
            flag = "★" if ins.promoted_to_soul else " "
            print(
                f" {flag} {ins.id}  conf={ins.confidence:.2f}  "
                f"[{ins.category}]  {ins.content}  "
                f"(+{len(ins.supporting_episodes)} / -{len(ins.contradicting_episodes)})"
            )
        return True
    if head == "/episodes":
        n = 5
        if len(parts) == 2 and parts[1].isdigit():
            n = max(1, int(parts[1]))
        for ep in agent.episodic.episodes[-n:]:
            print(("R " if ep.reflected else "  ") + ep.summary_for_prompt())
        return True
    if head == "/reflect":
        diff = agent.force_reflect()
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        return True
    if head == "/reindex":
        summary = doc_rag.reindex()
        print(
            f"reindexed: {summary['chunks']} chunks from "
            f"{summary['sources']} files."
        )
        if summary["skipped"]:
            print(f"  skipped (no extractable text): {', '.join(summary['skipped'])}")
        return True

    if head.startswith("/"):
        print(f"unknown command: {head}. Type /help.")
        return True
    return False


def main() -> None:
    print("Soul-Driven Agent — type /help for commands, /quit to exit.\n")
    agent = Agent()
    print(f"Loaded Soul v{agent.soul.version} with "
          f"{len(agent.episodic.episodes)} episodes, "
          f"{len(agent.insights.insights)} insights.\n")

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if _handle_command(agent, raw):
            continue

        try:
            out = agent.chat(raw)
        except Exception as exc:  # don't kill the REPL on a single bad turn
            print(f"[error] {type(exc).__name__}: {exc!s}")
            continue
        print(f"\nagent> {out['answer']}\n")


if __name__ == "__main__":
    main()
