"""Soul: a structured, human-readable description of who the agent is.

Two on-disk representations live side by side:

* ``data/soul.yaml`` — canonical machine format. Authoritative for
  ``evolution_log`` and for the snapshots under ``data/soul_history/``.
* ``data/soul.md``   — human-friendly format the user is meant to edit.
  Lists are bullet items under H2 headers; ``version`` and
  ``last_updated`` are simple ``key: value`` lines.

``Soul.load()`` picks whichever file has a newer mtime so manual edits to
``soul.md`` win on the next startup. ``Soul.save()`` rewrites *both*
files atomically, so reflection updates stay visible in the Markdown.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import config


SOUL_LIST_FIELDS = (
    "identity",
    "values",
    "knowledge_about_user",
    "learned_patterns",
    "open_questions",
)

# Mapping between Markdown H2 titles and dataclass field names. The title
# strings are kept user-facing (natural English). Lowercased compare on
# parse so casing in the .md doesn't have to be exact.
_MD_SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("identity", "Identity"),
    ("values", "Values"),
    ("knowledge_about_user", "What I know about the user"),
    ("learned_patterns", "Patterns I have learned"),
    ("open_questions", "Open questions"),
)
_MD_TITLE_TO_FIELD: dict[str, str] = {
    title.lower(): field for field, title in _MD_SECTION_TITLES
}
_MD_EMPTY_MARKERS = {"(empty)", "(none)", "—", "-"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Soul:
    version: int = 1
    last_updated: str = field(default_factory=_utcnow)
    identity: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    knowledge_about_user: list[str] = field(default_factory=list)
    learned_patterns: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    evolution_log: list[dict[str, Any]] = field(default_factory=list)

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "identity": list(self.identity),
            "values": list(self.values),
            "knowledge_about_user": list(self.knowledge_about_user),
            "learned_patterns": list(self.learned_patterns),
            "open_questions": list(self.open_questions),
            "evolution_log": [dict(e) for e in self.evolution_log],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Soul":
        if not isinstance(data, dict):
            raise ValueError("soul.yaml must contain a mapping at the top level")
        soul = cls(
            version=int(data.get("version", 1)),
            last_updated=str(data.get("last_updated") or _utcnow()),
        )
        for f in SOUL_LIST_FIELDS:
            raw = data.get(f) or []
            if not isinstance(raw, list):
                raise ValueError(f"soul field {f!r} must be a list")
            setattr(soul, f, [str(x) for x in raw])
        log = data.get("evolution_log") or []
        if not isinstance(log, list):
            raise ValueError("soul.evolution_log must be a list")
        soul.evolution_log = [dict(e) for e in log if isinstance(e, dict)]
        return soul

    @classmethod
    def load(cls, path: Path | None = None) -> "Soul":
        """Load a Soul.

        With an explicit ``path``: load that single file (YAML or Markdown
        chosen by extension). With no path: pick whichever of ``soul.md`` /
        ``soul.yaml`` was modified most recently so manual ``.md`` edits
        take effect on the next startup. The Markdown form does not encode
        ``evolution_log``; when ``.md`` wins, the log is copied over from
        the existing ``.yaml`` so history is never lost.
        """
        if path is not None:
            return cls._load_file(path)

        md, yml = config.SOUL_MD_PATH, config.SOUL_PATH
        if md.exists() and yml.exists():
            if md.stat().st_mtime > yml.stat().st_mtime:
                soul = cls._load_file(md)
                existing = cls._load_file(yml)
                if existing.evolution_log and not soul.evolution_log:
                    soul.evolution_log = existing.evolution_log
                soul.save()  # sync the manual edits back into yaml + md
                return soul
            return cls._load_file(yml)
        if yml.exists():
            return cls._load_file(yml)
        if md.exists():
            soul = cls._load_file(md)
            soul.save()  # materialise yaml from the .md seed
            return soul
        return cls()

    @classmethod
    def _load_file(cls, path: Path) -> "Soul":
        if not path.exists():
            return cls()
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".md":
            return cls.from_markdown(text)
        return cls.from_dict(yaml.safe_load(text) or {})

    def save(self, path: Path | None = None) -> None:
        """Persist this Soul.

        With an explicit ``path`` (used for snapshots): write a single
        file, format inferred from extension. With no path: write both
        ``soul.yaml`` (canonical) and ``soul.md`` (human-readable).
        """
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".md":
                path.write_text(self.to_markdown(), encoding="utf-8")
            else:
                with path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        self.to_dict(), f, allow_unicode=True, sort_keys=False, width=100
                    )
            return

        config.SOUL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.SOUL_PATH.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(), f, allow_unicode=True, sort_keys=False, width=100
            )
        config.SOUL_MD_PATH.write_text(self.to_markdown(), encoding="utf-8")

    def snapshot(self) -> Path:
        """Write the *current* soul to ``soul_history/v{N}.yaml``.

        Snapshots are always YAML so they round-trip exactly, including
        ``evolution_log``.
        """
        config.SOUL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        target = config.SOUL_HISTORY_DIR / f"v{self.version}.yaml"
        self.save(target)
        return target

    # ---- markdown (de)serialisation ---------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Soul",
            "",
            "<!--",
            "Edit any section. Lines beginning with \"- \" are list items.",
            "On startup the agent reloads this file if it is newer than soul.yaml,",
            "and it rewrites this file after every reflection update.",
            "The Evolution log section is read-only (managed by the reflection engine).",
            "-->",
            "",
            f"version: {self.version}",
            f"last_updated: {self.last_updated}",
        ]
        for fname, title in _MD_SECTION_TITLES:
            items: list[str] = getattr(self, fname)
            lines.append("")
            lines.append(f"## {title}")
            if not items:
                lines.append("- (empty)")
            else:
                for item in items:
                    lines.append(f"- {item}")
        if self.evolution_log:
            lines.append("")
            lines.append("## Evolution log")
            lines.append("<!-- read-only; managed by the reflection engine -->")
            # Only the last 10 entries — older changes stay in the YAML.
            for entry in self.evolution_log[-10:]:
                fv = entry.get("from_version", "?")
                tv = entry.get("to_version", "?")
                at = entry.get("at", "")
                change = entry.get("change", "")
                lines.append(f"- v{fv} → v{tv} ({at}): {change}")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> "Soul":
        """Parse a Markdown Soul. Forgiving: unknown sections are ignored.

        ``version`` defaults to 1 if missing; ``last_updated`` defaults to
        now. The Evolution log section is intentionally skipped on read
        because its structured fields can't survive the lossy Markdown
        rendering — the YAML is the source of truth for history.
        """
        soul = cls()
        current_field: str | None = None
        version_set = False

        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            # Strip HTML comments inline (rare but harmless).
            if stripped.startswith("<!--") and stripped.endswith("-->"):
                continue

            m = re.match(r"^##\s+(.+?)\s*$", line)
            if m:
                title = m.group(1).strip().lower()
                if title == "evolution log":
                    current_field = None  # skip
                else:
                    current_field = _MD_TITLE_TO_FIELD.get(title)
                continue

            if not version_set:
                mv = re.match(r"^version\s*:\s*(\d+)\s*$", stripped, re.IGNORECASE)
                if mv:
                    soul.version = int(mv.group(1))
                    version_set = True
                    continue
            mt = re.match(r"^last_updated\s*:\s*(.+?)\s*$", stripped, re.IGNORECASE)
            if mt:
                soul.last_updated = mt.group(1).strip().strip('"\'')
                continue

            if current_field and stripped.startswith("- "):
                item = stripped[2:].strip()
                if not item or item.lower() in _MD_EMPTY_MARKERS:
                    continue
                getattr(soul, current_field).append(item)

        return soul

    def copy(self) -> "Soul":
        return Soul.from_dict(copy.deepcopy(self.to_dict()))

    # ---- prompt rendering --------------------------------------------------

    def to_prompt(self) -> str:
        lines: list[str] = [
            f"# Agent Soul (version {self.version}, updated {self.last_updated})",
            "Behave consistently with this Soul. It is the stable description of",
            "who you are and what you have learned about the user.",
        ]
        sections = [
            ("Identity", self.identity),
            ("Values", self.values),
            ("What you know about the user", self.knowledge_about_user),
            ("Patterns you have learned", self.learned_patterns),
            ("Open questions to keep observing", self.open_questions),
        ]
        for title, items in sections:
            if not items:
                continue
            lines.append("")
            lines.append(f"## {title}")
            for item in items:
                lines.append(f"- {item}")
        return "\n".join(lines)

    # ---- diffing -----------------------------------------------------------

    def diff(self, other: "Soul") -> dict[str, dict[str, list[str]]]:
        """Field-wise added/removed strings between ``self`` (old) and ``other``."""
        out: dict[str, dict[str, list[str]]] = {}
        for f in SOUL_LIST_FIELDS:
            old = set(getattr(self, f))
            new = set(getattr(other, f))
            added = sorted(new - old)
            removed = sorted(old - new)
            if added or removed:
                out[f] = {"added": added, "removed": removed}
        return out
