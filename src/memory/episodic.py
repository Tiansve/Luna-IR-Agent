"""Episodic memory: one JSONL record per past conversation.

Each Episode captures the user's query, the tool calls the agent made, and
the final answer, along with an embedding for similarity search. The store
keeps everything in RAM during a session; only writes are persisted.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from . import embed as embed_mod


@dataclass
class Episode:
    id: str
    timestamp: str
    user_query: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    user_feedback: str | None = None
    reflected: bool = False

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            id=str(data["id"]),
            timestamp=str(data["timestamp"]),
            user_query=str(data.get("user_query", "")),
            tool_calls=list(data.get("tool_calls") or []),
            final_answer=str(data.get("final_answer", "")),
            user_feedback=data.get("user_feedback"),
            reflected=bool(data.get("reflected", False)),
        )

    def summary_for_prompt(self) -> str:
        # Short rendering used when injecting past episodes into the prompt.
        tool_names = ", ".join(c.get("name", "?") for c in self.tool_calls) or "—"
        answer = self.final_answer.strip().replace("\n", " ")
        if len(answer) > 240:
            answer = answer[:237] + "..."
        return (
            f"[{self.id} @ {self.timestamp}] "
            f"Q: {self.user_query.strip()} | tools: {tool_names} | A: {answer}"
        )


class EpisodicStore:
    """Append-only JSONL with an in-memory embedding matrix."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.EPISODES_PATH
        self.episodes: list[Episode] = []
        self._matrix: np.ndarray | None = None  # built lazily
        self._load()

    # ---- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.episodes.append(Episode.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[episodic] skipping malformed line: {exc!s}")

    def _append_to_disk(self, ep: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(ep.to_jsonl() + "\n")

    def _rewrite_all(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(ep.to_jsonl() + "\n")

    # ---- public API --------------------------------------------------------

    def add(
        self,
        user_query: str,
        tool_calls: list[dict[str, Any]],
        final_answer: str,
    ) -> Episode:
        ep = Episode(
            id=f"ep_{uuid.uuid4().hex[:10]}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            user_query=user_query,
            tool_calls=tool_calls,
            final_answer=final_answer,
        )
        self.episodes.append(ep)
        self._append_to_disk(ep)
        self._matrix = None  # invalidate cache
        return ep

    def mark_reflected(self, ids: list[str]) -> None:
        idset = set(ids)
        touched = False
        for ep in self.episodes:
            if ep.id in idset and not ep.reflected:
                ep.reflected = True
                touched = True
        if touched:
            self._rewrite_all()

    def unreflected(self) -> list[Episode]:
        return [e for e in self.episodes if not e.reflected]

    def _ensure_matrix(self) -> np.ndarray:
        # Invalidate the cache if the embedding backend has switched dims
        # (e.g. remote endpoint was disabled mid-run).
        if self._matrix is not None and self._matrix.shape[1] != embed_mod.current_dim():
            self._matrix = None
        if self._matrix is None:
            texts = [self._embed_text(e) for e in self.episodes]
            self._matrix = embed_mod.embed_texts(texts)
        return self._matrix

    @staticmethod
    def _embed_text(ep: Episode) -> str:
        return f"{ep.user_query}\n{ep.final_answer}"

    def search(self, query: str, k: int) -> list[tuple[Episode, float]]:
        if not self.episodes or k <= 0:
            return []
        qvec = np.asarray(embed_mod.embed_one(query), dtype=np.float32)
        matrix = self._ensure_matrix()
        return [(self.episodes[i], score) for i, score in embed_mod.cosine_topk(qvec, matrix, k)]
