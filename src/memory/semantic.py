"""Semantic memory: insights distilled from episodes by the reflection engine.

Insights live in a JSONL file. Each has a confidence in [0, 1] that the
reflection engine adjusts when later episodes support or contradict it.
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


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Insight:
    id: str
    content: str
    category: str = "general"
    confidence: float = 0.6
    supporting_episodes: list[str] = field(default_factory=list)
    contradicting_episodes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow)
    last_updated: str = field(default_factory=_utcnow)
    promoted_to_soul: bool = False

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Insight":
        return cls(
            id=str(data["id"]),
            content=str(data.get("content", "")),
            category=str(data.get("category", "general")),
            confidence=float(data.get("confidence", 0.6)),
            supporting_episodes=list(data.get("supporting_episodes") or []),
            contradicting_episodes=list(data.get("contradicting_episodes") or []),
            created_at=str(data.get("created_at") or _utcnow()),
            last_updated=str(data.get("last_updated") or _utcnow()),
            promoted_to_soul=bool(data.get("promoted_to_soul", False)),
        )


class InsightStore:
    """Insight CRUD + embedding-based recall.

    Confidence is updated additively with small step sizes so a single
    contradicting episode cannot wipe out a hard-earned belief.
    """

    CONF_DELTA_UP = 0.10
    CONF_DELTA_DOWN = 0.15
    DROP_BELOW = 0.15

    PROMOTE_CONF = 0.80
    PROMOTE_MIN_EVIDENCE = 2

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.INSIGHTS_PATH
        self.insights: list[Insight] = []
        self._matrix: np.ndarray | None = None
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
                    self.insights.append(Insight.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[insights] skipping malformed line: {exc!s}")

    def _save_all(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for ins in self.insights:
                f.write(ins.to_jsonl() + "\n")
        self._matrix = None

    # ---- public API --------------------------------------------------------

    def by_id(self, ins_id: str) -> Insight | None:
        for ins in self.insights:
            if ins.id == ins_id:
                return ins
        return None

    def add(
        self,
        content: str,
        category: str,
        supporting_episodes: list[str],
        confidence: float = 0.6,
    ) -> Insight:
        ins = Insight(
            id=f"ins_{uuid.uuid4().hex[:8]}",
            content=content.strip(),
            category=category.strip() or "general",
            confidence=max(0.0, min(1.0, confidence)),
            supporting_episodes=list(supporting_episodes),
        )
        self.insights.append(ins)
        self._save_all()
        return ins

    def reinforce(self, ins_id: str, episode_ids: list[str]) -> None:
        ins = self.by_id(ins_id)
        if ins is None:
            return
        ins.confidence = min(1.0, ins.confidence + self.CONF_DELTA_UP)
        for eid in episode_ids:
            if eid not in ins.supporting_episodes:
                ins.supporting_episodes.append(eid)
        ins.last_updated = _utcnow()
        self._save_all()

    def weaken(self, ins_id: str, episode_ids: list[str]) -> None:
        ins = self.by_id(ins_id)
        if ins is None:
            return
        ins.confidence = max(0.0, ins.confidence - self.CONF_DELTA_DOWN)
        for eid in episode_ids:
            if eid not in ins.contradicting_episodes:
                ins.contradicting_episodes.append(eid)
        ins.last_updated = _utcnow()
        # Drop only if its confidence collapsed AND it never made it into Soul.
        if ins.confidence < self.DROP_BELOW and not ins.promoted_to_soul:
            self.insights = [i for i in self.insights if i.id != ins.id]
        self._save_all()

    def mark_promoted(self, ins_ids: list[str]) -> None:
        idset = set(ins_ids)
        touched = False
        for ins in self.insights:
            if ins.id in idset and not ins.promoted_to_soul:
                ins.promoted_to_soul = True
                touched = True
        if touched:
            self._save_all()

    def promotion_candidates(self) -> list[Insight]:
        return [
            ins
            for ins in self.insights
            if not ins.promoted_to_soul
            and ins.confidence >= self.PROMOTE_CONF
            and len(ins.supporting_episodes) >= self.PROMOTE_MIN_EVIDENCE
        ]

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is not None and self._matrix.shape[1] != embed_mod.current_dim():
            self._matrix = None
        if self._matrix is None:
            texts = [ins.content for ins in self.insights]
            self._matrix = embed_mod.embed_texts(texts)
        return self._matrix

    def search(self, query: str, k: int) -> list[tuple[Insight, float]]:
        if not self.insights or k <= 0:
            return []
        qvec = np.asarray(embed_mod.embed_one(query), dtype=np.float32)
        matrix = self._ensure_matrix()
        return [(self.insights[i], score) for i, score in embed_mod.cosine_topk(qvec, matrix, k)]
