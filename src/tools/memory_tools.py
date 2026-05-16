"""Memory-side IR tools: episode recall, insight recall, soul note write-back.

These are "action-as-IR": the agent can use them to fetch from or write into
its own memory. The soul-note action is intentionally restricted to the two
fields where the agent is *allowed* to write directly (open observations);
durable changes to identity/values still flow through the reflection engine.
"""
from __future__ import annotations

from typing import Any

from .. import config


_SOUL_WRITABLE_FIELDS = ("open_questions", "knowledge_about_user")


class MemoryTools:
    """Bundles tool callables together with the stores they touch."""

    def __init__(self, episodic, insights, soul, soul_save_cb) -> None:
        self.episodic = episodic
        self.insights = insights
        self.soul = soul
        self._soul_save_cb = soul_save_cb

    # ---- callables ---------------------------------------------------------

    def recall_episodes(self, query: str, k: int = config.TOP_K_EPISODES) -> dict[str, Any]:
        if not query or not query.strip():
            return {"error": "empty query"}
        k = max(1, min(10, int(k)))
        hits = self.episodic.search(query, k)
        return {
            "query": query,
            "results": [
                {"id": ep.id, "score": round(score, 4), "summary": ep.summary_for_prompt()}
                for ep, score in hits
            ],
        }

    def recall_insights(self, query: str, k: int = config.TOP_K_INSIGHTS) -> dict[str, Any]:
        if not query or not query.strip():
            return {"error": "empty query"}
        k = max(1, min(10, int(k)))
        hits = self.insights.search(query, k)
        return {
            "query": query,
            "results": [
                {
                    "id": ins.id,
                    "score": round(score, 4),
                    "content": ins.content,
                    "category": ins.category,
                    "confidence": round(ins.confidence, 2),
                }
                for ins, score in hits
            ],
        }

    def update_soul_note(self, field: str, content: str) -> dict[str, Any]:
        if field not in _SOUL_WRITABLE_FIELDS:
            return {
                "error": (
                    f"field {field!r} is not writable by tool. Allowed: "
                    f"{', '.join(_SOUL_WRITABLE_FIELDS)}"
                )
            }
        content = (content or "").strip()
        if not content:
            return {"error": "content is empty"}
        bucket: list[str] = getattr(self.soul, field)
        if content in bucket:
            return {"ok": True, "note": "already present", "field": field}
        bucket.append(content)
        self._soul_save_cb()
        return {"ok": True, "field": field, "added": content}

    # ---- schemas -----------------------------------------------------------

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "recall_episodes",
                    "description": (
                        "Retrieve summaries of past conversations relevant to a query. "
                        "Use to check whether the user has asked something similar before."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to recall."},
                            "k": {"type": "integer", "description": "1-10.", "default": 3},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_insights",
                    "description": (
                        "Retrieve distilled insights about the user or about effective "
                        "approaches. Use before answering to honour what was already learned."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to recall."},
                            "k": {"type": "integer", "description": "1-10.", "default": 5},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_soul_note",
                    "description": (
                        "Append a short note to a writable Soul field. Use ONLY for "
                        "'open_questions' (things to keep watching) or "
                        "'knowledge_about_user' (a freshly volunteered fact). "
                        "Do not duplicate existing notes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "enum": list(_SOUL_WRITABLE_FIELDS),
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["field", "content"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
