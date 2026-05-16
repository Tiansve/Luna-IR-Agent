"""Extract insights from a batch of recent episodes via the reflection LLM."""
from __future__ import annotations

import json
from typing import Any

from .. import config, llm_client
from ..memory.episodic import Episode
from ..memory.semantic import Insight, InsightStore


_SYS = (
    "You are the reflection module of a learning agent. "
    "You read recent conversations and propose a short, conservative set of "
    "insights about the user and about effective answering strategies. "
    "Output strict JSON only."
)


def _episode_brief(ep: Episode) -> str:
    tools = ", ".join(c.get("name", "?") for c in ep.tool_calls) or "—"
    ans = ep.final_answer.strip().replace("\n", " ")
    if len(ans) > 400:
        ans = ans[:397] + "..."
    return f"id={ep.id} | Q: {ep.user_query.strip()} | tools: {tools} | A: {ans}"


def _existing_brief(insights: list[Insight], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"id": i.id, "content": i.content, "category": i.category, "confidence": round(i.confidence, 2)}
        for i in insights[:limit]
    ]


def extract_insights(episodes: list[Episode], store: InsightStore) -> dict[str, Any]:
    """Call the reflection LLM and apply the result to the insight store.

    Returns a diff summary suitable for logging / display.
    """
    if not episodes:
        return {"new": [], "reinforced": [], "weakened": [], "skipped": "no episodes"}

    user_payload = {
        "existing_insights": _existing_brief(store.insights),
        "recent_episodes": [_episode_brief(e) for e in episodes],
        "schema_hint": {
            "new_insights": [
                {
                    "content": "short, behaviour-level statement",
                    "category": "user_preference | user_fact | strategy | other",
                    "supporting_episode_ids": ["ep_..."],
                }
            ],
            "reinforced_insight_ids": ["ins_..."],
            "weakened_insight_ids": ["ins_..."],
        },
        "rules": [
            "Propose at most 3 new insights.",
            "Each new insight needs >= 1 supporting episode id from the batch.",
            "Reuse an existing insight id instead of restating it.",
            "Prefer patterns (how the user reacts) over surface facts.",
        ],
    }

    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        resp = llm_client.chat(
            messages,
            model=config.REFLECTION_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception as exc:
        return {"error": f"reflection LLM call failed: {exc!s}"}

    content = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {"error": f"reflection JSON parse failed: {exc!s}", "raw": content[:500]}

    valid_ep_ids = {e.id for e in episodes}
    diff: dict[str, Any] = {"new": [], "reinforced": [], "weakened": []}

    for new in parsed.get("new_insights") or []:
        if not isinstance(new, dict):
            continue
        text = str(new.get("content", "")).strip()
        if not text:
            continue
        supports = [
            str(x) for x in (new.get("supporting_episode_ids") or []) if str(x) in valid_ep_ids
        ]
        if not supports:
            continue
        ins = store.add(
            content=text,
            category=str(new.get("category", "general")),
            supporting_episodes=supports,
            confidence=0.65,
        )
        diff["new"].append({"id": ins.id, "content": ins.content})

    batch_ids = [e.id for e in episodes]
    existing_ids = {i.id for i in store.insights}
    for rid in parsed.get("reinforced_insight_ids") or []:
        rid = str(rid)
        if rid in existing_ids:
            store.reinforce(rid, batch_ids)
            diff["reinforced"].append(rid)
    for wid in parsed.get("weakened_insight_ids") or []:
        wid = str(wid)
        if wid in existing_ids:
            store.weaken(wid, batch_ids)
            diff["weakened"].append(wid)

    return diff
