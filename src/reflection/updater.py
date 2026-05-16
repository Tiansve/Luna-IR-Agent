"""Soul updater: promote stable insights into the Soul, conservatively.

The LLM proposes additions/modifications, but the updater enforces hard
constraints:
  * each Soul list field is capped at MAX_PER_FIELD
  * removals require strictly stronger evidence than the existing entry
    (i.e. a non-empty reason and only one removal at a time)
  * the previous Soul is always snapshotted to soul_history/ before write
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .. import config, llm_client
from ..memory.semantic import Insight, InsightStore
from ..soul import SOUL_LIST_FIELDS, Soul


MAX_PER_FIELD = 15
ALLOWED_TARGET_FIELDS = ("identity", "values", "knowledge_about_user", "learned_patterns")


_SYS = (
    "You decide whether to evolve the agent's persistent Soul. Be conservative: "
    "additions are easier than removals; do not exceed 15 items per list field; "
    "do not duplicate semantically equivalent entries. Output strict JSON."
)


def _candidate_payload(cands: list[Insight]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "content": c.content,
            "category": c.category,
            "confidence": round(c.confidence, 2),
            "supporting_count": len(c.supporting_episodes),
        }
        for c in cands
    ]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def maybe_update_soul(
    soul: Soul, insight_store: InsightStore
) -> tuple[Soul, dict[str, Any]]:
    """Return (possibly new) Soul plus a structured diff.

    Caller is responsible for persisting the returned Soul.
    """
    candidates = insight_store.promotion_candidates()
    if not candidates:
        return soul, {"skipped": "no promotion candidates"}

    payload = {
        "current_soul": soul.to_dict(),
        "candidates": _candidate_payload(candidates),
        "schema_hint": {
            "additions": [
                {"field": "values", "content": "...", "from_insight_id": "ins_..."}
            ],
            "removals": [{"field": "values", "content": "exact existing string", "reason": "..."}],
            "log_entry": "one short sentence summarising this change",
        },
        "constraints": {
            "allowed_fields": list(ALLOWED_TARGET_FIELDS),
            "max_per_field": MAX_PER_FIELD,
            "removal_policy": "only remove if the candidate evidence strictly contradicts the existing entry",
        },
    }

    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    try:
        resp = llm_client.chat(
            messages,
            model=config.REFLECTION_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception as exc:
        return soul, {"error": f"soul-updater LLM failed: {exc!s}"}

    content = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return soul, {"error": f"soul-updater JSON parse failed: {exc!s}", "raw": content[:500]}

    additions = parsed.get("additions") or []
    removals = parsed.get("removals") or []
    log_entry = str(parsed.get("log_entry") or "").strip()

    new_soul = soul.copy()
    applied_adds: list[dict[str, str]] = []
    applied_removes: list[dict[str, str]] = []
    promoted_ids: list[str] = []
    cand_ids = {c.id for c in candidates}

    # ---- additions ---------------------------------------------------------
    for add in additions:
        if not isinstance(add, dict):
            continue
        field = str(add.get("field", ""))
        text = str(add.get("content", "")).strip()
        ins_id = str(add.get("from_insight_id", ""))
        if field not in ALLOWED_TARGET_FIELDS or not text:
            continue
        if ins_id and ins_id not in cand_ids:
            # only promote real candidates
            continue
        bucket: list[str] = getattr(new_soul, field)
        if any(text.lower() == existing.lower() for existing in bucket):
            continue
        if len(bucket) >= MAX_PER_FIELD:
            continue
        bucket.append(text)
        applied_adds.append({"field": field, "content": text, "from_insight_id": ins_id})
        if ins_id:
            promoted_ids.append(ins_id)

    # ---- removals (capped + must specify reason) ---------------------------
    for rem in removals[:2]:  # never accept more than two removals at once
        if not isinstance(rem, dict):
            continue
        field = str(rem.get("field", ""))
        text = str(rem.get("content", "")).strip()
        reason = str(rem.get("reason", "")).strip()
        if field not in SOUL_LIST_FIELDS or not text or not reason:
            continue
        bucket: list[str] = getattr(new_soul, field)
        for i, existing in enumerate(bucket):
            if existing.strip().lower() == text.lower():
                del bucket[i]
                applied_removes.append({"field": field, "content": text, "reason": reason})
                break

    if not applied_adds and not applied_removes:
        return soul, {
            "skipped": "no actionable changes proposed",
            "candidates_seen": len(candidates),
        }

    new_soul.version = soul.version + 1
    new_soul.last_updated = _utcnow()
    new_soul.evolution_log.append(
        {
            "from_version": soul.version,
            "to_version": new_soul.version,
            "change": log_entry or "soul updated by reflection engine",
            "added": applied_adds,
            "removed": applied_removes,
            "at": new_soul.last_updated,
        }
    )

    # Snapshot the OLD soul (the one being replaced) before persisting new.
    soul.snapshot()
    insight_store.mark_promoted(promoted_ids)

    diff = {
        "from_version": soul.version,
        "to_version": new_soul.version,
        "additions": applied_adds,
        "removals": applied_removes,
        "log_entry": log_entry,
    }
    return new_soul, diff
