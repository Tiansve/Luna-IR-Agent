"""Offline smoke test: exercises everything that does NOT need a live LLM.

Run from the project root:
    & "C:\\Anaconda\\envs\\IR_P_env\\python.exe" -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.memory import embed as embed_mod  # noqa: E402
from src.memory.episodic import EpisodicStore  # noqa: E402
from src.memory.semantic import InsightStore  # noqa: E402
from src.soul import Soul  # noqa: E402
from src.tools import doc_rag  # noqa: E402
from src.tools.memory_tools import MemoryTools  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    # --- Soul: load, render, snapshot, diff --------------------------------
    soul = Soul.load()
    rendered = soul.to_prompt()
    assert "Agent Soul" in rendered, "Soul prompt rendering failed"
    snap = soul.snapshot()
    assert snap.exists(), "snapshot did not write"

    soul_b = soul.copy()
    soul_b.values.append("__test_value__")
    diff = soul.diff(soul_b)
    assert "values" in diff and "__test_value__" in diff["values"]["added"]
    print(f"[ok] soul v{soul.version}: render ({len(rendered)} chars), snapshot, diff")

    # --- Markdown round-trip -----------------------------------------------
    md = soul.to_markdown()
    assert "## Identity" in md and "version:" in md
    reparsed = Soul.from_markdown(md)
    assert reparsed.identity == soul.identity, "identity lost on md round-trip"
    assert reparsed.values == soul.values, "values lost on md round-trip"
    assert reparsed.open_questions == soul.open_questions
    assert reparsed.version == soul.version
    # The Evolution log is intentionally not round-tripped through .md.
    print("[ok] soul markdown round-trip preserves list fields and version")

    # --- Embeddings: fallback works, cosine_topk sane ----------------------
    vecs = embed_mod.embed_texts(["hello world", "good morning", "BM25 ranking function"])
    assert vecs.shape[0] == 3
    q = vecs[2]
    top = embed_mod.cosine_topk(q, vecs, 2)
    assert top[0][0] == 2, f"self-similarity should rank highest, got {top}"
    print(f"[ok] embeddings shape={vecs.shape}, top2={top}")

    # --- doc_rag: build + search -------------------------------------------
    summary = doc_rag.reindex()
    assert summary["chunks"] > 0, "expected at least one doc chunk"
    hit = doc_rag.search_docs("what is BM25?", k=2)
    assert "results" in hit and hit["results"], "search_docs returned no hits"
    print(
        f"[ok] doc_rag: {summary['chunks']} chunks from {summary['sources']} files"
        + (f", skipped {len(summary['skipped'])}" if summary["skipped"] else "")
    )

    # --- Episodic store: add, search, mark reflected -----------------------
    eps = EpisodicStore(path=config.DATA_DIR / "_smoke_episodes.jsonl")
    try:
        e1 = eps.add("what is bm25?", [{"name": "search_docs"}], "BM25 is a probabilistic ranker.")
        eps.add("how do I tokenize chinese?", [], "Use jieba or character n-grams.")
        results = eps.search("bm25", k=1)
        assert results and results[0][0].id == e1.id, "episodic search broken"
        eps.mark_reflected([e1.id])
        assert e1.reflected
        print("[ok] episodic store add/search/mark_reflected")
    finally:
        eps.path.unlink(missing_ok=True)

    # --- Insight store: add, reinforce, weaken, drop -----------------------
    ins_store = InsightStore(path=config.DATA_DIR / "_smoke_insights.jsonl")
    try:
        ins = ins_store.add("user prefers analogies", "user_preference", ["ep_a"], confidence=0.5)
        ins_store.reinforce(ins.id, ["ep_b"])
        ins_store.reinforce(ins.id, ["ep_c"])
        assert ins_store.by_id(ins.id).confidence > 0.5
        weak = ins_store.add("disposable", "test", ["ep_z"], confidence=0.2)
        ins_store.weaken(weak.id, ["ep_y"])
        assert ins_store.by_id(weak.id) is None, "low-confidence insight should be dropped"
        cands = ins_store.promotion_candidates()
        print(f"[ok] insight store: 1 alive, promotion_candidates={len(cands)}")
    finally:
        ins_store.path.unlink(missing_ok=True)

    # --- Memory tools (action-as-IR write to Soul) -------------------------
    tools = MemoryTools(
        episodic=EpisodicStore(path=config.DATA_DIR / "_smoke_episodes2.jsonl"),
        insights=InsightStore(path=config.DATA_DIR / "_smoke_insights2.jsonl"),
        soul=soul,
        soul_save_cb=lambda: None,
    )
    try:
        ok = tools.update_soul_note("open_questions", "smoke-test: does this round-trip?")
        assert ok.get("ok"), ok
        bad = tools.update_soul_note("identity", "trying to bypass reflection")
        assert "error" in bad, "must refuse writes to non-whitelisted fields"
        empty = tools.update_soul_note("open_questions", "")
        assert "error" in empty
        # cleanup the side-effect on the in-memory soul
        if "smoke-test: does this round-trip?" in soul.open_questions:
            soul.open_questions.remove("smoke-test: does this round-trip?")
        print("[ok] memory_tools whitelist + empty-content guard")
    finally:
        tools.episodic.path.unlink(missing_ok=True)
        tools.insights.path.unlink(missing_ok=True)

    print("\nall smoke checks passed.")


if __name__ == "__main__":
    main()
