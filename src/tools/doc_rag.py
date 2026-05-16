"""Tiny RAG over the local ``docs/`` folder.

On first use, every ``*.md`` / ``*.txt`` file under ``docs/`` is split into
overlapping chunks, embedded, and held in memory. The corpus is small enough
that there is no value in a persistent vector store.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..memory import embed as embed_mod


CHUNK_CHARS = 600
CHUNK_OVERLAP = 120
_SUPPORTED_EXT = {".md", ".txt"}


class _DocIndex:
    def __init__(self) -> None:
        self.chunks: list[dict[str, str]] = []  # {"source", "text"}
        self.matrix: np.ndarray = np.zeros((0, config.HASH_EMBED_DIM), dtype=np.float32)
        self.loaded: bool = False

    def build(self) -> None:
        chunks: list[dict[str, str]] = []
        if config.DOCS_DIR.exists():
            for path in sorted(config.DOCS_DIR.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXT:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    print(f"[doc_rag] skipping {path}: {exc!s}")
                    continue
                rel = str(path.relative_to(config.DOCS_DIR))
                for piece in _split(text, CHUNK_CHARS, CHUNK_OVERLAP):
                    chunks.append({"source": rel, "text": piece})
        self.chunks = chunks
        self.matrix = (
            embed_mod.embed_texts([c["text"] for c in chunks])
            if chunks
            else np.zeros((0, config.HASH_EMBED_DIM), dtype=np.float32)
        )
        self.loaded = True


def _split(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        out.append(text[start : start + size])
        if start + size >= len(text):
            break
    return out


_index = _DocIndex()


def search_docs(query: str, k: int = 4) -> dict[str, Any]:
    if not query or not query.strip():
        return {"error": "empty query"}
    if not _index.loaded:
        _index.build()
    if not _index.chunks:
        return {"query": query, "results": [], "note": "no documents indexed"}
    k = max(1, min(10, int(k)))
    qvec = np.asarray(embed_mod.embed_one(query), dtype=np.float32)
    hits = embed_mod.cosine_topk(qvec, _index.matrix, k)
    results = [
        {"source": _index.chunks[i]["source"], "score": round(score, 4), "text": _index.chunks[i]["text"]}
        for i, score in hits
    ]
    return {"query": query, "results": results}


def reindex() -> int:
    """Force a rebuild and return the chunk count (used by ``/reindex``)."""
    _index.loaded = False
    _index.build()
    return len(_index.chunks)


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Retrieve passages from the agent's local Markdown/TXT knowledge base "
            "under docs/. Use for stable reference material the user added."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "k": {"type": "integer", "description": "How many passages (1-10).", "default": 4},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
