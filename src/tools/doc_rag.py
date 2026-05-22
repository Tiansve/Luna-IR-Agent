"""Tiny RAG over the local ``docs/`` folder.

Supported file types: ``.md``, ``.txt``, ``.pdf``. On first use every file
is read, split into overlapping character chunks, embedded, and held in
memory. The corpus is small enough that there is no value in a persistent
vector store.

PDF text is extracted with :mod:`pypdf`. Scanned (image-only) PDFs yield
empty text and are skipped with a warning — they would need OCR, which is
out of scope.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

# pypdf is chatty about non-fatal recoveries from malformed PDFs. The
# warnings appear once per affected page and have no actionable content
# for our use case — silence them at import time.
logging.getLogger("pypdf").setLevel(logging.ERROR)

from .. import config
from ..memory import embed as embed_mod


# Chunk size has to stay under the embedding model's token window. Berget's
# 1024-d embedding model caps inputs at 512 tokens; ~500 English characters
# is a safe budget (≈125 tokens) and leaves headroom for code/maths pages
# where tokenisation is denser.
CHUNK_CHARS = 500
CHUNK_OVERLAP = 100
_SUPPORTED_EXT = {".md", ".txt", ".pdf"}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 so we never crash on stray bytes.
        return path.read_text(encoding="latin-1", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        print(
            f"[doc_rag] cannot read {path.name}: pypdf is not installed "
            f"({exc!s}). Run: pip install pypdf"
        )
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # malformed / encrypted PDF
        print(f"[doc_rag] failed to open {path.name}: {exc!s}")
        return ""

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:  # pypdf is occasionally fragile
            print(f"[doc_rag] {path.name} page {i + 1}: extract failed ({exc!s})")
            txt = ""
        txt = _clean_pdf_text(txt)
        if txt:
            pages.append(txt)
    return "\n\n".join(pages)


_WS_RE = re.compile(r"[ \t]+")
_MANY_NL_RE = re.compile(r"\n{3,}")


def _clean_pdf_text(text: str) -> str:
    """Tidy common pypdf artefacts: stray hyphenation, ragged whitespace."""
    if not text:
        return ""
    # Join hyphen-broken words at line ends ("informa-\ntion" -> "information").
    text = re.sub(r"-\n(?=\w)", "", text)
    # Collapse single newlines that sit between words into spaces.
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
    text = _WS_RE.sub(" ", text)
    text = _MANY_NL_RE.sub("\n\n", text)
    return text.strip()


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return _read_text_file(path)


# Persistent cache for the embedded index. Re-embedding ~2.5k PDF chunks
# via a remote API takes 1–2 minutes; reusing the cache makes subsequent
# starts near-instant. The cache key fingerprints every input that could
# change the output (file mtime+size, chunk params, embedding model).
_CACHE_PATH = config.DATA_DIR / "doc_index.npz"


def _corpus_fingerprint(paths: list[Path]) -> str:
    parts: list[str] = [
        f"chunk_chars={CHUNK_CHARS}",
        f"chunk_overlap={CHUNK_OVERLAP}",
        f"embed_model={config.EMBED_MODEL or ''}",
        f"hash_dim={config.HASH_EMBED_DIM}",
    ]
    for p in sorted(paths):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{p.name}:missing")
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def _discover_files() -> list[Path]:
    if not config.DOCS_DIR.exists():
        return []
    return [
        p for p in sorted(config.DOCS_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXT
    ]


class _DocIndex:
    def __init__(self) -> None:
        self.chunks: list[dict[str, str]] = []  # {"source", "text"}
        self.matrix: np.ndarray = np.zeros((0, config.HASH_EMBED_DIM), dtype=np.float32)
        self.loaded: bool = False
        self.skipped: list[str] = []
        self.fingerprint: str = ""

    # ---- cache I/O --------------------------------------------------------

    def _try_load_cache(self, fingerprint: str) -> bool:
        if not _CACHE_PATH.exists():
            return False
        try:
            with np.load(_CACHE_PATH, allow_pickle=False) as data:
                cached_fp = str(data["fingerprint"].item())
                if cached_fp != fingerprint:
                    return False
                matrix = data["matrix"].astype(np.float32, copy=False)
                meta_raw = str(data["meta"].item())
        except (OSError, KeyError, ValueError) as exc:
            print(f"[doc_rag] cache unreadable, will rebuild ({exc!s}).")
            return False
        try:
            meta = json.loads(meta_raw)
            chunks = list(meta["chunks"])
            skipped = list(meta.get("skipped") or [])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[doc_rag] cache metadata corrupt, will rebuild ({exc!s}).")
            return False
        if matrix.shape[0] != len(chunks):
            return False
        if matrix.shape[0] and matrix.shape[1] != embed_mod.current_dim():
            return False  # backend dim changed since we wrote the cache
        self.chunks = chunks
        self.matrix = matrix
        self.skipped = skipped
        self.fingerprint = fingerprint
        self.loaded = True
        return True

    def _save_cache(self) -> None:
        if not self.chunks:
            return
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            meta = json.dumps(
                {"chunks": self.chunks, "skipped": self.skipped}, ensure_ascii=False
            )
            np.savez(
                _CACHE_PATH,
                matrix=self.matrix,
                fingerprint=np.array(self.fingerprint),
                meta=np.array(meta),
            )
        except OSError as exc:
            print(f"[doc_rag] could not write cache ({exc!s}); index kept in memory only.")

    # ---- build ------------------------------------------------------------

    def build(self, *, verbose: bool = True) -> None:
        files = _discover_files()
        fingerprint = _corpus_fingerprint(files)

        if self._try_load_cache(fingerprint):
            if verbose:
                print(
                    f"[doc_rag] loaded cached index: {len(self.chunks)} chunks from "
                    f"{len({c['source'] for c in self.chunks})} files."
                )
            return

        chunks: list[dict[str, str]] = []
        skipped: list[str] = []
        if verbose and files:
            print(f"[doc_rag] reading {len(files)} files...")
        for path in files:
            try:
                text = _read_file(path)
            except OSError as exc:
                print(f"[doc_rag] skipping {path.name}: {exc!s}")
                skipped.append(path.name)
                continue
            rel = str(path.relative_to(config.DOCS_DIR))
            pieces = _split(text, CHUNK_CHARS, CHUNK_OVERLAP)
            if not pieces:
                print(f"[doc_rag] {rel}: no extractable text (possibly scanned)")
                skipped.append(rel)
                continue
            for piece in pieces:
                chunks.append({"source": rel, "text": piece})

        if verbose and chunks:
            print(f"[doc_rag] embedding {len(chunks)} chunks (this can take a minute)...")
        t0 = time.time()
        matrix = (
            embed_mod.embed_texts([c["text"] for c in chunks])
            if chunks
            else np.zeros((0, config.HASH_EMBED_DIM), dtype=np.float32)
        )
        if verbose and chunks:
            print(f"[doc_rag] embedded {len(chunks)} chunks in {time.time() - t0:.1f}s.")

        self.chunks = chunks
        self.skipped = skipped
        self.matrix = matrix
        self.fingerprint = fingerprint
        self.loaded = True
        self._save_cache()


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
    # Rebuild if the embedding backend dim drifted (remote → hash fallback).
    if _index.matrix.shape[1] != embed_mod.current_dim():
        print("[doc_rag] embedding dim changed; rebuilding index.")
        _index.loaded = False
        _index.build()
        if not _index.chunks:
            return {"query": query, "results": [], "note": "no documents indexed"}
    k = max(1, min(10, int(k)))
    qvec = np.asarray(embed_mod.embed_one(query), dtype=np.float32)
    hits = embed_mod.cosine_topk(qvec, _index.matrix, k)
    excerpt_chars = max(80, config.TOOL_CHUNK_EXCERPT_CHARS)
    results = []
    for i, score in hits:
        text = _index.chunks[i]["text"]
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars].rstrip() + "..."
        results.append(
            {"source": _index.chunks[i]["source"], "score": round(score, 4), "text": text}
        )
    return {"query": query, "results": results}


def ensure_index() -> None:
    """Build (or load from cache) the doc index. Safe to call repeatedly."""
    if not _index.loaded:
        _index.build()


def reindex() -> dict[str, Any]:
    """Force a rebuild from scratch, ignoring the cache."""
    _index.loaded = False
    try:
        _CACHE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    _index.build()
    sources = sorted({c["source"] for c in _index.chunks})
    return {
        "chunks": len(_index.chunks),
        "sources": len(sources),
        "skipped": _index.skipped,
        "files": sources,
    }


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Retrieve passages from the agent's local knowledge base under docs/ "
            "(supports .md, .txt, .pdf). Use for course slides, lecture notes, "
            "or any stable reference material the user added."
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
