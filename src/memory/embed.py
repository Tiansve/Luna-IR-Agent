"""Embedding helper: remote first, deterministic hash-vector fallback.

The fallback uses signed feature hashing over whitespace tokens. It is
intentionally simple — at lab scale (<1k items) it is enough for relevance
ranking and lets the system run without any embedding model configured.

Backend selection is sticky for the whole process. On first use we probe
the remote endpoint with a tiny request and lock the dimension. If the
remote call later fails mid-run, :func:`llm_client.embed` sets a sticky
disable flag; subsequent calls return hash vectors. Callers that cache a
similarity matrix should re-check :func:`current_dim` between builds and
queries — see ``EpisodicStore`` and ``_DocIndex`` for the pattern.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .. import config, llm_client


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _hash_vector(text: str, dim: int = config.HASH_EMBED_DIM) -> np.ndarray:
    """Signed feature hashing: deterministic, no model required."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


_current_dim: int | None = None


def current_dim() -> int:
    """Return the dimension of vectors :func:`embed_texts` will produce now.

    Probes the remote endpoint exactly once. After the probe the answer is
    cached for the rest of the process unless :mod:`llm_client` later marks
    remote embeddings disabled, in which case we drop to the hash dim.
    """
    global _current_dim
    if _current_dim is not None and llm_client.remote_embeddings_disabled() and _current_dim != config.HASH_EMBED_DIM:
        # Remote was working, then turned off — switch dim.
        _current_dim = config.HASH_EMBED_DIM
    if _current_dim is not None:
        return _current_dim
    if llm_client.remote_embeddings_disabled():
        _current_dim = config.HASH_EMBED_DIM
        return _current_dim
    probe = llm_client.embed(["probe"])
    if probe and probe[0]:
        _current_dim = len(probe[0])
    else:
        _current_dim = config.HASH_EMBED_DIM
    return _current_dim


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an ``(N, D)`` float32 matrix of L2-normalised embeddings.

    Always produces vectors of width :func:`current_dim`. If the remote
    endpoint refuses this batch we fall back to hash vectors *and* the
    sticky flag in :mod:`llm_client` flips so future calls skip the remote.
    """
    dim = current_dim()
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if not llm_client.remote_embeddings_disabled():
        remote = llm_client.embed(texts)
        if remote is not None:
            arr = np.asarray(remote, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return arr / norms
    # Hash fallback — but the matrix dim must match current_dim, which may
    # still be the (now stale) remote dim if we already built a matrix
    # earlier. Refresh it.
    global _current_dim
    _current_dim = config.HASH_EMBED_DIM
    return np.vstack([_hash_vector(t) for t in texts])


def embed_one(text: str) -> tuple[float, ...]:
    """Single-text variant. Not LRU-cached so we never serve a stale-dim vector."""
    return tuple(embed_texts([text])[0].tolist())


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return ``[(index, score), ...]`` sorted by descending cosine score.

    Both inputs are assumed L2-normalised, so cosine == dot product.
    """
    if matrix.size == 0 or k <= 0:
        return []
    scores = matrix @ query_vec
    k = min(k, scores.shape[0])
    # argpartition for speed on large arrays, then sort the small slice.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
