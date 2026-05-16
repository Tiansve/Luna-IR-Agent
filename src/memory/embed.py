"""Embedding helper: remote first, deterministic hash-vector fallback.

The fallback uses signed feature hashing over whitespace tokens. It is
intentionally simple — at lab scale (<1k items) it is enough for relevance
ranking and lets the system run without any embedding model configured.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

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


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an ``(N, D)`` float32 matrix of L2-normalised embeddings."""
    if not texts:
        return np.zeros((0, config.HASH_EMBED_DIM), dtype=np.float32)
    remote = llm_client.embed(texts)
    if remote is not None:
        arr = np.asarray(remote, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms
    return np.vstack([_hash_vector(t) for t in texts])


@lru_cache(maxsize=256)
def embed_one(text: str) -> tuple[float, ...]:
    """Single-text variant with a small LRU cache for repeated queries."""
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
