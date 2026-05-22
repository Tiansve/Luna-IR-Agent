"""Thin wrapper around the OpenAI-compatible SDK.

Everything in this project that talks to an LLM goes through ``chat`` or
``embed`` so that the base URL, key, and model are read in one place. This
is what makes the codebase portable across Berget.AI, OpenAI, vLLM, etc.
"""
from __future__ import annotations

from typing import Any, Iterable

from openai import OpenAI

from . import config


_client: OpenAI | None = None

# Sticky flag: once the remote embedding endpoint refuses a request we stop
# attempting it for the rest of the process. This prevents mixing remote
# vectors (one dim) and local hash vectors (different dim) inside the same
# similarity matrix, which crashes at matmul time.
_embed_disabled: bool = False


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        # Wall-clock timeout per request — protects against silent stalls
        # on the provider side, which would otherwise hang the REPL.
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=1,
        )
    return _client


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.3,
) -> Any:
    """One LLM call. Returns the raw ``ChatCompletion`` object."""
    kwargs: dict[str, Any] = {
        "model": model or config.OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format is not None:
        kwargs["response_format"] = response_format
    return get_client().chat.completions.create(**kwargs)


# Embedding-call tuning. Many OpenAI-compat providers cap inputs at 512
# tokens; we hard-truncate by characters as a cheap proxy. Batches are
# kept modest to stay well under any per-request token budget.
_EMBED_MAX_CHARS = 1800           # ≈ 450 tokens, safe for 512-token caps
_EMBED_BATCH_SIZE = 32
_EMBED_MIN_BATCH = 1


def _prep_embed_inputs(texts: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in texts:
        s = t if t else " "
        if len(s) > _EMBED_MAX_CHARS:
            s = s[:_EMBED_MAX_CHARS]
        out.append(s)
    return out


def _embed_call(inputs: list[str]) -> list[list[float]]:
    resp = get_client().embeddings.create(model=config.EMBED_MODEL, input=inputs)
    return [d.embedding for d in resp.data]


def embed(texts: Iterable[str]) -> list[list[float]] | None:
    """Best-effort embedding via the configured OpenAI-compat endpoint.

    Handles three real-world failure modes seen on smaller providers:

    * Whole-batch 400 because *one* item exceeded the model's token window
      — we recursively halve the batch and finally embed the offending
      items one by one with extra-aggressive truncation.
    * Soft batch-size limits — kept under control by ``_EMBED_BATCH_SIZE``.
    * Hard endpoint outages — after a single unrecoverable failure we set
      the sticky disable flag and return ``None`` so the caller falls back
      to local hash vectors for the rest of the process.
    """
    global _embed_disabled
    if _embed_disabled or not config.EMBED_MODEL:
        return None
    inputs = _prep_embed_inputs(texts)
    if not inputs:
        return []
    try:
        return _embed_batched(inputs, _EMBED_BATCH_SIZE)
    except Exception as exc:
        _embed_disabled = True
        print(
            f"[embed] remote embedding failed ({exc!s}); "
            "falling back to hash vectors for the rest of this run."
        )
        return None


def _embed_batched(inputs: list[str], batch_size: int) -> list[list[float]]:
    """Embed ``inputs`` in groups of ``batch_size``; halve on per-batch failure."""
    out: list[list[float]] = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        out.extend(_embed_with_retry(chunk))
    return out


def _embed_with_retry(chunk: list[str]) -> list[list[float]]:
    """Embed a single batch; on 400-style errors split & retry, then truncate."""
    try:
        return _embed_call(chunk)
    except Exception as exc:
        if len(chunk) > _EMBED_MIN_BATCH:
            mid = len(chunk) // 2
            return _embed_with_retry(chunk[:mid]) + _embed_with_retry(chunk[mid:])
        # Single item still failing — try one last time with a hard truncation
        # to ~half the normal budget. If that fails, surface the exception so
        # ``embed`` can flip the sticky disable and fall back.
        salvaged = [chunk[0][: _EMBED_MAX_CHARS // 2]]
        print(f"[embed] truncating one oversized input: {exc!s}")
        return _embed_call(salvaged)


def remote_embeddings_disabled() -> bool:
    """True iff the remote embedding endpoint has been turned off."""
    return _embed_disabled or not config.EMBED_MODEL
