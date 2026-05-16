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


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
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


def embed(texts: Iterable[str]) -> list[list[float]] | None:
    """Best-effort embedding via the configured OpenAI-compat endpoint.

    Returns ``None`` if no embedding model is configured or the call fails.
    The caller is responsible for falling back to a local hash vector.
    """
    if not config.EMBED_MODEL:
        return None
    inputs = [t if t else " " for t in texts]
    if not inputs:
        return []
    try:
        resp = get_client().embeddings.create(model=config.EMBED_MODEL, input=inputs)
    except Exception as exc:  # network / auth / model-not-found
        print(f"[embed] remote embedding failed ({exc!s}); falling back to hash vectors.")
        return None
    return [d.embedding for d in resp.data]
