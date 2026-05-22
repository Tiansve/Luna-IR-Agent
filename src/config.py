"""Central configuration loaded from environment variables.

Everything that varies between machines (API keys, model names, paths,
tuning knobs) lives here. Modules read constants from this file; they
never read os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
SOUL_PATH = DATA_DIR / "soul.yaml"
SOUL_MD_PATH = DATA_DIR / "soul.md"
SOUL_HISTORY_DIR = DATA_DIR / "soul_history"
EPISODES_PATH = DATA_DIR / "episodes.jsonl"
INSIGHTS_PATH = DATA_DIR / "insights.jsonl"

OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
OPENAI_MODEL = _get("OPENAI_MODEL") or "gpt-4o-mini"
REFLECTION_MODEL = _get("REFLECTION_MODEL") or OPENAI_MODEL

EMBED_MODEL = _get("EMBED_MODEL")
BRAVE_API_KEY = _get("BRAVE_API_KEY")

REFLECT_EVERY = _get_int("REFLECT_EVERY", 3)
TOP_K_EPISODES = _get_int("TOP_K_EPISODES", 3)
TOP_K_INSIGHTS = _get_int("TOP_K_INSIGHTS", 5)
MAX_TOOL_ITERS = _get_int("MAX_TOOL_ITERS", 6)
# Seconds before an LLM request is aborted. Keeps the REPL responsive when
# the upstream provider stalls instead of returning an error.
REQUEST_TIMEOUT = _get_int("REQUEST_TIMEOUT", 60)
# Max characters returned by ``search_docs`` *per chunk* inside the tool
# response payload. Long chunks bloat the next LLM turn for no benefit —
# the agent only needs a useful excerpt, not the full passage.
TOOL_CHUNK_EXCERPT_CHARS = _get_int("TOOL_CHUNK_EXCERPT_CHARS", 320)

# Hashed-vector fallback dimensionality. Small enough to stay cheap, large
# enough that random collisions don't dominate similarity.
HASH_EMBED_DIM = 512


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOUL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
