# Soul-Driven Agent

A ReAct-style information-retrieval agent built for the IR Agent lab. The
twist: it treats **its own memory** as an IR problem. Conversations are
retrieved with vector search, distilled into reusable insights, and — when
those insights become stable — promoted into a human-readable, hand-editable
**Soul** that ships into every system prompt.

> 中文使用手册见 [`README.zh.md`](README.zh.md).

---

## Table of contents

1. [What this project is, and what makes it different](#1-what-this-project-is-and-what-makes-it-different)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [The three memory layers](#3-the-three-memory-layers)
4. [The reflection engine](#4-the-reflection-engine)
5. [Tools (including Action-as-IR)](#5-tools-including-action-as-ir)
6. [The Soul: `soul.md` ⇄ `soul.yaml`](#6-the-soul-soulmd--soulyaml)
7. [Embeddings: remote, with a hash fallback](#7-embeddings-remote-with-a-hash-fallback)
8. [Local doc RAG with persistent index cache](#8-local-doc-rag-with-persistent-index-cache)
9. [Installation and configuration](#9-installation-and-configuration)
10. [Running the agent](#10-running-the-agent)
11. [Slash-command reference](#11-slash-command-reference)
12. [Extending the agent](#12-extending-the-agent)
13. [Troubleshooting](#13-troubleshooting)
14. [Project layout](#14-project-layout)
15. [Improvements over OpenClaw-style gateways](#15-improvements-over-openclaw-style-gateways)

---

## 1. What this project is, and what makes it different

A vanilla RAG agent fetches passages from a static corpus and writes nothing
back. This agent does that — but it also:

* keeps an **episodic log** of every past conversation and retrieves from it
  via vector search at the start of each new turn;
* runs a **reflection pass** every *N* turns that compresses recent
  episodes into typed insights with a confidence score;
* **promotes** stable, high-confidence insights into a structured Soul
  document that defines who the agent is and what it has learned about you;
* persists every Soul change as a versioned YAML snapshot so you can
  diff history and roll back if reflection misbehaves;
* lets you **hand-edit** the Soul directly through `data/soul.md`;
* exposes a write-back tool (`update_soul_note`) that the agent can use
  to nudge two whitelisted Soul fields without going through reflection —
  this is the "**actions count as IR**" line from the assignment.

The whole system is OpenAI-compatible: switch `OPENAI_BASE_URL` and you can
run on Berget.AI, vLLM, llama.cpp, OpenAI proper, or anything else that
speaks the standard chat-completions + tool-use schema.

---

## 2. Architecture at a glance

```
                     ┌───────────────────────┐
                     │     User input        │
                     └───────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │           Context Builder  (rebuilt every turn)           │
   │   • Soul.to_prompt()       — system-prompt header         │
   │   • top-K insights         — semantic recall over JSONL   │
   │   • top-K episode digests  — semantic recall over JSONL   │
   │   • tool schemas           — five callables               │
   └───────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │   Agent loop (ReAct, capped at MAX_TOOL_ITERS steps)      │
   │                                                           │
   │   LLM ─►─ tool_calls? ─►─ dispatch ─►─ tool result ─►─┐   │
   │     ▲                                                 │   │
   │     └─────────── append message, loop ────────────────┘   │
   │                                                           │
   │   exits when the model returns a final answer with no     │
   │   tool calls, OR when MAX_TOOL_ITERS is hit               │
   └───────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │   Episode recorder  →  data/episodes.jsonl                │
   └───────────────────────────────────────────────────────────┘
                                │
              unreflected episodes ≥ REFLECT_EVERY?
                                │ yes
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │   Reflection engine                                       │
   │   ① extractor  : episodes → new / reinforced / weakened   │
   │                   insights  (LLM, JSON mode)              │
   │   ② updater    : promotion-eligible insights → Soul       │
   │                   patch     (LLM, JSON mode, gated)       │
   │   ③ snapshot   : old Soul → data/soul_history/vN.yaml     │
   └───────────────────────────────────────────────────────────┘
```

Map of the source tree to the diagram:

| Module | Role |
| --- | --- |
| `src/agent.py`           | ReAct loop, context assembly, reflection orchestration |
| `src/soul.py`            | Soul dataclass, YAML/Markdown I/O, prompt rendering, diff |
| `src/llm_client.py`      | OpenAI-compatible chat + embedding wrapper (timeouts, retries, batching, truncation) |
| `src/memory/embed.py`    | Embedding façade: probes remote dim once, falls back to hash vectors |
| `src/memory/episodic.py` | Episode store (JSONL) + lazy cosine matrix |
| `src/memory/semantic.py` | Insight store + confidence dynamics + promotion candidates |
| `src/reflection/extractor.py` | LLM call that turns episodes into insight ops |
| `src/reflection/updater.py`   | LLM call + guard-rails that mutate the Soul |
| `src/tools/doc_rag.py`   | PDF/MD/TXT RAG with a persistent npz cache |
| `src/tools/web_search.py`| Brave Web Search tool |
| `src/tools/memory_tools.py` | `recall_*` and `update_soul_note` (action-as-IR) |
| `src/main.py`            | CLI REPL + slash commands |

---

## 3. The three memory layers

```
  Conversation
       │ written every turn
       ▼
  Episodic  (raw)            data/episodes.jsonl
       │ distilled every REFLECT_EVERY turns
       ▼
  Insights  (compressed)     data/insights.jsonl
       │ promoted when stable enough
       ▼
  Soul      (stable persona) data/soul.yaml  +  data/soul.md
```

| Layer | What it holds | Lifetime | Read path |
| --- | --- | --- | --- |
| **Episodic** | one record per past turn: query, tool calls, final answer | append-only on disk | cosine similarity on a lazily-built matrix |
| **Insights** | typed `(content, category, confidence, supports)` facts the reflection engine extracted | confidence-weighted; auto-dropped below 0.15 unless already promoted | cosine similarity by content text |
| **Soul** | identity / values / user-knowledge / learned patterns / open questions / evolution log | hand-editable and version-snapshotted | rendered straight into the system prompt |

### Insight confidence dynamics (`InsightStore`)

| Event | Effect |
| --- | --- |
| Reflection marks insight as reinforced by a new batch | `confidence += 0.10` (capped at 1.0) |
| Reflection marks insight as weakened | `confidence -= 0.15` (floored at 0.0) |
| `confidence < 0.15` **and** not yet promoted to Soul | dropped from JSONL |
| `confidence ≥ 0.80` **and** ≥ 2 supporting episodes **and** not yet promoted | becomes a Soul promotion candidate |

The bias is intentional: gaining a belief is slow, losing it is faster
(reflects how user feedback should hit harder than the agent's own
generalisations), but a single bad turn cannot blow away a hard-earned
insight because the floor and the drop threshold are different.

---

## 4. The reflection engine

Triggered either automatically (when `unreflected episodes ≥ REFLECT_EVERY`,
default 3) or manually via `/reflect`. Runs in two LLM calls, both in
strict JSON mode.

### 4.1 Insight extractor (`reflection/extractor.py`)

Inputs sent to the model:

* a brief of every existing insight (`id`, `content`, `category`, `confidence`)
* a brief of every unreflected episode (id, query, tool list, answer)
* a JSON schema hint and explicit rules ("propose ≤ 3 new insights",
  "every new insight needs ≥ 1 supporting episode id from this batch",
  "reuse an existing id instead of restating")

Expected output:

```json
{
  "new_insights": [
    {"content": "...", "category": "user_preference|user_fact|strategy|other",
     "supporting_episode_ids": ["ep_..."]}
  ],
  "reinforced_insight_ids": ["ins_..."],
  "weakened_insight_ids":   ["ins_..."]
}
```

The extractor refuses to apply any operation referencing an unknown id, so
hallucinated reinforcements are silent no-ops. If the model returns
malformed JSON, the whole pass is skipped — the live agent is unaffected.

### 4.2 Soul updater (`reflection/updater.py`)

Runs only if `InsightStore.promotion_candidates()` is non-empty. Sends the
full current Soul plus the candidates and asks the model to propose:

* `additions[]` — `{field, content, from_insight_id}`
* `removals[]`  — `{field, content, reason}`
* `log_entry`   — a one-line description for the evolution log

Hard guards (enforced in code, not via prompt):

| Guard | Why |
| --- | --- |
| `field ∈ {identity, values, knowledge_about_user, learned_patterns}` | reflection cannot touch `open_questions`; that field is reserved for the human/`update_soul_note` |
| Each Soul list capped at `MAX_PER_FIELD = 15` | prevents unbounded growth |
| Duplicate content (case-insensitive) is silently skipped | no accidental restatements |
| `removals[:2]` | the model cannot delete more than two entries per pass |
| Every removal requires a non-empty `reason` | forces explicit rationale on the way out |
| Additions must reference a candidate insight id | the model can't smuggle in proposals that were not promoted |
| Old Soul is snapshotted before write | every change is reversible via `/soul revert N` |

When `additions` or `removals` is non-empty:

1. `version` is incremented by 1.
2. `last_updated` is set to now (UTC).
3. A new `evolution_log` entry is appended capturing the from/to versions,
   what was added and removed, and the model's one-line `change` description.
4. The previous Soul is snapshotted to `data/soul_history/vN.yaml`.
5. `soul.yaml` *and* `soul.md` are rewritten — the Markdown form is kept
   in lockstep so manual editors always see the current state.
6. The promoted insights are marked so they aren't re-proposed next pass.

---

## 5. Tools (including Action-as-IR)

The LLM sees five tools via the standard `tools=[...]` / `tool_choice=auto`
protocol. Three are read-side IR, one is web IR, and one is **write-side
IR (action-as-IR)**.

| Tool | Side | What it does |
| --- | --- | --- |
| `web_search(query, count)`            | external IR | Brave Search API; falls back to a clear error if `BRAVE_API_KEY` is unset |
| `search_docs(query, k)`               | local IR    | top-k cosine over the cached doc index; returns a truncated excerpt per hit |
| `recall_episodes(query, k)`           | memory IR   | top-k cosine over past episodes; returns short digests |
| `recall_insights(query, k)`           | memory IR   | top-k cosine over insights; returns content + confidence |
| `update_soul_note(field, content)`    | **action-as-IR** | append to **whitelisted** Soul fields only: `open_questions` and `knowledge_about_user`. Refuses other fields with a clear error |

The `update_soul_note` design choice matters for the rubric. The model is
allowed a fast, low-risk path to write into the very-soft fields the user
explicitly volunteered (a new fact) or the agent wants to keep watching
(an open question), but it **cannot** rewrite identity/values/patterns
without the reflection engine and its guard-rails. Durable persona
changes still go through the slow, vetted promotion path.

### Tool result hygiene

Every tool result is JSON-serialised back to the model. To prevent
context blow-up across a multi-step ReAct turn:

* `search_docs` truncates each chunk's `text` to
  `TOOL_CHUNK_EXCERPT_CHARS` characters (default 320).
* Episode digests are pre-truncated to 240 characters in the store.
* Tool errors surface as `{"error": "..."}` rather than raising —
  the model can read the error and recover instead of the loop crashing.

---

## 6. The Soul: `soul.md` ⇄ `soul.yaml`

Two on-disk representations are kept in sync:

| File | Authority for | Edited by |
| --- | --- | --- |
| `data/soul.md`   | identity, values, knowledge_about_user, learned_patterns, open_questions | **you**, with your editor of choice |
| `data/soul.yaml` | the structured `evolution_log` and snapshots | the reflection engine |

### Load resolution

`Soul.load()` looks at both files:

* both exist → pick whichever has a newer `mtime`. If `soul.md` wins, the
  `evolution_log` is copied across from the YAML (Markdown is lossy for
  that structure), then both files are rewritten in lockstep.
* only one exists → use it; the other is materialised on first save.
* neither exists → start from a blank `Soul()`.

### Save behaviour

`Soul.save()` with no path writes **both** files. With an explicit path
(used for `soul_history/vN.yaml` snapshots) the format is inferred from
the extension.

### Markdown grammar

* Top of file: `version: N` and `last_updated: ISO8601`.
* Each H2 (`## Identity`, `## Values`, ...) starts a section.
* Items beginning with `- ` become list entries; `(empty)`, `(none)`,
  `—` and `-` are recognised as placeholders.
* The `## Evolution log` section is **rendered for display only** —
  parsing skips it because its structured fields don't round-trip cleanly.
* HTML comments (`<!-- ... -->`) are stripped.
* Section titles are matched case-insensitively, and any unrecognised
  H2 is ignored, so you can add scratch notes inside the file.

### Editing workflow

```
edit data/soul.md
       │
       ▼
/soul reload   (or restart the agent)
       │
       ▼
agent's system prompt now uses the new Soul on the next turn
```

If you only want to look at the current state without restarting, run
`/soul`. If you want to compare snapshots, run `/soul diff A B`. If
reflection made a change you dislike, run `/soul revert N`.

---

## 7. Embeddings: remote, with a hash fallback

`src/memory/embed.py` is a single façade that all stores call. It has two
backends:

1. **Remote** — calls the configured `EMBED_MODEL` through the
   OpenAI-compatible `/embeddings` endpoint.
2. **Hash fallback** — deterministic signed feature hashing on whitespace
   tokens; produces `HASH_EMBED_DIM = 512` dimensional vectors with zero
   network dependencies.

Decisions made for robustness:

* **First-call probe.** `current_dim()` makes one tiny request with the
  token `"probe"` to learn the remote dimension, then caches it.
* **Sticky disable.** If a remote embedding call fails, `llm_client`
  flips an in-process flag and every later call goes straight to the
  hash backend — no per-call retry-then-fail latency.
* **Batched, self-healing calls.** `llm_client.embed` chunks input
  in batches of 32 (default). If one batch returns a 400 because a
  single item exceeded the model's token window (a real failure mode on
  Berget's 512-token model), `_embed_with_retry` recursively halves the
  batch and finally truncates the lone offending input. The whole
  process keeps working even with one weird page.
* **Per-input character cap.** Every input is hard-truncated to
  `_EMBED_MAX_CHARS = 1800` (≈ 450 tokens) before being sent. This is a
  cheap proxy for token counting and stops "one long paragraph"
  failures before they happen.
* **Cache invalidation across backends.** `EpisodicStore` and
  `InsightStore` keep their cosine matrices in memory. They check
  `embed_mod.current_dim()` before each query and rebuild if the
  backend switched dimensions mid-run (so a remote outage in the middle
  of a session doesn't produce a shape mismatch).

---

## 8. Local doc RAG with persistent index cache

`docs/` accepts `.md`, `.txt`, and `.pdf` files. The first build:

1. Walks `docs/`, extracts text per file (`pypdf` for PDFs; scanned PDFs
   are skipped with a clear log line).
2. Splits text into character chunks (`CHUNK_CHARS = 500`,
   `CHUNK_OVERLAP = 100`). The size is tuned to stay under typical
   embedding-model token windows even on dense math/code pages.
3. Embeds every chunk through the embedding façade.
4. Writes the matrix and chunk metadata to `data/doc_index.npz`.

Subsequent runs read the npz file and skip the slow path entirely. On
a corpus of 26 lecture PDFs (≈ 2500 chunks) this is the difference
between **~230 s** (first build) and **~2 s** (warm cache).

### Cache key

Every input that could change the embeddings is fingerprinted into a
BLAKE2b hash stored alongside the matrix:

* every file's `(name, size, mtime)`
* `CHUNK_CHARS`, `CHUNK_OVERLAP`
* `EMBED_MODEL`
* `HASH_EMBED_DIM`

Add a PDF, edit one, change chunk size, or switch embedding model → the
fingerprint changes → the cache is rebuilt automatically.

### Manual control

* `/reindex` — delete the cache file and rebuild from scratch.
* `ensure_index()` — called once during `Agent.__init__`, so the first
  `search_docs` mid-conversation is always fast.

---

## 9. Installation and configuration

### 9.1 Environment

The project runs inside conda env `IR_P_env` (Python 3.11). Outside that
env the dependency versions are not guaranteed.

```powershell
& "C:\Anaconda\envs\IR_P_env\python.exe" -m pip install -r requirements.txt
```

Dependencies are intentionally light: `openai`, `pyyaml`, `python-dotenv`,
`requests`, `numpy`, `pypdf`.

### 9.2 `.env`

Copy and edit:

```powershell
Copy-Item .env.example .env
```

| Variable | Required | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY`            | yes | API key for the LLM provider |
| `OPENAI_BASE_URL`           | yes | OpenAI-compatible endpoint (e.g. `https://api.berget.ai/v1`) |
| `OPENAI_MODEL`              | yes | chat model name |
| `REFLECTION_MODEL`          | no  | reflection model; defaults to `OPENAI_MODEL`. A stronger model here is recommended |
| `EMBED_MODEL`               | no  | OpenAI-compatible embedding model; if blank or unreachable, hash fallback is used |
| `BRAVE_API_KEY`             | no  | enables `web_search`; without it the tool returns a clear error |
| `REFLECT_EVERY`             | 3   | trigger reflection every N unreflected episodes |
| `TOP_K_EPISODES`            | 3   | how many past episode digests to inject into the prompt |
| `TOP_K_INSIGHTS`            | 5   | how many insights to inject |
| `MAX_TOOL_ITERS`            | 6   | upper bound on tool-call iterations per user turn |
| `REQUEST_TIMEOUT`           | 60  | seconds per LLM request before aborting |
| `TOOL_CHUNK_EXCERPT_CHARS`  | 320 | per-chunk character cap returned by `search_docs` |

---

## 10. Running the agent

```powershell
Set-Location "C:\Users\16083\Desktop\Study\IR_P\ass2\soul-agent"
& "C:\Anaconda\envs\IR_P_env\python.exe" -m src.main
```

First startup will spend time embedding everything under `docs/`. After
that the index is cached and subsequent starts are near-instant.

Offline self-check (no LLM calls):

```powershell
& "C:\Anaconda\envs\IR_P_env\python.exe" -m scripts.smoke_test
```

Sample turn:

```
you> Briefly compare BM25 and dense retrieval.
  [tool] recall_insights({"query": "BM25 vs dense"}) -> {"results": []}
  [tool] search_docs({"query": "BM25 dense retrieval"}) -> {"results": [...]}

agent> BM25 wins on rare-token queries (exact lexical match); dense wins on
       paraphrase. A hybrid of the two with reciprocal rank fusion usually
       beats either alone. (Source: ir_basics.md.)

[reflect] 3 new episodes -> extracting insights...
[reflect] insight diff: {"new":[{"id":"ins_3f1a","content":"User prefers ..."}],"reinforced":[],"weakened":[]}
[reflect] Soul unchanged ({"skipped":"no promotion candidates"})
```

---

## 11. Slash-command reference

| Command | What it does |
| --- | --- |
| `/help`               | print this list |
| `/soul`               | show the current Soul, rendered |
| `/soul history`       | list snapshot version numbers in `data/soul_history/` |
| `/soul diff A B`      | show added/removed list items between two snapshots |
| `/soul revert N`      | restore snapshot vN; bumps version monotonically and logs the revert in `evolution_log` |
| `/soul reload`        | re-read `soul.md` (or `soul.yaml` — newer wins) into the running agent |
| `/insights`           | list every insight with `id`, `confidence`, `category`, support/contradict counts; `★` marks promoted ones |
| `/episodes [N]`       | show last N episode digests (default 5); `R` prefix = reflected |
| `/reflect`            | force a reflection pass over all unreflected episodes |
| `/reindex`            | delete the doc index cache and rebuild |
| `/quit`, `/exit`      | exit (Ctrl-C also works) |

---

## 12. Extending the agent

### 12.1 Add a tool

1. Create `src/tools/<your_tool>.py` with:
   * a plain function returning a JSON-serialisable dict;
   * a module-level `SCHEMA: dict` in OpenAI tool format.
2. In `src/agent.py`, register both inside `Agent.__init__`:

   ```python
   from .tools import your_tool
   self._dispatch["your_tool"] = lambda **kw: your_tool.your_tool(**kw)
   self._schemas.append(your_tool.SCHEMA)
   ```

3. Restart. The model will discover it through the standard tool schema
   exposed in every chat completion.

### 12.2 Swap the embedding backend

`src/memory/embed.py` is the only place that needs to change. Implementing
`embed_texts(list[str]) -> np.ndarray` with a different backend (say
`sentence-transformers`) and a stable `current_dim()` is enough; all
downstream stores will pick it up. The doc index cache will rebuild
automatically because `EMBED_MODEL` is part of the fingerprint.

### 12.3 Tune reflection aggressiveness

* Faster Soul evolution: lower `REFLECT_EVERY`, lower
  `InsightStore.PROMOTE_CONF`, raise `CONF_DELTA_UP`.
* More conservative: do the opposite.

### 12.4 Multi-user

Currently single-user. Key `SOUL_PATH`, `EPISODES_PATH`, `INSIGHTS_PATH`,
and the cache file by user id and thread the id through `Agent`.

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RuntimeError: OPENAI_API_KEY is not set` | no `.env` or empty key | copy `.env.example` and fill it |
| `[embed] remote embedding failed (... 400 ... maximum context length ...)` | a chunk exceeded the embedding model's token window | already handled — the retry path truncates the offending input. If it keeps happening, drop `CHUNK_CHARS` further |
| First `search_docs` feels frozen for a minute | first-time embedding of all PDFs | wait; it only happens once. The `data/doc_index.npz` cache makes subsequent runs ~100× faster |
| `brave request failed: ...` | no/invalid `BRAVE_API_KEY` | leave blank and the agent simply won't use web search |
| `reflection JSON parse failed` | the chat model returned non-JSON | one bad pass is harmless. Use a stronger `REFLECTION_MODEL` if it persists |
| `unknown tool 'xxx'` in logs | LLM hallucinated a tool name | already handled — the dispatcher returns an error to the model so it can recover |
| Soul never evolves | not enough unreflected episodes, or insights aren't reaching `confidence ≥ 0.8` | run `/reflect` manually, or temporarily lower `InsightStore.PROMOTE_CONF` |
| Terminal shows mojibake (Chinese/Swedish) | PowerShell stdout is not UTF-8 | `$env:PYTHONIOENCODING = "utf-8"` before launching |
| Loop hits `MAX_TOOL_ITERS` | model can't converge | bump it slightly, or improve the system prompt — the bound exists to stop runaway loops, not as a soft cap |

---

## 14. Project layout

```
soul-agent/
├── README.md                   # this file
├── README.zh.md                # detailed Chinese manual
├── requirements.txt
├── .env.example
├── data/                       # runtime state, mostly auto-generated
│   ├── soul.md                 # human-editable Soul (primary entry point)
│   ├── soul.yaml               # machine-canonical Soul (authoritative evolution_log)
│   ├── soul_history/           # vN.yaml snapshots written before every Soul update
│   ├── episodes.jsonl          # episodic memory (one line per turn)
│   ├── insights.jsonl          # semantic memory (one line per insight)
│   └── doc_index.npz           # cached doc-RAG matrix + chunk metadata
├── docs/                       # local RAG corpus (.md, .txt, .pdf)
├── scripts/
│   └── smoke_test.py           # exercises every offline code path
└── src/
    ├── __init__.py
    ├── config.py               # .env → constants
    ├── llm_client.py           # OpenAI-compatible chat + embed wrapper
    ├── soul.py                 # Soul dataclass + YAML/Markdown I/O
    ├── agent.py                # ReAct loop + reflection orchestration
    ├── main.py                 # CLI REPL
    ├── memory/
    │   ├── embed.py            # remote/hash embedding façade
    │   ├── episodic.py
    │   └── semantic.py
    ├── reflection/
    │   ├── extractor.py        # episodes → insights
    │   └── updater.py          # insights → Soul (gated)
    └── tools/
        ├── doc_rag.py          # PDF/MD/TXT RAG + persistent index cache
        ├── web_search.py       # Brave Search
        └── memory_tools.py     # recall_* + update_soul_note (action-as-IR)
```

What ships to a server vs. what stays local:

* **ship**: `src/`, `requirements.txt`, `.env.example`, `README*.md`,
  `docs/`, `data/soul.md` (optional starter persona).
* **don't ship**: `.env`, `data/episodes.jsonl`, `data/insights.jsonl`,
  `data/soul_history/`, `data/doc_index.npz`, `__pycache__/`. These are
  local experiment state.

---

## 15. Improvements over OpenClaw-style gateways

| Dimension | OpenClaw-style baseline | This project |
| --- | --- | --- |
| Memory layers              | single KV / RAG slot                  | three layers with distinct timescales |
| Persistence                | session cache                          | JSONL + YAML + Markdown + versioned snapshots |
| Self-update                | external/manual                        | built-in reflection engine, automatic |
| Interpretability of state  | opaque vectors                         | Markdown the user can read and edit |
| User intervention          | hard                                   | edit `soul.md`, `/soul revert N`, `/soul reload` |
| Drift protection           | not addressed                          | field whitelist, per-field caps, reason-required removals, snapshot-before-write, manual revert |
| Action-as-IR               | not explicit                           | `update_soul_note` exposes memory writes as first-class tool calls |

Academic touchpoints worth citing in the report: *Generative Agents*
(Park et al., 2023) for the memory-stream + reflection idea, *MemGPT*
(Packer et al., 2023) for hierarchical paging, *ReAct* (Yao et al., 2022)
for the loop, and *Constitutional AI* for the spirit of the `values`
field.

---

*Designed and built for the IR Agent assignment in the Uppsala IR course.
The Soul is meant to grow, but you are always in charge of it.*
