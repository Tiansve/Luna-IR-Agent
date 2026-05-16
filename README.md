# Soul-Driven Agent

A lab project for the IR Agent assignment. A ReAct agent that **grows a
persistent, human-readable Soul** by reflecting on its own conversations.

The agent treats memory itself as an IR problem: it retrieves from three
memory layers — episodic, semantic (insights), and the Soul — and it can
take *actions* to write back into the upper two layers (action-as-IR).

```
  user turn
      │
      ▼
  Context Builder   ── Soul ── Insights ── Episodes
      │
      ▼
  ReAct loop  ──►  tools: web_search · search_docs ·
      │             recall_episodes · recall_insights ·
      │             update_soul_note
      ▼
  Episode written to JSONL
      │
      ▼  every N episodes
  Reflection Engine  ── new/strengthened/weakened insights
      │
      ▼  when an insight is stable enough
  Soul Updater  ── new soul.yaml + versioned snapshot
```

---

## Requirements

* Conda env **`IR_P_env`** (Python 3.11). The project does not run outside it.
* An OpenAI-compatible LLM endpoint (e.g. Berget.AI).
* Optional: a Brave Search API key for the web tool.

```powershell
# from the project root
& "C:\Anaconda\envs\IR_P_env\python.exe" -m pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill it in:

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.berget.ai/v1
OPENAI_MODEL=gemma-3-27b-it
# optional
REFLECTION_MODEL=
EMBED_MODEL=
BRAVE_API_KEY=
```

* `OPENAI_BASE_URL` is the only thing you need to change to point at a
  different OpenAI-compatible provider.
* `EMBED_MODEL` is optional. If left empty (or if the embedding call fails)
  the agent falls back to a deterministic hashed bag-of-words vector that
  is good enough for the lab-scale corpus.

## Run

```powershell
& "C:\Anaconda\envs\IR_P_env\python.exe" -m src.main
```

## Slash commands

| Command | What it does |
| --- | --- |
| `/soul` | print the current Soul |
| `/soul history` | list snapshot versions |
| `/soul diff A B` | diff list-fields between two versions |
| `/soul revert N` | restore a previous snapshot (bumps version) |
| `/insights` | list current insights with confidence |
| `/episodes [N]` | show last N episodes (default 5) |
| `/reflect` | force a reflection pass right now |
| `/reindex` | rebuild the local doc index |
| `/help`, `/quit` | … |

## Project layout

```
soul-agent/
├── data/
│   ├── soul.yaml           # current Soul (versioned)
│   ├── soul_history/       # snapshots vN.yaml (auto-written before each update)
│   ├── insights.jsonl      # semantic memory
│   └── episodes.jsonl      # episodic memory
├── docs/                   # local RAG corpus (Markdown / txt)
└── src/
    ├── config.py
    ├── llm_client.py       # OpenAI-compatible chat & embed
    ├── soul.py             # Soul dataclass, YAML I/O, prompt rendering
    ├── agent.py            # ReAct loop + reflection orchestration
    ├── main.py             # CLI entry
    ├── memory/
    │   ├── embed.py        # remote embed + hash-vector fallback
    │   ├── episodic.py
    │   └── semantic.py
    ├── reflection/
    │   ├── extractor.py    # episodes -> insights (LLM, JSON-mode)
    │   └── updater.py      # insights -> Soul (constrained)
    └── tools/
        ├── web_search.py   # Brave
        ├── doc_rag.py      # in-memory cosine over docs/
        └── memory_tools.py # recall_* + update_soul_note (action-as-IR)
```

## Design choices

* **Soul is YAML, not vectors.** Human-readable, hand-editable, diffable.
* **Three memory layers, three speeds.** Episodes (every turn) → insights
  (every N turns) → Soul (only when an insight is stable).
* **Anti-drift guards in the Soul updater.**
  * Each Soul list capped at 15 items.
  * At most two removals per pass; every removal needs an explicit `reason`.
  * Old Soul is snapshotted before every write.
  * `/soul revert` is the human escape hatch.
* **Action-as-IR.** `update_soul_note` lets the agent itself write short
  notes into two whitelisted Soul fields (`open_questions`,
  `knowledge_about_user`) without going through the reflection engine.
  Permanent changes to identity/values still require reflection.

## Known limitations

* Single-user only — multi-user support would require keying memory by user id.
* Reflection quality depends on the model. With a small model, prefer
  setting `REFLECTION_MODEL` to a stronger one.
* The hash-vector fallback is OK for similarity ranking on a small corpus
  but is not competitive with a real embedding model.
