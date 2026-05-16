"""ReAct-style agent loop.

The loop is deliberately small. All knobs (max iterations, top-k recall) live
in ``config``. The agent assembles a context from:

  * the current Soul (rendered as Markdown into the system prompt)
  * top-k insights relevant to the user query
  * top-k past episode summaries relevant to the user query

Then iterates ``LLM → tool dispatch → LLM`` until a final answer arrives or
``MAX_TOOL_ITERS`` is reached. Every tool call is recorded into an Episode.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import config, llm_client
from .memory.episodic import EpisodicStore
from .memory.semantic import InsightStore
from .reflection.extractor import extract_insights
from .reflection.updater import maybe_update_soul
from .soul import Soul
from .tools import doc_rag, web_search
from .tools.memory_tools import MemoryTools


_BASE_SYSTEM = (
    "You are a learning-companion agent for IR / NLP study. "
    "You have tools for web search, local document RAG, memory recall, and "
    "for writing short notes back into your own Soul. "
    "Always check recall_insights or recall_episodes before answering a "
    "non-trivial question — past conversations may already contain the answer. "
    "Use update_soul_note sparingly: only for genuinely new facts the user just "
    "volunteered, or for observations you want to keep watching."
)


class Agent:
    def __init__(self) -> None:
        config.ensure_dirs()
        self.soul: Soul = Soul.load()
        self.episodic = EpisodicStore()
        self.insights = InsightStore()
        self.memory_tools = MemoryTools(
            self.episodic, self.insights, self.soul, self._save_soul
        )

        self._dispatch: dict[str, Callable[..., Any]] = {
            "web_search": lambda **kw: web_search.web_search(**kw),
            "search_docs": lambda **kw: doc_rag.search_docs(**kw),
            "recall_episodes": self.memory_tools.recall_episodes,
            "recall_insights": self.memory_tools.recall_insights,
            "update_soul_note": self.memory_tools.update_soul_note,
        }
        self._schemas: list[dict[str, Any]] = [
            web_search.SCHEMA,
            doc_rag.SCHEMA,
            *MemoryTools.schemas(),
        ]

    # ---- Soul helpers ------------------------------------------------------

    def _save_soul(self) -> None:
        self.soul.save()

    # ---- prompt assembly ---------------------------------------------------

    def _build_system_prompt(self, user_query: str) -> str:
        parts: list[str] = [_BASE_SYSTEM, "", self.soul.to_prompt()]

        ins_hits = self.insights.search(user_query, config.TOP_K_INSIGHTS)
        if ins_hits:
            parts.append("")
            parts.append("# Relevant insights (from prior reflection)")
            for ins, score in ins_hits:
                parts.append(
                    f"- ({ins.category}, conf={ins.confidence:.2f}, sim={score:.2f}) {ins.content}"
                )

        ep_hits = self.episodic.search(user_query, config.TOP_K_EPISODES)
        if ep_hits:
            parts.append("")
            parts.append("# Relevant past episodes")
            for ep, score in ep_hits:
                parts.append(f"- (sim={score:.2f}) {ep.summary_for_prompt()}")

        return "\n".join(parts)

    # ---- main loop ---------------------------------------------------------

    def chat(self, user_query: str) -> dict[str, Any]:
        user_query = (user_query or "").strip()
        if not user_query:
            return {"answer": "", "tool_calls": [], "reflection": None}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(user_query)},
            {"role": "user", "content": user_query},
        ]
        tool_call_log: list[dict[str, Any]] = []

        final_text = ""
        for step in range(config.MAX_TOOL_ITERS):
            resp = llm_client.chat(messages, tools=self._schemas)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_text = (msg.content or "").strip()
                break

            # Echo the assistant turn (with tool_calls) into history.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                fn = self._dispatch.get(name)
                if fn is None:
                    result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        result = fn(**args)
                    except TypeError as exc:
                        result = {"error": f"bad arguments: {exc!s}"}
                    except Exception as exc:  # surface to model, don't crash
                        result = {"error": f"{type(exc).__name__}: {exc!s}"}

                print(f"  [tool] {name}({_short(args)}) -> {_short(result)}")
                tool_call_log.append({"name": name, "arguments": args, "result_preview": _short(result)})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        else:
            final_text = (
                f"(stopped after {config.MAX_TOOL_ITERS} tool iterations; "
                "the model did not converge to a final answer.)"
            )

        # Persist this turn as an Episode.
        self.episodic.add(user_query, tool_call_log, final_text)

        # Decide whether to reflect.
        reflection_diff: dict[str, Any] | None = None
        unreflected = self.episodic.unreflected()
        if len(unreflected) >= config.REFLECT_EVERY:
            reflection_diff = self._run_reflection(unreflected)

        return {"answer": final_text, "tool_calls": tool_call_log, "reflection": reflection_diff}

    # ---- reflection orchestration ------------------------------------------

    def _run_reflection(self, batch) -> dict[str, Any]:
        print(f"\n[reflect] {len(batch)} new episodes -> extracting insights...")
        ins_diff = extract_insights(batch, self.insights)
        print(f"[reflect] insight diff: {_short(ins_diff)}")

        new_soul, soul_diff = maybe_update_soul(self.soul, self.insights)
        if new_soul is not self.soul:
            self.soul = new_soul
            self.memory_tools.soul = new_soul  # keep MemoryTools pointed at current Soul
            self._save_soul()
            print(f"[reflect] Soul evolved: v{soul_diff['from_version']} -> v{soul_diff['to_version']}")
        else:
            print(f"[reflect] Soul unchanged ({_short(soul_diff)})")

        self.episodic.mark_reflected([e.id for e in batch])
        return {"insights": ins_diff, "soul": soul_diff}

    # ---- manual hooks (called by CLI) --------------------------------------

    def force_reflect(self) -> dict[str, Any]:
        batch = self.episodic.unreflected()
        if not batch:
            return {"skipped": "no unreflected episodes"}
        return self._run_reflection(batch)


def _short(obj: Any, limit: int = 200) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= limit else s[: limit - 3] + "..."
