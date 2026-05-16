# Agent Loops and ReAct

A ReAct-style agent alternates between *reasoning* (a thought step produced
by the LLM) and *acting* (a tool invocation). The standard loop is:

1. Build messages: system prompt + history + the new user turn.
2. Call the LLM with the tool schemas attached.
3. If the response contains tool_calls, dispatch each call and append the
   result as a `role=tool` message.
4. Loop until the model returns a final assistant message with no tool_calls.

Practical guards: cap iterations (e.g. 6), serialise tool errors back to
the model, and log every tool call for debuggability.

## Action-as-IR
Tool calls themselves count as information retrieval: a `search_docs` call
is an IR operation, and so is an `update_note` call that writes structured
context back into the agent's own memory store.
