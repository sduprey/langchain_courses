# tools_and_agent

A recode of [03-Functions-Tools-and-Agents-with-LangChain](../03-Functions-Tools-and-Agents-with-LangChain/) using **Ollama's cloud API** instead of OpenAI, in the same plain-script style as [src/call_minimax.py](../src/call_minimax.py) rather than notebooks.

Each `L1`..`L6` script is a standalone recode of the matching lesson notebook and can be run directly:

```
..\.venv\Scripts\python.exe L1_ollama_tool_calling_basics.py
..\.venv\Scripts\python.exe L2_lcel_chains.py
..\.venv\Scripts\python.exe L3_function_calling_langchain.py
..\.venv\Scripts\python.exe L4_tagging_and_extraction.py
..\.venv\Scripts\python.exe L5_tools_and_routing.py
..\.venv\Scripts\python.exe L6_conversational_agent.py          # interactive REPL
..\.venv\Scripts\python.exe L6_conversational_agent.py --demo   # scripted, non-interactive
```

They read the same `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` variables from the repo-root `.env` that `src/model.py` uses.

## What changed vs. the original course

- **Model**: `ChatOllama` (via `common.get_model()`) instead of `ChatOpenAI` / raw `openai`. L1 uses the raw `ollama` client directly, mirroring how the original L1 used the raw `openai` SDK before LangChain shows up in L2.
- **LangChain version**: the installed `langchain==1.3.14` is a major-version jump from what the course was written against. Several APIs the notebooks use no longer exist:
  - `convert_pydantic_to_openai_function`, `format_tool_to_openai_function`, `.bind(functions=...)` → `model.bind_tools([...])` (accepts Pydantic models, `BaseTool`s, or plain dicts directly).
  - `JsonOutputFunctionsParser`, `JsonKeyOutputFunctionsParser` → `model.with_structured_output(PydanticModel)`.
  - `OpenAIFunctionsAgentOutputParser`, `AgentAction`/`AgentFinish` → read `AIMessage.tool_calls` directly (empty = done, otherwise a list of `{name, args}` to run).
  - `AgentExecutor`, `format_to_openai_functions`, `ConversationBufferMemory` → `langchain.agents.create_agent` (LangGraph-based) with an `InMemorySaver` checkpointer keyed by `thread_id`.
- **`tool_choice` / forced function calls**: Ollama doesn't support forcing a specific tool call the way OpenAI's `function_call={"name": ...}` does. `langchain-ollama` accepts a `tool_choice` argument but documents it as ignored. L3 demonstrates this directly (the "forced" call has no effect) rather than pretending it works.
- **Embeddings** (L2's retriever demo): the Ollama cloud account behind this project's `.env` returns `unauthorized` for embedding models, so the toy retriever is a hand-rolled keyword matcher instead of `OpenAIEmbeddings` + a vector store. Swap in `OllamaEmbeddings` + `langchain_core.vectorstores.InMemoryVectorStore` if you have a local embedding model pulled and point `base_url` at a local Ollama instance.
- **Web content** (L4's "doing it for real" section): the original fetches a Lilian Weng blog post via `WebBaseLoader`; that host doesn't complete a TLS handshake from this environment, so this fetches a Wikipedia article via the `wikipedia` package instead (same source L5/L6 already depend on).
- **Wikipedia access**: Wikimedia rejects the default `wikipedia`-package User-Agent (empty/invalid JSON back). `common.configure_wikipedia()` sets a descriptive one per [Wikimedia's policy](https://meta.wikimedia.org/wiki/User-Agent_policy) before any lesson calls the package.
- **Panel GUI** (end of L6): dropped in favor of a plain terminal REPL - it's the same `agent.invoke(...)` call the notebook already demonstrates non-interactively, just wired to `input()`/`print()` instead of a widget.

## New dependencies

Added to [requirements.txt](../requirements.txt): `langchain`, `langchain-core`, `langchain-ollama`, `langchain-text-splitters`, `wikipedia`. None of these were previously installed - the rest of the repo uses `llama-index` instead of LangChain.

## Langfuse tracing

Every LLM call across `L1`-`L6` is logged to Langfuse using the `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_BASE_URL` already in `.env` (the same project `src/model.py` traces into) - open the Langfuse dashboard and filter by trace name (e.g. `L3:`, `L5:`) to see a given script's calls.

- **`tracing.py`** owns the plain Langfuse client (`langfuse = get_client()`) plus an `atexit`-registered `flush()`, so events aren't dropped when these short-lived scripts exit. It has no LangChain dependency, so L1 (raw `ollama` client) can import just this.
- **`common.py`** layers a `langfuse.langchain.CallbackHandler` on top for the LangChain scripts (L2-L6), exposed via `traced(run_name=..., **config)` - a `RunnableConfig` builder used at every `.invoke()` / `.batch()` / `.stream()` call site: `chain.invoke(x, config=traced("L3: single Pydantic tool"))`.
- **L1** has no LangChain runnable for the callback handler to attach to, so its `traced_chat()` helper wraps each `client.chat(...)` by hand in `langfuse.start_as_current_observation(..., as_type="generation")`, including token usage (`prompt_eval_count` / `eval_count`).
- **Why `config=` at invoke time, not baked into `get_model()`**: verified empirically that calling `.bind_tools()` / `.bind()` *after* `model.with_config(callbacks=[...])` silently drops the callback - `bind_tools` forwards to the underlying unwrapped model and returns a fresh, config-less binding. Since L3/L5/L6 all call `.bind_tools()` on the model returned by `get_model()`, baking the handler in there would only trace the scripts that never bind tools. Passing it via `config=` at each call site sidesteps that entirely and works uniformly across plain chains, bound-tool models, and `create_agent`'s LangGraph runnable.
- **L4's chunked extraction** is the one non-obvious case: the outer chain calls `paper_extraction_chain.batch(docs, ...)` from *inside* a plain Python function wrapped in `RunnableLambda`. Declaring a `config=None` parameter on that function is what makes LangChain inject the parent's config (and therefore the Langfuse callback) into it - verified by checking the resulting trace had all 6 chunk-level `ChatOllama` generations nested under one trace instead of untraced.
