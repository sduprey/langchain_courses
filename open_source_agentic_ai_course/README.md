# llm_application_development

A recode of [02-LangChain-for-LLM-Application-Development](../02-LangChain-for-LLM-Application-Development/) using **Ollama's cloud API** instead of OpenAI, in the same plain-script style as [tools_and_agent](../tools_and_agent/) (and [src/call_minimax.py](../src/call_minimax.py)) rather than notebooks.

Each `L1`..`L6` script is a standalone recode of the matching lesson notebook and can be run directly:

```
..\.venv\Scripts\python.exe L1_models_prompts_parsers.py
..\.venv\Scripts\python.exe L2_memory.py
..\.venv\Scripts\python.exe L3_chains.py
..\.venv\Scripts\python.exe L4_qna.py
..\.venv\Scripts\python.exe L5_evaluation.py
..\.venv\Scripts\python.exe L6_agents.py
```

They read the same `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` variables from the repo-root `.env` that `src/model.py` and `tools_and_agent` use, and trace every LLM call to Langfuse the same way `tools_and_agent` does (see that folder's README for how `tracing.py`/`common.py`'s `traced()` helper works - it's identical here).

## What changed vs. the original course

This course was written against a much older LangChain (`0.0.x`/`0.1.x`). The installed `langchain==1.3.14` removed entire modules the notebooks depend on - `langchain.chains`, `langchain.memory`, `langchain.evaluation`, and `langchain.agents`' old surface (`AgentExecutor`, `initialize_agent`, `AgentType`, `load_tools`, `create_python_agent`) are all gone, not just renamed. Every lesson here is a rebuild on what replaced them, not a syntax find-and-replace:

- **L1 - Models, prompts, parsers**: `ResponseSchema` + `StructuredOutputParser` → `model.with_structured_output(PydanticModel)`, same as tools_and_agent.
- **L2 - Memory**: `ConversationChain` + `ConversationBufferMemory` / `ConversationBufferWindowMemory` / `ConversationTokenBufferMemory` / `ConversationSummaryBufferMemory` were all deleted - modern LangChain has no "memory object" abstraction at all. Recoded as four small classes managing a plain `list[BaseMessage]` by hand (keep everything / keep last k / keep what fits a token budget / summarize the overflow via an LLM call). Token counts are a `len(text) // 4` approximation, not `tiktoken` - that counts OpenAI's tokenizer specifically, which says nothing about how the Ollama-hosted model actually tokenizes.
- **L3 - Chains**: `LLMChain`, `SimpleSequentialChain`, `SequentialChain`, `MultiPromptChain`/`LLMRouterChain` → LCEL throughout. Sequential-with-named-outputs is a stack of `RunnablePassthrough.assign(key=chain)` calls; the router is `model.with_structured_output(RouterDecision)` picking a destination name, then a plain dict lookup dispatches - no hand-parsed markdown JSON blob needed.
- **L4 - Q&A over documents** / **L5 - Evaluation**: `RetrievalQA`, `VectorstoreIndexCreator`, `CSVLoader`, `DocArrayInMemorySearch`, `QAGenerateChain`, `QAEvalChain` → an LCEL RAG chain (retriever output formatted straight into the prompt's `{context}`), `common.load_csv_documents` (stdlib `csv`, no `langchain_community`), `langchain_core.vectorstores.InMemoryVectorStore`, and `with_structured_output` again for both example-generation and LLM-as-judge grading.
- **L6 - Agents**: `initialize_agent`/`AgentType`/`load_tools` → `langchain.agents.create_agent` (LangGraph-based, same as tools_and_agent/L6). The built-in `"llm-math"` tool is replaced by a small `ast`-based safe arithmetic evaluator (no nested LLM call needed - a tool-calling model already turns "25% of 300" into `300 * 0.25` itself). `PythonREPLTool` is reimplemented directly with `exec()` + captured stdout - **no sandboxing**, exactly like the original tool had none; fine for this local exercise, never point it at untrusted input.

## Embeddings

L4 and L5 are built around vector search. The natural swap for `OpenAIEmbeddings` would be `OllamaEmbeddings`, but the Ollama cloud account behind this project's `.env` returns `unauthorized` for every embedding model tried (checked directly against `embeddinggemma`, `nomic-embed-text`, `mxbai-embed-large`) - the cloud key doesn't cover embeddings. `local_embeddings.py` implements a small, real (if simple) local TF-IDF vectorizer against LangChain's `Embeddings` interface instead of faking the retrieval step - see that module's docstring. It's good enough for these lessons' small, topically-distinct catalogs; swap in `OllamaEmbeddings` + a locally-pulled embedding model if you have Ollama running locally instead of the cloud endpoint.

## Data

`Data.csv` and `OutdoorClothingCatalog_1000.csv` from the original aren't included in this repo (per the top-level README, some notebooks' local data files didn't make the mirror). `data/reviews.csv` and `data/outdoor_clothing_catalog.csv` are small synthetic stand-ins with the same shape - a French review to exercise L3's translate/detect-language chains, and a handful of UPF/sun-protection items among non-matching ones in the catalog so L4/L5's retrieval demos are meaningful.

## New dependencies

Added to [requirements.txt](../requirements.txt): `pandas` (L3 reads a CSV the same way the original notebook does). Everything else (`langchain`, `langchain-ollama`, `wikipedia`, etc.) was already added for `tools_and_agent`.
