"""L2 - LangChain Expression Language (LCEL), recoded from L2-lcel-student.ipynb.

Same LCEL concepts as the original (`prompt | model | parser`, RunnableMap,
`.bind()`, fallbacks, the batch/stream/invoke interface) with `ChatOllama`
standing in for `ChatOpenAI`.

Note on the "more complex chain" section: the original notebook builds a
retriever with `OpenAIEmbeddings` + `DocArrayInMemorySearch`. The Ollama
cloud account behind this project's `.env` returns "unauthorized" for the
embeddings endpoint (checked directly - the cloud key doesn't cover
embedding models), so the retriever here is a tiny hand-rolled keyword
matcher instead. It plugs into the same `RunnableMap` shape; swap in
`OllamaEmbeddings` + `InMemoryVectorStore` (both ship with
langchain-ollama / langchain-core) if you have a local embedding model
pulled (`ollama pull nomic-embed-text`) and point `base_url` at
`http://localhost:11434`.
"""

import json

from common import get_model, strip_json_fence, traced
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap

# --- Simple Chain ---

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = get_model()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print("--- simple chain ---")
print(chain.invoke({"topic": "bears"}, config=traced("L2: simple joke chain")))


# --- More complex chain: RunnableMap feeding a retriever + question into a prompt ---

corpus = ["harrison worked at kensho", "bears like to eat honey"]


def toy_retriever(question: str) -> list[str]:
    """Keyword-overlap 'retriever' standing in for a real vector store (see module docstring)."""
    scored = sorted(
        corpus,
        key=lambda doc: len(set(doc.lower().split()) & set(question.lower().split())),
        reverse=True,
    )
    return scored[:2]


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)

rag_inputs = RunnableMap(
    {
        "context": lambda x: toy_retriever(x["question"]),
        "question": lambda x: x["question"],
    }
)

print("\n--- RunnableMap inputs only ---")
print(rag_inputs.invoke({"question": "where did harrison work?"}))

rag_chain = rag_inputs | rag_prompt | model | output_parser

print("\n--- RAG-style chain ---")
print(rag_chain.invoke({"question": "where did harrison work?"}, config=traced("L2: RAG-style chain")))


# --- Bind: attaching tools at runtime ---

weather_tool = {
    "name": "weather_search",
    "description": "Search for weather given an airport code",
    "parameters": {
        "type": "object",
        "properties": {
            "airport_code": {
                "type": "string",
                "description": "The airport code to get the weather for",
            },
        },
        "required": ["airport_code"],
    },
}

bind_prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
bound_model = get_model().bind_tools([weather_tool])
runnable = bind_prompt | bound_model

print("\n--- bind: single tool ---")
print(runnable.invoke({"input": "what is the weather in sf"}, config=traced("L2: bind single tool")).tool_calls)

sports_tool = {
    "name": "sports_search",
    "description": "Search for news of recent sport events",
    "parameters": {
        "type": "object",
        "properties": {
            "team_name": {
                "type": "string",
                "description": "The sports team to search for",
            },
        },
        "required": ["team_name"],
    },
}

bound_model = get_model().bind_tools([weather_tool, sports_tool])
runnable = bind_prompt | bound_model

print("\n--- bind: model picks between two tools ---")
print(
    runnable.invoke(
        {"input": "how did the patriots do yesterday?"}, config=traced("L2: bind two tools")
    ).tool_calls
)


# --- Fallbacks ---
# The original pairs a "simple" completion-style model (prone to producing
# non-JSON text around its answer) with a chat model as a fallback. Here the
# "flaky" step is a plain chat model piped straight into `json.loads` - it
# tends to add prose around the JSON and fails to parse. The fallback swaps
# in `format="json"` plus `strip_json_fence` (this model still wraps JSON in
# ```json fences even in JSON mode, same quirk src/model.py works around).
# The pattern itself is the point: `runnable.with_fallbacks([other_runnable])`.

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

flaky_model = get_model()  # no format="json" -> usually wraps the answer in prose
flaky_chain = flaky_model | StrOutputParser() | json.loads

reliable_model = get_model(format="json")
reliable_chain = reliable_model | StrOutputParser() | strip_json_fence | json.loads

final_chain = flaky_chain.with_fallbacks([reliable_chain])

print("\n--- fallback chain ---")
print(final_chain.invoke(challenge, config=traced("L2: fallback chain")))


# --- Interface: invoke / batch / stream ---

prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
model = get_model()
chain = prompt | model | StrOutputParser()

print("\n--- invoke ---")
print(chain.invoke({"topic": "bears"}, config=traced("L2: interface invoke")))

print("\n--- batch ---")
print(chain.batch([{"topic": "bears"}, {"topic": "frogs"}], config=traced("L2: interface batch")))

print("\n--- stream ---")
for chunk in chain.stream({"topic": "bears"}, config=traced("L2: interface stream")):
    print(chunk, end="", flush=True)
print()
