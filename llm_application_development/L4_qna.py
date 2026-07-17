"""L4 - Q&A over documents, recoded from L4-QnA.ipynb.

`VectorstoreIndexCreator`, `RetrievalQA`, `CSVLoader` and `DocArrayInMemorySearch`
all live in `langchain_community`/`langchain.chains`, neither of which is
installed or (for the chains) even exists in this LangChain version anymore.
The modern replacement for `RetrievalQA` is a small LCEL RAG chain: format
whatever the retriever returns into the prompt's `{context}`, same idea as
tools_and_agent/L2's RunnableMap-fed prompt.

`CSVLoader` is replaced by `common.load_csv_documents` (stdlib `csv`, no
`langchain_community` needed) and `DocArrayInMemorySearch` by
`langchain_core.vectorstores.InMemoryVectorStore` - both already used this
way in tools_and_agent.

Real embeddings would normally come from `OllamaEmbeddings`, but the Ollama
cloud account behind this project's `.env` returns "unauthorized" for every
embedding model tried. `local_embeddings.LocalTfidfEmbeddings` is a real (if
simple) local TF-IDF vectorizer plugged into the same `Embeddings` interface
instead - see that module's docstring for the full explanation.

`OutdoorClothingCatalog_1000.csv` from the original isn't included in this
repo; data/outdoor_clothing_catalog.csv is a small synthetic stand-in with
the same shape (a handful of items explicitly UPF/sun-protection rated, the
rest not, so retrieval is meaningful).
"""

from common import get_model, load_csv_documents, traced
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from local_embeddings import LocalTfidfEmbeddings

docs = load_csv_documents("data/outdoor_clothing_catalog.csv")
print(f"--- loaded {len(docs)} catalog rows ---")
print(docs[0])

model = get_model()

vectorstore = InMemoryVectorStore.from_documents(docs, LocalTfidfEmbeddings())

query = "Please suggest a shirt with sunblocking"
hits = vectorstore.similarity_search(query, k=4)
print(f"\n--- similarity_search: {len(hits)} hits for {query!r} ---")
for doc in hits:
    print(" -", doc.page_content.splitlines()[0])

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(retrieved) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved)


# --- The manual version: stuff every retrieved doc into one prompt ---

full_query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
qdocs = format_docs(retriever.invoke(full_query))
manual_response = model.invoke(
    f"{qdocs}\n\nQuestion: {full_query}", config=traced("L4: manual stuffed-context answer")
)
print("\n--- manual retrieve-then-stuff-into-prompt ---")
print(manual_response.content)


# --- The RetrievalQA equivalent: an LCEL RAG chain ---

rag_prompt = ChatPromptTemplate.from_template(
    "Use the following product catalog excerpts to answer the question. "
    "If the answer isn't in the excerpts, say you don't know.\n\n"
    "{context}\n\nQuestion: {question}"
)

qa_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
    | rag_prompt
    | model
    | StrOutputParser()
)

print("\n--- RetrievalQA equivalent (LCEL RAG chain) ---")
print(qa_chain.invoke({"question": full_query}, config=traced("L4: RAG chain answer")))
