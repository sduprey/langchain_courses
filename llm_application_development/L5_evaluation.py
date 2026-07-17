"""L5 - Evaluation, recoded from L5-Evaluation.ipynb.

Same three-part outline as the original (hand-written examples, LLM-generated
examples, LLM-assisted grading), rebuilt on primitives that still exist:

- `QAGenerateChain` (LLM writes a Q&A pair from a document) -> a Pydantic
  `QAPair` schema through `model.with_structured_output(...)`, `.batch()`'d
  over the source documents.
- `QAEvalChain` (LLM grades a predicted answer against the real one) -> the
  same structured-output pattern, this time with a `Grade` schema
  (`CORRECT` / `INCORRECT` + reasoning).
- `langchain.debug = True` (the original's way to see what the chain sent
  the model) doesn't exist anymore - Langfuse tracing (already wired into
  every call here) gives the same visibility and more, in the dashboard
  rather than stdout.

Rebuilds the same retrieval QA chain as L4 - see that file's docstring for
why the retriever is a local TF-IDF `Embeddings` implementation instead of
`OllamaEmbeddings`.
"""

from typing import Literal

from common import get_model, load_csv_documents, traced
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from local_embeddings import LocalTfidfEmbeddings
from pydantic import BaseModel, Field

docs = load_csv_documents("data/outdoor_clothing_catalog.csv")
model = get_model()

vectorstore = InMemoryVectorStore.from_documents(docs, LocalTfidfEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(retrieved) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved)


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


# --- Hard-coded examples ---

examples = [
    {"query": "Does the Sun Shield Shirt have UPF 50+ sun protection?", "answer": "Yes"},
    {"query": "What organization recommends the Sun Shield Shirt?", "answer": "The Skin Cancer Foundation"},
]


# --- LLM-generated examples ---


class QAPair(BaseModel):
    """A question-and-answer pair for evaluating a QA system, based on a document."""

    query: str = Field(description="A question that can be answered using only the given document")
    answer: str = Field(description="The correct answer to that question, based only on the document")


qa_gen_prompt = ChatPromptTemplate.from_template(
    "You are a teacher coming up with questions to ask on a quiz.\n"
    "Given the following document, come up with a question and answer pair that "
    "can be used to evaluate a question-answering system. The question must be "
    "answerable using only the information in the document.\n\n"
    "Document:\n{doc}"
)
qa_gen_chain = qa_gen_prompt | model.with_structured_output(QAPair, method="function_calling")

print("--- generating Q&A pairs from the first 5 catalog rows ---")
generated = qa_gen_chain.batch(
    [{"doc": doc.page_content} for doc in docs[:5]],
    config=traced("L5: generate QA pairs"),
)
for pair in generated:
    print(f"  Q: {pair.query}\n  A: {pair.answer}\n")

examples += [{"query": pair.query, "answer": pair.answer} for pair in generated]


# --- Manual evaluation ---

print("--- manual evaluation: one example, traced in Langfuse ---")
print("query:", examples[0]["query"])
print("result:", qa_chain.invoke({"question": examples[0]["query"]}, config=traced("L5: manual eval example")))


# --- LLM-assisted evaluation ---


class Grade(BaseModel):
    """Grade a predicted answer against the real (expected) answer."""

    grade: Literal["CORRECT", "INCORRECT"] = Field(
        description="CORRECT if the predicted answer is factually consistent with the real answer, else INCORRECT"
    )
    reasoning: str = Field(description="One sentence explaining the grade")


eval_prompt = ChatPromptTemplate.from_template(
    "You are grading the following question:\n{query}\n\n"
    "Here is the real answer:\n{answer}\n\n"
    "Here is the predicted answer:\n{result}\n\n"
    "Grade the predicted answer as either CORRECT or INCORRECT, based on whether it "
    "is factually consistent with the real answer. Minor wording differences are fine."
)
eval_chain = eval_prompt | model.with_structured_output(Grade, method="function_calling")

print("\n--- LLM-assisted evaluation ---")
predictions = [
    {**eg, "result": qa_chain.invoke({"question": eg["query"]}, config=traced(f"L5: predict - {eg['query'][:30]}"))}
    for eg in examples
]
grades = eval_chain.batch(predictions, config=traced("L5: grade predictions"))

for i, (prediction, grade) in enumerate(zip(predictions, grades)):
    print(f"Example {i}:")
    print("Question:", prediction["query"])
    print("Real Answer:", prediction["answer"])
    print("Predicted Answer:", prediction["result"])
    print("Predicted Grade:", grade.grade, "-", grade.reasoning)
    print()
