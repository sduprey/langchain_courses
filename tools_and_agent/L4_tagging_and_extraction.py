"""L4 - Tagging and extraction, recoded from L4-tagging-and-extraction-student.ipynb.

The original chains `convert_pydantic_to_openai_function(...)`,
`model.bind(functions=..., function_call={...})` and
`JsonOutputFunctionsParser()` / `JsonKeyOutputFunctionsParser(key_name=...)`
by hand. Modern LangChain collapses all of that into one call:
`model.with_structured_output(PydanticModel)` returns a Runnable that
outputs a validated instance of the model directly.

`with_structured_output` defaults to `method="json_schema"` (Ollama's
native structured-output mode), but that came back as unparsed prose
against this project's cloud model/account - pinning `method="function_calling"`
(same tool-calling machinery as L1/L3) is what actually works reliably here.

The "doing it for real" section swaps the original's `WebBaseLoader` fetch of
a Lilian Weng blog post for a Wikipedia article via the `wikipedia` package
(same source L5/L6 use for their `search_wikipedia` tool) - the blog's host
doesn't complete a TLS handshake from this environment, and this keeps the
lessons to one fetch mechanism instead of two.
"""

from typing import Optional

import wikipedia
from common import configure_wikipedia, get_model, traced
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

configure_wikipedia()
model = get_model()


def structured(schema):
    return model.with_structured_output(schema, method="function_calling")


# --- Tagging ---


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


tagging_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}"),
    ]
)
tagging_chain = tagging_prompt | structured(Tagging)

print("--- tagging: English ---")
print(tagging_chain.invoke({"input": "I love langchain"}, config=traced("L4: tagging English")))

print("\n--- tagging: Italian ---")
print(tagging_chain.invoke({"input": "non mi piace questo cibo"}, config=traced("L4: tagging Italian")))


# --- Extraction ---


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(description="person's name")
    age: Optional[int] = Field(default=None, description="person's age")


class Information(BaseModel):
    """Information to extract."""

    people: list[Person] = Field(description="List of info about people")


extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the relevant information, if not explicitly provided do not guess. Extract partial info",
        ),
        ("human", "{input}"),
    ]
)
extraction_chain = extraction_prompt | structured(Information)

print("\n--- extraction ---")
result = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}, config=traced("L4: extraction"))
print(result.people)


# --- Doing it for real: tag and extract from a live Wikipedia article ---

article_title = "Large language model"
page_content = wikipedia.page(title=article_title, auto_suggest=False).content[:10000]

print(f"\n--- fetched {len(page_content)} chars from the '{article_title}' Wikipedia article ---")
print(page_content[:300])


class Overview(BaseModel):
    """Overview of a section of text."""

    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")


overview_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}"),
    ]
)
overview_chain = overview_prompt | structured(Overview)

print("\n--- overview tagging of the full article ---")
print(overview_chain.invoke({"input": page_content}, config=traced("L4: overview tagging")))


class Paper(BaseModel):
    """Information about a paper mentioned in the text."""

    title: str
    author: Optional[str] = None


class Info(BaseModel):
    """Information to extract."""

    papers: list[Paper]


paper_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A article will be passed to you. Extract from it all papers that are "
            "mentioned by this article. Do not extract the name of the article itself. "
            "If no papers are mentioned that's fine - you don't need to extract any! "
            "Just return an empty list. Do not make up or guess ANY extra information. "
            "Only extract what exactly is in the text.",
        ),
        ("human", "{input}"),
    ]
)
paper_extraction_chain = paper_prompt | structured(Info) | RunnableLambda(lambda info: info.papers)

print("\n--- paper extraction on the first 10k chars ---")
print(paper_extraction_chain.invoke({"input": page_content}, config=traced("L4: paper extraction (single)")))


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
splits = text_splitter.split_text(page_content)
print(f"\n--- split article into {len(splits)} chunks ---")

prep = RunnableLambda(lambda x: [{"input": doc} for doc in text_splitter.split_text(x)])


def batch_tolerating_bad_chunks(docs, config=None):
    # Real text occasionally makes the model emit a shape that doesn't match
    # the schema (e.g. `author` as a list instead of a string) - `.map()`
    # would let one bad chunk crash the whole extraction, so batch with
    # `return_exceptions=True` and drop only the chunks that failed to parse.
    # RunnableLambda injects the parent `config` (declared as a param here),
    # so the Langfuse callback passed to `full_extraction_chain.invoke(...)`
    # below reaches this nested `.batch()` call too.
    results = paper_extraction_chain.batch(docs, config=config, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]


full_extraction_chain = prep | RunnableLambda(batch_tolerating_bad_chunks) | flatten

print("\n--- paper extraction mapped over every chunk ---")
print(full_extraction_chain.invoke(page_content, config=traced("L4: paper extraction (chunked)")))
