"""Your prompt registry. Keep prompts here (not scattered in notebooks) so they
are versioned, diffable, and easy to evaluate. Bump VERSION when you change one."""

VERSION = "0.1.0"

SYSTEM = """You are a careful assistant.
Answer only from the provided context. Cite sources as [1], [2].
If the answer is not in the context, say "I don't know"."""

ANSWER_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with citations):"""


def build_answer_prompt(context: str, question: str) -> str:
    return ANSWER_TEMPLATE.format(context=context, question=question)
