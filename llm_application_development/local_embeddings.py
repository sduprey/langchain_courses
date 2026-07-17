"""A tiny local TF-IDF `Embeddings` implementation, no network, no extra deps.

L4 (QnA) and L5 (Evaluation) are built around vector search over documents.
The original notebooks use `OpenAIEmbeddings`; the natural swap would be
`OllamaEmbeddings`, but the Ollama cloud account behind this project's
`.env` returns "unauthorized" for every embedding model tried (checked
directly against embeddinggemma / nomic-embed-text / mxbai-embed-large) -
the cloud key just doesn't cover embeddings.

Rather than fake the retrieval step, this is a real (if simple) bag-of-words
TF-IDF vectorizer wired into LangChain's `Embeddings` interface, so
`InMemoryVectorStore` and `.as_retriever()` behave exactly like they would
with a real embedding model - just with cosine similarity over TF-IDF
vectors instead of a learned embedding space. Good enough for the small,
topically-distinct catalogs these lessons search over; swap in
`OllamaEmbeddings` + a locally-pulled embedding model if you have Ollama
running locally.
"""

import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LocalTfidfEmbeddings(Embeddings):
    """Fit on `embed_documents` (the corpus), then project queries into that space."""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._vocab: list[str] = []

    def _fit(self, texts: list[str]) -> None:
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(_tokenize(text)))
        n_docs = len(texts)
        self._vocab = sorted(doc_freq)
        self._idf = {
            term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term in self._vocab
        }

    def _vectorize(self, text: str) -> list[float]:
        counts = Counter(_tokenize(text))
        vec = [counts.get(term, 0) * self._idf.get(term, 0.0) for term in self._vocab]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._fit(texts)
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)
