"""BM25, TF-IDF, dense, and hybrid passage retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize and apply conservative plural normalization for lexical matching."""
    tokens = []
    for match in _TOKEN.findall(text):
        token = match.lower()
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("sses"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))
        self.average_length = sum(map(len, self.tokens)) / max(len(chunks), 1)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = []
        n_docs = len(self.chunks)
        for index, frequencies in enumerate(self.frequencies):
            score = 0.0
            length = len(self.tokens[index])
            for term in tokenize(query):
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return _rank(self.chunks, np.asarray(scores), top_k, positive_only=True)


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        return _rank(self.chunks, scores, top_k, positive_only=True)


class DenseRetriever:
    """Sentence-transformer embeddings, using FAISS when available and NumPy otherwise."""

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires the optional ML dependencies: pip install -r requirements-ml.txt"
            ) from exc
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = None
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except ImportError:
            pass

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = np.asarray(
            self.model.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        if self.index is not None:
            scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
            return [
                RetrievedChunk(self.chunks[int(index)], float(score))
                for score, index in zip(scores[0], indices[0])
                if index >= 0 and score > 0
            ]
        scores = self.embeddings @ query_vector[0]
        return _rank(self.chunks, scores, top_k, positive_only=True)


class HybridRetriever:
    """Reciprocal-rank fusion of lexical and dense rankings."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
        if not retrievers:
            raise ValueError("Hybrid retrieval needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, result in enumerate(retriever.retrieve(query, max(top_k * 3, 20)), start=1):
                chunk_id = result.chunk.chunk_id
                chunk_by_id[chunk_id] = result.chunk
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [RetrievedChunk(chunk_by_id[chunk_id], fused[chunk_id]) for chunk_id in ranked]


def build_retriever(
    chunks: list[Chunk], method: str = "tfidf", embedding_model: str = "all-MiniLM-L6-v2"
) -> Retriever:
    if not chunks:
        raise ValueError("Cannot build a retriever without document chunks")
    method = method.lower()
    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "tfidf":
        return TfidfRetriever(chunks)
    if method == "dense":
        return DenseRetriever(chunks, embedding_model)
    if method == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, embedding_model)])
    raise ValueError(f"Unknown retrieval method: {method}")


def _rank(
    chunks: list[Chunk], scores: np.ndarray, top_k: int, positive_only: bool = False
) -> list[RetrievedChunk]:
    if top_k < 1:
        return []
    indices = np.argsort(-scores, kind="stable")[:top_k]
    results = []
    for index in indices:
        score = float(scores[index])
        if positive_only and score <= 0:
            continue
        results.append(RetrievedChunk(chunks[int(index)], score))
    return results
