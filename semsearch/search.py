"""Hybrid product search: BM25 for exact terms (brands, SKUs, sizes), dense vectors for meaning
("couch" finds sofas), fused with Reciprocal Rank Fusion.

`ElasticBackend` builds the Elasticsearch 8 request (a `knn` clause plus a `query` clause, fused
with RRF). `MemoryBackend` implements the same ranking in-process so relevance can be evaluated
offline and in CI without a cluster.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "for", "with", "and", "of", "in", "to", "on", "my", "i", "need", "want", "some", "best"}


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


@dataclass(frozen=True)
class Product:
    id: str
    title: str
    description: str
    category: str
    brand: str
    price_cents: int
    in_stock: bool = True

    @property
    def text(self) -> str:
        return f"{self.title}. {self.brand}. {self.category}. {self.description}"


@dataclass
class Filters:
    category: str | None = None
    max_price_cents: int | None = None
    in_stock_only: bool = True

    def ok(self, p: Product) -> bool:
        return ((self.category is None or p.category == self.category)
                and (self.max_price_cents is None or p.price_cents <= self.max_price_cents)
                and (not self.in_stock_only or p.in_stock))


class Embedder(Protocol):
    dims: int
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LsaEmbedder:
    """TF-IDF + truncated SVD (latent semantic analysis): a dependency-light stand-in for a sentence
    embedding model. Words that co-occur in catalog text ("sofa", "couch", "loveseat") land near
    each other, which is what makes vocabulary-mismatch queries work."""

    def __init__(self, dims: int = 16):  # chosen on the tuning query set (seed 11); see README
        self.dims = dims

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True, min_df=1, token_pattern=None)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dims, X.shape[1] - 1), random_state=0).fit(X)
        self.dims = self.svd.n_components
        return self

    def embed(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(n == 0, 1, n)


class Bm25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out[i] = s
        return out


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: robust to the two scorers having incomparable scales."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda t: -t[1])


@dataclass
class Hit:
    id: str
    score: float
    title: str


class SearchBackend(Protocol):
    def search(self, query: str, filters: Filters, k: int = 10, mode: str = "hybrid") -> list[Hit]: ...


@dataclass
class MemoryBackend:
    products: list[Product]
    embedder: Embedder = field(default_factory=LsaEmbedder)
    window: int = 50  # candidates per retriever before fusion (Elasticsearch's num_candidates / rank_window_size)

    def __post_init__(self):
        texts = [p.text for p in self.products]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts)
        self.bm25 = Bm25([tokenize(t) for t in texts])
        self.by_id = {p.id: p for p in self.products}

    def _rank(self, scores: np.ndarray, allowed: np.ndarray, require_positive: bool) -> list[str]:
        idx = [i for i in np.argsort(-scores) if allowed[i] and (not require_positive or scores[i] > 0)]
        return [self.products[i].id for i in idx[: self.window]]

    def search(self, query, filters=Filters(), k=10, mode="hybrid"):
        allowed = np.array([filters.ok(p) for p in self.products])  # filters apply inside each retriever (pre-filter)
        lex = self._rank(self.bm25.scores(tokenize(query)), allowed, True)
        if mode == "bm25":
            fused = [(d, 1.0 / (61 + r)) for r, d in enumerate(lex)]
        else:
            sem = self._rank(self.vectors @ self.embedder.embed([query])[0], allowed, False)
            fused = [(d, 1.0 / (61 + r)) for r, d in enumerate(sem)] if mode == "vector" else rrf([lex, sem])
        return [Hit(d, s, self.by_id[d].title) for d, s in fused[:k]]


class ElasticBackend:
    """Elasticsearch 8.x: `products` index with `embedding` as dense_vector (cosine, HNSW)."""

    def __init__(self, es, embedder: Embedder, index: str = "products"):
        self.es, self.embedder, self.index = es, embedder, index

    @staticmethod
    def build_request(query: str, vector: list[float], filters: Filters, k: int = 10, window: int = 50) -> dict:
        must_filter = []
        if filters.category:
            must_filter.append({"term": {"category": filters.category}})
        if filters.max_price_cents is not None:
            must_filter.append({"range": {"price_cents": {"lte": filters.max_price_cents}}})
        if filters.in_stock_only:
            must_filter.append({"term": {"in_stock": True}})
        return {
            "size": k,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {"standard": {"query": {"bool": {
                            "must": {"multi_match": {"query": query, "fields": ["title^3", "brand^2", "category", "description"], "type": "best_fields"}},
                            "filter": must_filter}}}},
                        {"knn": {"field": "embedding", "query_vector": vector, "k": window, "num_candidates": window * 4,
                                 "filter": must_filter}},  # same filter inside kNN, so filtered results aren't starved
                    ],
                    "rank_window_size": window,
                    "rank_constant": 60,
                }
            },
            "_source": ["id", "title", "price_cents"],
        }

    def search(self, query, filters=Filters(), k=10, mode="hybrid"):
        vec = self.embedder.embed([query])[0].tolist()
        res = self.es.search(index=self.index, body=self.build_request(query, vec, filters, k))
        return [Hit(h["_source"]["id"], h["_score"], h["_source"]["title"]) for h in res["hits"]["hits"]]

    @staticmethod
    def mapping(dims: int) -> dict:
        return {"mappings": {"properties": {
            "id": {"type": "keyword"}, "title": {"type": "text"}, "brand": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "category": {"type": "keyword"}, "description": {"type": "text"}, "price_cents": {"type": "integer"},
            "in_stock": {"type": "boolean"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine",
                          "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100}},
        }}}


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(relevant.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0
