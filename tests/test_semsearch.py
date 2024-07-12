import os

os.environ["SEMSEARCH_AUTOSTART"] = "0"

import pytest
from fastapi.testclient import TestClient

from semsearch.api import create_app
from semsearch.catalog import build_catalog, judged_queries
from semsearch.search import Bm25, ElasticBackend, Filters, MemoryBackend, ndcg_at_k, rrf, tokenize


@pytest.fixture(scope="module")
def backend():
    return MemoryBackend(build_catalog())


def evaluate(b, queries):
    by = {}
    f = Filters(in_stock_only=False)
    for q, kind, rel in queries:
        by.setdefault(kind, []).append(ndcg_at_k([h.id for h in b.search(q, f, 10, MODE)], rel))
    return {k: sum(v) / len(v) for k, v in by.items()}


MODE = "hybrid"


def scores(b, mode, queries):
    global MODE
    MODE = mode
    r = evaluate(b, queries)
    r["all"] = sum(r[k] * sum(1 for q in queries if q[1] == k) for k in r) / len(queries)
    return r


def test_hybrid_beats_bm25_on_held_out_queries(backend):
    held_out = judged_queries(backend.products, seed=23, n=90)  # dims were tuned on seed 11
    bm25, hybrid = scores(backend, "bm25", held_out), scores(backend, "hybrid", held_out)
    assert hybrid["all"] > bm25["all"] + 0.05, (bm25, hybrid)
    assert hybrid["semantic"] > bm25["semantic"] + 0.2   # "settee" finds sofas
    assert hybrid["lexical"] >= 0.95                      # brand queries stay precise


def test_vocabulary_mismatch_example(backend):
    f = Filters(in_stock_only=False)
    top = backend.search("settee", f, 10)
    assert sum(backend.by_id[h.id].category == "sofas" for h in top) >= 9
    assert len(backend.search("settee", f, 10, "bm25")) < 10  # BM25 only finds listings whose copy says "settee"


def test_known_limit_of_corpus_trained_embeddings(backend):
    # "carpet" never appears in the catalog, so LSA can't place it; a pretrained embedding model can.
    # This is why production swaps LsaEmbedder for a hosted sentence-embedding model behind the same interface.
    assert not any("carpet" in p.text.lower() for p in backend.products)
    top = backend.search("carpet", Filters(in_stock_only=False), 10)
    assert sum(backend.by_id[h.id].category == "rugs" for h in top) < 5


def test_filters_apply_before_ranking(backend):
    hits = backend.search("lamp", Filters(category="lighting", max_price_cents=50_000), 20)
    assert hits and all(backend.by_id[h.id].category == "lighting" and backend.by_id[h.id].price_cents <= 50_000 and backend.by_id[h.id].in_stock for h in hits)


def test_rrf_and_bm25_basics():
    assert [d for d, _ in rrf([["a", "b", "c"], ["c", "a"]])] == ["a", "c", "b"]
    b = Bm25([tokenize("oak bookcase"), tokenize("walnut bookcase with five shelves"), tokenize("linen sofa")])
    s = b.scores(tokenize("oak bookcase"))
    assert s[0] > s[1] > s[2] == 0
    assert ndcg_at_k(["x", "y"], {"x": 3, "y": 1}) == pytest.approx(1.0)
    assert ndcg_at_k(["y", "x"], {"x": 3, "y": 1}) < 1


def test_elasticsearch_request_uses_rrf_with_filters_in_both_retrievers():
    req = ElasticBackend.build_request("navy couch", [0.1, 0.2], Filters(category="sofas", max_price_cents=100_000), k=12)
    r = req["retriever"]["rrf"]
    std, knn = r["retrievers"][0]["standard"], r["retrievers"][1]["knn"]
    assert req["size"] == 12 and r["rank_constant"] == 60
    assert knn["field"] == "embedding" and knn["num_candidates"] >= knn["k"]
    assert knn["filter"] == std["query"]["bool"]["filter"]
    assert {"term": {"category": "sofas"}} in knn["filter"] and {"term": {"in_stock": True}} in knn["filter"]
    m = ElasticBackend.mapping(16)["mappings"]["properties"]["embedding"]
    assert m["type"] == "dense_vector" and m["dims"] == 16 and m["similarity"] == "cosine"


def test_api(backend):
    c = TestClient(create_app(backend))
    r = c.get("/search", params={"q": "duvet", "max_price": 600}).json()
    assert r["mode"] == "hybrid" and r["hits"]
    assert c.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422
    assert c.get("/search", params={"q": ""}).status_code == 422
