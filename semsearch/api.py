"""FastAPI search endpoint. Backend is Elasticsearch in production, in-memory for dev/tests."""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, Query

from .search import Filters, MemoryBackend, SearchBackend


def create_app(backend: SearchBackend) -> FastAPI:
    app = FastAPI(title="SemSearch")

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=200),
               category: str | None = None,
               max_price: int | None = Query(default=None, ge=0, description="dollars"),
               include_out_of_stock: bool = False,
               mode: str = Query(default="hybrid", pattern="^(hybrid|bm25|vector)$"),
               k: int = Query(default=10, ge=1, le=50)):
        t0 = time.perf_counter()
        f = Filters(category=category, max_price_cents=None if max_price is None else max_price * 100,
                    in_stock_only=not include_out_of_stock)
        hits = backend.search(q, f, k, mode)
        return {"query": q, "mode": mode, "took_ms": round((time.perf_counter() - t0) * 1000, 2),
                "hits": [h.__dict__ for h in hits]}

    return app


def _default_backend() -> SearchBackend:
    if os.environ.get("ES_URL"):
        from elasticsearch import Elasticsearch

        from .catalog import build_catalog
        from .search import ElasticBackend, LsaEmbedder
        emb = LsaEmbedder().fit([p.text for p in build_catalog()])  # production: a hosted sentence-embedding model
        return ElasticBackend(Elasticsearch(os.environ["ES_URL"]), emb)
    from .catalog import build_catalog
    return MemoryBackend(build_catalog())


app = create_app(_default_backend()) if os.environ.get("SEMSEARCH_AUTOSTART", "1") == "1" else None
