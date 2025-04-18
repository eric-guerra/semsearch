# SemSearch

Portfolio reference implementation of a 2024 project, built to show hybrid product search for a home-goods catalog. Shoppers type "settee" or "bookshelf"; listings say "Sofa" and "Bookcase". Lexical search alone misses them, and vector search alone loses precision on brand and size queries.

**Stack:** FastAPI, Elasticsearch 8 (BM25 + `dense_vector` int8 HNSW kNN, fused with the RRF retriever), NumPy, scikit-learn.

## How it ranks
1. **BM25** over title^3, brand^2, category and description, which keeps exact terms precise.
2. **kNN** over product embeddings, which handles meaning.
3. **Reciprocal Rank Fusion** (k = 60) over the top 50 from each retriever. RRF fuses ranks, not raw scores, so BM25's and cosine's incomparable scales don't matter.

Filters (category, maximum price, in stock) are applied **inside both retrievers**: as a filter on the kNN clause, not afterwards. Post-filtering kNN results starves filtered queries of results.

`ElasticBackend.build_request()` emits the ES 8 `retriever.rrf` request. `MemoryBackend` implements the same ranking in-process for offline evaluation and CI.

## Evaluation (NDCG@10, held-out queries)
Synthetic catalog: 320 products in 8 categories. Only about 25% of listings use the shopper's alternate word anywhere. There are 90 judged queries in three kinds: *semantic* (a synonym never used in the title), *lexical* (brand + product type) and *attribute* (color + synonym). The embedding size (16) was tuned on a separate query set (seed 11) and reported here on seed 23.

| Mode | Semantic | Lexical | Attribute | All |
|---|---|---|---|---|
| BM25 | 0.565 | **0.996** | 0.459 | 0.673 |
| Vector | 0.825 | 0.963 | 0.458 | 0.749 |
| **Hybrid (RRF)** | **0.832** | 0.970 | **0.500** | **0.768** |

Hybrid recovers the vocabulary-mismatch queries while keeping brand queries precise.

**Known limit (covered by a test):** the bundled `LsaEmbedder` (TF-IDF + SVD) learns only from catalog text. A word the catalog never uses ("carpet") can't be placed at all. In production a pretrained sentence-embedding model sits behind the same `Embedder` interface. LSA is here so the pipeline and the evaluation run without a model download.

## Run and test
```bash
pip install -r requirements.txt
pytest -q                                     # 7 tests incl. the held-out relevance bar
uvicorn semsearch.api:app                     # in-memory backend; set ES_URL to use Elasticsearch
curl 'localhost:8000/search?q=navy+settee&max_price=900'
```
