# PLAN: FastAPI `POST /query` endpoint for DocMind

Wraps the existing `query.py` script (hybrid retrieve → cross-encoder rerank → LLM answer) as a FastAPI endpoint. As with `upload_api/` (see [../upload_api/PLAN.md](../upload_api/PLAN.md)), the actual code lands flat under `api/` (`api/routers/`, `api/services/`, `api/tests/`) alongside the existing upload endpoint — `query_api/` holds only this planning doc.

## 0. Pre-existing issue found — fix first

Running `uv run pytest` today fails at collection:

```
api/tests/conftest.py:7: in <module>
    from upload_api.app import app
E   ModuleNotFoundError: No module named 'upload_api'
```

`app.py` actually lives at `api/app.py`, but `conftest.py`, `test_upload.py`, `pyproject.toml`'s `testpaths`, and the upload README's run/uvicorn instructions all still reference `upload_api.*` — a leftover from before the code was flattened into `api/`. **No tests currently run at all.** Fixing this is a prerequisite for adding query tests:

- `api/tests/conftest.py` and `api/tests/test_upload.py`: `from upload_api.app import app` → `from api.app import app`.
- `pyproject.toml`: `testpaths = ["upload_api/tests"]` → `testpaths = ["api/tests"]`.
- `api/README.md`: `uv run uvicorn upload_api.app:app --reload` → `uv run uvicorn api.app:app --reload`.

## 1. Scope

- Accept `{"question": str}` — **no namespace in the request.** Retrieval fans out across *every* namespace currently in the Pinecone index and merges the results, so the answer can draw on whichever uploaded document(s) are actually relevant.
- Reuse the existing pipeline: hybrid retrieval (dense via Pinecone + sparse via BM25) → RRF fusion → cross-encoder rerank → OpenAI chat completion — now run per-namespace for the dense/sparse steps and merged before rerank.
- **Multi-document, namespace-transparent to the caller**: the API discovers namespaces itself (via `describe_index_stats`) rather than the client naming one, unlike `upload_api`'s `namespace`/`filename`-scoped model. Each returned source still reports which document/namespace it came from.
- Out of scope for v1: auth, conversational/multi-turn queries (each request is a single independent question), streaming responses, letting the caller scope a query to a specific document (could be added later as an optional filter, not required now).

## 2. Files touched/added

```
DocMind/
├── query.py                    # existing script — kept as reference or removed once API replaces it
└── api/
    ├── config.py                # extend: chat model, cross-encoder model, retrieval top_k's, rerank_k, rrf_k
    ├── app.py                   # extend lifespan: load CrossEncoder once; init empty BM25 cache; import update (see below)
    ├── routers/
    │   └── router.py            # renamed from upload.py — now holds both POST /upload and POST /query routes
    ├── services/
    │   ├── retrieval.py         # dense_retrieve, sparse_retrieve, reciprocal_rank_fusion, hybrid_retrieve — all namespace-fan-out aware
    │   ├── bm25_index.py        # per-namespace BM25 corpus: build from input/<namespace>, cache in app.state; list_namespaces() helper
    │   ├── reranking.py         # rerank_with_cross_encoder
    │   └── generation.py        # generate_answer (chat completion)
    └── tests/
        ├── conftest.py          # add: mocked CrossEncoder, mocked chat completion, fixture PDF in a temp input dir
        └── test_query.py
```

**Router consolidation:** `routers/upload.py` gets renamed to `routers/router.py`, and the new `POST /query` route is added to the same file/`APIRouter` instance rather than a separate `query.py` — both endpoints now live together. This requires updating `api/app.py`'s import: `from .routers.upload import router as upload_router` → `from .routers.router import router`, and `app.include_router(upload_router)` → `app.include_router(router)` (one router, both routes). Test files stay split by endpoint (`test_upload.py`, `test_query.py`) — only the router module itself is merged.

Dependencies: no new ones — `rank-bm25` and `sentence-transformers` are already in `pyproject.toml` (added for `query.py`).

## 3. API contract

**`POST /query`**

- Added to the same `routers/router.py` `APIRouter` that already handles `/upload` (renamed from `upload.py` — see §2), included in `app.py` with no prefix, same convention as today.
- Request (JSON body):
  ```json
  { "question": "What is a SQL warehouse?" }
  ```
- Validation:
  - `question` non-empty after stripping → else `400`.
- Response `200`:
  ```json
  {
    "question": "What is a SQL warehouse?",
    "answer": "A SQL warehouse is ...",
    "sources": [
      {
        "text": "full text of the reranked chunk ...",
        "score": 4.21,
        "namespace": "databricks_architecture_governance.pdf",
        "source_filename": "databricks_architecture_governance.pdf",
        "page_number": 3
      }
    ]
  }
  ```
  - `sources` is the same set of chunks passed to the LLM as context (post-rerank, `rerank_k` of them — default 5), each with the **full chunk text** (not truncated), the cross-encoder score, page metadata, and which `namespace` it came from — since a single answer may now be synthesized from more than one document, the top-level response has no single `namespace` field, only per-source.
- Error responses:
  - `400` — empty `question`.
  - `404` — the Pinecone index has no namespaces at all (nothing has ever been uploaded) — nothing to search.
  - `502` — upstream OpenAI or Pinecone call failed.
  - `500` — unexpected.
  - Note: a namespace with vectors but no matching local file under `input/` is **not** an error for the request as a whole — it's silently skipped for sparse retrieval and still contributes dense results (see §6/§7, changed from the earlier per-namespace-404 design).

## 4. Processing flow inside the endpoint

1. Validate `question` (`400` if blank).
2. List current namespaces via `index.describe_index_stats().namespaces` (same call `replace_namespace_if_exists` already uses). `404` if there are none.
3. Embed the question **once** (single OpenAI embeddings call, reused for every namespace's dense query — no need to re-embed per namespace).
4. For each namespace: `dense_retrieve` (Pinecone `query(..., namespace=ns)` with the shared embedding, top `dense_top_k`), tagging each result with its `namespace`. Sequential loop for v1 (§5 notes parallelization as a later option).
5. For each namespace: resolve its BM25 index from `app.state.bm25_cache`, building and caching it on first use (§5) by locating `input/<namespace>` — expected to always exist per the project's workflow (§7). If it's ever missing anyway, **skip sparse retrieval for that namespace only** (log a warning) rather than failing the request, as a defensive fallback — that namespace still contributes dense results.
6. Concatenate all namespaces' dense results into one list and all sparse results into another, then `reciprocal_rank_fusion` the two combined lists — merged by a **composite key** of `(namespace, chunk_id)`, not `chunk_id` alone (§6: per-upload chunk IDs restart at 0, so raw IDs collide across documents too, not just across dense/sparse).
7. `rerank_with_cross_encoder` the fused candidates (now potentially spanning multiple documents) using the startup-loaded model; keep top `rerank_k` globally.
8. `generate_answer` from the top chunks (context may mix chunks from different source documents — the existing prompt already just concatenates chunk texts, which still works, but doesn't currently attribute a chunk to its document in the prompt itself; worth a light prompt tweak, e.g. prefixing each context block with its source filename, so the LLM doesn't conflate facts from different documents).
9. Return `answer` + the reranked chunks (with `namespace`/`source_filename`/`page_number`/`score`) as `sources`.

Define the route as `def` (not `async def`), same reasoning as `/upload` — FastAPI runs it in the threadpool so the blocking OpenAI/Pinecone/CrossEncoder calls don't stall the event loop.

## 5. Client/model lifecycle

- `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` loaded **once** in `lifespan`, stored on `app.state.cross_encoder` — it's the slowest thing to initialize (same rationale as `SemanticChunker` today).
- BM25 indexes are **per-namespace** and expensive to (re)build (PDF load + split + tokenize), so they're built lazily and cached in `app.state.bm25_cache: dict[str, BM25NamespaceIndex]` for the life of the process. Since every query now touches every namespace, the cache fills in fully after the first request rather than per-document-as-queried; still lazy (built on first sight of a namespace, not at startup) since new namespaces can appear later via `/upload`. No eviction for v1. As the number of documents grows, per-namespace dense/sparse fan-out (§4 steps 4–5) becomes the main latency cost — sequential looping is fine for a handful of documents; parallelizing the per-namespace Pinecone queries (e.g. `concurrent.futures.ThreadPoolExecutor`) is a natural follow-up if that becomes slow, not needed for v1.
- New `config.py` settings: `chat_model` (`gpt-4o-mini`), `cross_encoder_model`, `dense_top_k` (10), `sparse_top_k` (10), `hybrid_top_k` (20), `rerank_k` (5), `rrf_k` (60) — same per-namespace defaults as today's hardcoded values in `query.py`, now applied per namespace before the cross-namespace merge.

## 6. Bugs / correctness issues found while porting `query.py`

- **Critical — RRF merges unrelated chunks, now two ways.** `dense_retrieve` returns Pinecone vector IDs (`chunk-{i}`, assigned by `upload_api`'s `hybrid_chunk` pass at upload time, restarting at 0 for *every* upload/namespace). `sparse_retrieve` in the script builds a *separate* BM25 corpus by re-splitting the PDF with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` — different boundaries entirely — and also IDs it `chunk-{idx}`, again restarting at 0. `reciprocal_rank_fusion` merges by that shared `id` key, so (a) `chunk-3` from dense and `chunk-3` from sparse get treated as the same document even though they're different text spans from two unrelated chunkings, and now that retrieval fans out across namespaces, (b) `chunk-3` from *document A* and `chunk-3` from *document B* would also collide, since IDs aren't globally unique. **Fix:** the BM25 corpus for a namespace must be built with the *exact same* `hybrid_chunk()` call (same `min_chars`/`max_chars`/`sem_chunker` config) used at upload time, in the same order, assigned the same `chunk-{i}` IDs — so a given ID means the same chunk within that namespace in both retrieval paths — **and** every merge/fusion key must be the composite `(namespace, chunk_id)`, not `chunk_id` alone, to keep documents from colliding with each other. As a more robust alternative to positional-index matching (doesn't depend on chunking being perfectly deterministic across processes), key by `(namespace, sha1(text))` instead — recommended.
- `import numpy as npup` (query.py:3) but line 51 calls `np.argsort(...)` — `np` is never defined; `sparse_retrieve` would raise `NameError` the moment it runs with real (non-degenerate) scores. Fix the import name when porting.
- Module-level side effects at import time (`PyPDFLoader(...).load()`, `BM25Okapi(...)`, `CrossEncoder(...)` all run at `import query`) — restructured into the lifespan/lazy-cache pattern in §5, not run at module import.
- Minimal `STOPWORDS` set and decorative unicode-box comments (`# ■■ Sparse retrieval ...`) — drop the decorative comments; stopword list can be ported as-is (not a functional bug, just worth a glance).

## 7. Project invariant: documents live in `input/`

Confirmed: for this project, it's mandatory that every document is manually placed under `input/` (matching how the original `upload.py`/`query.py` scripts already worked) — `POST /upload`'s per-request temp file (written by `pdf_loader.py`, deleted in a `finally` block right after ingestion) is not the expected way documents enter the system here. Given that, BM25 re-parsing `input/<namespace>` per namespace (§4 step 3 of the earlier draft, now step 5) isn't a gap to design around — it's expected to always succeed, since a namespace only exists in Pinecone because someone put the corresponding file in `input/` first.

The "skip sparse retrieval for a namespace whose file is missing" behavior (§4 step 5) is kept as a **defensive safety net**, not an expected code path — e.g. covering someone renaming/deleting a file in `input/` after it was already indexed. It should never trigger in normal operation; if it does, that's a sign of drift between `input/` and the Pinecone index worth investigating, not a routine degraded mode.

(For the record, in case the workflow changes later: `POST /upload` embeds/upserts into Pinecone — which already stores each chunk's `text` in vector metadata via `vector_store.py`'s `upsert_chunks` — independently of `input/`. A future revision could build BM25 from that stored metadata instead of the local file, removing the dependency on `input/` altogether. Not needed given the current mandatory-manual-placement workflow.)

## 8. Decisions

- **Document scope:** multi-document, **no `namespace` in the request** — the endpoint discovers all namespaces via `describe_index_stats` and fans out retrieval across every one of them, merging results before rerank (§1, §3, §4).
- **BM25 corpus source:** re-parse the local PDF under `input/<namespace>`, per namespace — expected to always be present since manual placement in `input/` is mandatory for this project (§7); a missing file is handled defensively (skip that namespace's sparse retrieval) but shouldn't occur in normal operation.
- **Model lifecycle:** `CrossEncoder` loaded once at startup; BM25 indexes built lazily per namespace (as namespaces are encountered) and cached in memory for the process lifetime (§5).
- **Response shape:** full chunk text (not truncated) plus score/namespace/filename/page metadata per source; no top-level `namespace` since one answer can span documents (§3).
- **Chunk ID alignment fix:** merge/fusion keys must be `(namespace, chunk_id)` (or `(namespace, content_hash)`), not `chunk_id` alone — needed both to align dense/sparse for the same document and to keep different documents from colliding (§6).
- **Router file:** `routers/upload.py` renamed to `routers/router.py`; both `/upload` and `/query` are defined on the same `APIRouter` in that one file (§2).

## 9. Testing plan

- Unit test `reciprocal_rank_fusion` directly (pure function, no mocks needed) — including the case where the same `chunk_id` appears in two different namespaces and must **not** be merged together (the cross-document collision from §6).
- Unit test `sparse_retrieve`/BM25 index building against a small fixture PDF (reuse `conftest.py`'s `make_minimal_pdf` helper, written to a temp `input/`-like dir monkeypatched via config).
- Integration test `POST /query` with **multiple namespaces** present: mock `OpenAI` embeddings + chat completion, mock Pinecone `query`/`describe_index_stats` to return 2+ namespaces, mock (or use a tiny real) `CrossEncoder`, fixture PDFs for BM25. Assert the question is embedded once and reused, `query` is called once per namespace, and `sources` can include chunks tagged with different `namespace` values.
- Integration test for the defensive fallback: a namespace present in Pinecone with **no** matching file in `input/` (simulating drift from the mandatory-`input/`-placement invariant, §7) still contributes dense results and doesn't fail the request — assert no `404`/`500`, and that namespace's sources (if any) came from dense retrieval only.
- Validation tests: empty `question` → `400`; **zero namespaces in the index** → `404`.
- Regression test for the `np`/`npup` import bug (§6): a request path that exercises `sparse_retrieve` with a non-degenerate score should not raise `NameError`.
- After §0's import-path fix, confirm `uv run pytest` collects and runs both `test_upload.py` and the new `test_query.py`.
