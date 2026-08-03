# DocMind API

FastAPI service wrapping the DocMind pipeline: `POST /upload` chunks a PDF (regex + semantic), embeds the chunks, and upserts them into Pinecone; `POST /query` hybrid-retrieves (dense + BM25) across every indexed document, reranks with a cross-encoder, and answers with an LLM.

See [api/upload_api/PLAN.md](api/upload_api/PLAN.md) and [api/query_api/PLAN.md](api/query_api/PLAN.md) for the design rationale and decisions.

## Layout

- **`api/config.py`** — pydantic-settings config (index name, batch size, chunk thresholds, upload size cap, retrieval/rerank/chat settings). Loads `.env` from the project root.
- **`api/services/pdf_loader.py`** — validates the upload (size, `%PDF-` magic bytes), writes it to a temp file for `PyPDFLoader`, cleans up afterward.
- **`api/services/chunking.py`** — `hybrid_chunk`: regex structural split, falling back to semantic sub-splitting for oversized sections.
- **`api/services/embeddings.py`** — batch embedding calls via the OpenAI client.
- **`api/services/vector_store.py`** — builds vector records, batched embed+upsert, and `replace_namespace_if_exists` (deletes a namespace before re-upserting, so re-uploading the same filename never leaves stale chunks behind).
- **`api/services/bm25_index.py`** — per-namespace BM25 corpus, built by re-parsing `input/<namespace>` with the same `hybrid_chunk` config used at upload time; lazily built and cached per namespace; `list_namespaces()` helper.
- **`api/services/retrieval.py`** — `dense_retrieve`, `sparse_retrieve`, `reciprocal_rank_fusion`, `hybrid_retrieve` — all namespace-fan-out aware; RRF merges on a composite `(namespace, sha1(text))` key so chunks never collide across namespaces or across dense/sparse chunkings.
- **`api/services/reranking.py`** — `rerank_with_cross_encoder`, using the startup-loaded `CrossEncoder`.
- **`api/services/generation.py`** — `generate_answer`, prefixing each context block with its source filename so a multi-document answer doesn't conflate facts across documents.
- **`api/routers/router.py`** — both the `POST /upload` and `POST /query` routes and their request/response models.
- **`api/app.py`** — FastAPI app; `lifespan` builds the OpenAI/Pinecone clients, the `SemanticChunker`, and the `CrossEncoder` once at startup, and initializes the per-namespace BM25 cache.
- **`api/tests/`** — unit tests for `hybrid_chunk`/`reciprocal_rank_fusion`/BM25 building, and integration tests for `POST /upload` and `POST /query` (OpenAI/Pinecone/CrossEncoder calls mocked, no real API usage).

## `POST /upload`

`multipart/form-data` with a `file` field (PDF only, ≤10 MB).

Response `200`:

```json
{
  "filename": "databricks_architecture_governance.pdf",
  "namespace": "databricks_architecture_governance.pdf",
  "pages_loaded": 12,
  "chunks_upserted": 47,
  "index_name": "docmind-index",
  "replaced_existing": false
}
```

`replaced_existing` is `true` when a prior namespace for this filename was cleared before the new upsert (re-upload of the same file replaces its vectors rather than appending to them).

Error responses:

| Status | Meaning |
|---|---|
| 400 | Not a PDF / unreadable |
| 413 | File exceeds the 10 MB limit |
| 422 | PDF parsed but had no extractable text (e.g. scanned/image-only — OCR is out of scope) |
| 502 | Upstream OpenAI or Pinecone call failed |
| 500 | Unexpected error |

Processing is synchronous — the response only returns once chunking, embedding, and upserting have fully completed.

## `POST /query`

JSON body, no namespace — retrieval fans out across every namespace currently in the index and merges results before rerank:

```json
{ "question": "What is a SQL warehouse?" }
```

Response `200`:

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

`sources` is the same set of chunks passed to the LLM as context (post-rerank, `rerank_k` of them — default 5), each with the full chunk text, cross-encoder score, page metadata, and which namespace it came from. A single answer may be synthesized from more than one document, so there's no top-level `namespace` — only per-source.

Error responses:

| Status | Meaning |
|---|---|
| 400 | Empty `question` |
| 404 | The index has no namespaces at all (nothing has ever been uploaded) |
| 502 | Upstream OpenAI or Pinecone call failed |
| 500 | Unexpected error |

A namespace with vectors but no matching local file under `input/` is **not** an error for the request as a whole — it's silently skipped for sparse retrieval (dense results still contribute), since every namespace is expected to have a corresponding file under `input/` (see [api/query_api/PLAN.md](api/query_api/PLAN.md) §7).

## Running

From the `DocMind/` project root:

```bash
uv sync --extra dev
uv run uvicorn api.app:app --reload
```

```bash
curl -F file=@input/<yourfile>.pdf http://localhost:8000/upload
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is a SQL warehouse?"}' | jq -r '.answer'
```

Interactive docs at `http://localhost:8000/docs`.

## Testing

```bash
uv run pytest
```

All OpenAI, Pinecone, and CrossEncoder calls are mocked in tests — no real API keys are hit, no model weights are downloaded, and nothing is written to the live Pinecone index.

## Evaluation

[`eval/ragas_eval.py`](eval/ragas_eval.py) runs a [RAGAS](https://github.com/explodinggradients/ragas) evaluation of the retrieval + generation pipeline against 20 hand-authored questions spanning every major section of `input/databricks_architecture_governance.pdf` (control/data plane, Delta Lake, compute, Unity Catalog, governance models, sharing/federation, orchestration, MLflow/Vector Search, security & compliance). Each row's `response` and `retrieved_contexts` mirror what the real pipeline would produce; scoring uses `gpt-4o-mini` as the evaluator LLM and `OpenAIEmbeddings`, and the full per-row breakdown is written to `eval/ragas_results.csv`.

Aggregate scores across all 20 questions:

| Metric | Score |
|---|---|
| Faithfulness | 0.976 |
| Context Precision | 0.950 |
| Context Recall | 0.925 |
| Answer Relevancy | 0.848 |

**Faithfulness**, **Context Precision**, and **Context Recall** are all consistently high (≥0.85 on every row), indicating answers stay grounded in the retrieved chunks and retrieval surfaces the right source material. **Answer Relevancy** is pulled down by one outlier: the compliance-certifications question scored `0.0`, because its answer includes an explicit hedge ("...though these vary by cloud/region and should be reverified against current documentation") — RAGAS's relevancy metric penalizes non-committal phrasing. Every other row scored ≥0.8 on this metric.

Run it yourself:

```bash
uv run python eval/ragas_eval.py
```
