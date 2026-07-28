# DocMind Upload API

FastAPI service wrapping the DocMind ingestion pipeline (`../upload.py`): accepts a PDF over HTTP, chunks it (regex + semantic), embeds the chunks, and upserts them into Pinecone.

See [PLAN.md](PLAN.md) for the design rationale and decisions.

## Layout

- **`config.py`** — pydantic-settings config (index name, batch size, chunk thresholds, upload size cap). Loads `.env` from the project root.
- **`services/pdf_loader.py`** — validates the upload (size, `%PDF-` magic bytes), writes it to a temp file for `PyPDFLoader`, cleans up afterward.
- **`services/chunking.py`** — `hybrid_chunk`: regex structural split, falling back to semantic sub-splitting for oversized sections.
- **`services/embeddings.py`** — batch embedding calls via the OpenAI client.
- **`services/vector_store.py`** — builds vector records, batched embed+upsert, and `replace_namespace_if_exists` (deletes a namespace before re-upserting, so re-uploading the same filename never leaves stale chunks behind).
- **`routers/upload.py`** — the `POST /upload` route and its response model.
- **`app.py`** — FastAPI app; `lifespan` builds the OpenAI/Pinecone clients and the `SemanticChunker` once at startup rather than per-request.
- **`tests/`** — unit tests for `hybrid_chunk` and integration tests for `POST /upload` (OpenAI/Pinecone calls mocked, no real API usage).

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

## Running

From the `DocMind/` project root:

```bash
uv sync --extra dev
uv run uvicorn upload_api.app:app --reload
```

```bash
curl -F file=@input/<yourfile>.pdf http://localhost:8000/upload
```

Interactive docs at `http://localhost:8000/docs`.

## Testing

```bash
uv run pytest
```

All OpenAI and Pinecone calls are mocked in tests — no real API keys are hit and nothing is written to the live Pinecone index.
