# PLAN: FastAPI `POST /upload` endpoint for DocMind

Wraps the existing `upload.py` script (PDF → hybrid chunk → embed → upsert to Pinecone) in a FastAPI service. The script becomes reusable service functions; the endpoint accepts an uploaded PDF instead of a hardcoded local path.

## 1. Scope

- Accept a PDF via `multipart/form-data` upload.
- Run the same pipeline: `PyPDFLoader` → `hybrid_chunk` (regex + `SemanticChunker`) → OpenAI embeddings → Pinecone upsert (namespaced by filename, as today).
- Return a JSON summary (chunk count, namespace, per-batch upsert status).
- Out of scope for v1: auth, multi-tenant namespacing beyond filename, OCR for scanned PDFs (empty-text PDFs are rejected with a clear error instead — see §3).

## 2. Project layout

```
DocMind/
├── upload.py                  # existing script — kept as-is or removed once API replaces it
├── pinecone_index_create.py
└── upload_api/
    ├── PLAN.md
    ├── app.py                 # FastAPI() instance, lifespan (init OpenAI/Pinecone clients once)
    ├── config.py               # pydantic-settings: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME, BATCH_SIZE, chunk thresholds
    ├── routers/
    │   └── upload.py           # POST /upload route, request/response models
    ├── services/
    │   ├── pdf_loader.py        # save UploadFile to temp path, load via PyPDFLoader, cleanup
    │   ├── chunking.py          # hybrid_chunk(), moved verbatim from upload.py (with bug fix, see §6)
    │   ├── embeddings.py        # get_embeddings() batch helper
    │   └── vector_store.py      # build vector records + namespaced upsert
    └── tests/
        ├── conftest.py          # fixture PDF, mocked OpenAI/Pinecone clients
        └── test_upload.py
```

Dependencies to add to `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `python-multipart` (required by FastAPI for form/file uploads), `pydantic-settings`.

## 3. API contract

**`POST /upload`**

- Route path is exactly `/upload` — `routers/upload.py`'s `APIRouter` must be included in `app.py` with **no prefix** (`app.include_router(upload.router)`), so the module filename doesn't accidentally double up the path (e.g. `/upload/upload`).
- Request: `multipart/form-data`, field `file` (PDF only).
- Validation before processing:
  - content-type is `application/pdf` (belt-and-suspenders: also sniff the `%PDF-` magic bytes, since content-type is client-supplied and unreliable).
  - file size under **10 MB** (configurable via `config.py`) — reject early with `413`.
- Response `200`:
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
  `replaced_existing` is `true` when a prior namespace for this filename was cleared before the new upsert (see §4 clean-replace step).
- Error responses:
  - `400` — not a PDF / unreadable.
  - `413` — over the 10 MB limit.
  - `422` — PDF parsed but no extractable text on any page (e.g. scanned/image-only PDF); response body names this explicitly, e.g. `{"detail": "No extractable text found in PDF — scanned/image-only PDFs are not supported."}`. OCR fallback is out of scope for now (§1).
  - `502` — OpenAI or Pinecone call failed after retries.
  - `500` — unexpected.

## 4. Processing flow inside the endpoint

1. Validate upload (type, size — reject over 10 MB with `413`).
2. Write `UploadFile` to a temp file (`tempfile.NamedTemporaryFile`) — `PyPDFLoader` needs a filesystem path, not a stream.
3. Load pages, run `hybrid_chunk` per page (reuse existing logic). If no chunk survives across all pages (empty/scanned PDF), skip straight to cleanup and return `422`.
4. Compute the target namespace (sanitized filename). **Clean replace:** query/delete any existing vectors in that namespace (`index.delete(delete_all=True, namespace=ns)`) *before* upserting the new ones, so a re-upload never leaves stale chunks from a previous version behind. Track whether a delete actually happened, for the `replaced_existing` response field.
5. Build vector records; batch-embed and upsert to Pinecone exactly as the script does today, batched by `BATCH_SIZE`.
6. Clean up temp file in a `finally` block regardless of success/failure.
7. Return summary response.

**Synchronous processing (decided):** the route responds only once chunking/embedding/upsert fully completes — no background job/status polling. Define the route as `def` (not `async def`) so FastAPI runs it in its threadpool automatically rather than blocking the event loop. Given the 10 MB cap, worst-case latency stays bounded (tens of seconds), so this stays simple for v1.

## 5. Client/config lifecycle

- Initialize `OpenAI()`, `Pinecone()`, `pc.Index(INDEX_NAME)`, and `OpenAIEmbeddings`/`SemanticChunker` **once** at app startup (FastAPI `lifespan`), not per-request — the current script re-creates them at import time, which is fine for a script but wasteful per-request in a server.
- Pull `INDEX_NAME`, `BATCH_SIZE`, chunking thresholds (`min_chars`, `max_chars`, `sem_threshold_type`, `sem_threshold_amt`) into `config.py` via `pydantic-settings`, defaulting to today's hardcoded values.

## 6. Bugs / cleanup to fix while porting (found in `upload.py`)

- `raw_sections_final.append(raw_sections)` (upload.py:47) appends the *entire list* on every iteration and the variable is never read afterward — dead code, drop it during the port.
- Vector IDs are `chunk-{i}` using a counter that restarts at 0 for every run/upload. This is now moot for the re-upload case: §4's clean-replace step deletes the whole namespace before re-upserting, so no stale vectors can survive a re-upload regardless of chunk-count drift. IDs only need to be unique *within* a single upload's namespace, which `chunk-{i}` already satisfies.
- Debug `print(chunk)` / `print(stats)` statements — replace with proper logging (or drop) in the service layer.
- `embeddings` (langchain `OpenAIEmbeddings`, used only inside `SemanticChunker`) and `openai_client` (raw `OpenAI()`, used for the actual stored embeddings) are two separate clients hitting the same model — intentional but worth a one-line comment in the ported code so it doesn't look like duplication.

## 7. Decisions

- **Large-file handling:** synchronous response (see §4).
- **Re-upload behavior:** clean replace — delete the existing namespace's vectors before upserting the new ones (see §4).
- **File size limit:** 10 MB.
- **Scanned PDFs:** out of scope; endpoint returns `422` with an explicit "no extractable text" message instead of attempting OCR (see §3).

## 8. Testing plan

- Unit test `hybrid_chunk` with small/large synthetic text (no API calls).
- Integration test for `POST /upload` with a small fixture PDF, mocking `OpenAI` embeddings and `Pinecone` upsert/delete calls (via `unittest.mock` or `respx`/`pytest-mock`) so tests don't hit real APIs or cost money.
- Test validation paths: non-PDF upload → `400`; oversized upload → `413`; empty/scanned PDF → `422`.
- Test clean-replace: upload the same filename twice with different content, assert the namespace's vector count/content reflects only the second upload (`delete` called before the second `upsert`).
