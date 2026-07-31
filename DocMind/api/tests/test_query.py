from types import SimpleNamespace
from unittest.mock import MagicMock

from api.app import app
from api.services.bm25_index import BM25NamespaceIndex, build_bm25_index
from api.services.retrieval import reciprocal_rank_fusion, sparse_retrieve


def make_multi_page_pdf(texts: list[str]) -> bytes:
    """Multi-page hand-rolled PDF (à la conftest's make_minimal_pdf). BM25's
    classic idf, log((N - n + 0.5) / (n + 0.5)), is <= 0 whenever a term appears
    in half or more of the corpus's documents — with rank_bm25's BM25Okapi that
    means at least 3 pages/chunks are needed before a term unique to one of them
    gets a positive (i.e. non-filtered) score."""
    n = len(texts)
    kids = " ".join(f"{i} 0 R" for i in range(3, 3 + n))
    font_obj_num = 3 + 2 * n  # after n page objs + n content-stream objs

    def content(text: str) -> bytes:
        return f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode()

    contents = [content(t) for t in texts]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
    ]
    for i, c in enumerate(contents):
        content_obj_num = 3 + n + i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_obj_num} 0 R >>".encode()
        )
    for c in contents:
        objects.append(
            b"<< /Length " + str(len(c)).encode() + b" >>\nstream\n" + c + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    parts = [b"%PDF-1.4\n"]
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(sum(len(p) for p in parts))
        parts.append(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = sum(len(p) for p in parts)
    xref = [b"xref\n", f"0 {len(objects) + 1}\n".encode(), b"0000000000 65535 f \n"]
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n".encode())
    parts.extend(xref)
    parts.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF".encode()
    )
    return b"".join(parts)


# -- reciprocal_rank_fusion: pure function, no mocks needed -----------------


def test_rrf_merges_matching_keys_and_sums_rank_scores():
    dense = [{"key": ("ns", "a"), "text": "chunk a"}, {"key": ("ns", "b"), "text": "chunk b"}]
    sparse = [{"key": ("ns", "b"), "text": "chunk b"}, {"key": ("ns", "a"), "text": "chunk a"}]

    merged = reciprocal_rank_fusion([dense, sparse], k=60)

    assert len(merged) == 2
    # Both keys appear once in each list (ranks 1 and 2), so both get the same
    # combined score and either ordering is valid — just confirm no duplication.
    assert {doc["key"] for doc in merged} == {("ns", "a"), ("ns", "b")}


def test_rrf_composite_key_prevents_cross_namespace_collision():
    # Same bare id ("hash1") reused across two different namespaces — the fix in
    # PLAN.md §6 requires the composite (namespace, id) key so these two distinct
    # chunks from different documents are never merged into one entry.
    dense = [
        {"key": ("doc_a.pdf", "hash1"), "text": "chunk from doc A", "namespace": "doc_a.pdf"},
    ]
    sparse = [
        {
            "key": ("doc_b.pdf", "hash1"),
            "text": "different chunk from doc B",
            "namespace": "doc_b.pdf",
        },
    ]

    merged = reciprocal_rank_fusion([dense, sparse])

    assert len(merged) == 2
    assert {doc["namespace"] for doc in merged} == {"doc_a.pdf", "doc_b.pdf"}


# -- BM25 index building / sparse_retrieve -----------------------------------


def test_build_bm25_index_and_sparse_retrieve(tmp_path):
    namespace = "sample.pdf"
    (tmp_path / namespace).write_bytes(
        make_multi_page_pdf([
            "Alpha paragraph about hybrid retrieval systems in DocMind indexing.",
            "Beta paragraph about completely unrelated gardening topics outdoors.",
            "Gamma paragraph about weather patterns and ocean currents worldwide.",
        ])
    )

    settings = SimpleNamespace(input_dir=str(tmp_path), chunk_min_chars=10, chunk_max_chars=2000)
    bm25_index = build_bm25_index(namespace, settings, sem_chunker=MagicMock())

    assert isinstance(bm25_index, BM25NamespaceIndex)
    assert len(bm25_index.chunks) == 3

    # Regression guard for the query.py `np`/`npup` import bug (PLAN.md §6) — this
    # exercises sparse_retrieve's score-ranking path with real, non-degenerate
    # BM25 scores and must not raise NameError.
    results = sparse_retrieve(bm25_index, "hybrid retrieval", namespace, top_k=5)
    assert results
    assert all(r["namespace"] == namespace for r in results)
    assert all(r["score"] > 0 for r in results)
    assert "Alpha" in results[0]["text"]


def test_build_bm25_index_missing_file_returns_none(tmp_path):
    settings = SimpleNamespace(input_dir=str(tmp_path), chunk_min_chars=10, chunk_max_chars=2000)
    assert build_bm25_index("missing.pdf", settings, sem_chunker=MagicMock()) is None


# -- POST /query integration --------------------------------------------------


def _fake_query_returning(namespace_texts: dict) -> callable:
    def fake_query(*, vector, top_k, include_metadata, namespace):
        text = namespace_texts[namespace]
        match = SimpleNamespace(
            id="chunk-0",
            score=0.9,
            metadata={"text": text, "source_filename": namespace, "page_number": 1},
        )
        return SimpleNamespace(matches=[match])

    return fake_query


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


def test_query_no_namespaces_returns_404(client):
    app.state.pinecone_index.describe_index_stats.return_value = SimpleNamespace(namespaces={})
    response = client.post("/query", json={"question": "What is a SQL warehouse?"})
    assert response.status_code == 404


def test_query_multi_namespace_fans_out_and_merges_sources(client):
    app.state.pinecone_index.describe_index_stats.return_value = SimpleNamespace(
        namespaces={
            "doc_a.pdf": SimpleNamespace(vector_count=1),
            "doc_b.pdf": SimpleNamespace(vector_count=1),
        }
    )
    app.state.pinecone_index.query.side_effect = _fake_query_returning(
        {"doc_a.pdf": "Relevant content from doc A", "doc_b.pdf": "Relevant content from doc B"}
    )

    response = client.post("/query", json={"question": "What is in these documents?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is in these documents?"
    assert body["answer"] == "mocked answer"
    assert {source["namespace"] for source in body["sources"]} == {"doc_a.pdf", "doc_b.pdf"}

    # Question embedded once and reused across every namespace's dense query.
    assert app.state.openai_client.embeddings.create.call_count == 1
    assert app.state.pinecone_index.query.call_count == 2
    called_namespaces = {
        call.kwargs["namespace"] for call in app.state.pinecone_index.query.call_args_list
    }
    assert called_namespaces == {"doc_a.pdf", "doc_b.pdf"}


def test_query_missing_local_file_falls_back_to_dense_only(client):
    # "ghost.pdf" has vectors in Pinecone but no file under input/ — simulates
    # drift from the mandatory input/-placement invariant (PLAN.md §7). Must not
    # fail the request; that namespace just contributes dense-only results.
    app.state.pinecone_index.describe_index_stats.return_value = SimpleNamespace(
        namespaces={"ghost.pdf": SimpleNamespace(vector_count=1)}
    )
    app.state.pinecone_index.query.side_effect = _fake_query_returning(
        {"ghost.pdf": "Some dense-only content"}
    )

    response = client.post("/query", json={"question": "Anything in here?"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["namespace"] == "ghost.pdf"


def test_query_upstream_pinecone_failure_returns_502(client):
    from pinecone.exceptions import PineconeException

    app.state.pinecone_index.describe_index_stats.side_effect = PineconeException("boom")
    response = client.post("/query", json={"question": "What is a SQL warehouse?"})
    assert response.status_code == 502
