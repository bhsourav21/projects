from types import SimpleNamespace

from upload_api.app import app


def test_upload_pdf_happy_path(client, pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["namespace"] == "sample.pdf"
    assert body["pages_loaded"] == 1
    assert body["chunks_upserted"] == 1
    assert body["replaced_existing"] is False

    app.state.pinecone_index.upsert.assert_called_once()
    app.state.pinecone_index.delete.assert_not_called()


def test_upload_rejects_non_pdf(client, not_a_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", not_a_pdf_bytes, "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, pdf_bytes):
    app.state.settings.max_upload_mb = 0  # force the size check to fail
    try:
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 413
    finally:
        app.state.settings.max_upload_mb = 10


def test_upload_rejects_empty_text_pdf(client, empty_text_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("scanned.pdf", empty_text_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_clean_replace_on_reupload(client, pdf_bytes):
    empty_stats = SimpleNamespace(namespaces={})
    existing_stats = SimpleNamespace(namespaces={"sample.pdf": SimpleNamespace(vector_count=1)})
    app.state.pinecone_index.describe_index_stats.side_effect = [empty_stats, existing_stats]

    first = client.post("/upload", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
    assert first.status_code == 200
    assert first.json()["replaced_existing"] is False
    app.state.pinecone_index.delete.assert_not_called()

    second = client.post("/upload", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
    assert second.status_code == 200
    assert second.json()["replaced_existing"] is True
    app.state.pinecone_index.delete.assert_called_once_with(delete_all=True, namespace="sample.pdf")
