from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from upload_api.app import app


def make_minimal_pdf(text: str) -> bytes:
    """Hand-rolled single-page PDF with a text stream, for tests that need a real,
    parseable PDF without pulling in a PDF-writing dependency."""
    stream_content = f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream_content)).encode() + b" >>\nstream\n"
        + stream_content + b"\nendstream",
    ]
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


@pytest.fixture
def pdf_bytes() -> bytes:
    paragraph = "Hello World, this is a DocMind test document. " * 5
    return make_minimal_pdf(paragraph)


@pytest.fixture
def empty_text_pdf_bytes() -> bytes:
    return make_minimal_pdf("hi")


@pytest.fixture
def not_a_pdf_bytes() -> bytes:
    return b"this is not a pdf file"


def fake_embeddings_create(*, input, model):
    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 5) for _ in input])


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.openai_client = MagicMock()
        app.state.openai_client.embeddings.create.side_effect = fake_embeddings_create
        app.state.pinecone_index = MagicMock()
        app.state.pinecone_index.describe_index_stats.return_value = SimpleNamespace(namespaces={})
        yield test_client
