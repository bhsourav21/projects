import tempfile
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader

PDF_MAGIC = b"%PDF-"


class InvalidPDFError(ValueError):
    pass


class PDFTooLargeError(ValueError):
    pass


def load_pdf_pages(file: UploadFile, max_upload_mb: int) -> list:
    max_bytes = max_upload_mb * 1024 * 1024
    contents = file.file.read()

    if len(contents) > max_bytes:
        raise PDFTooLargeError(f"File exceeds {max_upload_mb} MB limit")
    if not contents.startswith(PDF_MAGIC):
        raise InvalidPDFError("Uploaded file is not a valid PDF")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        return PyPDFLoader(str(tmp_path)).load()
    finally:
        tmp_path.unlink(missing_ok=True)
