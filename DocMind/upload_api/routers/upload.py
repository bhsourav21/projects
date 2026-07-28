from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from openai import OpenAIError
from pinecone.exceptions import PineconeException
from pydantic import BaseModel

from ..services.chunking import hybrid_chunk
from ..services.pdf_loader import InvalidPDFError, PDFTooLargeError, load_pdf_pages
from ..services.vector_store import replace_namespace_if_exists, sanitize_namespace, upsert_chunks

router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    namespace: str
    pages_loaded: int
    chunks_upserted: int
    index_name: str
    replaced_existing: bool


@router.post("/upload", response_model=UploadResponse)
def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a PDF")

    settings = request.app.state.settings

    try:
        pages = load_pdf_pages(file, settings.max_upload_mb)
    except PDFTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        chunks_final = []
        for page in pages:
            for chunk in hybrid_chunk(
                page.page_content,
                sem_chunker=request.app.state.sem_chunker,
                min_chars=settings.chunk_min_chars,
                max_chars=settings.chunk_max_chars,
            ):
                chunk["metadata"] = page.metadata
                chunks_final.append(chunk)

        if not chunks_final:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "No extractable text found in PDF — "
                    "scanned/image-only PDFs are not supported."
                ),
            )

        namespace = sanitize_namespace(file.filename or "unnamed.pdf")
        index = request.app.state.pinecone_index

        replaced = replace_namespace_if_exists(index, namespace)

        chunks_upserted = upsert_chunks(
            index=index,
            openai_client=request.app.state.openai_client,
            chunks=chunks_final,
            namespace=namespace,
            embedding_model=settings.embedding_model,
            batch_size=settings.batch_size,
        )
    except (OpenAIError, PineconeException) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream embedding or vector store call failed",
        ) from exc

    return UploadResponse(
        filename=file.filename or "unnamed.pdf",
        namespace=namespace,
        pages_loaded=len(pages),
        chunks_upserted=chunks_upserted,
        index_name=settings.index_name,
        replaced_existing=replaced,
    )
