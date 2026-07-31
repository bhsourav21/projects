import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from openai import OpenAIError
from pinecone.exceptions import PineconeException
from pydantic import BaseModel

from ..services.bm25_index import get_or_build_bm25_index, list_namespaces
from ..services.chunking import hybrid_chunk
from ..services.embeddings import get_embeddings
from ..services.generation import generate_answer
from ..services.pdf_loader import InvalidPDFError, PDFTooLargeError, load_pdf_pages
from ..services.reranking import rerank_with_cross_encoder
from ..services.retrieval import hybrid_retrieve
from ..services.vector_store import replace_namespace_if_exists, sanitize_namespace, upsert_chunks

logger = logging.getLogger(__name__)

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


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    text: str
    score: float
    namespace: str
    source_filename: str
    page_number: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


@router.post("/query", response_model=QueryResponse)
def query_documents(request: Request, body: QueryRequest) -> QueryResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question must not be empty"
        )

    settings = request.app.state.settings
    index = request.app.state.pinecone_index

    try:
        namespaces = list_namespaces(index)
    except PineconeException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream vector store call failed"
        ) from exc

    if not namespaces:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No documents have been indexed yet"
        )

    try:
        question_vector = get_embeddings(
            request.app.state.openai_client, [question], settings.embedding_model
        )[0]

        def get_bm25_index(namespace: str):
            return get_or_build_bm25_index(request.app.state, namespace)

        fused = hybrid_retrieve(
            index=index,
            namespaces=namespaces,
            question=question,
            question_vector=question_vector,
            get_bm25_index=get_bm25_index,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            hybrid_top_k=settings.hybrid_top_k,
            rrf_k=settings.rrf_k,
        )
        reranked = rerank_with_cross_encoder(
            request.app.state.cross_encoder, question, fused, settings.rerank_k
        )
        answer = generate_answer(
            request.app.state.openai_client, settings.chat_model, question, reranked
        )
    except (OpenAIError, PineconeException) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream OpenAI or vector store call failed",
        ) from exc

    return QueryResponse(
        question=question,
        answer=answer,
        sources=[
            SourceChunk(
                text=chunk["text"],
                score=chunk["score"],
                namespace=chunk["namespace"],
                source_filename=chunk["source_filename"],
                page_number=chunk["page_number"],
            )
            for chunk in reranked
        ],
    )
