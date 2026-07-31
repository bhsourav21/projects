import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi

from .chunking import hybrid_chunk

logger = logging.getLogger(__name__)

STOPWORDS = {"a", "an", "the", "is", "in", "it", "of", "and", "or", "to"}


def tokenise(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [word for word in cleaned.split() if word not in STOPWORDS and len(word) > 1]


class BM25NamespaceIndex:
    def __init__(self, bm25: BM25Okapi, chunks: list[dict]):
        self.bm25 = bm25
        self.chunks = chunks


def build_bm25_index(namespace: str, settings, sem_chunker) -> BM25NamespaceIndex | None:
    """Rebuilds the BM25 corpus for a namespace from its local PDF under input/,
    using the exact same hybrid_chunk() config as upload time so chunk text lines
    up with what's stored in Pinecone (see PLAN.md §6)."""
    pdf_path = Path(settings.input_dir) / namespace
    if not pdf_path.exists():
        logger.warning(
            "BM25 build skipped: no local file for namespace %r at %s", namespace, pdf_path
        )
        return None

    pages = PyPDFLoader(str(pdf_path)).load()
    chunks: list[dict] = []
    for page in pages:
        for chunk in hybrid_chunk(
            page.page_content,
            sem_chunker=sem_chunker,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
        ):
            chunks.append({
                "text": chunk["text"],
                "source_filename": namespace,
                "page_number": page.metadata.get("page", 0) + 1,
            })

    bm25 = BM25Okapi([tokenise(chunk["text"]) for chunk in chunks])
    return BM25NamespaceIndex(bm25=bm25, chunks=chunks)


def get_or_build_bm25_index(app_state, namespace: str) -> BM25NamespaceIndex | None:
    cache = app_state.bm25_cache
    if namespace not in cache:
        cache[namespace] = build_bm25_index(namespace, app_state.settings, app_state.sem_chunker)
    return cache[namespace]


def list_namespaces(index) -> list[str]:
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    return [ns for ns, summary in namespaces.items() if summary.vector_count > 0]
