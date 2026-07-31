import hashlib
import logging
from typing import Callable

from .bm25_index import BM25NamespaceIndex, tokenise

logger = logging.getLogger(__name__)


def _composite_key(namespace: str, text: str) -> tuple[str, str]:
    """(namespace, sha1(text)) — robust to dense/sparse chunkings drifting apart,
    unlike matching on the bare 'chunk-{i}' id, which collides both across
    dense/sparse and across documents (see PLAN.md §6)."""
    return namespace, hashlib.sha1(text.encode("utf-8")).hexdigest()


def dense_retrieve(index, namespace: str, vector: list[float], top_k: int) -> list[dict]:
    results = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
    retrieved = []
    for match in results.matches:
        text = match.metadata.get("text", "")
        retrieved.append({
            "key": _composite_key(namespace, text),
            "text": text,
            "score": match.score,
            "namespace": namespace,
            "source_filename": match.metadata.get("source_filename", namespace),
            "page_number": match.metadata.get("page_number", 0),
        })
    return retrieved


def sparse_retrieve(
    bm25_index: BM25NamespaceIndex, question: str, namespace: str, top_k: int
) -> list[dict]:
    scores = bm25_index.bm25.get_scores(tokenise(question))
    top_idx = scores.argsort()[::-1][:top_k]
    retrieved = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        chunk = bm25_index.chunks[idx]
        retrieved.append({
            "key": _composite_key(namespace, chunk["text"]),
            "text": chunk["text"],
            "score": float(scores[idx]),
            "namespace": namespace,
            "source_filename": chunk["source_filename"],
            "page_number": chunk["page_number"],
        })
    return retrieved


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], key: str = "key", k: int = 60
) -> list[dict]:
    """Merge multiple ranked lists using RRF. ranked_lists: each a list of dicts
    sorted by score descending. Returns a single merged list sorted by RRF score
    descending."""
    rrf_scores: dict = {}
    doc_store: dict = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_key = doc[key]
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1 / (k + rank)
            doc_store[doc_key] = doc

    merged = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [{**doc_store[doc_key], "rrf_score": score} for doc_key, score in merged]


def hybrid_retrieve(
    index,
    namespaces: list[str],
    question: str,
    question_vector: list[float],
    get_bm25_index: Callable[[str], BM25NamespaceIndex | None],
    dense_top_k: int,
    sparse_top_k: int,
    hybrid_top_k: int,
    rrf_k: int,
) -> list[dict]:
    dense_results: list[dict] = []
    sparse_results: list[dict] = []
    for namespace in namespaces:
        dense_results.extend(dense_retrieve(index, namespace, question_vector, dense_top_k))
        bm25_index = get_bm25_index(namespace)
        if bm25_index is not None:
            sparse_results.extend(sparse_retrieve(bm25_index, question, namespace, sparse_top_k))
        else:
            logger.warning("Skipping sparse retrieval for namespace %r (no BM25 index)", namespace)

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)
    return fused[:hybrid_top_k]
