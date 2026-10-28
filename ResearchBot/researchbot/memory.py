"""Short-term (message window) and long-term (Pinecone) memory."""

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from researchbot.config import settings

# ---------------------------------------------------------------------------
# Short-term memory helpers (pure functions, no API calls)
# ---------------------------------------------------------------------------


def split_pairs(messages: list[AnyMessage]) -> tuple[list[list[AnyMessage]], list[AnyMessage]]:
    """Split messages into completed Q&A pairs and the pending (unanswered) tail.

    A pair starts at a HumanMessage and is completed once an AIMessage follows it.
    The trailing question that has not been answered yet is returned as `pending`,
    so it is never summarised away.
    """
    pairs: list[list[AnyMessage]] = []
    current: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and current:
            pairs.append(current)
            current = []
        current.append(msg)
    if current:
        if any(isinstance(m, AIMessage) for m in current):
            pairs.append(current)
            current = []
    return pairs, current


def recent_pairs(
    messages: list[AnyMessage], n: int = settings.short_term_pairs
) -> list[AnyMessage]:
    """The last `n` completed Q&A pairs, flattened."""
    pairs, _ = split_pairs(messages)
    return [m for pair in pairs[-n:] for m in pair]


def format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "(no earlier conversation)"


# ---------------------------------------------------------------------------
# Long-term memory (Pinecone)
# ---------------------------------------------------------------------------


@lru_cache
def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


@lru_cache
def _index():
    pc = Pinecone()
    if not pc.has_index(settings.index_name):
        print(f"[memory] creating Pinecone index '{settings.index_name}' ...")
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc.Index(settings.index_name)


def embed(text: str) -> list[float]:
    return _embeddings().embed_query(text)


def search_memory(query: str, top_k: int = settings.memory_top_k) -> list[dict]:
    """Semantic search over both namespaces, merged and sorted by score."""
    vector = embed(query)
    hits = []
    for namespace in (settings.research_namespace, settings.conversation_namespace):
        res = _index().query(
            vector=vector, top_k=top_k, namespace=namespace, include_metadata=True
        )
        for match in res.matches:
            meta = dict(match.metadata or {})
            hits.append(
                {
                    "namespace": namespace,
                    "score": float(match.score),
                    "text": meta.get("text", ""),
                    "question": meta.get("question", ""),
                    "date": meta.get("date", ""),
                    "urls": meta.get("urls", []),
                    "report_path": meta.get("report_path", ""),
                }
            )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def store_conversation_summary(summary: str, thread_id: str) -> None:
    _index().upsert(
        vectors=[
            {
                "id": f"conv-{uuid.uuid4()}",
                "values": embed(summary),
                "metadata": {"text": summary, "thread_id": thread_id, "date": _now()},
            }
        ],
        namespace=settings.conversation_namespace,
        show_progress=False,
    )


def _is_duplicate(first_chunk_vector: list[float]) -> bool:
    """True if a stored report's first chunk is near-identical to this one."""
    res = _index().query(
        vector=first_chunk_vector,
        top_k=1,
        namespace=settings.research_namespace,
        filter={"chunk_index": 0},
    )
    return bool(res.matches) and res.matches[0].score >= settings.dedup_threshold


def store_report(
    report: str, question: str, urls: list[str], report_path: str, thread_id: str
) -> int:
    """Chunk, embed and upsert a report.

    Returns the number of chunks stored, or 0 if a near-identical report already exists.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_text(report)
    vectors = _embeddings().embed_documents(chunks)
    if _is_duplicate(vectors[0]):
        return 0
    doc_id = uuid.uuid4().hex
    date = _now()
    _index().upsert(
        vectors=[
            {
                "id": f"{doc_id}-{i}",
                "values": vec,
                "metadata": {
                    "text": chunk,
                    "question": question,
                    "date": date,
                    "urls": urls or ["none"],
                    "report_path": report_path,
                    "thread_id": thread_id,
                    "chunk_index": i,
                },
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ],
        namespace=settings.research_namespace,
        show_progress=False,
    )
    return len(chunks)
