import os

from .embeddings import get_embeddings


def sanitize_namespace(filename: str) -> str:
    return os.path.basename(filename).strip()


def replace_namespace_if_exists(index, namespace: str) -> bool:
    stats = index.describe_index_stats()
    existing = stats.namespaces or {}
    summary = existing.get(namespace)
    if summary and summary.vector_count > 0:
        index.delete(delete_all=True, namespace=namespace)
        return True
    return False


def upsert_chunks(
    index,
    openai_client,
    chunks: list[dict],
    namespace: str,
    embedding_model: str,
    batch_size: int,
) -> int:
    vectors = []
    for i, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        source = metadata.get("source", "")
        vectors.append({
            "id": f"chunk-{i}",
            "values": None,  # filled in batch below
            "metadata": {
                "text": chunk["text"],
                "source": source,
                "page": metadata.get("page", 0),
                "source_filename": os.path.basename(source),
                "page_number": metadata.get("page", 0) + 1,  # 0-indexed -> 1-indexed
            },
        })

    for batch_start in range(0, len(vectors), batch_size):
        batch = vectors[batch_start : batch_start + batch_size]
        texts = [chunks[batch_start + j]["text"] for j in range(len(batch))]
        batch_embeddings = get_embeddings(openai_client, texts, embedding_model)
        for vec, emb in zip(batch, batch_embeddings):
            vec["values"] = emb
        index.upsert(vectors=batch, namespace=namespace)

    return len(vectors)
