def rerank_with_cross_encoder(
    cross_encoder, question: str, chunks: list[dict], rerank_k: int
) -> list[dict]:
    if not chunks:
        return []

    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [{**chunk, "score": float(score)} for score, chunk in ranked[:rerank_k]]
