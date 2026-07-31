def generate_answer(openai_client, chat_model: str, question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {chunk['source_filename']}]\n{chunk['text']}" for chunk in chunks
    )
    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the provided context. "
                "If the context doesn't contain the answer, say you don't know.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content
