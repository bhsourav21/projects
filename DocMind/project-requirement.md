# DocMind — Document Q&A with FastAPI Backend

## Requirements

1. Accept any PDF upload via a FastAPI `POST /upload` endpoint. Chunk, embed, and index it automatically.
2. Expose a `POST /query` endpoint: takes a question and document ID, returns the answer + source chunk excerpts.
3. Implement hybrid search + cross-encoder re-ranking in the retrieval step.
4. Run RAGAS evaluation on 20 test questions. Include the scores in your README.
5. Dockerize the FastAPI app and verify it runs with `docker-compose up`.
