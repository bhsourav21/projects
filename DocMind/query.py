import os
import re
import numpy as npup
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

load_dotenv()

INDEX_NAME = "docmind-index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PDF_PATH = os.path.join(os.path.dirname(__file__), "input", "databricks_architecture_governance.pdf")
NAMESPACE = os.path.basename(PDF_PATH)

openai_client = OpenAI()
pc = Pinecone()
index = pc.Index(INDEX_NAME)

model_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

STOPWORDS = {"a","an","the","is","in","it","of","and","or","to"}
def tokenise(text):
    t = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [w for w in t.split() if w not in STOPWORDS and len(w)>1]

pages = PyPDFLoader(PDF_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunk_texts = [chunk.page_content for chunk in splitter.split_documents(pages)]

bm25 = BM25Okapi([tokenise(t) for t in chunk_texts])

def dense_retrieve(query: str, top_k: int = 10) -> list[dict]:
    emb = openai_client.embeddings.create(input=[query], model=EMBED_MODEL)
    vector = emb.data[0].embedding
    results = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=NAMESPACE)
    return [
    {"id": m.id, "score": m.score,
    "text": m.metadata.get("text",""), "source": "dense"}
    for m in results.matches
    ]

# ■■ Sparse retrieval (BM25) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def sparse_retrieve(query: str, top_k: int = 5) -> list[dict]:
    scores = bm25.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
    {"id": f"chunk-{idx}", "score": float(scores[idx]),
    "text": chunk_texts[idx], "source": "bm25"}
    for idx in top_idx if scores[idx] > 0
    ]

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
    ) -> list[dict]:
    """
    Merge multiple ranked lists using RRF.
    ranked_lists: each is a list of dicts sorted by score descending.
    Returns a single merged list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_store[doc_id] = doc # keep the doc for reference
    
    # Sort by RRF score descending
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
    {**doc_store[doc_id], "rrf_score": score}
    for doc_id, score in merged
    ]

# ■■ Full hybrid retrieval ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def hybrid_retrieve(query: str, top_k: int = 20) -> list[dict]:
    dense_results = dense_retrieve(query, top_k=10)
    sparse_results = sparse_retrieve(query, top_k=10)
    # Fuse using RRF
    merged = reciprocal_rank_fusion([dense_results, sparse_results])
    return merged[:top_k]

def rerank_with_cross_encoder(question: str, chunks: list[str]) -> list[tuple[float, str]]:
    pairs = [(question, chunk) for chunk in chunks]
    scores = model_cross_encoder.predict(pairs)
    return sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
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

def answer_query(query: str, top_k: int = 20, rerank_k: int = 5) -> str:
    retrieved = hybrid_retrieve(query, top_k=top_k)
    reranked = rerank_with_cross_encoder(query, [r["text"] for r in retrieved])
    top_chunks = [chunk for _, chunk in reranked[:rerank_k]]
    return generate_answer(query, top_chunks)

queries = [
    "What is a sql warehouse?",
    # "What does DSPy stand for / do?",
    # "What is Gemini 1.5 Pro's context window size?",
    # "Which tool is Anthropic's model with a 200K context window?",
    # "What primary tool stack does a Data Scientist use, per the comparison chart?"
]

for query in queries:
    answer = answer_query(query)
    print(f"\nQuery: {query}\nAnswer: {answer}")

