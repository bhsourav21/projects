import re
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/projects/DocMind/input'
loader = PyPDFLoader(f"{path}/databricks_architecture_governance.pdf")
pages = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def hybrid_chunk(
    text: str,
    section_pattern: str = r"\n(?=#{1,6}\s)|\n\n+",
    min_chars: int = 100,
    max_chars: int = 2000,
    sem_threshold_type: str = "percentile",
    sem_threshold_amt: float = 95,
    ) -> list[dict]:

    """
    Phase 1: regex structural split.
    Phase 2: semantic sub-split for oversized sections.
    Returns list of {'text', 'chars', 'method'} dicts.
    """

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    sem_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=sem_threshold_type,
        breakpoint_threshold_amount=sem_threshold_amt,
    )

    # Phase 1
    raw_sections_final = []
    final_chunks = []

    raw_sections = [s.strip() for s in re.split(section_pattern, text) if s.strip()]
    for raw_section in raw_sections:
        raw_sections_final.append(raw_sections)
    
    for section in raw_sections:
        if len(section) < min_chars:
            continue # too short, skip
        elif len(section) <= max_chars:
            final_chunks.append({ # just right
                "text": section, "chars": len(section), "method": "regex"
            })
        else:
            # Phase 2: semantic sub-split
            sub = sem_chunker.split_text(section)
            for s in sub:
                if len(s.strip()) >= min_chars:
                    final_chunks.append({
                        "text": s.strip(),
                        "chars": len(s.strip()),
                        "method": "regex+semantic",
                    })

    return final_chunks

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

chunks_final = []
for page in pages:
    chunks = hybrid_chunk(page.page_content)
    for chunk in chunks:
        chunk["metadata"] = page.metadata
        chunks_final.append(chunk)


# Upsert into Pinecone
INDEX_NAME = "docmind-index"
BATCH_SIZE = 100
# CATEGORY = "ai-ecosystem"

openai_client = OpenAI()
pc = Pinecone()                  # reads PINECONE_API_KEY from env automatically
index = pc.Index(INDEX_NAME)

vectors = []
for i, chunk in enumerate(chunks_final):
    print(f"chunk")
    print(f"type(chunk): {type(chunk)}")
    print(chunk)
    print("----------------------")
    vectors.append({
        "id": f"chunk-{i}",
        "values": None,          # filled in batch below
        "metadata": {
            "text": chunk["text"],
            "source": chunk["metadata"].get("source", ""),
            "page": chunk["metadata"].get("page", 0),
            "source_filename": os.path.basename(chunk["metadata"].get("source", "")),
            "page_number": chunk["metadata"].get("page", 0) + 1,  # 0-indexed → 1-indexed
        }
    })

# Embed and upsert in batches
for batch_start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[batch_start : batch_start + BATCH_SIZE]
    texts = [chunks_final[batch_start + j]["text"] for j in range(len(batch))]
    embeddings = get_embeddings(texts)
    for vec, emb in zip(batch, embeddings):
        vec["values"] = emb
    groups = {}
    for vec in batch:
        ns = vec["metadata"]["source_filename"]
        groups.setdefault(ns, []).append(vec)
    for ns, vecs in groups.items():
        index.upsert(vectors=vecs, namespace=ns)
    print(f"Upserted chunks {batch_start + 1} – {batch_start + len(batch)}")


stats = index.describe_index_stats()
print(stats)


print(f"\nDone. Total chunks upserted: {len(vectors)}")
