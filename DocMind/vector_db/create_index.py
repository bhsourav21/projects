from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone()

if not pc.has_index("docmind-index"):
    pc.create_index(
        name="docmind-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("docmind-index created")
else:
    print("docmind-index already exists, skipping create")
