from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone()

indexes = pc.list_indexes()
for index in indexes:
    print("--------------------")
    print(index)
    print("--------------------")

pc.create_index(
    name="docmind-index",
    dimension=1536,        # text-embedding-3-small output dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

indexes = pc.list_indexes()
for index in indexes:
    print(index)
