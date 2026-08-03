from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone()

if pc.has_index("docmind-index"):
    pc.delete_index("docmind-index")
    print("docmind-index deleted")
else:
    print("docmind-index does not exist, nothing to delete")
