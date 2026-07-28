from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone

from .config import settings
from .routers.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.openai_client = OpenAI()

    pc = Pinecone()
    app.state.pinecone_index = pc.Index(settings.index_name)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    app.state.sem_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.chunk_sem_threshold_type,
        breakpoint_threshold_amount=settings.chunk_sem_threshold_amt,
    )

    yield


app = FastAPI(title="DocMind Upload API", lifespan=lifespan)
app.include_router(upload_router)
