from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    index_name: str = "docmind-index"
    embedding_model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_upload_mb: int = 10

    chunk_min_chars: int = 100
    chunk_max_chars: int = 1000
    chunk_sem_threshold_type: str = "percentile"
    chunk_sem_threshold_amt: float = 95

    input_dir: str = str(Path(__file__).resolve().parent.parent / "input")
    chat_model: str = "gpt-4o-mini"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dense_top_k: int = 10
    sparse_top_k: int = 10
    hybrid_top_k: int = 20
    rerank_k: int = 5
    rrf_k: int = 60


settings = Settings()
