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


settings = Settings()
