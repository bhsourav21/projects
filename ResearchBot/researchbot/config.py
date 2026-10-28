from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    # Models
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Pinecone
    index_name: str = "researchbot-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    research_namespace: str = "research"
    conversation_namespace: str = "conversation"
    memory_top_k: int = 5
    strong_hit_threshold: float = 0.75  # a research hit above this means "mostly covered already"
    dedup_threshold: float = 0.95  # a new report this close to an old one is not stored again

    # Short-term memory: number of completed Q&A pairs kept verbatim
    short_term_pairs: int = 3

    # Search / read
    max_queries: int = 4
    results_per_query: int = 3
    max_pages: int = 8
    max_page_chars: int = 12000
    max_search_retries: int = 2

    # Report
    max_report_tokens: int = 2000
    chunk_size: int = 1000
    chunk_overlap: int = 100
    reports_dir: Path = PROJECT_ROOT / "reports"


settings = Settings()
