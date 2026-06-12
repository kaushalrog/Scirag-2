"""
config/settings.py
------------------
Centralised application settings via pydantic-settings.
All values are read from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq Cloud API key")
    llm_model: str = Field("llama-3.1-70b-versatile", description="Groq model name")
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, gt=0)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        "all-MiniLM-L6-v2", description="SentenceTransformers model"
    )
    embedding_dim: int = Field(384, description="Embedding vector dimension")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_db")
    chroma_collection: str = Field("scirag_papers")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(5, ge=1, le=20)
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0, description="MMR diversity param")
    bm25_weight: float = Field(0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)

    # ── Uncertainty ───────────────────────────────────────────────────────────
    confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)
    abstention_threshold: float = Field(0.40, ge=0.0, le=1.0)
    entropy_threshold: float = Field(2.5, ge=0.0)

    # ── Data paths ────────────────────────────────────────────────────────────
    data_raw_dir: str = Field("./data/raw")
    data_processed_dir: str = Field("./data/processed")
    benchmark_dir: str = Field("./data/benchmarks")

    # ── Semantic Scholar ──────────────────────────────────────────────────────
    s2_api_key: str = Field("", description="Semantic Scholar API key (optional)")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)
    debug: bool = Field(True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")
    log_file: str = Field("./logs/scirag.log")

    @property
    def raw_dir(self) -> Path:
        p = Path(self.data_raw_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_dir(self) -> Path:
        p = Path(self.data_processed_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def bench_dir(self) -> Path:
        p = Path(self.benchmark_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
