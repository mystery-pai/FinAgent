"""
Configuration management for the financial RAG system.
核心配置管理
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Model configurations
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"  # or "cuda" if available

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval settings
    bm25_k: int = 10  # BM25 top-k results
    vector_k: int = 10  # Vector search top-k results
    hybrid_k: int = 5  # Final top-k after fusion

    # ChromaDB settings
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "financial_reports"

    # Data paths
    raw_data_path: str = "./data/raw/aapl_10k.json"
    processed_data_path: str = "./data/processed"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
