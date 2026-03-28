"""
Configuration management using pydantic-settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings"""

    # LLM Configuration
    llm_provider: Literal["deepseek", "ollama"] = Field(default="deepseek")
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="deepseek-coder:33b")

    # Embedding settings
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # Retrieval settings
    bm25_top_k: int = Field(default=10)
    vector_top_k: int = Field(default=10)
    final_top_k: int = Field(default=5)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Streamlit settings
    streamlit_host: str = Field(default="0.0.0.0")
    streamlit_port: int = Field(default=8501)

    # Paths
    data_raw_path: str = Field(default="data/raw")
    data_processed_path: str = Field(default="data/processed")
    indexes_path: str = Field(default="indexes")
    embeddings_path: str = Field(default="embeddings")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
