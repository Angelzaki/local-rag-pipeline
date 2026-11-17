from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docs_dir: Path = Path("samples/docs")
    index_dir: Path = Path("data/index")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    use_ollama: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    chunk_size: int = 280
    chunk_overlap: int = 40
    top_k: int = 1
    min_score: float = 0.35
    min_overlap: float = 0.5
    enable_guard: bool = True
    guard_threshold: float = 0.55


def get_settings() -> Settings:
    return Settings()
