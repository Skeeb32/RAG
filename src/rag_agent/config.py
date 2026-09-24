"""Environment-backed application configuration."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    judge_model: str = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    index: Path = Path(".rag/index")
    memory: Path = Path(".rag/preferences.sqlite3")
    traces: Path = Path(".rag/traces")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            model=os.getenv("OLLAMA_MODEL", cls.model),
            judge_model=os.getenv("JUDGE_MODEL", cls.judge_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            index=Path(os.getenv("RAG_INDEX", ".rag/index")),
            memory=Path(os.getenv("RAG_MEMORY", ".rag/preferences.sqlite3")),
            traces=Path(os.getenv("RAG_TRACES", ".rag/traces")),
        )
