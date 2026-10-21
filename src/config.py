"""Central settings, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Hugging Face commit of all-MiniLM-L6-v2, pinned so vectors are reproducible.
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _default_revision() -> str:
    model = _env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return DEFAULT_EMBEDDING_REVISION if model == DEFAULT_EMBEDDING_MODEL else "main"


@dataclass(frozen=True)
class Settings:
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    reports_dir: Path = PROJECT_ROOT / "reports"
    embedding_model: str = _env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_revision: str = _env("EMBEDDING_REVISION", _default_revision())
    top_k: int = int(_env("TOP_K", "3"))
    min_score: float = float(_env("MIN_SCORE", "0.30"))
    # Local on-disk Qdrant folder; ignored by git and rebuilt by `build-index`.
    vector_storage_path: Path = PROJECT_ROOT / _env("VECTOR_STORAGE_PATH", "storage/qdrant")
    collection_name: str = "northstar_support"
    llm_provider: str = _env("LLM_PROVIDER", "none")
    chunk_max_chars: int = 600
    chunk_overlap_sentences: int = 1


settings = Settings()
