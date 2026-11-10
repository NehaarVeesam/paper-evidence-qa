"""Shared configuration. Shell variables take precedence over .env."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def setting(name, default=""):
    return os.getenv(name, "").strip() or default


def project_path(name, default):
    path = Path(setting(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def optional_bool(name):
    value = setting(name).lower()
    if not value:
        return None
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, or blank.")


PDF_PATH = project_path("PDF_PATH", "docs/document.pdf")
VECTOR_DB_PATH = project_path("VECTOR_DB_PATH", "data/chromadb")
COLLECTION_NAME = setting("COLLECTION_NAME", "document_data")
EMBEDDING_MODEL_NAME = setting(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(setting("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(setting("CHUNK_OVERLAP", "50"))
TOP_K = int(setting("TOP_K", "5"))

QWEN_BASE_URL = setting("QWEN_BASE_URL").rstrip("/")
QWEN_API_KEY = setting("QWEN_API_KEY")
QWEN_MODEL = setting("QWEN_MODEL")
QWEN_TIMEOUT = float(setting("QWEN_TIMEOUT", "120"))
QWEN_MAX_TOKENS = int(setting("QWEN_MAX_TOKENS", "1024"))
QWEN_TEMPERATURE = float(setting("QWEN_TEMPERATURE", "0.2"))
QWEN_ENABLE_THINKING = optional_bool("QWEN_ENABLE_THINKING")


def pipeline_metadata():
    """Identify the processing settings used to create a collection."""
    return {
        "pipeline_version": 1,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "normalized_embeddings": True,
    }
