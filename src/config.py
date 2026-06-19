from pathlib import Path
from dotenv import load_dotenv
import os

#Loading .env from project root 
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# Paths 
DATA_DIR = _ROOT / "data"
PDF_PATH = DATA_DIR / "paper.pdf"
CHROMA_DIR = _ROOT / ".chroma"

# Chunking & Retrieval 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#LLM
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME","qwen2.5:3b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL","http://127.0.0.1:11434/v1")

#Chroma collection
CHROMA_COLLECTION_NAME = "research-paper-rag"
