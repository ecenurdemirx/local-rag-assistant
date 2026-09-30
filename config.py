import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("RAG_DOCUMENTS_DIR", BASE_DIR / "documents"))
DATABASE_PATH = Path(os.getenv("RAG_DATABASE_PATH", BASE_DIR / "rag.db"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen2.5-0.5b")
APP_NAME = "foundry_local_rag"

SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.55"))
