"""On-device embeddings provided by Microsoft Foundry Local."""

import config
from foundry_runtime import get_manager

_embedding_model = None
_embedding_client = None

def _get_client():
    """Load the configured embedding model once and return its native client."""
    global _embedding_model, _embedding_client
    if _embedding_client is None:
        manager = get_manager()
        _embedding_model = manager.catalog.get_model(config.EMBEDDING_MODEL)
        _embedding_model.download(lambda _progress: None)
        _embedding_model.load()
        _embedding_client = _embedding_model.get_embedding_client()
    return _embedding_client

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a non-empty batch of texts on device."""
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding input cannot be empty.")
    response = _get_client().generate_embeddings(texts)
    vectors = [item.embedding for item in response.data]
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding model returned an unexpected vector count.")
    return vectors

def get_embedding(text: str) -> list[float]:
    """Generate one embedding with the same model used during ingestion."""
    if not text.strip():
        raise ValueError("Embedding input cannot be empty.")
    response = _get_client().generate_embedding(text)
    return response.data[0].embedding

def unload_model():
    """Release the embedding model while keeping its cached weights."""
    global _embedding_model, _embedding_client
    if _embedding_model is not None and _embedding_model.is_loaded:
        _embedding_model.unload()
    _embedding_model = None
    _embedding_client = None
