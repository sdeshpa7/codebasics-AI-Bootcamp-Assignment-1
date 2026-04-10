"""
app/embeddings.py
Encodes document chunks into dense vector embeddings using sentence-transformers.

The model is loaded once and reused across calls (lazy initialisation).
Embeddings are L2-normalised so cosine similarity == dot product,
which is required by most vector stores (including Qdrant).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.chunking import Chunk


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """A Chunk paired with its dense vector embedding."""
    chunk: Chunk
    embedding: list[float]      # normalised, ready to be indexed into a vector store

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict[str, Any]:
        """Flat metadata dict — used when upserting into a vector store."""
        return {
            "source_file":      self.chunk.source_file,
            "collection":       self.chunk.collection,
            "chunk_type":       self.chunk.chunk_type,
            "section_title":    self.chunk.section_title,
            "page_number":      self.chunk.page_number,
            "parent_chunk_id":  self.chunk.parent_chunk_id,
            "access_roles":     self.chunk.access_roles,
        }


# ---------------------------------------------------------------------------
# Model (lazy singleton)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model on first use and cache it."""
    global _model
    if _model is None:
        print(f"Loading embedding model '{settings.embedding_model}'…")
        try:
            _model = SentenceTransformer(settings.embedding_model)
        except Exception:
            # Network unavailable — load from local cache
            print("Network unavailable, loading from local cache…")
            _model = SentenceTransformer(settings.embedding_model, local_files_only=True)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
    """
    Encode a list of Chunk objects into EmbeddedChunk objects.

    Args:
        chunks:     Chunks produced by chunking.chunk_all()
        batch_size: Number of texts encoded per forward pass.
                    Increase for GPU; keep lower for CPU memory.

    Returns:
        List of EmbeddedChunk — each wraps the original Chunk plus its
        normalised embedding vector (as a plain Python list of floats).
    """
    if not chunks:
        return []

    model = _get_model()
    texts = [chunk.text for chunk in chunks]

    print(f"Encoding {len(texts)} chunks with '{settings.embedding_model}'…")
    raw: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
        convert_to_numpy=True,
    )

    embedded = [
        EmbeddedChunk(chunk=chunk, embedding=vec.tolist())
        for chunk, vec in zip(chunks, raw)
    ]

    dim = len(embedded[0].embedding) if embedded else 0
    print(f"Done — {len(embedded)} embeddings, dimension: {dim}")
    return embedded
