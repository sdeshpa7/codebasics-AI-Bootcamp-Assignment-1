"""
app/vector_store.py
Stores embedded chunks in Qdrant with full metadata as payload.

Collection is created automatically on first run.
Points are upserted in configurable batches for memory efficiency.
Supports both local Qdrant (Docker) and Qdrant Cloud (API key in .env).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.config import settings
from app.embeddings import EmbeddedChunk


# ---------------------------------------------------------------------------
# Client (lazy singleton)
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Return a shared QdrantClient, creating it on first call.

    Priority:
      1. use_local_qdrant=True  → in-memory client (no server needed, data lost on exit)
      2. QDRANT_API_KEY set     → Qdrant Cloud
      3. fallback               → local server at QDRANT_URL (e.g. Docker)
    """
    global _client
    if _client is None:
        if settings.use_local_qdrant:
            import os
            os.makedirs("data/qdrant_local", exist_ok=True)
            _client = QdrantClient(path="data/qdrant_local")
            print("Using persistent local Qdrant (stored in data/qdrant_local).")
        elif settings.qdrant_api_key:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            print(f"Connected to Qdrant Cloud at {settings.qdrant_url}")
        else:
            _client = QdrantClient(url=settings.qdrant_url)
            print(f"Connected to local Qdrant at {settings.qdrant_url}")
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(vector_size: int, collection: str = settings.qdrant_collection) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Args:
        vector_size: Dimensionality of the embedding vectors (e.g. 384).
        collection:  Collection name (defaults to settings.qdrant_collection).
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection in existing:
        print(f"Collection '{collection}' already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,   # pairs with normalize_embeddings=True in embed step
            ),
        )
        print(f"Created collection '{collection}' (dim={vector_size}, distance=COSINE).")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_embeddings(
    embedded_chunks: list[EmbeddedChunk],
    collection: str = settings.qdrant_collection,
    batch_size: int = settings.qdrant_batch_size,
) -> None:
    """
    Upsert a list of EmbeddedChunk objects into Qdrant.

    Each point includes:
      - id       : chunk UUID
      - vector   : normalised embedding (list[float])
      - payload  : all chunk metadata + the full text (for retrieval)

    Points are sent in batches to avoid memory spikes on large corpora.

    Args:
        embedded_chunks: Output of embeddings.embed_chunks()
        collection:      Target Qdrant collection name
        batch_size:      Points per upsert call
    """
    if not embedded_chunks:
        print("No embedded chunks to upsert.")
        return

    client = get_client()
    vector_size = len(embedded_chunks[0].embedding)
    ensure_collection(vector_size, collection)

    total   = len(embedded_chunks)
    batches = range(0, total, batch_size)

    print(f"Upserting {total} points into '{collection}' in batches of {batch_size}…")

    for start in batches:
        batch = embedded_chunks[start : start + batch_size]

        points = [
            PointStruct(
                id=ec.id,
                vector=ec.embedding,
                payload={
                    "text": ec.text,          # stored for retrieval / RAG context
                    **ec.metadata,            # source_file, folder, chunk_type,
                                              # heading, page_number,
                                              # parent_chunk_id, access_roles
                },
            )
            for ec in batch
        ]

        client.upsert(collection_name=collection, points=points)
        end = min(start + batch_size, total)
        print(f"  Upserted points {start + 1}–{end} / {total}")

    print(f"Done — {total} points stored in '{collection}'.")


# ---------------------------------------------------------------------------
# Convenience: count / info
# ---------------------------------------------------------------------------

def collection_info(collection: str = settings.qdrant_collection) -> dict:
    """Return basic stats about the collection."""
    client = get_client()
    info = client.get_collection(collection)
    return {
        "collection":   collection,
        "point_count":  info.points_count,
        "vector_size":  info.config.params.vectors.size,
        "distance":     info.config.params.vectors.distance,
        "status":       info.status,
    }
