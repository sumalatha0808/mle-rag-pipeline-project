"""Embed chunks and index them in a local Qdrant collection.

Backend: Qdrant in embedded "local mode" (``QdrantClient(path=...)``). It is a
real vector database collection persisted to ``storage/qdrant`` with no server
to run. Every ``build-index`` drops and recreates the collection from the
Markdown sources, so the index can never drift from the approved documents.

Vectors are L2-normalised and the collection uses cosine distance, so a score
is a cosine similarity in [-1, 1] (higher = more similar).
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import settings
from src.ingestion import Chunk, load_chunks


class Embedder(Protocol):
    name: str
    version: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (384 dimensions) at a pinned Hugging Face revision."""

    def __init__(
        self,
        name: str = settings.embedding_model,
        revision: str = settings.embedding_revision,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # slow import

        self.name = name
        self.version = revision
        self._model = SentenceTransformer(name, revision=revision, device="cpu")
        self.dimension = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.astype(np.float32)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    title: str
    chunk_index: int
    text: str
    score: float
    rank: int


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\n{chunk.text}\n".encode())
    return digest.hexdigest()[:12]


class VectorIndex:
    def __init__(
        self,
        embedder: Embedder,
        storage_path: Path | None = settings.vector_storage_path,
        collection: str = settings.collection_name,
    ) -> None:
        """``storage_path=None`` keeps the index in memory (used by tests)."""
        self.embedder = embedder
        self.collection = collection
        self.storage_path = storage_path
        if storage_path is None:
            self.client = QdrantClient(":memory:")
        else:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))

    def build(self, chunks: list[Chunk]) -> dict:
        """Recreate the collection from ``chunks`` and return a manifest."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        vectors = self.embedder.embed([c.embedding_text for c in chunks])
        if vectors.shape != (len(chunks), self.embedder.dimension):
            raise ValueError(f"Unexpected embedding shape {vectors.shape}")

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=chunk.to_payload(),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        manifest = {
            "collection": self.collection,
            "backend": "qdrant-local",
            "distance": "cosine",
            "embedding_model": self.embedder.name,
            "embedding_revision": self.embedder.version,
            "dimension": self.embedder.dimension,
            "chunk_count": len(chunks),
            "document_count": len({c.document_id for c in chunks}),
            "corpus_fingerprint": corpus_fingerprint(chunks),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if self.storage_path is not None:
            (Path(self.storage_path) / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        return manifest

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    def search(self, query: str, k: int = settings.top_k) -> list[SearchHit]:
        if self.count() == 0:
            raise RuntimeError("Index is empty. Run `python -m src.cli build-index` first.")
        query_vector = self.embedder.embed([query])[0]
        points = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=k,
            with_payload=True,
        ).points
        return [
            SearchHit(
                chunk_id=p.payload["chunk_id"],
                document_id=p.payload["document_id"],
                title=p.payload["title"],
                chunk_index=p.payload["chunk_index"],
                text=p.payload["text"],
                score=float(p.score),
                rank=rank,
            )
            for rank, p in enumerate(points, start=1)
        ]

    def close(self) -> None:
        """Release the on-disk lock so another process can open the index."""
        self.client.close()


def build_index(storage_path: Path | None = settings.vector_storage_path) -> dict:
    index = VectorIndex(SentenceTransformerEmbedder(), storage_path)
    try:
        return index.build(load_chunks())
    finally:
        index.close()
