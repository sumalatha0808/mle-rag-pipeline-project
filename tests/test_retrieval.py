import numpy as np
import pytest

from src.ingestion import load_chunks
from src.retrieval import SentenceTransformerEmbedder, VectorIndex


class BagOfWordsEmbedder:
    """Tiny deterministic embedder so store tests run without the model."""

    name = "test-bag-of-words"
    version = "1"
    dimension = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, sum(map(ord, word.strip(".,`"))) % self.dimension] += 1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_index_stores_every_chunk_with_metadata():
    chunks = load_chunks()
    index = VectorIndex(BagOfWordsEmbedder(), storage_path=None)
    manifest = index.build(chunks)
    assert index.count() == len(chunks) == manifest["chunk_count"]
    hit = index.search("password reset link expires", k=1)[0]
    assert hit.document_id == "account-password"
    assert hit.chunk_id.startswith("account-password#") and hit.title


def test_rebuild_replaces_instead_of_duplicating():
    chunks = load_chunks()
    index = VectorIndex(BagOfWordsEmbedder(), storage_path=None)
    index.build(chunks)
    index.build(chunks)
    assert index.count() == len(chunks)


def test_search_on_empty_index_explains_what_to_do():
    index = VectorIndex(BagOfWordsEmbedder(), storage_path=None)
    with pytest.raises(RuntimeError, match="build-index"):
        index.search("anything")


@pytest.fixture(scope="module")
def minilm_index():
    index = VectorIndex(SentenceTransformerEmbedder(), storage_path=None)
    index.build(load_chunks())
    yield index
    index.close()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which payment details must I include when reporting a duplicate charge?", "billing-refunds"),
        ("Where should I open a ticket to replace a lost authenticator?", "account-mfa"),
        ("How should my integration handle rate limiting from the Northstar API?", "technical-api-limits"),
    ],
)
def test_minilm_retrieves_expected_document_in_top_3(minilm_index, question, expected):
    hits = minilm_index.search(question, k=3)
    assert expected in [h.document_id for h in hits]
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
