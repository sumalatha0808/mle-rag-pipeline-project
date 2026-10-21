"""Load the Markdown knowledge base and split it into citable chunks.

Chunking strategy
-----------------
* Each document is split on blank lines into paragraphs. In this corpus a
  paragraph holds one policy rule, so paragraphs are never merged: merging
  would blur which rule a citation supports.
* A paragraph longer than ``max_chars`` is split on sentence boundaries, and
  the last ``overlap_sentences`` sentences are repeated at the start of the
  next piece so a rule that spans the cut keeps its context.
* The document title is stored as metadata and prefixed to the text that is
  embedded, so a chunk like "Each reset link expires after 20 minutes" still
  matches queries about "password reset".
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import PROJECT_ROOT, settings

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`])")


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    body: str
    path: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    chunk_index: int
    paragraph_index: int
    text: str
    source_path: str

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model; the title adds topical context."""
        return f"{self.title}. {self.text}"

    def to_payload(self) -> dict:
        return asdict(self)


def load_documents(source_dir: Path = settings.knowledge_base_dir) -> list[Document]:
    """Read every ``*.md`` file; ``document_id`` is the filename stem."""
    documents = []
    for path in sorted(Path(source_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path.name}: first line must be a '# Title' heading")
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            raise ValueError(f"{path.name}: document has no body text")
        try:
            rel_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        documents.append(Document(path.stem, title, body, rel_path))
    if not documents:
        raise ValueError(f"No Markdown files found in {source_dir}")
    return documents


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def split_paragraph(paragraph: str, max_chars: int, overlap_sentences: int) -> list[str]:
    """Return the paragraph whole, or sentence-aligned pieces of <= max_chars."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(paragraph):
        if current and len(" ".join(current + [sentence])) > max_chars:
            pieces.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_documents(
    documents: list[Document],
    max_chars: int = settings.chunk_max_chars,
    overlap_sentences: int = settings.chunk_overlap_sentences,
) -> list[Chunk]:
    if max_chars <= 0 or overlap_sentences < 0:
        raise ValueError("max_chars must be > 0 and overlap_sentences >= 0")
    chunks = []
    for doc in documents:
        paragraphs = [" ".join(p.split()) for p in doc.body.split("\n\n") if p.strip()]
        index = 0
        for p_index, paragraph in enumerate(paragraphs):
            for piece in split_paragraph(paragraph, max_chars, overlap_sentences):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}#{index}",
                        document_id=doc.document_id,
                        title=doc.title,
                        chunk_index=index,
                        paragraph_index=p_index,
                        text=piece,
                        source_path=doc.path,
                    )
                )
                index += 1
    return chunks


def load_chunks() -> list[Chunk]:
    return chunk_documents(load_documents())


def summarize(chunks: list[Chunk]) -> str:
    """Chunk counts per document plus one sample chunk (Milestone 1 evidence)."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    lines = [f"{len(chunks)} chunks from {len(counts)} documents", ""]
    lines += [f"  {doc_id:<26} {n}" for doc_id, n in counts.items()]
    lines += ["", "Sample chunk:"]
    lines += [f"  {k}: {v}" for k, v in chunks[0].to_payload().items()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(load_chunks()))
