from src.ingestion import chunk_documents, load_documents, split_paragraph

EXPECTED_IDS = {
    "account-mfa", "account-password", "billing-refunds", "invoice-corrections",
    "privacy-data", "programme-attendance", "programme-extensions",
    "retrieved-content-safety", "technical-api-limits", "technical-incidents",
}


def test_loads_every_document_with_title():
    docs = load_documents()
    assert {d.document_id for d in docs} == EXPECTED_IDS
    assert all(d.title and not d.body.startswith("#") for d in docs)


def test_every_chunk_keeps_citation_metadata():
    chunks = chunk_documents(load_documents())
    assert {c.document_id for c in chunks} == EXPECTED_IDS
    for c in chunks:
        assert c.document_id and c.title and c.text
        assert c.chunk_id == f"{c.document_id}#{c.chunk_index}"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_indexes_are_sequential_per_document():
    by_doc: dict[str, list[int]] = {}
    for c in chunk_documents(load_documents()):
        by_doc.setdefault(c.document_id, []).append(c.chunk_index)
    for indexes in by_doc.values():
        assert indexes == list(range(len(indexes)))


def test_long_paragraph_splits_on_sentences_with_overlap():
    paragraph = "First rule applies. Second rule applies. Third rule applies."
    pieces = split_paragraph(paragraph, max_chars=40, overlap_sentences=1)
    assert pieces == [
        "First rule applies. Second rule applies.",
        "Second rule applies. Third rule applies.",
    ]


def test_short_paragraph_is_kept_whole():
    assert split_paragraph("One sentence.", 600, 1) == ["One sentence."]
