import pytest

from src.evaluation import (
    answer_relevance,
    citation_segments,
    full_recall_at_k,
    literal_faithfulness,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_and_reciprocal_rank():
    retrieved = ["invoice-corrections", "billing-refunds", "billing-refunds"]
    assert recall_at_k(retrieved, {"billing-refunds"}) == 1.0
    assert reciprocal_rank(retrieved, {"billing-refunds"}) == 0.5
    assert recall_at_k(retrieved, {"privacy-data"}) == 0.0
    assert reciprocal_rank(retrieved, {"privacy-data"}) == 0.0
    assert full_recall_at_k(retrieved, {"billing-refunds", "invoice-corrections"}) == 1.0
    assert full_recall_at_k(retrieved, {"billing-refunds", "privacy-data"}) == 0.0


def test_answer_relevance_counts_required_terms_case_insensitively():
    answer = "Provide the Invoice Number and both payment\nreferences. [billing-refunds]"
    assert answer_relevance(answer, ["invoice number", "payment references"]) == 1.0
    assert answer_relevance(answer, ["invoice number", "30 calendar days"]) == 0.5


def test_citation_segments_keep_uncited_tail():
    segments = citation_segments("A rule. [doc-a] Another. [doc-b] Made up.")
    assert segments == [("A rule.", "doc-a"), ("Another.", "doc-b"), ("Made up.", "")]


def test_literal_faithfulness_requires_text_from_the_cited_document():
    context = {"doc-a": ["Links expire after 20 minutes. They work once."]}
    assert literal_faithfulness("Links expire after 20 minutes. [doc-a]", context) == 1.0
    # Correct citation but altered fact: citation correctness alone is not faithfulness.
    assert literal_faithfulness("Links expire after 60 minutes. [doc-a]", context) == 0.0
    # Real text attributed to the wrong document is not supported either.
    assert literal_faithfulness("They work once. [doc-b]", context) == 0.0
    assert literal_faithfulness("They work once. [doc-a] Extra claim.", context) == pytest.approx(0.5)
