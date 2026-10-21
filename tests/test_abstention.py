from src.abstention import content_terms, decide, stem, term_coverage
from src.calibration import Candidate, evaluate_candidate, select
from src.retrieval import SearchHit


def hit(document_id: str, text: str, score: float, rank: int = 1) -> SearchHit:
    return SearchHit(f"{document_id}#0", document_id, "Title", 0, text, score, rank)


REFUNDS = hit("billing-refunds", "Northstar refunds duplicate payments for the same invoice.", 0.45)


def test_stem_matches_word_forms():
    assert stem("limiting") == stem("limits") == "limit"
    assert stem("charges") == stem("charge")


def test_content_terms_drop_stopwords():
    assert content_terms("What is the refund policy?") == {"refund", "policy"}


def test_coverage_only_counts_the_top_document():
    other = hit("invoice-corrections", "A subscription policy for annual plans.", 0.40, rank=2)
    query = "annual subscription refund policy"
    assert term_coverage(query, [REFUNDS, other]) == 0.25  # only "refund" is in the top doc


def test_low_similarity_abstains():
    decision = decide("refund duplicate payments", [REFUNDS], threshold=0.5, min_coverage=0.0)
    assert not decision.answer and decision.reason == "low_similarity"


def test_similar_but_inapplicable_question_abstains_on_coverage():
    decision = decide("annual subscription cancellation refund", [REFUNDS], 0.3, 0.4)
    assert not decision.answer and decision.reason == "low_coverage"


def test_supported_question_answers():
    decision = decide("refund for duplicate payments", [REFUNDS], 0.3, 0.4)
    assert decision.answer and decision.reason == "evidence_ok"


def test_no_hits_abstains():
    assert decide("anything", [], 0.0, 0.0).reason == "no_results"


def test_candidate_margin_and_selection():
    cases = [
        {"id": "a", "question": "refund duplicate payments", "should_abstain": False,
         "expected_sources": ["billing-refunds"]},
        {"id": "b", "question": "student discount", "should_abstain": True,
         "expected_sources": []},
    ]
    hits = {"a": [REFUNDS], "b": [hit("billing-refunds", "unrelated text", 0.20)]}
    good = evaluate_candidate(cases, hits, k=1, threshold=0.3, min_coverage=0.5)
    assert good.accuracy == 1.0 and good.recall_at_k == 1.0
    # supported: min(0.45-0.3, 1.0-0.5) = 0.15; unsupported: max(0.3-0.2, 0.5-0.0) = 0.5
    assert good.margin == 0.15
    bad = evaluate_candidate(cases, hits, k=1, threshold=0.5, min_coverage=0.0)
    assert select([bad, good]) == good


def test_selection_prefers_smaller_k_on_ties():
    a = Candidate(5, 0.3, 0.4, 1.0, 1.0, 0.2)
    b = Candidate(3, 0.3, 0.4, 1.0, 1.0, 0.2)
    assert select([a, b]).k == 3
