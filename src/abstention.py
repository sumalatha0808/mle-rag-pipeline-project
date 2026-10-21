"""Decide whether retrieved evidence is strong enough to answer.

Two checks, both chosen on ``data/evaluation/calibration.json``:

1. **Similarity**: the best chunk's cosine score must reach ``threshold``.
   This catches questions about topics the knowledge base never mentions.
2. **Coverage**: at least ``min_coverage`` of the question's content terms
   must appear in the retrieved chunks of the top-ranked document. A close
   vector only shows that the topic is similar; coverage checks that the
   specific thing asked about (e.g. "annual", "cancellation") is in the
   evidence. Only the top document counts, because pooling words from several
   documents lets an unsupported question borrow "refund" from one policy and
   "subscription" from another.
"""

import re
from dataclasses import dataclass

from src.retrieval import SearchHit

# Question words, function words and the brand name. They appear in nearly
# every question or document, so they carry no evidence about applicability.
STOPWORDS = frozenset(
    """a about after an and any are as at be because been before being by can
    could did do does for from had has have how i if in into is it its me my
    no not of on or our should so than that the their them then there these
    they this to up us was we were what when where which who why will with
    would you your must northstar get got need want""".split()
)
_SUFFIXES = ("ing", "ed", "es", "s", "e")


def stem(word: str) -> str:
    """Crude suffix stripping: 'limiting'/'limits' -> 'limit', 'charges'/'charge' -> 'charg'."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {stem(w) for w in words if w not in STOPWORDS and len(w) > 1}


def term_coverage(query: str, hits: list[SearchHit]) -> float:
    """Share of the query's content terms found in the top document's retrieved chunks."""
    query_terms = content_terms(query)
    if not query_terms or not hits:
        return 0.0
    top_document = hits[0].document_id
    evidence_terms: set[str] = set()
    for hit in hits:
        if hit.document_id == top_document:
            evidence_terms |= content_terms(f"{hit.title} {hit.text}")
    return len(query_terms & evidence_terms) / len(query_terms)


@dataclass(frozen=True)
class Decision:
    answer: bool
    reason: str  # evidence_ok | no_results | low_similarity | low_coverage
    top_score: float
    coverage: float


def decide(query: str, hits: list[SearchHit], threshold: float, min_coverage: float) -> Decision:
    if not hits:
        return Decision(False, "no_results", 0.0, 0.0)
    top_score = hits[0].score
    coverage = term_coverage(query, hits)
    if top_score < threshold:
        return Decision(False, "low_similarity", top_score, coverage)
    if coverage < min_coverage:
        return Decision(False, "low_coverage", top_score, coverage)
    return Decision(True, "evidence_ok", top_score, coverage)
