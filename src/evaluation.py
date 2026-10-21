"""Evaluate the end-to-end pipeline on data/evaluation/questions.json.

Every question goes through the same ``RagPipeline.ask`` used by the CLI,
with the k / threshold / min_coverage frozen by ``calibrate``. The run refuses
to start if the calibration file changed after the settings were frozen.

Metric rubrics
--------------
recall@k            Supported questions only. 1 if at least one expected
                    source document is among the k retrieved chunks (before
                    the safety filter), else 0. Averaged. ``full_recall_at_k``
                    instead requires every expected document. Both are
                    document-level: the right document can be retrieved
                    through a chunk that does not contain the answer.
MRR                 Supported questions only. 1 / rank of the first retrieved
                    chunk whose document is expected; 0 if none. Averaged.
decision accuracy   All questions. Correct if the pipeline answered a
                    supported question or abstained on an unsupported one.
answer relevance    Supported questions only. Share of the labelled
                    ``required_terms`` that appear in the answer
                    (case-insensitive, whitespace-normalised). An abstention
                    on a supported question scores 0.
faithfulness        Answered questions only. The answer is split into
                    segments, one per citation ("text [doc-id]"). A segment
                    is supported if its text appears verbatim in a retrieved
                    context chunk from the cited document. Score = share of
                    supported segments. Literal matching only works for
                    extractive answers, so LLM answers are reported as
                    "unrated" rather than guessed.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.calibration import SETTINGS_FILE
from src.config import settings
from src.generation import CITATION
from src.pipeline import RagPipeline, RagResult

QUESTIONS_FILE = settings.evaluation_dir / "questions.json"


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def recall_at_k(retrieved_docs: list[str], expected: set[str]) -> float:
    return float(any(doc in expected for doc in retrieved_docs))


def full_recall_at_k(retrieved_docs: list[str], expected: set[str]) -> float:
    return float(expected <= set(retrieved_docs))


def reciprocal_rank(retrieved_docs: list[str], expected: set[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def answer_relevance(answer: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 1.0
    text = _normalise(answer)
    return sum(_normalise(term) in text for term in required_terms) / len(required_terms)


def citation_segments(answer: str) -> list[tuple[str, str]]:
    """Split "A. [doc-a] B. [doc-b]" into [("A.", "doc-a"), ("B.", "doc-b")]."""
    segments, start = [], 0
    for match in CITATION.finditer(answer):
        text = answer[start:match.start()].strip()
        if text:
            segments.append((text, match.group(1)))
        start = match.end()
    trailing = answer[start:].strip()
    if trailing:
        segments.append((trailing, ""))  # uncited text is never supported
    return segments


def literal_faithfulness(answer: str, context: dict[str, list[str]]) -> float:
    """Share of cited segments found verbatim in a context chunk of the cited document."""
    segments = citation_segments(answer)
    if not segments:
        return 0.0
    supported = sum(
        any(_normalise(text) in _normalise(chunk) for chunk in context.get(doc, []))
        for text, doc in segments
    )
    return supported / len(segments)


@dataclass
class CaseResult:
    id: str
    question: str
    should_abstain: bool
    expected_sources: list[str]
    required_terms: list[str]
    retrieved: list[dict]
    answered: bool
    reason: str
    answer: str
    citations: list[str]
    safety_events: list[str]
    decision_correct: bool
    top_score: float
    coverage: float
    recall_at_k: float | None
    full_recall_at_k: float | None
    reciprocal_rank: float | None
    answer_relevance: float | None
    faithfulness: float | str | None
    cites_expected_source: bool | None
    model: str


def score_case(case: dict, result: RagResult) -> CaseResult:
    expected = set(case["expected_sources"])
    supported = not case["should_abstain"]
    retrieved_docs = [h.document_id for h in result.retrieved]
    context: dict[str, list[str]] = {}
    for hit in result.context:
        context.setdefault(hit.document_id, []).append(hit.text)

    faithfulness: float | str | None = None
    if result.answered:
        faithfulness = (literal_faithfulness(result.answer, context)
                        if result.mode == "extractive" else "unrated")

    return CaseResult(
        id=case["id"],
        question=case["question"],
        should_abstain=case["should_abstain"],
        expected_sources=case["expected_sources"],
        required_terms=case["required_terms"],
        retrieved=[{"chunk_id": h.chunk_id, "document_id": h.document_id,
                    "score": round(h.score, 4)} for h in result.retrieved],
        answered=result.answered,
        reason=result.reason,
        answer=result.answer,
        citations=result.citations,
        safety_events=[f"{e.chunk_id}:{e.rule}" for e in result.safety_events],
        decision_correct=result.answered == supported,
        top_score=round(result.decision.top_score, 4),
        coverage=round(result.decision.coverage, 3),
        recall_at_k=recall_at_k(retrieved_docs, expected) if supported else None,
        full_recall_at_k=full_recall_at_k(retrieved_docs, expected) if supported else None,
        reciprocal_rank=reciprocal_rank(retrieved_docs, expected) if supported else None,
        answer_relevance=(answer_relevance(result.answer, case["required_terms"])
                          if supported and result.answered else (0.0 if supported else None)),
        faithfulness=faithfulness,
        cites_expected_source=(bool(expected & set(result.citations))
                               if supported and result.answered else None),
        model=result.model,
    )


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(values) / len(values), 3) if values else None


def summarise(cases: list[CaseResult]) -> dict:
    supported = [c for c in cases if not c.should_abstain]
    unsupported = [c for c in cases if c.should_abstain]
    answered = [c for c in cases if c.answered]
    return {
        "questions": len(cases),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "recall_at_k": _mean(c.recall_at_k for c in supported),
        "full_recall_at_k": _mean(c.full_recall_at_k for c in supported),
        "mrr": _mean(c.reciprocal_rank for c in supported),
        "decision_accuracy": _mean(float(c.decision_correct) for c in cases),
        "supported_answered_rate": _mean(float(c.answered) for c in supported),
        "unsupported_abstained_rate": _mean(float(not c.answered) for c in unsupported),
        "answer_relevance": _mean(c.answer_relevance for c in supported),
        "answer_relevance_when_answered": _mean(
            c.answer_relevance for c in supported if c.answered),
        "faithfulness": _mean(c.faithfulness for c in answered),
        "faithfulness_rated": sum(isinstance(c.faithfulness, float) for c in answered),
        "faithfulness_unrated": sum(c.faithfulness == "unrated" for c in answered),
        "cites_expected_source_rate": _mean(
            float(c.cites_expected_source) for c in supported if c.answered),
        "safety_events": sum(len(c.safety_events) for c in cases),
    }


def _check_frozen_settings() -> dict:
    if not SETTINGS_FILE.exists():
        raise RuntimeError("No frozen settings. Run `python -m src.cli calibrate` first.")
    frozen = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    calibration_path = settings.evaluation_dir / frozen["calibration_file"]
    current = hashlib.sha256(calibration_path.read_bytes()).hexdigest()[:12]
    if current != frozen["calibration_file_sha256_12"]:
        raise RuntimeError("calibration.json changed after freezing. Re-run calibrate.")
    return frozen


def evaluate(mode: str = "extractive", questions_path: Path = QUESTIONS_FILE) -> dict:
    frozen = _check_frozen_settings()
    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    pipeline = RagPipeline()
    try:
        results = [score_case(case, pipeline.ask(case["question"], mode)) for case in cases]
    finally:
        pipeline.close()

    models = sorted({c.model for c in results} - {"none"})
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "generation_model": ", ".join(models) or "none (extractive)",
        "settings": {key: frozen[key] for key in ("k", "threshold", "min_coverage", "frozen_at",
                                                  "embedding_model", "embedding_revision")},
        "questions_file_sha256_12": hashlib.sha256(questions_path.read_bytes()).hexdigest()[:12],
        "summary": summarise(results),
        "cases": [vars(c) for c in results],
    }
    out = settings.reports_dir / f"evaluation_results_{mode}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_file"] = str(out)
    return report
