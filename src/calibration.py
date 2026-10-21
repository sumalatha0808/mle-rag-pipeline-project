"""Choose k, the similarity threshold and min_coverage on calibration.json.

Only the calibration set is used here. The chosen values are frozen to
``reports/calibration_settings.json`` and the final evaluation reads them
from that file, so ``questions.json`` never influences the settings.

Selection rule (lexicographic):
1. highest decision accuracy (answer supported, abstain unsupported);
2. highest recall@k on supported questions;
3. largest margin: the smallest distance, over all calibration questions,
   between a question and the decision boundary. A supported question needs
   both checks to pass, so its margin is min(score - t, coverage - c). An
   unsupported question is caught by either check, so its margin is
   max(t - score, c - coverage). Scores and coverage both lie in [0, 1];
4. smaller k (less unrelated text in the answer context);
5. larger threshold (the more conservative floor among equal margins).
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.abstention import decide, term_coverage
from src.config import settings
from src.retrieval import SearchHit, SentenceTransformerEmbedder, VectorIndex

K_VALUES = (1, 3, 5)
THRESHOLDS = [round(0.01 * i, 2) for i in range(0, 81)]
COVERAGES = [round(0.05 * i, 2) for i in range(0, 21)]
SETTINGS_FILE = settings.reports_dir / "calibration_settings.json"


@dataclass(frozen=True)
class Candidate:
    k: int
    threshold: float
    min_coverage: float
    accuracy: float
    recall_at_k: float
    margin: float


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def evaluate_candidate(cases, hits_by_id, k, threshold, min_coverage) -> Candidate:
    correct, recalls, margins = 0, [], []
    for case in cases:
        hits = hits_by_id[case["id"]][:k]
        decision = decide(case["question"], hits, threshold, min_coverage)
        correct += decision.answer != case["should_abstain"]
        if case["should_abstain"]:
            margins.append(max(threshold - decision.top_score, min_coverage - decision.coverage))
        else:
            margins.append(min(decision.top_score - threshold, decision.coverage - min_coverage))
            expected = set(case["expected_sources"])
            recalls.append(float(any(h.document_id in expected for h in hits)))
    return Candidate(
        k=k,
        threshold=threshold,
        min_coverage=min_coverage,
        accuracy=correct / len(cases),
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        margin=round(min(margins), 4),
    )


def select(candidates: list[Candidate]) -> Candidate:
    return max(
        candidates,
        key=lambda c: (c.accuracy, c.recall_at_k, c.margin, -c.k, c.threshold, -c.min_coverage),
    )


def calibrate(calibration_path: Path = settings.evaluation_dir / "calibration.json") -> dict:
    cases = json.loads(calibration_path.read_text(encoding="utf-8"))
    embedder = SentenceTransformerEmbedder()
    index = VectorIndex(embedder)
    try:
        hits_by_id: dict[str, list[SearchHit]] = {
            case["id"]: index.search(case["question"], k=max(K_VALUES)) for case in cases
        }
    finally:
        index.close()

    candidates = [
        evaluate_candidate(cases, hits_by_id, k, t, c)
        for k in K_VALUES
        for t in THRESHOLDS
        for c in COVERAGES
    ]
    best = select(candidates)

    rows = []
    for case in cases:
        hits = hits_by_id[case["id"]][: best.k]
        decision = decide(case["question"], hits, best.threshold, best.min_coverage)
        rows.append(
            {
                "id": case["id"],
                "should_abstain": case["should_abstain"],
                "top_document": hits[0].document_id,
                "top_score": round(decision.top_score, 4),
                "coverage": round(term_coverage(case["question"], hits), 3),
                "decision": "answer" if decision.answer else "abstain",
                "reason": decision.reason,
                "correct": decision.answer != case["should_abstain"],
            }
        )

    def sweep(**fixed):
        return [
            {key: getattr(c, key) for key in ("threshold", "min_coverage", "accuracy")}
            for c in candidates
            if all(getattr(c, name) == value for name, value in fixed.items())
        ]

    frozen = {
        "k": best.k,
        "threshold": best.threshold,
        "min_coverage": best.min_coverage,
        "calibration_accuracy": round(best.accuracy, 3),
        "calibration_recall_at_k": round(best.recall_at_k, 3),
        "calibration_margin": best.margin,
        "best_by_k": {
            str(k): vars(select([c for c in candidates if c.k == k])) for k in K_VALUES
        },
        "selection_rule": (
            "max decision accuracy, then recall@k, then the largest worst-case margin to "
            "the decision boundary, then smaller k, then larger threshold"
        ),
        "embedding_model": embedder.name,
        "embedding_revision": embedder.version,
        "calibration_file": calibration_path.name,
        "calibration_file_sha256_12": _file_sha(calibration_path),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_rows": rows,
        "threshold_sensitivity": sweep(k=best.k, min_coverage=best.min_coverage),
        "coverage_sensitivity": sweep(k=best.k, threshold=best.threshold),
    }
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def load_frozen_settings() -> dict:
    """Settings used by the pipeline. Falls back to .env values if not calibrated yet."""
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {"k": data["k"], "threshold": data["threshold"],
                "min_coverage": data["min_coverage"], "source": SETTINGS_FILE.name}
    return {"k": settings.top_k, "threshold": settings.min_score,
            "min_coverage": 0.0, "source": ".env defaults (not calibrated)"}
