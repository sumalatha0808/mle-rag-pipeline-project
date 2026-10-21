"""End-to-end question answering: retrieve, filter, decide, generate, validate.

    question
      -> retrieve top-k chunks (Qdrant)
      -> drop chunks containing injected instructions (safety events)
      -> abstain unless the remaining evidence passes the frozen threshold rule
      -> generate a cited answer (extractive or LLM)
      -> reject answers with missing/unknown citations or unsafe output
"""

from dataclasses import asdict, dataclass, field

from src.abstention import Decision, decide
from src.calibration import load_frozen_settings
from src.generation import SYSTEM_PROMPT, generate, validate_citations
from src.retrieval import SearchHit, SentenceTransformerEmbedder, VectorIndex
from src.safety import SafetyEvent, scan_chunk, scan_output

ABSTAIN_MESSAGE = (
    "I can't answer this from the approved support articles. "
    "Please check with the responsible team or escalate the request."
)


@dataclass
class RagResult:
    question: str
    answered: bool
    answer: str
    reason: str
    citations: list[str]
    mode: str
    model: str
    decision: Decision
    retrieved: list[SearchHit]
    context: list[SearchHit]
    safety_events: list[SafetyEvent] = field(default_factory=list)
    rejected_answer: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, index: VectorIndex | None = None, frozen: dict | None = None) -> None:
        self.index = index or VectorIndex(SentenceTransformerEmbedder())
        self.frozen = frozen or load_frozen_settings()

    def close(self) -> None:
        self.index.close()

    def ask(self, question: str, mode: str = "extractive") -> RagResult:
        retrieved = self.index.search(question, self.frozen["k"])

        events: list[SafetyEvent] = []
        context: list[SearchHit] = []
        for hit in retrieved:
            found = scan_chunk(hit.chunk_id, hit.document_id, hit.text)
            events.extend(found)
            if not found:
                context.append(hit)

        decision = decide(question, context, self.frozen["threshold"], self.frozen["min_coverage"])

        def result(answered, answer, reason, citations=(), mode_=mode, model="none", rejected=None):
            return RagResult(question, answered, answer, reason, list(citations), mode_, model,
                             decision, retrieved, context, events, rejected)

        if not decision.answer:
            return result(False, ABSTAIN_MESSAGE, decision.reason)

        generation = generate(question, context, mode)
        if generation.abstained:
            return result(False, ABSTAIN_MESSAGE, "generator_abstained",
                          model=generation.model, rejected=generation.text or None)

        blocked = scan_output(generation.text, SYSTEM_PROMPT)
        if blocked:
            return result(False, ABSTAIN_MESSAGE, blocked,
                          model=generation.model, rejected=generation.text)

        check = validate_citations(generation.text, {h.document_id for h in context})
        if not check.valid:
            return result(False, ABSTAIN_MESSAGE, f"citation_rejected:{check.reason}",
                          check.cited, model=generation.model, rejected=generation.text)

        return result(True, generation.text, "answered", check.cited, model=generation.model)
