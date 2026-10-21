"""Treat retrieved text as untrusted data (OWASP LLM01: prompt injection).

Defences, in pipeline order:
1. ``scan_chunk`` flags retrieved chunks that contain instruction-like text
   (e.g. "ignore all prior instructions", "reveal the system prompt"). Flagged
   chunks are removed from the answer context and recorded as safety events,
   as the knowledge base's own ``retrieved-content-safety`` policy requires.
2. The LLM prompt wraps the remaining chunks in <source> tags and tells the
   model they are data, never instructions (see ``generation.py``).
3. ``scan_output`` blocks an answer that leaks secrets or repeats injected
   instructions, in case steps 1 and 2 are bypassed.
"""

import os
import re
from dataclasses import dataclass

INJECTION_PATTERNS = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(prior|previous|above|earlier|all|system)\b"
        r".{0,20}\b(instructions?|rules|prompts?)\b",
        re.IGNORECASE,
    ),
    "reveal_secrets": re.compile(
        # Targets the assistant's own secrets only; policy text such as "support
        # cannot disclose a password" must stay usable as evidence.
        r"\b(reveal|print|show|output|disclose|leak)\b.{0,30}\b(system prompt|api key|secret|credential)s?\b",
        re.IGNORECASE,
    ),
    "role_hijack": re.compile(
        r"\b(you are now|act as|new instructions:|from now on you)\b", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class SafetyEvent:
    chunk_id: str
    document_id: str
    rule: str
    excerpt: str


def scan_text(text: str) -> list[tuple[str, str]]:
    """Return (rule, matched text) for every injection pattern found."""
    return [(rule, m.group(0)) for rule, pattern in INJECTION_PATTERNS.items()
            if (m := pattern.search(text))]


def scan_chunk(chunk_id: str, document_id: str, text: str) -> list[SafetyEvent]:
    return [SafetyEvent(chunk_id, document_id, rule, excerpt) for rule, excerpt in scan_text(text)]


def _secret_values() -> list[str]:
    return [v for k, v in os.environ.items()
            if k.endswith(("_API_KEY", "_TOKEN")) and len(v) >= 8]


def scan_output(answer: str, system_prompt: str = "") -> str | None:
    """Return a reason string if the generated answer must be blocked."""
    if any(secret in answer for secret in _secret_values()):
        return "output_contains_secret"
    if system_prompt and system_prompt[:80] in answer:
        return "output_contains_system_prompt"
    # Only the override pattern applies to outputs: a correct answer may say
    # "the assistant must not reveal the system prompt", and real leaks are
    # caught by the exact secret / prompt checks above.
    if INJECTION_PATTERNS["override_instructions"].search(answer):
        return "output_contains_injection_text"
    return None
