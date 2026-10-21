# TODO — Northstar RAG Project

Work top to bottom. Each phase ends with something you can show (a command, a test, or a report section).
Do **not** edit `data/` files. Do **not** look at `questions.json` results while tuning — use `calibration.json` only.

---

## Phase 0 — Setup (30 min)

- [x] `uv sync` — install dependencies from `pyproject.toml` / `uv.lock`
- [x] Check `.env` is in `.gitignore` (never commit API keys)
- [x] Create folders: `src/`, `tests/`, `reports/`
- [x] Add `src/__init__.py` and `src/config.py` (reads `.env`: `EMBEDDING_MODEL`, `TOP_K`, `MIN_SCORE`, `VECTOR_STORAGE_PATH`, `LLM_PROVIDER`)
- [x] Create a git branch (`feat/rag-pipeline-phases-0-4`) — PR not opened yet
- [x] Decide what to do with the `northstar-rag/` folder — its `config/settings.json` references doc IDs (`billing-invoices`, `incident-response`…) that don't exist in this knowledge base, so it looks like it came from somewhere else

## Phase 1 — Ingestion & Chunking → `src/ingestion.py` (Milestone 1)

- [x] Load all 10 `.md` files from `data/knowledge_base/`
- [x] `document_id` = filename without `.md` (e.g. `billing-refunds`), `title` = first `# ` heading
- [x] Split into chunks by heading / paragraph (docs are short: 8–14 lines each)
  - [x] Pick chunk size + overlap (e.g. ~500 chars, ~1 sentence overlap) and **write down why**
  - [x] Never cut in the middle of a sentence; keep the section heading with its paragraph
- [x] Each chunk stores: `chunk_id`, `document_id`, `title`, `section`, `chunk_index`, `text`
- [x] Print a summary: chunks per document + one sample chunk with metadata

## Phase 2 — Embeddings & Vector Store → `src/retrieval.py` (Milestone 1)

- [x] Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` (record model name **and revision**)
- [x] Index in Qdrant (`qdrant-client`, in-memory or local path) — cosine distance, vector size 384
- [x] Payload = chunk metadata from Phase 1
- [x] CLI command to rebuild the index, e.g. `uv run python -m src.cli build-index`

## Phase 3 — Retrieval & Abstention Threshold (Milestone 2/3)

- [x] `retrieve(query, k)` → top-k chunks with scores
- [x] Abstention rule — answer only if:
  - [x] top score ≥ `MIN_SCORE`, **and**
  - [x] (recommended) enough query keywords appear in the evidence — similarity alone doesn't prove the policy applies
    (calibration has "nearby" traps: free extension due to outage, annual cancellation refunds, student discount)
- [x] Script: run every `calibration.json` question, sweep thresholds (and k = 1, 3, 5), record accuracy per value
- [x] Choose settings, **freeze them** (save to a file with a date), then never change them after the final run

## Phase 4 — Grounded Generation & Guardrails → `src/generation.py` (Milestone 2)

- [x] **Dry/extractive mode (no API key, deterministic)** — build the answer from retrieved sentences + `[document_id]`
- [x] Optional LLM mode (Groq/Mistral via `.env`) — system prompt: "answer only from context, cite `[doc-id]`, say you don't know otherwise"
- [x] Wrap retrieved text in clear delimiters and label it as **untrusted data**
- [x] Detect/neutralize injection text (e.g. "ignore previous instructions") — see `retrieved-content-safety.md`
- [x] Citation validator: parse `[...]` in the answer, **reject** any ID not in the retrieved chunks
- [x] Abstention message when the threshold rule fails
- [x] CLI: `uv run python -m src.cli ask "How do I get a refund for a duplicate charge?"`

## Phase 5 — Tests → `tests/` (Milestone 2)

- [x] `test_ingestion.py` — every chunk has `document_id`, `title`, `chunk_index`; all 10 docs loaded
- [x] `test_retrieval.py` — a known question returns the right document in top-k
- [x] `test_generation.py` — fabricated citation `[fake-doc]` is rejected; unsupported question abstains
- [x] `test_safety.py` — injection text in a chunk is not followed
- [x] `uv run pytest` passes

## Phase 6 — Evaluation → `src/evaluation.py` (Milestone 3)

- [x] Run all 16 questions in `questions.json` through the **same** end-to-end pipeline
- [x] Metrics:
  - [x] **recall@k** — expected source in top-k?
  - [x] **MRR** — 1 / rank of first correct source
  - [x] **Decision accuracy** — answered when it should, abstained when it should
  - [x] **Answer relevance** — rubric, e.g. fraction of `required_terms` present in the answer
  - [x] **Faithfulness** — rubric, e.g. every answer sentence is literally found in a retrieved chunk (LLM paraphrases = "unrated")
- [x] Save per-question results (JSON/CSV) for the report

## Phase 7 — Report → `reports/evaluation_report.md` (Milestone 3)

- [x] Configuration: chunking, embedding model + version, vector store, k, threshold, generation mode
- [x] Metrics table + per-question results table
- [x] Threshold chosen + calibration evidence (sweep table)
- [x] Analysis of ≥ 2 failures or fragile successes (show the retrieved chunks/scores)
- [x] Any label disagreements (documented, labels not changed)
- [x] 3 prioritized improvement recommendations, tied to the failures
- [x] Answer the 4 submission questions:
  1. Which retrieval failures matter most for support, and why?
  2. How did you select the no-answer threshold?
  3. What evidence shows an answer is faithful?
  4. What would you change before live use?

## Phase 8 — Final Polish

- [x] Update `README.md`: setup from scratch, all commands (build-index, ask, evaluate, test)
- [x] Fresh clone → `uv sync` → commands work
- [ ] Merge PRs, final check of the acceptance checklist in `project-description.md`

---

**Tip:** `solution/` contains a working reference implementation (`solution/rag_project/`, `solution/reports/reference_evaluation.md`).
Use it to get unstuck, but build and justify your own design.
