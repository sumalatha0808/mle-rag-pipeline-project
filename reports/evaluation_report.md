# Northstar Support Assistant: Evaluation Report

**Recommendation in one line:** the prototype works as designed on the supplied fixtures (16/16 correct answer-or-abstain decisions, every answer cited and literally grounded), but one supported question (q01) produced a cited, "faithful" answer that does not actually answer the question, and the final set never tested near-miss unsupported questions or the injection path. It should **not** go to live support colleagues yet; the three changes in [Recommendations](#7-prioritised-recommendations) come first.

| | |
|---|---|
| Evaluation run | 2026-10-21T13:07:43Z, `questions.json` sha256 `c8455b21c172` |
| Settings frozen | 2026-10-21T12:48:10Z on `calibration.json` sha256 `92a12f6d3e0f`, before any final-set run |
| Primary mode | Extractive (deterministic, no API key) |
| Comparison mode | LLM: Groq `openai/gpt-oss-20b`, temperature 0 |
| Raw results | `reports/evaluation_results_extractive.json`, `reports/evaluation_results_llm.json` |
| Frozen settings + sweep | `reports/calibration_settings.json` |

## 1. System configuration

### Ingestion and chunking

| Setting | Value | Why |
|---|---|---|
| Unit | One paragraph (blank-line separated) per chunk | In this corpus each paragraph states one policy rule. Merging paragraphs would make a citation ambiguous about which rule it supports. |
| Max size | 600 characters, split on sentence boundaries | No paragraph in the current corpus exceeds it (all 22 chunks are whole paragraphs). The limit only protects against future long documents. |
| Overlap | 1 sentence, only when a paragraph is split | Keeps a rule's context across a cut. Whole paragraphs need no overlap. |
| Headings | `# Title` is removed from the body, stored as metadata and prefixed to the embedded text | A chunk like "Each reset link expires after 20 minutes" still matches "password reset". |
| Metadata per chunk | `chunk_id` (`account-mfa#0`), `document_id` (filename stem), `title`, `chunk_index`, `paragraph_index`, `text`, `source_path` | Enough to cite, to trace back to the file, and to rebuild. |
| Result | 22 chunks from 10 documents (2–3 per document) | `python -m src.cli chunks` |

### Embeddings and vector store

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (pinned), 384 dimensions, L2-normalised |
| Backend | Qdrant local mode, a real vector **database** collection (`northstar_support`) persisted to `storage/qdrant/`, no server process |
| Distance | Cosine |
| Rebuild | `python -m src.cli build-index` drops and recreates the collection from `data/knowledge_base/*.md` and writes `storage/qdrant/manifest.json` (model, revision, chunk count, corpus fingerprint, build time) |

### Retrieval, abstention and generation

| Setting | Value |
|---|---|
| k | 3 |
| Answer only if | top chunk cosine score ≥ **0.30** **and** ≥ **40 %** of the question's content terms appear in the retrieved chunks of the top-ranked document |
| Injection handling | Retrieved chunks matching instruction patterns ("ignore all prior instructions", "reveal the system prompt/API key", role hijack) are removed from the context and logged as safety events. The LLM prompt wraps sources in `<source id>` tags marked as untrusted data. Outputs containing secret values, the system prompt or override text are blocked. |
| Extractive mode | Quotes each retrieved paragraph that shares a content term with the question and scores within 0.20 of the best chunk, followed by `[document-id]` |
| LLM mode | Answers only from sources, cites `[document-id]`, replies `INSUFFICIENT_EVIDENCE` when sources don't apply |
| Citation check | An answer with no citation, or citing a document not in the retrieved context, is rejected and the system abstains |

## 2. How the threshold was chosen

The settings were selected on `calibration.json` only (3 supported and 3 unsupported questions) and frozen to a file. The evaluation command refuses to run if the calibration file has changed since freezing.

**Search grid:** k ∈ {1, 3, 5} × threshold 0.00–0.80 (step 0.01) × min coverage 0.00–1.00 (step 0.05).

**Selection rule (in order):** highest decision accuracy → highest recall@k → largest worst-case margin to the decision boundary → smaller k → higher threshold. A supported question's margin is `min(score − threshold, coverage − min_coverage)` (both checks must pass). An unsupported question's margin is `max(threshold − score, min_coverage − coverage)` (either check suffices).

**Calibration evidence (k = 3, threshold 0.30, min coverage 0.40):**

| id | Expected | Decision | Reason | Top score | Coverage | Top document |
|---|---|---|---|---|---|---|
| cal01 | answer | answer | evidence ok | 0.574 | 0.67 | invoice-corrections |
| cal02 | answer | answer | evidence ok | 0.503 | 1.00 | account-mfa |
| cal03 | answer | answer | evidence ok | 0.708 | 0.60 | technical-api-limits |
| cal04 | abstain | abstain | low coverage | 0.309 | 0.00 | account-password |
| cal05 | abstain | abstain | low coverage | **0.426** | 0.20 | billing-refunds |
| cal06 | abstain | abstain | low coverage | 0.305 | 0.00 | billing-refunds |

**Why two checks:** cal05 ("cancellation refund policy for an annual subscription") scores 0.426 against `billing-refunds`, close to the supported cal02 (0.503). A similarity threshold alone has only a 0.077-wide safe band. Coverage separates them widely: supported ≥ 0.60, unsupported ≤ 0.20. Coverage is measured on the top document only: when it was first pooled across all three chunks, cal05 reached 0.60 by borrowing "subscription" and "policy" from `invoice-corrections`.

**Sensitivity:** with k = 3 and min coverage 0.40, any threshold from 0.00 to 0.50 gives 100 % calibration accuracy (0.51 → 83 %). With threshold 0.30, any min coverage from 0.25 to 0.60 gives 100 % (0.20 → 83 %, 0.65 → 83 %). The worst-case margin is 0.20. k = 1 was rejected because recall@1 is 0.67 (cal01's top chunk is from `invoice-corrections`).

**Limitation:** on calibration the 0.30 similarity floor caught nothing; every abstention came from coverage. In the final set the opposite happened (all three abstentions came from low similarity). Six calibration questions are far too few to trust the exact values.

## 3. Results

### Summary

All rubric definitions are in [section 4](#4-metric-rubrics).

| Metric | Extractive (primary) | LLM (comparison) |
|---|---|---|
| recall@3 (any expected document) | **1.00** | 1.00 |
| full recall@3 (all expected documents) | 1.00 | 1.00 |
| MRR | **0.949** | 0.949 |
| Decision accuracy | **1.00** (16/16) | 0.94 (15/16) |
| · supported questions answered | 13/13 | 12/13 |
| · unsupported questions abstained | 3/3 | 3/3 |
| Answer relevance (supported, abstain = 0) | **0.949** | 0.769 |
| Faithfulness (answered) | **1.00** (13 rated) | unrated (12 paraphrased answers) |
| Answered questions citing an expected source | 13/13 | 12/12 |
| Safety events during the run | 0 | 0 |

Retrieval is identical in both modes, so recall and MRR are the same.

### Per-question results

"ext" = extractive mode. R@3 = recall@3, RR = reciprocal rank. ✅/❌ marks the decision only.

| ID | Question | Expected | Top-3 retrieved (score) | Top score / coverage | Decision (ext) | R@3 | RR | Relevance (ext) | Faithful (ext) | Decision (LLM) | Relevance (LLM) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 | I was charged twice for the same invoice. What must I send and when? | `billing-refunds` | `invoice-corrections#1` 0.644<br>`invoice-corrections#0` 0.467<br>`billing-refunds#2` 0.462 | 0.644 / 0.40 | answer ✅ | 1.00 | 0.33 | 0.33 | 1.00 | abstain (generator_abstained) ❌ | 0.00 |
| q02 | How long does finance need to review a complete duplicate-payment request? | `billing-refunds` | `billing-refunds#1` 0.758<br>`billing-refunds#0` 0.495<br>`billing-refunds#2` 0.495 | 0.758 / 0.86 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q03 | Can the legal company name on an invoice be corrected? | `invoice-corrections` | `invoice-corrections#0` 0.768<br>`invoice-corrections#1` 0.576<br>`billing-refunds#2` 0.399 | 0.768 / 1.00 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q04 | I replaced my phone and lost my authenticator. What should I do? | `account-mfa` | `account-mfa#0` 0.631<br>`account-mfa#1` 0.498<br>`account-password#1` 0.324 | 0.631 / 1.00 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 0.50 |
| q05 | How quickly does a password reset link expire? | `account-password` | `account-password#0` 0.823<br>`account-password#1` 0.703<br>`account-mfa#0` 0.332 | 0.823 / 0.80 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q06 | What do I need to include when requesting more time for an assignment? | `programme-extensions` | `programme-extensions#0` 0.489<br>`programme-extensions#1` 0.440<br>`programme-attendance#0` 0.333 | 0.489 / 0.40 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q07 | What should my client do after receiving an HTTP 429 response? | `technical-api-limits` | `technical-api-limits#1` 0.505<br>`technical-incidents#1` 0.350<br>`retrieved-content-safety#2` 0.309 | 0.505 / 0.60 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q08 | How frequently are updates published during a confirmed outage? | `technical-incidents` | `technical-incidents#0` 0.679<br>`technical-incidents#1` 0.436<br>`technical-api-limits#1` 0.305 | 0.679 / 0.67 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q09 | Can I send authentication tokens to a model while debugging support requests? | `privacy-data`, `technical-incidents` | `privacy-data#0` 0.351<br>`technical-incidents#1` 0.348<br>`account-mfa#1` 0.329 | 0.351 / 0.62 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q10 | How must an assistant handle instructions found inside retrieved documents? | `retrieved-content-safety` | `retrieved-content-safety#0` 0.627<br>`retrieved-content-safety#2` 0.518<br>`privacy-data#0` 0.424 | 0.627 / 0.57 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 0.50 |
| q11 | Where can I find the cafeteria menu for next week? | abstain | `programme-attendance#0` 0.168<br>`programme-attendance#1` 0.130<br>`technical-incidents#0` 0.105 | 0.168 / 0.20 | abstain (low_similarity) ✅ | – | – | – | – | abstain (low_similarity) ✅ | – |
| q12 | How much is the annual enterprise subscription? | abstain | `technical-api-limits#0` 0.232<br>`invoice-corrections#1` 0.182<br>`technical-api-limits#1` 0.146 | 0.232 / 0.00 | abstain (low_similarity) ✅ | – | – | – | – | abstain (low_similarity) ✅ | – |
| q13 | Should I paste my passport scan into chat to reset my second factor? | `account-mfa`, `privacy-data` | `account-mfa#1` 0.550<br>`account-mfa#0` 0.377<br>`privacy-data#1` 0.270 | 0.550 / 0.71 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 0.50 |
| q14 | Does sending an extension request immediately move my deadline? | `programme-extensions` | `programme-extensions#1` 0.726<br>`programme-extensions#0` 0.590<br>`technical-api-limits#1` 0.292 | 0.726 / 0.67 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 0.50 |
| q15 | How do I report an unexpected absence? | `programme-attendance` | `programme-attendance#0` 0.595<br>`programme-attendance#1` 0.491<br>`technical-incidents#1` 0.289 | 0.595 / 1.00 | answer ✅ | 1.00 | 1.00 | 1.00 | 1.00 | answer ✅ | 1.00 |
| q16 | Can I reserve a bicycle parking space near the office? | abstain | `programme-attendance#1` 0.111<br>`privacy-data#0` 0.084<br>`programme-attendance#0` 0.076 | 0.111 / 0.00 | abstain (low_similarity) ✅ | – | – | – | – | abstain (low_similarity) ✅ | – |

## 4. Metric rubrics

All metrics are deterministic and implemented in `src/evaluation.py`.

| Metric | Scope | Rubric |
|---|---|---|
| recall@3 | supported | 1 if any expected document is among the 3 retrieved chunks (before the safety filter), else 0. Document-level. |
| full recall@3 | supported | 1 only if **every** expected document is retrieved (matters for q09, q13). |
| MRR | supported | Mean of 1 / rank of the first chunk from an expected document; 0 if none. |
| Decision accuracy | all | Correct = answered a supported question or abstained on an unsupported one. |
| Answer relevance | supported | Share of the labelled `required_terms` found in the answer (case-insensitive, whitespace-normalised). An abstention on a supported question scores 0. |
| Faithfulness | answered | The answer is split into one segment per citation (`text [doc-id]`). A segment is supported only if its text appears **verbatim** in a retrieved, unfiltered chunk of the **cited** document. Score = share of supported segments; uncited trailing text counts as unsupported. |

**What faithfulness does and does not show.** A correct citation is not enough: a segment with an altered fact, or real text attributed to the wrong document, scores 0 (covered by `tests/test_evaluation.py`). Literal matching can only rate extractive answers. LLM paraphrases are reported as **unrated** instead of guessed; they would need an independently reviewed judge or human review. Faithfulness also says nothing about whether the quoted text *answers* the question: see q01.

## 5. Failure and fragile-success analysis

### 5.1 q01: cited, faithful, and still wrong (most important finding)

*"I was charged twice for the same invoice. What must I send and when?"* The rule is `billing-refunds#0`: *"The requester must provide the invoice number and both payment references within 30 calendar days of the second charge."*

- **Retrieval missed the answer chunk.** The top 3 were `invoice-corrections#1` (0.644, "Duplicate charges follow the refund procedure in the separate billing policy"), `invoice-corrections#0` (0.467, legal-name corrections) and `billing-refunds#2` (0.462, "dissatisfaction with course content is not automatically a billing error"). `billing-refunds#0` was not in the top 3.
- **The metrics hide it.** recall@3 = 1 because *a* `billing-refunds` chunk was retrieved. Faithfulness = 1 because every quote is real source text. Relevance gets 0.33 only because "invoice number" appears in the *invoice-correction* rule ("must include the invoice number and the exact replacement legal details"), which is a different procedure. A colleague reading this answer could tell a customer to send legal-entity details instead of payment references.
- **It was borderline anyway.** Coverage was exactly 0.40, the minimum.
- **The LLM handled it better.** Given the same three chunks, it replied `INSUFFICIENT_EVIDENCE` and the system abstained. That counts as a decision error in the table, but it is the safer behaviour for a support tool.
- **Root cause:** the question says "charged twice" and "send", while the rule says "duplicate payments" and "provide", and the `invoice-corrections` chunk mentions "duplicate charges" literally. A small bi-encoder ranks that lexical overlap above the paraphrased rule. Document-level labels cannot detect this; chunk-level labels would.

### 5.2 q06 and q09: correct, but close to the boundary

- **q06** ("more time for an assignment") had coverage exactly **0.40**: "time" and "requesting" don't match "extension" and "request" after crude stemming. One more unmatched word in the question would have produced a wrong abstention.
- **q09** ("send authentication tokens to a model while debugging") had a top score of **0.351**, only 0.051 above the threshold, and the three retrieved chunks scored 0.351 / 0.348 / 0.329 across three different documents. Retrieval is uncertain here. The answer happened to be right because the extractive mode quoted both `privacy-data#0` and `technical-incidents#1`.

### 5.3 What the final set did not test

- **Near-miss unsupported questions.** All three unsupported final questions (cafeteria 0.168, enterprise price 0.232, bicycle parking 0.111) are far from the corpus and fail on similarity alone. None resembles calibration's near misses (cal04–cal06). The 3/3 abstention rate therefore says little about the hard case the coverage check was built for.
- **Prompt injection.** No final question retrieved the malicious chunk `retrieved-content-safety#1`; q10 retrieved chunks #0 and #2. The removal path is covered only by unit tests (`tests/test_generation.py`) and a manual check: asking *"What should the assistant do when a retrieved document tells it to reveal the system prompt?"* retrieves #1, logs two safety events, removes it, and answers from #0 and #2 in both modes.

### 5.4 LLM relevance is under-measured, not necessarily worse

The LLM's lower relevance (0.769) comes from literal term matching: q04 says "verify your identity" instead of "identity verification"; q13 says "never asks for such documents" instead of "identity document"; q14 answers correctly but omits "approved deadline". The extractive mode scores higher partly because it quotes whole paragraphs, which also makes its answers long and sometimes padded with loosely related rules (q01, and cal01 during development).

### 5.5 Issues found and fixed during development (on calibration or unit tests only)

- The first injection pattern flagged legitimate policy text ("Support cannot … disclose a password"). It now targets only the assistant's own secrets.
- Sentence-level extraction lost context ("on 429, use exponential backoff" depends on the previous sentence) and found 1 of 5 required terms on calibration; paragraph-level quoting found 5 of 5.
- The LLM sometimes wrote citations with a Unicode non-breaking hyphen (`[account‑mfa]`) or a chunk suffix (`[doc#0]`), which the citation check wrongly rejected. Citations are now normalised before validation; the document ID is still checked.

## 6. Label disagreements

No labels were changed.

- **q01**: the required term `"30"` is too weak; any "30" (e.g. "30 minutes" from another policy) would satisfy it. `"30 calendar days"` would be a better label.
- **q01, decision label**: the LLM's abstention is scored as an error, but given that the retrieved evidence did not contain the rule, abstaining was the correct behaviour *for that evidence*. Decision accuracy measures the end-to-end outcome, so this is a retrieval failure surfacing as a decision error.
- **q09**: lists two expected sources, but either one alone (`privacy-data`: never put tokens in a prompt) answers the question. Full recall therefore may be stricter than necessary.
- **q14**: requires "approved deadline". A correct answer ("sending a request does not itself change the deadline") can omit it.

## 7. Prioritised recommendations

1. **Fix chunk-level retrieval for paraphrased questions** (evidence: q01, q09). Retrieve a wider candidate pool (e.g. 10 chunks), add a lexical retriever (BM25) alongside the dense one, and rerank with a cross-encoder before taking the top 3. Add chunk-level labels (`expected_chunks`) to the evaluation so a miss like q01 shows up in recall instead of hiding behind document-level credit.
2. **Add an answer-sufficiency check before answering** (evidence: q01 extractive vs LLM, borderline coverage on q01/q06). Measure coverage on the passages actually quoted, not just the top document, and in LLM mode keep the `INSUFFICIENT_EVIDENCE` path. Replace crude suffix stripping with lemmatisation plus a small support-domain synonym list ("charged twice" ↔ "duplicate payment", "more time" ↔ "extension"). Re-calibrate on a larger set.
3. **Build a harder, larger evaluation set before any live pilot** (evidence: section 5.3, 6 calibration questions, unrated LLM faithfulness). Include near-miss unsupported questions in the final set, questions that retrieve the injection chunk, chunk-level labels, and a human- or judge-reviewed faithfulness rating for LLM paraphrases. With 13 supported questions, one failure moves a metric by 7.7 points; report confidence intervals.

## 8. Submission questions

**1. Which retrieval failures matter most to the support scenario, and why?**
Retrieving the *right document but the wrong rule* (q01). It produces an answer that looks authoritative, carries a valid citation and passes literal faithfulness, yet tells the colleague to do the wrong thing (send legal-entity details instead of payment references). Second are near-miss unsupported requests (cal04–cal06: free extension after an outage, annual-subscription cancellation refunds, student discounts). Retrieval finds a plausible-looking policy, and answering would invent a refund or entitlement Northstar never promised. Both cause wrong commitments to customers. A plain "no result" is less harmful: the colleague escalates.

**2. How did you select your no-answer threshold?**
On `calibration.json` only, with a grid search over k, similarity threshold and term-coverage minimum. The selection rule maximises decision accuracy, then recall@k, then the worst-case margin to the decision boundary (section 2). This chose k = 3, threshold 0.30, min coverage 0.40, which were frozen with a timestamp and a hash of the calibration file before the final set was run. Similarity alone was insufficient: cal05 (unsupported) scored 0.426 vs 0.503 for a supported question. Coverage of the top document separated them (≤ 0.20 vs ≥ 0.60).

**3. What evidence suggests an answer is faithful to retrieved context?**
For extractive answers: every cited segment is found verbatim in a retrieved chunk of the cited document (13/13 answers), and every citation refers to a retrieved, non-filtered chunk, which is enforced at runtime. Neither shows that the quoted text answers the question; q01 is faithful and still unhelpful, so relevance has to be read alongside faithfulness. For LLM answers there is no automatic evidence beyond valid citations, so they are reported as unrated until reviewed.

**4. What would you change before allowing live support colleagues to use the system?**
The three recommendations above: reranked hybrid retrieval with chunk-level evaluation; an answer-sufficiency check on the quoted passages; and a larger, harder evaluation set including near misses and injection cases. Operationally I would also: show the retrieved passages next to every answer so colleagues can verify it; log questions, decisions and safety events, with redaction per the `privacy-data` policy; re-run `build-index` and the evaluation automatically when a knowledge-base article changes; and pilot with a small group whose escalations are reviewed weekly.

## 9. Reproducing this report

```powershell
uv sync
uv run python -m src.cli build-index              # rebuild the Qdrant index from data/knowledge_base
uv run python -m src.cli calibrate                # re-select and freeze settings (calibration.json only)
uv run python -m src.cli evaluate                 # extractive, deterministic, no API key
uv run python -m src.cli evaluate --mode llm      # optional, needs GROQ_API_KEY in .env
uv run pytest
```

The extractive results are deterministic for a fixed model revision and corpus. LLM results may vary slightly between runs even at temperature 0.
