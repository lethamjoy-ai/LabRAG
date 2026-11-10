# Grounded conversational answers

Two explicit answer-generation modes are available:

- `extractive_v1`: the existing source-quote answer, unchanged; evaluation default.
- `grounded_conversational_v1`: typed catalog facts rendered with constrained
  natural-language templates; chat default.

## Files

Created `src/rag/conversational_answer.py` and its regression tests in
`tests/test_conversational_answers.py`. Updated `src/rag/rag_graph.py` and
`src/rag/rag_state.py` for the post-validation stage; `src/ui_day4.py` for chat
style/citations/audit display; `src/ui_day5.py`, `src/evaluation/evaluation_runner.py`
and `src/evaluation/report_generator.py` for explicit signed modes and exports.
`tests/test_day4_ui.py` now selects legacy mode explicitly when verifying the
original extractive chat behavior. This document describes the new path.

The new mode does **not** ask a model to rewrite clinical prose freely. The
currently selected Groq/OpenAI service still selects verified excerpt IDs as
before. A separate LangGraph node runs only after the existing all-or-nothing
exact-quote validator passes. It adds no model calls, tokens or provider fallback.

## Evidence and final-answer validation

`src/rag/conversational_answer.py` recognizes a deliberately bounded set of Mayo
layouts: specimen types; preferred/acceptable collection tubes; collection versus
submission containers; requested versus minimum volumes; canonical fasting and
biotin instructions; specimen/temperature/stability tables; and recognized
collection/processing instructions. Table-carried specimen identities and special
container conditions remain attached to their rows. Celsius spelling, quantities,
units, qualifiers and durations are preserved. No unit conversions are performed.

Each typed fact carries a fact ID, stable per-turn source ID, test identity, field
role, source offsets and values/quantities. Offsets refer to the scoped retrieved
source text saved with the turn, not PDF coordinates. A fact cannot cross a test
region or source boundary. Ambiguous Preferred labels without a collection heading
are not treated as tube instructions. Unrecognized or conditional preparation
text is not simplified into unconditional requirements.

Before display, fact claims are re-derived from the exact verified source spans
and compared by typed role, values, units, test identity and provenance. Only a
closed set of templates can render accepted claims. A grammar integrity check
then rejects any changed sentence or additional unsupported prose. This final
text comparison is not the evidence-support test: the support decision uses the
independently reconstructed typed facts and mappings. Arbitrary model paraphrases
are never accepted merely because their numbers match.

Explicitly requested unsupported fields, competing field values, missing comparison
identities, unsupported units/conditions or failed quote validation cause abstention.
Processing-and-storage queries require both. Broad answers identify that only the
verified fields shown are summarized; supporting evidence may contain more detail.
Source identity/evidence gates and patient-treatment policy remain unchanged.

## Chat presentation and audit

Choose Answer generation mode in Advanced Chat Settings. The new conversational
mode shows readable prose without raw `[S…]` markers, followed by concise verified
citations. The internal answer retains its source mappings for evaluation.
View supporting evidence contains typed claims, the original validated extractive
answer and stable citation references. View Retrieval Details remains available,
collapsed, with complete retrieved passages and metadata. Copy answer copies the
readable version. No raw JSON or long PDF quotation is visible by default.

The integrated Laboratory question / Sample Questions / Send row, provider/model
selection, conversation history and follow-up context remain unchanged. A sample
only fills the draft; Send remains the sole request event. Existing quota/error
handling applies to the original selection call; synthesis cannot trigger another
paid request or hide an upstream failure.

## Evaluation versions and historical compatibility

Evaluation defaults to extractive mode. A new explicit selector enables
conversational mode for new RAG runs. Generation mode is stored in signed
parameters, run metadata, records, JSON/CSV/Markdown reports and the dashboard.
Resume rejects a different mode. The synthesis implementation is also included
in the pipeline fingerprint. No ground truth, retrieval, ranking or existing
answer-quality calculation is changed.

Historical reports remain byte-for-byte unchanged and display as legacy extractive
mode when version metadata is absent. Old checkpoints lacking the newly signed
mode are not silently migrated; use a separate new run for conversational scoring.
Do not edit signatures to bypass this check. Old lexical completeness/faithfulness
checks can undercount controlled paraphrases; their limitations remain visible,
and typed synthesis validation is separate evidence, not a replacement metric.
Never compare answer scores across generation modes without identifying the mode.

## Manual review

1. With no active evaluation, load the updated app and leave Grounded conversational
   selected in Advanced Chat Settings. Verify the input/sample/Send row is intact.
2. When ready to authorize a live question, ask an indexed BUN specimen question;
   review the direct answer, requested volume, collection/submission distinctions
   and citations. Open supporting evidence to inspect exact source mappings.
3. Ask Calcium minimum volume, C-Peptide fasting/biotin and processing/storage
   questions, then a two-test collection comparison. Check units and qualifiers.
4. Verify unsupported clinical advice, absent facts or unresolved layouts abstain.
5. Switch provider explicitly if desired; only the selection call uses that provider.
6. For evaluation, create a new RAG run with an explicit generation mode. Keep A/B/C/D
   and retrieval settings fixed for a deliberate mode comparison. Do not resume or
   overwrite a historical benchmark to test the new style.

## Limits

This is conservative template-based synthesis, not general medical entailment or
an unconstrained writing model. Unsupported layouts may abstain even when a human
can interpret the PDF. New catalog layouts and instruction variants require
separately reviewed rules/tests. Structured extraction does not validate the
clinical correctness or currency of the source document itself. Verify current
official instructions before clinical use. Full interpretation/reference ranges,
orderability/billing questions, patient-specific decisions and free-form medical
advice are not supported by the conversational fact grammar. Extractive mode
remains available under the unchanged existing evidence/quote rules.

Offline tests cover numerical/role/unit tampering, preparation exceptions, specimen
distinctions, storage, comparisons, exact quotes, audit citations, providers,
UI samples and versioned evaluation. No live requests or evaluations are initiated.
