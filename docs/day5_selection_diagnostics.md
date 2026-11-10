# Answer-selection and quote-validation observability

## Scope and behavior preservation

No changes to retrieval depth, ranking, evidence selection, generation prompts,
ground-truth labels, Pinecone data or exact-quote validation rules. The diagnostic
helper calls the existing `validate_selection()` decision. A selection containing
an invalid excerpt still fails as a whole. Existing whitespace restoration remains
unchanged. Malformed schema/parser responses still fail execution; they now carry
a sanitized diagnostic category rather than a provider payload.

## Recorded fields

New graph state and evaluation records contain `selection_diagnostics`:

- `structured_response`, `generation_executed`, `excerpt_count`, `source_ids`.
- `stage`: evidence gate, policy gate, structured generation or quote validation.
- `quote_validation`: passed, rejected or not attempted.
- `failure_category` and, for structured selections, `failure_categories`.
- Per-excerpt safe source identifier, quote retention flag, SHA-256 and failure.
- `evidence_gate_passed` when the graph reached final validation.
- `final_answer_status`: answered, abstained or failed.

The evaluator also records `raw_selection_diagnostics` before the existing
whitespace restoration. The graph diagnostic observes the restored selection.
This allows accepted whitespace-only repair to be distinguished from rejection
without changing the quote rules. Historical reports are never backfilled.

Failure categories distinguish No excerpts selected, Unknown source identifier,
Quote does not match the cited passage, Invalid structured generation response,
Evidence insufficient and Other validation failure. Safety policy is separate.
Malformed outputs do not expose unknown excerpt counts, source IDs or raw text as
if they were valid structured content.

## Privacy boundaries

No raw provider responses, exception bodies, prompts, free-form model notes or
request payloads are added to diagnostics. Invalid identifiers are redacted.
Unknown/nonmatching quotes are withheld and represented by hashes. Exact selected
quotes are retained only when grounded in explicitly synthetic evidence or a
declared public catalog source on the allowlisted Mayo hosts. Whitespace-only
differences can be retained when all normalized text is source-grounded. Private
or provenance-unknown source quotes are withheld even when validation succeeds.
This depends on truthful source provenance; it is not a general-purpose PHI
detector. Use only laboratory catalog documents, never patient information.
Existing generation-evidence fields are not broadened by this change.

## Abstention reporting

An evidence-insufficient abstention remains distinct from post-generation
abstention (no excerpts) and post-generation rejection (invalid quote/source).
A lexical evidence-gate pass alone no longer causes an automatic “Incorrect
refusal despite sufficient evidence” classification. Missing diagnostics leave
the cause unmeasured. Correctness and faithfulness remain unmeasured for empty
answers; existing completeness behavior is unchanged.

New RAG result tables display validation stage and selection failure reason. The
record inspector shows complete safe diagnostics. JSON/CSV exports include both
diagnostic fields. The model's original structured selection is not blindly
dumped: only permitted source-grounded excerpts and sanitized metadata are retained.

## Manual Streamlit review

1. Refresh Evaluation. Open an older saved pilot run: its diagnostics should show
   “Historical — not recorded”/“Not recorded in this historical result”, not an
   inferred failure reason. Original historical classifications remain unchanged.
2. After separate approval, explicitly start a new v3 RAG pilot evaluation. No live
   evaluation was initiated for this implementation.
3. Inspect TC12.1 A/C. An empty structured selection should show No excerpts selected;
   a nonmatching quote or unknown ID should show the corresponding rejection.
4. Compare raw and restored diagnostics for whitespace-only normalization. Successful
   exact quotes must retain their original accepted source references.
5. A pre-generation evidence failure must show Evidence insufficient and generation
   not executed. A malformed response must be Failed with a sanitized structured
   generation diagnostic, not a fabricated generic abstention cause.

## Files

Updated `src/rag/citation_manager.py`, `src/rag/rag_graph.py`,
`src/rag/rag_state.py`, `src/rag/answer_generator.py`, `src/llm/groq_service.py`,
`src/evaluation/evaluation_runner.py`, `src/evaluation/answer_quality.py`,
`src/evaluation/report_generator.py`, `src/ui_day5.py` and the existing abstention
expectation in `tests/test_generated_answer_quality.py`. Added
`tests/test_selection_diagnostics.py` and this document.

Old TC12.1 selections cannot be reconstructed from absent logs. A new explicitly
authorized evaluation is needed to diagnose its future generation/validation
outcomes. The lexical evidence gate still does not prove exhaustive completeness.

## Validation

Final command: `python -m pytest -q -m "not integration"` — **308 passed,
4 integration tests deselected, 0 failures**. Tests cover empty selections, unknown
IDs, nonmatching quotes, exact matches, evidence abstention without generation,
schema/parser failures, sensitive-payload withholding, evaluation persistence,
historical immutability and the revised abstention classification. No live calls,
new evaluations, model downloads or historical report rewrites were performed.
