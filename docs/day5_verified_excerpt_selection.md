# Verified excerpt selection

New Groq answer generation uses `verified_excerpt_ids_v1`. The application builds
a menu of contiguous spans within each supplied retrieved passage. Spans cover
the original text, are at most 1,200 characters, and prefer paragraph, line or
word boundaries. They retain original whitespace. This is an answer-selection
representation, not a change to retrieval, indexed chunks or ranking.

Groq returns only `selected_excerpt_ids` and optional missing/conflicting evidence
flags. Its strict structured schema rejects extra quote, source or offset fields.
Every identifier must exist in the per-request menu. The application verifies
the menu against original source text and deterministic boundaries, then resolves
IDs to original source slices. Unknown IDs reject the response; multiple excerpts
remain separate. No missing words are repaired, sources reassigned or cross-chunk
quotes assembled. The existing all-or-nothing citation validator still runs.

Historical quote selections remain readable with protocol `exact_quote_v1`.
Injected legacy generators continue through existing exact validation and
whitespace restoration. Production Groq requests with a menu do not fall back to
free-form quote responses. Selection diagnostics record the protocol; the updated
system prompt changes the existing evaluation prompt fingerprint/resume signature.
Historical reports and datasets are unchanged.

## Limitations

The model can still propose invalid IDs, but none can be accepted. It can choose
irrelevant or incomplete valid spans or abstain. Exact grounding alone does not
guarantee answer completeness. Fixed span boundaries may require multiple IDs,
and spans may include extra context. The menu increases prompt size. This offline
change does not establish improved live TC12.1 performance; a separately authorized
pilot is required. Compare generated-answer quality with attention to the changed
selection protocol and prompt fingerprint.

## Manual pilot review (requires separate live-run authorization)

1. Restart/refresh Streamlit and open Evaluation > Day 5.
2. Select the current compatible dataset, TC12.1, RAG mode, Test-Context Retrieval,
   and A/B/C/D with unchanged Top K and candidate count.
3. Start a new run rather than resuming an older prompt fingerprint.
4. Inspect each configuration's evidence and answer details. Confirm
   `selection_protocol=verified_excerpt_ids_v1`, exact source text, original
   source IDs, passed quote validation or explicit abstention/rejection.
5. Review CA/CAI identity, collection facts, completeness and extra context.
   Keep the new report separate from historical benchmarks.

No live evaluation is started by this implementation.
