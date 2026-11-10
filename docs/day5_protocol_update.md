# Day 5 retrieval protocols

Raw Similarity Retrieval (default, `raw`) preserves A/B top-5 and C/D top-15
reranked to 5. Test-Context Retrieval (`test_context`) calls the Explorer's
`search_with_context` for either fixed or section chunks, then optionally
reranks its final candidate pool. The candidate count is the context pipeline's
output cap, not a cap on its internal identity/page-neighbor searches.
Context mode includes identity enrichment and test/facet filtering before reranking.
Its latency includes all these searches; embedding/search sub-timings are unavailable
when the underlying context pipeline does not expose them.

Select the protocol in Evaluation → Configure Day 5 evaluation. CLI callers can
use `--protocol test_context`. Live execution still requires explicit opt-in.
No live evaluation was run for this change. Indexes and stored vectors are untouched.

Verification: `python -m pytest -q -m "not integration"` completed with
245 passed and 4 integration tests deselected. Targeted protocol, runner,
dashboard, metric, reranker and small-chunk tests previously completed with
40 passed and 2 optional tests skipped. All provider responses and reranker
predictions in the new tests were injected fixtures; no live quality results
or production improvements are asserted.

New JSON parameters/records, CSV, Markdown and resume signatures record the
protocol. Reports without a protocol are displayed as raw. Historical files are
not rewritten. Old signatures will not resume under changed pipeline code: start
a new run. Compare quality and reranking effects within the same protocol and
dataset fingerprint; context and raw are different experiments.

Retrieval-only answers remain empty in machine-readable JSON (no LLM was called).
The UI and CSV/Markdown display “Not generated — retrieval-only mode”.
Candidate misses, losses from candidate pool to final ranking, and unresolved
identity metadata are independent diagnostics, not proof of causal attribution.

## TC07.1 label review — proposed revision, not applied

Historical ground truth is preserved. It currently labels both requested volume
(0.5 mL) and minimum volume (0.25 mL) as relevant, although TC07.1 asks only for
minimum volume. Do not interpret the historical metric as minimum-only accuracy.

For a future separately versioned dataset, retain only the expected fact
`Specimen Minimum Volume: 0.25 mL` and these verified minimum-volume chunk IDs:

- Fixed: `b9378669a52f1c784654ca2ab4118035d59949a366b6382761d0da0a6b878a42`
- Section: `c3a5974f4be4ed95b4c9536adca0c0dcdb79a8e0800e5f23d1b6105eb072aedc`

Exclude the 0.5 mL-only chunks from that question's positive pool. Revalidate
the source/snapshot references and recompute its dataset fingerprint. Save to a
new dataset filename; never overwrite the historical dataset or its reports.

## Manual validation (live execution requires approval)

1. Open Evaluation; confirm Raw Similarity Retrieval is the default.
2. Inspect historical runs: absent protocol displays raw and answers show the
   retrieval-only explanation. Opening the page must not initiate retrieval.
3. Select TC07.1, Test-Context Retrieval, A/B/C/D, top-k 5 and candidates 15.
4. After approval, explicitly initiate a new retrieval-only run. Inspect CA,
   Calcium, Total, Serum and the 0.25 mL minimum-volume evidence for both strategies.
5. Check saved protocol, unique run ID, metadata preservation, timings, and
   separate failure diagnostics. No outcome or quality improvement is guaranteed
   by offline tests; cloud coverage may differ from local snapshots.
