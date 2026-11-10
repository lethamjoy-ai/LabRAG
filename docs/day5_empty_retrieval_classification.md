# Day 5 empty-result classification

Compatibility now has explicit states `compatible`, `mismatched` and `unverified`.
The existing boolean remains available for callers. An empty final candidate pool
alone proves neither compatibility nor incompatibility.

Context retrieval records available configuration metadata from expanded documents
before test scoping. Evidence includes actual chunk ID, document ID, source-content
hash, strategy, size and overlap, with no inferred values. Independent evidence is
accepted only for the dataset's source document, with nonempty chunk/source IDs.
Observed configuration conflicts still block scoring; incomplete metadata remains
unverified. This checks observed configuration, not complete index content coverage.

With independently compatible evidence, empty final results are Completed retrieval
misses and receive zero Hit Rate, MRR and Precision. Judged-pool recall is zero for
verified non-exhaustive positive pools; exhaustive pools use exact recall. No
evidence yields an Unverified unscored record. Genuine mismatches remain Incompatible
and unscored. Excluded and failed records remain distinct. Synthetic test fixtures
retain their explicitly marked injected compatibility behavior.

TC06.1 A still uses a depth of five; C still uses fifteen. The classification fix
does not make A retrieve CRP or improve its score. A future empty result will score
as a miss only when newly recorded intermediate metadata establishes compatibility.
The old saved A record has only intermediate IDs, so it remains unchanged; no
historical metadata or scores are fabricated.

The dashboard shows retrieval outcome separately from execution status, Unverified
counts/filtering, and a common-question intersection table whenever configuration
scored sets differ. Common-set coverage and IDs are explicit. Original aggregate
charts/metrics remain unchanged. Markdown exports also include this separate view.
Changed code fingerprints require new runs; no live run is initiated automatically.

Files changed: `src/evaluation/chunk_compatibility.py`,
`src/retrieval/semantic_retriever.py` (diagnostic metadata only),
`src/evaluation/evaluation_runner.py`, `src/evaluation/dashboard_reporting.py`,
`src/evaluation/report_generator.py`, `src/ui_day5.py`,
`scripts/run_day5_evaluation.py`, `tests/test_dashboard_reporting.py`.
Added `tests/test_empty_retrieval_classification.py` and this document. The existing
comparator already carries the full diagnostics, so no comparator change is needed.
No ground-truth dataset, retrieval/reranking algorithm, index or historical report
is changed.

## Manual validation

1. Refresh Evaluation and select the saved TC06.1 run. Its A status must remain
   historically Incompatible; original 20/21 aggregates must stay unchanged.
2. Inspect the separately labeled common-question table: 20 common IDs, 76.92%
   selected-question coverage. TC06.1 is not in the intersection.
3. Confirm Unverified is available as a status filter and the question table shows
   retrieval outcome separately. Empty compatible misses are numeric zero on the
   heatmap; unverified records are nonnumeric status-labeled cells.
4. Only after separate approval, start a new v3 retrieval evaluation. Inspect A's
   TC06.1 configuration evidence. A compatible empty result should now score zero,
   not be removed from the denominator. A genuine mismatch must remain unscored.

No rerun is needed to view common-set historical aggregates. A new approved run is
needed to capture intermediate metadata for corrected complete benchmark scoring.

## Offline validation

Final command: `python -m pytest -q -m "not integration"` — **281 passed,
4 integration tests deselected, 0 failures**. New regressions cover compatible
empty misses, partial-pool zero recall, missing/foreign/incomplete evidence,
true mismatches, unchanged A=5/C=15 depths, retained positive metadata, historical
immutability, common-set denominators and Unverified dashboard filtering.
The earlier focused run had 46 passes and one whole-app dashboard timeout; that
test and the new dashboard test passed in the full run. No provider requests,
model downloads or live evaluations were initiated.
