# Day 5 coverage and question-level reporting

The dashboard reads the selected saved run, independently of the dataset currently
selected for a future evaluation. Coverage cards count unique selected, scored,
excluded, incompatible and failed questions. Coverage is scored / selected.
For mixed-status configurations, status counts can overlap; scored means scored
in at least one configuration. A separate per-configuration table shows exact
record counts and coverage. A warning identifies different scored sets. Interrupted
runs retain their recorded selected denominator. If old reports omit the selection,
only observed IDs can be inferred and the UI explains that limitation.

Exclusions show precise reasons and fixed/section positive label counts. Historical
details are recovered only from a fingerprint-verified matching dataset. Missing
facts remain unknown, not false or zero. New runs snapshot eligibility metadata;
existing datasets and historical reports are not rewritten. Excluded retrieval is
displayed as “Not executed — excluded”. Exclusions are not retrieval failures.

The question comparison has question, configuration and status filters, status,
Hit Rate@K, MRR@K, Precision, first relevant rank, retrieved codes and exclusion
details. Aggregate charts remain full-run summaries. The filtered reciprocal-rank
heatmap displays scores 0–1; null cells carry explicit Excluded/Incompatible/Failed
or other unscored status labels on a gray background. Genuine zero remains numeric
zero. No metric or retrieval algorithm changed.

Precision divides known-positive hits by actual unique returned IDs up to K, not
always K. Hit Rate averages only scored questions. Excluded questions are neither
successes nor failures. Judged-pool recall is not exhaustive recall.

## Manual Streamlit review

1. Refresh Evaluation and select the saved 26-question run. Confirm 26 selected,
   14 scored, 12 excluded, zero incompatible/failed and 53.85% coverage; the table
   should show 14 scored/12 excluded for each A/B/C/D configuration.
2. Inspect exclusion details: eight missing-section-label questions plus the
   unverified, refusal, missing-information and mock-only cases.
3. Filter questions to TC07.1 and configurations A/B/C/D; inspect ranks and heatmap.
4. Filter Excluded. Confirm no numeric ranking scores, status-labeled gray heatmap
   cells and “Not executed — excluded”. Genuine zero scores should remain 0.00.
5. Switch the form's dataset while retaining the historical run; coverage must not
   change. Select older saved runs to verify unknown metadata stays unknown.
6. Opening/filtering reports must not initiate an evaluation or external API call.

## Scope

Changed `src/ui_day5.py`, `src/evaluation/evaluation_runner.py` (eligibility snapshot
only), `src/evaluation/report_generator.py` (derived coverage in exports).
Added `src/evaluation/dashboard_reporting.py`, `tests/test_dashboard_reporting.py`
and this document. Existing stored metrics, labels, retrieval/reranking, Pinecone
data and historical report files are preserved. Saved dataset/capture availability
limits historical detail; no cloud audit or new live evaluation is performed.

Validation: `python -m pytest -q -m "not integration"` completed with
**266 passed, 4 integration tests deselected**. The earlier focused reporting,
dashboard, protocol, runner and metric suite passed 38 tests with one integration
test deselected. Tests check saved-run denominators, exclusions, mismatched
datasets, missing historical metadata, null versus zero heatmap cells, filters,
report immutability and existing metric behavior. No external API calls were made.
