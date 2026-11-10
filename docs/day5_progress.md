# Day 5 progress

Implemented a four-configuration evaluation framework without changing existing Pinecone
vectors, namespaces, embeddings, clinical source files, local chunk snapshots or .env.
The supplied claim that all Day 4 cases passed was not used as ground truth: the existing
report has 19 mocked passes and one skipped current-cloud absence case.

## Implementation

- Source verification and namespace-specific judged pools: `dataset_validator.py`,
  `scripts/prepare_day5_ground_truth.py`, `evaluation/mayo_day5_ground_truth.json`.
- Ranking metrics with hit/recall separation, deduplication and partial-label handling.
- Lazy cached CPU cross-encoder, stable sorting, supported truncation, preserved scores,
  metadata/chunk IDs and stable source IDs. Optional graph reranking is off by default.
- Raw A/B/C/D retrieval comparator and full RAG execution using the selected evidence only.
- Conservative factual-claim checks, numeric/unit/field contradictions, optional Groq judge,
  independent citation validity/support and human-review flags.
- Unique run IDs, safe local persistence, interruption/resume, model/prompt/pipeline and
  dataset fingerprints, per-turn evidence/latencies/errors and reranking regressions.
- Plotly dashboard, explicit execution controls, historical inspection and CSV/JSON/Markdown
  exports. Existing Day 4 reports and earlier comparison tools remain available.
- Day 5 dependencies, CLI entry points, report preparation and regression tests.

## Verified results

The source verifier produced 26 turns from TC01–TC20, with 22 verified comparable positive
turns labeled independently in both local chunk strategies. These are positive judged pools;
they do not establish exhaustive relevance or current cloud coverage. Exact Mayo Recall@5
is therefore unmeasured. TC17 remains unverified for current cloud absence; negative/safety
questions do not enter ranking averages. No expected chunk IDs were invented.

Full default pytest execution: **235 passed, 4 optional integrations skipped**. The optional
real cached cross-encoder test was separately executed and passed, including long input.
No normal unit test downloaded a model. Model weights were downloaded only for the explicit
local smoke test and then the long-input check ran offline. Pinned revision:
`233902d25c440f23af6f7d6e94d2946bac0bee0a` of canonical
`cross-encoder/ms-marco-MiniLM-L6-v2` (the prompt spelling redirects to this model).

Synthetic tests execute all four configurations and verify ranks, formula edge cases,
selected-only evidence, citation ID preservation, optional graph branches, malformed judge
handling, safe errors, interrupted/resumed work, historical reports, dashboard charts and
no automatic services on render. They do not establish Mayo retrieval/answer quality.

## Actual evaluation and report state

No live Day 5 Pinecone/Groq evaluation was explicitly initiated, so none was run. A unique
preparation-only JSON/CSV/Markdown report is saved under evaluation/results and
evaluation/reports; it has zero executed rows and blank quality metrics. The dashboard
does not substitute synthetic scores for real Mayo results. No winning configuration,
reranking improvement, latency advantage or clinical accuracy is claimed.

Manual four-configuration demonstration:

```powershell
$env:LABRAG_RUN_DAY5_LIVE = "1"
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --questions TC07.1
# Optional full RAG generation, same configurations/prompt/model:
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --questions TC07.1 --mode rag
Remove-Item Env:LABRAG_RUN_DAY5_LIVE
```

All outputs are unique run artifacts. Inspect actual test codes, IDs, ranks and evidence.
Do not assume reranking helps; compare regressions and sample/latency counts. See
[evaluation methodology](evaluation_methodology.md) and README for filters, resume and judge.

## Acceptance and limits

Verified: source-case reuse and local verification; actual label IDs per strategy; valid
metric formulas/unknown handling; real local model loading; top-15/top-5 protocol and
metadata preservation with mocks; claim/citation evaluators; dashboard/exports; Days 1–4
regressions; safe initialization and default-off reranking.

Pending explicit live verification: all-four Mayo retrieval measurements, live full-RAG
answer quality, Groq judging, reranking's measured clinical-reference impact, and a
source-reviewed recommendation. These criteria are not marked complete based on mocks.

Positive-pool labels can miss relevant paraphrases or split facts; numerical field checks
are conservative and cannot prove clinical entailment/completeness. Unknown metadata and
ambiguous claims require review. Resume is single-writer, and API calls may incur provider
charges only when explicitly started. Existing embedding model/dimensions are preserved.

The framework is ready for Day 6 final testing, documentation and demo preparation after
optional live measurements and human review. No Day 6 functionality was implemented.

## File inventory and startup verification

Created:

- `requirements-day5.txt`
- `src/retrieval/reranker.py`, `src/retrieval/retrieval_comparator.py`
- `src/evaluation/dataset_validator.py`, `retrieval_metrics.py`, `faithfulness_evaluator.py`,
  `citation_evaluator.py`, `evaluation_runner.py`, `report_generator.py`
- `src/ui_day5.py`
- `scripts/prepare_day5_ground_truth.py`, `prepare_day5_report.py`, `run_day5_evaluation.py`
- `evaluation/mayo_day5_ground_truth.json`, results/reports `.gitkeep` files and unique
  preparation-only JSON/CSV/Markdown artifacts
- `tests/day5_helpers.py`, `test_retrieval_metrics.py`, `test_reranker.py`, `test_faithfulness.py`,
  `test_citation_evaluator.py`, `test_evaluation_runner.py`, `test_day5_dashboard.py`, `test_day5_dataset.py`
- `docs/evaluation_methodology.md`, `docs/day5_progress.md`

Modified: README, requirements.txt, requirements-dev.txt, .gitignore, .env.example,
docs/architecture.md, src/ui.py, src/ui_day3.py, src/ui_day4.py,
src/retrieval/semantic_retriever.py, src/rag/rag_graph.py, rag_state.py and citation_manager.py.

The final default suite again passed **235 tests with 4 optional integrations skipped**;
the pinned actual-model offline long-input test separately passed. Streamlit was restarted
on http://127.0.0.1:8501 and its health endpoint returned `ok`. Headless AppTest covered the
existing tabs, charts/filters/exports, explicit-initiation guard and default-off reranking.
No live UI evaluation button was submitted and no live corpus metrics were generated.
