# Day 5 generated-answer evaluation

## Reused components and scope

The existing RAG graph, Groq quote-selection generator, A/B/C/D configurations,
raw/test-context retrieval, reranker, faithfulness classifier, citation evaluator,
claim-level optional judge, latency traces and historical exports are reused.
No retrieval, chunking, ranking, namespace, ground-truth or Pinecone data changes
were made. New runs have unique IDs and changed evaluation-code fingerprints;
old report files remain untouched. Opening the dashboard never starts evaluation.

## Post-generation checks and denominators

The generator receives only the resolved question and actual retrieved evidence.
Its exact input evidence is recorded in `generation_evidence`. Ground-truth facts
and labels are not included in that prompt. Recorded selected chunks and scoped
`source_views` are retained separately because generation can use a test-region
fragment of an original retrieved chunk. Citations are checked against recorded
chunk/document identity and text containment.

- Correctness uses verified-reference claim checks. Score = Supported /
  (Supported + Unsupported + Contradicted); Unverifiable claims are counted but
  excluded. Missing references produce null scores, not assumed correctness.
  Numeric field checks distinguish requested/minimum volume, units, frozen and
  refrigerated durations. Ambiguous paraphrases need review.
- Faithfulness reuses existing retrieved-evidence classifications. The new answer
  metric uses Supported / evaluable claims and explicitly reports denominator and
  Unverifiable count. Legacy `supported_claim_rate` is preserved with its original
  all-claim denominator; it must not be confused with this new metric.
- Completeness is present / measurable verified required facts. Exact fact matching
  normalizes case/whitespace; semicolon clauses can be combined only within the same
  test attribution. Multiple-test comparisons require source-backed ownership.
  Minimum-only questions do not require separate requested-volume facts. Unknown
  ownership is unmeasured. Other rubrics remain bounded by the curated expected
  facts and may need human refinement; no extra clinical facts are invented.
- Citation validity remains separate from support. Fabricated IDs, incorrect
  metadata and sources outside recorded retrieval are invalid. New verified validity
  excludes unresolved-metadata citations from its denominator and reports their
  count. The legacy structural validity rate remains for historical compatibility.
  Citation support retains per-claim Supported/Unsupported/Contradicted/Unverifiable
  details, missing-reference counts and its original strict support fraction.
- Abstention records appropriate abstention, unsupported answer, incorrect refusal
  despite sufficient evidence, appropriate evidence-grounded answer, or unmeasured
  behavior. It uses the existing evidence gate, not an oracle. An appropriate refusal
  after a retrieval miss does not become an answer-quality success or perfect
  faithfulness. Completeness can be zero even when abstention is appropriate.

Answer-quality chart scores are means over measured questions. Summary fields record
measured-question counts; per-row details retain claim/fact denominators. Unmeasured
scores remain null and are omitted from charts. These are conservative deterministic
reference checks, not clinical correctness certification.

## Pilot and special categories

The pilot selects TC07.1 (CA minimum 0.25 mL, distinct from requested 0.5 mL),
TC04.1 (CPR fasting/biotin preparation) and TC12.1 (CA/CAI collection comparison).
Existing v3 verified facts supply the references; no labels were edited.

TC11.1 may receive post-generation reference checks when evidence is verified, but
never enters positive ranking averages without shared labels. A refusal does not
prove absence from the full catalog. TC17.1 remains unscorable without verified
reference evidence. TC18.1 is a separate safety/refusal category and bypasses
retrieval and generation. TC19.1 checks abstention about an unsupported exact Celsius
temperature; no numeric value is manufactured. TC20.1 remains excluded from live
Day 5 execution. Use the existing Day 4 mocked prompt-injection validation for that
case; its instructions are never executed or indexed.

## Supplemental judge

Disabled by default. Enabling it in explicitly started live RAG evaluation may make
both existing claim-level and new answer-rubric Groq calls. The new structured rubric
separately scores correctness, faithfulness, completeness and citation support,
requiring explanations and actual source IDs for measured scores. Missing reference
facts prohibit measured correctness/completeness. Malformed, missing-dimension,
out-of-range or fabricated-evidence responses return a sanitized error without
scores. Model and prompt version `day5-answer-quality-v1` are recorded. No judge
assessment replaces deterministic results or constitutes verified clinical truth.

## Reports and dashboard

JSON/CSV retain actual generation evidence, answers, source views, citations, verified
required facts, correctness/completeness/abstention results, optional judge output,
eligibility category and latency. Generation and RAG end-to-end timings come from
the existing graph; `answer_evaluation_seconds` measures post-generation validation
and optional judging separately. Provider token usage remains unmeasured when absent.
Retrieval summaries and historical metrics are preserved. New answer-quality tables
and charts are displayed separately; detailed JSON exposes unsupported claims,
missing facts, citation failures and judge assessments.

## Exact pilot steps — live execution requires explicit initiation

1. Refresh localhost and open Evaluation.
2. Select `mayo_day5_v3_overlap15_ground_truth.json`.
3. Expand Configure Day 5 evaluation. Select the **Generated-answer pilot
   (TC07.1, TC04.1, TC12.1)** preset.
4. Confirm A/B/C/D, the three question IDs, mode `rag`, protocol `test_context`,
   final top-k 5, initial candidates 15. Leave optional judge unchecked initially.
5. Confirm only public catalog questions are used; do not enter patient data or PHI.
   To authorize live execution, select the explicit initiation checkbox and click
   Start Day 5 Evaluation. This implementation did not perform that step.
6. Select the newly saved run. Inspect generated answers, actual generation evidence,
   attribution, required facts, score denominators and generation/end-to-end timings.
   A/B/C/D remain different complete configurations, not a pure reranker ablation.
7. Export new JSON/CSV/Markdown artifacts. Historical retrieval-only runs keep their
   original results and display answer quality as unavailable rather than zero.

CLI alternative (an explicit live opt-in, not part of automated tests):

```powershell
$env:LABRAG_RUN_DAY5_LIVE='1'
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --dataset evaluation/mayo_day5_v3_overlap15_ground_truth.json --questions TC07.1 TC04.1 TC12.1 --mode rag --protocol test_context --top-k 5 --candidates 15
Remove-Item Env:LABRAG_RUN_DAY5_LIVE
```

## Files

Added `src/evaluation/answer_quality.py`, `src/evaluation/answer_judge.py`,
`tests/test_generated_answer_quality.py` and this document. Updated
`src/evaluation/evaluation_runner.py`, `src/evaluation/citation_evaluator.py`,
`src/evaluation/report_generator.py` and `src/ui_day5.py`.

No live quality improvement is claimed. Exact matching can miss valid paraphrases;
required-fact pools are incomplete; lexical evidence sufficiency and attribution
need human review. Existing refusal rules and injection checks provide safeguards,
not validation for clinical decision support.

## Offline validation results

The complete offline suite (`python -m pytest -q -m "not integration"`) had a clean
run with **293 passed, 4 integration tests deselected**. After final export and
answer-success guard additions, the repeated full suite had 292 passes and one
timeout in the existing chunk-inspection UI test. That test plus all generated-answer
tests then passed on isolated retry (**13 passed**). The focused export/dashboard/
runner suite also passed **32 tests, 1 integration test deselected**. No retrieval,
reranker or chunk-inspection implementation was changed. The timeout is recorded
rather than treated as a quality result. No live Pinecone/Groq calls were made.
