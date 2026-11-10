# Day 5 overlap-15 ground truth

New dataset: `evaluation/mayo_day5_v2_overlap15_ground_truth.json`.
Version: `v2-overlap15-saved-evidence`. The historical overlap-10 dataset and
all historical evaluation reports are unchanged.

## Evidence and reconciliation

`scripts/reconcile_day5_ground_truth.py` reads historical, fingerprint-verified
local JSONL snapshots and saved live evaluation captures. It makes no external
requests and refuses to overwrite its output. Each accepted ID is copied from
a saved capture, never invented. Acceptance requires the correct namespace,
strategy, size 100/overlap 15, source document ID, source-content hash, raw
passage SHA-256 reconstructed from exact span boundaries, and independently
parsed test code/name from contiguous local page spans. Expected source facts
must occur inside that test's portion of the captured span. Conflicting or
incomplete local coverage is rejected. Capture file hashes and verified passage
spans are retained as provenance. No scoring-time remapping is performed.

The dataset has 26 turns, with 14 comparable questions having verified positive
pools for both strategies. Unresolved positive questions are TC03.1, TC04.1,
TC05.1, TC06.1, TC10.1, TC11.1, TC14.1 and TC14.2. They remain unscorable rather
than being assigned guessed IDs. TC17–TC20 also remain outside ranking metrics
as unverified, negative or safety cases. Partial single-strategy labels can be
retained for inspection, but do not make a question comparable.

TC07.1 now expects only `Specimen Minimum Volume: 0.25 mL`:

- Fixed: `d8221455f53a56375ca735bc022a28b44b41d128819e01e28b26d978f6463d4f`
- Section: `b880ceb28ffcb506e83296ca4e56abdf154d318b18b175567d61b44e19e322dd`

The separate 0.5 mL requested-volume passage is not a positive for this question.
Historical TC07.1 metrics used a different pool and are not directly comparable.

## Compatibility and reporting

The UI prefers v2 when available and displays its version, label configuration,
and latest saved indexed configuration. The latter is explicitly a historical
observation, not a live cloud audit. The evaluator checks all returned candidate
configurations before scoring; the RAG wrapper checks before generation too.
Mismatches receive `Incompatible ground-truth labels`, status `Incompatible`,
and null/unscored metrics. Missing configuration evidence is unverified and
also withheld from scoring. Synthetic fixtures are explicitly distinguished.
Legacy label configuration is inferred only from hash-matching local snapshots;
if those snapshots are unavailable, compatibility cannot be verified.

Configuration agreement does not prove that the entire cloud index matches the
dataset. It establishes agreement only for observed returned candidates. Empty
pools cannot independently establish compatibility. Future corpus drift or
rechunking requires separately versioned verified labels, not relaxed ID metrics.

New runs record the dataset version, fingerprint, label configuration, per-row
compatibility, and retrieval diagnostics. Context diagnostics distinguish identity
search IDs, instruction search IDs, merged seed IDs, expanded neighboring chunks,
pre-cap instruction candidates and final returned context candidates. The reranker
sees only the final returned context pool. Both raw/context protocols and A/B/C/D
remain available. Historical reports load without these newer optional fields.

## Manual Streamlit validation

1. Open Evaluation and select `mayo_day5_v2_overlap15_ground_truth.json`.
2. Confirm version, size 100/overlap 15 and compatibility with the saved observation.
3. Select the historical dataset: its overlap 10 should show incompatibility with
   the saved overlap-15 observation. Do not interpret past scores as new results.
4. Inspect historical reports to confirm they still load unchanged.
5. With separate approval for live calls, start a **new** TC07.1 retrieval-only
   test-context run across A/B/C/D. Inspect CA/page 590/0.25 mL, exact IDs,
   per-row compatibility and candidate-stage diagnostics. Repeat raw only as an
   explicitly separate protocol comparison. No live run was made for this change.

## Limitations

Saved result pools bias which positives can be reconciled. These are incomplete
positive labels, not exhaustive relevance judgments or proof of production quality.
No live improvement claim is made. Existing resumes are guarded by changed
pipeline/dataset fingerprints; start a new run. No Pinecone data or index changes
are necessary.

## Files and offline validation

Added `src/evaluation/chunk_compatibility.py`,
`scripts/reconcile_day5_ground_truth.py`, the v2 dataset, this document and
`tests/test_chunk_compatibility.py`. Updated `src/evaluation/evaluation_runner.py`,
`src/evaluation/report_generator.py`, `src/retrieval/semantic_retriever.py`,
`src/retrieval/retrieval_comparator.py`, `src/ui_day5.py`,
`scripts/run_day5_evaluation.py` and `tests/test_day5_protocols.py`.

Final command: `python -m pytest -q -m "not integration"`:
**254 passed, 4 integration tests deselected**. Focused verification: 28 passed,
1 integration test deselected. An earlier full run reported 253 passes and one
failure in the unrelated existing PDF exact-size-boundary test; that test passed
on retry and the final full suite passed. PDF splitter code was not changed.
No external API calls, model downloads or live evaluations were initiated.
