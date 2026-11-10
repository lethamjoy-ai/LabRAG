# Day 5 quote mismatch diagnostics

Quote diagnostics describe rejection without changing citation acceptance. Exact
quotes and the existing whitespace restoration retain their previous behavior.
One invalid excerpt still rejects the selection. Diagnostics never reassign a
source or accept a paraphrase, fuzzy match, or cross-chunk quotation.

## Recorded fields and classification

Each selection excerpt includes `mismatch_diagnostics`: a sanitized cited source
ID, alternate matching source IDs, possible boundary source IDs, token insertion,
deletion and substitution counts, numeric offsets, truncation status, and
whitespace restoration possibility/success. No rejected quote text or token
values are added by this helper.

Exact and whitespace-only matches are checked first. An exact or whitespace-only
match to another supplied source identifies wrong-source attribution. Compatible
adjacent fragments can identify a sequence spanning their boundary. Remaining
local alignments distinguish punctuation, absent headings, substitutions,
insertions and deletions. Disjoint or truncated comparisons remain undetermined.
Alignment is heuristic and cannot establish the model's intent. Counts refer to
the locally aligned passage, not necessarily the entire source chunk.

Boundary diagnostics require equal document ID, source hash and test code, plus
compatible page/character boundaries. They do not create combined answer evidence.
Analysis is bounded to 2,048 tokens per text and 100 supplied sources. Missing
metadata or text may leave the cause unknown. Restoration success is recorded
only when the evaluator has both raw and restored selections; otherwise it is
unknown.

## Privacy and historical reports

New mismatch fields contain only fixed categories, sanitized IDs, numbers and
booleans. Existing quote retention restrictions remain in effect. Arbitrary
rejected model text, provider payloads and credentials are not persisted by the
new helper. Historical reports are read without backfilling missing diagnostics;
their original rejected text cannot be reconstructed from hashes.

## Manual validation

1. Open a saved Day 5 evaluation in Streamlit and expand the evidence and answer
   details. Historical runs should show that detailed diagnostics were not recorded.
2. Review the raw and after-restoration diagnostic tables for a newly authorized
   evaluation. Check source IDs, categories, edit counts, stage and final status.
3. Confirm a mismatched excerpt still results in rejection; whitespace-only repair
   uses the existing validator. No live evaluation is initiated by these changes.

Offline verification: run `.venv\Scripts\python.exe -m pytest -q -m "not integration"`.
Regression fixtures cover edits, headings, punctuation, whitespace, wrong sources,
boundaries, redaction, whole-selection rejection and historical report loading.
