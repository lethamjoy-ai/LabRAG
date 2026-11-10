# Day 5 ground-truth coverage v3

Dataset: `evaluation/mayo_day5_v3_overlap15_ground_truth.json`.
Version: `v3-overlap15-source-page-reconciliation`; parent is the unchanged v2
fingerprint. Existing 14 eligible questions and all non-target cases retain their
v2 question records verbatim. Seven newly verified questions raise eligibility to
21/26 (80.77%), from 14/26 (53.85%). These are dataset eligibility counts, not new
live evaluation results or retrieval success rates.

## Independent source verification

Historical local JSONL file hashes are checked before reconstructing source pages.
Some section snapshots omit whitespace between sections; the original v2 validator
required a completely covered section page and consequently rejected them. V3 may
use a complete fixed-snapshot source page only when document ID, source-content
hash, page and every observed character in the section snapshot agree. Conflicting
or incomplete alternative source coverage remains rejected. This uses original
source text, not another strategy's chunk IDs or retrieved scores.

For each accepted saved overlap-15 ID, validation checks strategy/namespace,
size 100, overlap 15, document identity, source hash, raw text SHA-256 at exact
start/end boundaries, independently parsed test identity and exact supporting
question-specific facts inside that test region. Saved capture hashes and verified
passages are recorded in the new dataset. IDs come from saved captures, not newly
computed or guessed vectors. Positive pools remain incomplete.

Document: `1606cf9fe9722101af96520196c3becce14757fe6f7ab65e2fb1bedb3fd14f21`.
Source-content hash: `a81828a9a219651f36b286827a8406e0a140bd55cb7d35b84f50c5afec4417ef`.
Printed pages below are distinct from local PDF page ordinals.

| Question | Test | Printed page | Section labels | Specific evidence | Decision |
|---|---|---:|---|---|---|
| TC03.1 | BAP | 505 | S1 | Serum gel/red top; requested 0.6 mL; minimum 0.5 mL | Eligible |
| TC04.1 | CPR | 557 | S2 | Fasting 8 hours; avoid biotin-containing supplements for 12 hours | Eligible |
| TC05.1 | CPR | 557 | S3, S4 | Centrifuge/aliquot within 2 hours; frozen preferred 30 days; refrigerated 7 days | Eligible |
| TC06.1 | CRP | 558 | S5, S6 | Serum gel/red top; requested and minimum volume 0.5 mL | Eligible |
| TC10.1 | BPRPV | 509 | S7, S8, S9 | Preferred nasopharyngeal swab and media; acceptable aspirate/wash with specified container | Eligible |
| TC11.1 | BMAPC | 507 | None | Billing-only/non-orderable source fact exists, but no verified saved overlap-15 section candidate ID | Excluded |
| TC14.1 | CPR | 557 | S2 | Fasting 8 hours required | Eligible |
| TC14.2 | CPR | 557 | S4 | Frozen preferred 30 days; refrigerated 7 days; fasting-only chunk not labeled | Eligible |

## Verified section chunk IDs and source spans

Spans use the indexed start/end character convention within the local source page.

| Key | Chunk ID | Span |
|---|---|---|
| S1 | `1cfedc5c05cf7d48f87747c9a0307924d0849cd342b269527df79e671c4fa03b` | 379–859 |
| S2 | `602d282688ace8ce86228995de11294e9ea6638bdc8b5d71d21941f92772b66c` | 0–452 |
| S3 | `42918871c98ce80bed262df22eadc4c80bf58f0fe45a2ff4978a6063fb548de6` | 369–772 |
| S4 | `688b43d90495aa541a0243b8947565b169d4d6febc2f521e016893ce550dcee2` | 772–947 |
| S5 | `a6fb2d82355821d8a080c4ffeb45a973a90722596b030ed5486157e015c6ede0` | 834–1109 |
| S6 | `98ee2485ae5bfb116712d503591ac2cc360f24a7978beb55bf2c56aded98c70d` | 1109–1501 |
| S7 | `41799017e06515f2872aa0a76dd20fce3073d56f417b07b2b66dff5ec99cea49` | 0–155 |
| S8 | `4d83ba5b7c4d995f2e7997dfdf6a4428f63fc39a6dd27e6850d0e456b9c1f4e5` | 155–574 |
| S9 | `7f98205d25a82a11c79974d02a09c204d95944d9ffcb012bc9c9e04773df1d20` | 937–1290 |

Each newly eligible question also has independently verified fixed labels. Tests
require its expected facts to be supported across the positive pool of each strategy.
A positive passage may support part of a multi-fact question; a hit alone does not
establish complete answer coverage.

## Remaining exclusions and use

TC11.1 remains unresolved. TC17.1, TC18.1, TC19.1 and TC20.1 retain their v2
unverified/non-ranking/safety/mock-only exclusions. All A/B/C/D configurations
continue to share the same eligible question set. No scoring-time ID remapping,
retrieval/reranking edits, Pinecone changes, reindexing or external API calls occurred.

In Streamlit, explicitly select `mayo_day5_v3_overlap15_ground_truth.json` in the
Day 5 dataset selector. The form may still default to v2. Confirm 21 comparable
questions and size 100/overlap 15. Historical run coverage should remain unchanged
until a separately approved new evaluation is executed; dataset choice does not
rewrite selected historical results. New v3 results are not directly comparable
to v2 aggregate metrics because their question sets differ.

Files: updated `scripts/reconcile_day5_ground_truth.py` with an opt-in source-page
fallback (legacy behavior stays default); added `scripts/improve_day5_coverage.py`,
the v3 dataset, `tests/test_day5_coverage_v3.py` and this report. The builder refuses
to overwrite output files. Local snapshots and saved captures are required for
reproduction; no fresh cloud audit was performed.

Validation: `python -m pytest -q -m "not integration"` completed with **270 passed,
4 integration tests deselected**. Coverage includes historical dataset loading,
exact saved-ID provenance/configuration, question-specific facts in both strategies,
unchanged non-target records, rejection of conflicting source spans and existing
offline Mayo source verification. No live test or model download was initiated.
