# Day 5 evaluation methodology

LabRAG evaluates only the supplied Mayo catalog subset (printed approximately 500–999).
The full Mayo catalog is not assumed indexed. Existing vectors, embeddings, dimensions,
namespaces and source PDFs are read-only throughout Day 5 evaluation.

## Dataset and source verification

`scripts/prepare_day5_ground_truth.py` re-reads the actual local PDF through the Day 4
source verifier; a mocked Day 4 pass is not ground truth. Each clinical excerpt is
checked within the correct test entry. Day 4 expectations and per-turn expectations
must match the freshly verified data. Missing PDFs or changed facts cause unverified
labels, not guessed answers. Dataset content is fingerprinted and checked on load.

TC01–TC20 produce 26 per-turn records. Follow-ups have the original user context and
a resolved standalone question; switching tests does not reuse the prior test's values.
The current local files yield 22 comparable, verified answerable turns. Negative, safety,
missing-value and injection cases are separate from ranking averages. TC17's current
cloud absence remains unverified; local absence and small top-k results cannot prove it.
TC20 stays mock-only and is never inserted into clinical namespaces.

For each strategy, the builder reads the complete local JSONL snapshot, verifies the
source document's byte hash against the PDF, and uses contiguous indexed page spans to
recover test identity in memory. A chunk enters a positive relevance pool only when it
contains a source-verified expected fact inside that test's region. Actual document and
chunk IDs, physical namespace, test/section/page references and snapshot hashes are stored.
Fixed and section strategies have separate chunk-ID sets. Printed catalog pages are
distinct from one-based PDF ordinals. The whole catalog document is never labeled relevant
for every question merely because it contains one correct test somewhere.

These are **positive judged pools**, not exhaustive relevance labels. A relevant paraphrase
or fact split across chunks can be missed by literal labeling. Local snapshots are not
proof of current cloud contents. Both strategies must have positive labels for a turn
to enter comparable ranking metrics. Changed labels require regeneration/review.

## Four retrieval configurations

| Key | Initial retrieval | Final selection |
|---|---|---|
| A | Fixed top 5 | Original order |
| B | Section top 5 | Original order |
| C | Fixed top 15 | Local cross-encoder top 5 |
| D | Section top 15 | Local cross-encoder top 5 |

Counts are configurable and recorded; metric names use the actual k. All four use the
same question set, embedding model, filters and scoring rules. Evaluation uses raw
Pinecone retrieval without Day 4 heading/neighbor expansion, so the comparison does not
silently change candidate definitions. Normal chat retains its working context path
when reranking is disabled. When enabled, chat retrieves raw candidates, reranks locally,
validates selected evidence and sends only selected/scoped chunks to Groq.

The local CPU model is `cross-encoder/ms-marco-MiniLM-L6-v2`, the canonical identifier
to which the prompt's `MiniLM-L-6-v2` spelling redirects. The successfully loaded model
revision is pinned to `233902d25c440f23af6f7d6e94d2946bac0bee0a`. See the
[model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) and
[CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
Loading is lazy, CPU inference is cached per process, and model files live under
`.model-cache/rerankers`. The tokenizer applies a 512-token pair limit; full source text
is preserved for citations. Ties preserve initial order, duplicate chunk IDs are removed,
non-finite scores fail safely, and model errors never trigger a silent configuration
fallback. Similarity and reranker scores are separate and are not clinical probabilities.

## Ranking metrics

Duplicate retrieved IDs are deduplicated in original order before the top-k cutoff.
For a verified positive set R and unique returned top-k list L:

- Hit Rate@k: 1 if any item in L is in R, otherwise 0; aggregate by mean.
- MRR@k: reciprocal rank of the first known relevant item, otherwise 0; mean over questions.
- Precision@k: known relevant hits / actual unique returned count (up to k); empty retrieval is 0.
- Recall@k: known relevant hits / all relevant items, **only when labels are exhaustive**.
- Judged-pool recall: hits / known positives when labels are partial; never labeled exact Recall.

With partial labels, hit/MRR/precision describe known-positive retrieval and can understate
true relevance when unjudged items are useful. The denominator and label scope are
recorded. Unverified questions and missing positive pools produce null metrics, not zeros
or perfect scores. Current Mayo exact Recall is unmeasured because labels are not exhaustive.
Synthetic tests verify formulas with fully labeled examples and distinct namespace IDs.

## Answer faithfulness and citation support

Full RAG evaluation executes the actual LangGraph pipeline on each configuration's selected
evidence. The same Groq model, temperature, output limit and prompt hash are recorded;
temperature zero does not guarantee deterministic generation. Optional reranking keeps
original candidate source IDs stable even when rank changes.

Answers are split into cited factual units, with citation IDs retained. Deterministic
checks compare exact text and labeled numeric fields, units and test identity. A conflicting
number/unit in the same field is Contradicted; absence of evidence is not contradiction.
Requested and minimum volumes are separate fields. Supported, Unsupported, Contradicted
and Unverifiable classifications are recorded per claim. Ambiguous paraphrases require
human review rather than being silently accepted. This limited lexical/field checker is
not medical entailment; clinical completeness and interpretation require review.

Supported-claim rate is supported / all extracted claims. Contradictions are also counted
separately. Abstentions and empty answers have no claim score, so refusal does not inflate
faithfulness. Expected-fact omissions and expected response behavior are recorded separately:
an incomplete answer can still be faithful to the few facts it states.

Citation **validity** verifies the retrieved source/chunk/document/test/page association.
Unknown metadata is flagged for review; a page is never invented. Duplicate citations
do not inflate counts. Citation **support** checks the claim against its cited passages,
independent of structural validity. An existing ID can still cite the wrong test or fail
to support a claim. Missing citations and fabricated IDs are recorded explicitly.

Optional Groq judging requires explicitly enabled live RAG mode and a judge option.
The judge receives question, claims and cited evidence, not general medical authority.
Its model, prompt version and structured raw judgment are saved as estimates. Malformed
judgments or disagreements are flagged for review; they do not overwrite deterministic
classifications or become definitive clinical validation.

## Latency, reproducibility and failures

`perf_counter()` measures embedding/search, reranking, generation, retrieval pipeline,
RAG answer completion and total evaluation time. Total evaluation also includes quality
checks; RAG answer completion excludes them. Retrieval pipeline time covers initial search
through final selection. First-use model loading is included where it occurs. Provider
token counts are null when unavailable; none are invented. p50/p95 require at least 20
scored queries per configuration. Caches, warm-up and execution order can affect timings.
Compare retrieval and RAG modes separately and report sample counts and failures.

Run IDs are unique; JSON/CSV records and Markdown reports retain history. Resume validates
dataset/model/prompt/pipeline fingerprints, namespaces, thresholds, filters, counts and
question selection. Completed/excluded rows are retained; failed rows can be retried with
their prior errors preserved. Interruption saves completed work. No fallback or provider
body/credential logging occurs. The current runner is intended for a single trusted writer.

Failure inspection distinguishes wrong tests, missing sections, unresolved metadata,
candidate-ranking misses, generation/omission issues and citation problems. It does not
assume missing top-k evidence means an unindexed document. Chunking/coverage suspicions
need source and namespace review. Baseline/reranked pairs are checked for measured
regressions; no improvement or winning configuration is asserted without live evidence.

## Execution and dashboard

Normal pytest uses deterministic injected retrieval, generators and cross-encoders;
it neither downloads model weights nor invokes Pinecone/Groq. The optional local model
test is separately enabled. Live comparisons require an explicit UI initiation checkbox
and Start action, or `LABRAG_RUN_DAY5_LIVE=1` in the CLI. Judge generation is independently
optional. No evaluation runs on page load. All cloud operations are read-only.

The dashboard renders summaries and Plotly charts only for computed values, offers
configuration/result filters, evidence and failure inspection, and CSV/JSON/Markdown exports.
Synthetic fixture results are prominently labeled and cannot masquerade as Mayo accuracy.
A preparation-only report has zero executed questions and blank quality metrics.

Live comparison, reranking impact and clinical-quality conclusions remain unmeasured
until a developer explicitly initiates a run. Preserve historical runs and review results
against the current official catalog before any clinical use.
