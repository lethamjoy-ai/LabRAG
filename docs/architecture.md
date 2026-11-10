# Architecture

The supported Day 4 entry point is `app.py` → `src.ui`. It loads configuration and opens three tabs without constructing service clients. Knowledge Base validates extension, signatures where practical, size, and extractable text. Parsers return LangChain Documents with SHA-256 content identity, format, filename, leading heading where available, and one-based PDF page references. Content is never executed.

On explicit Save, required metadata is validated, duplicate bytes and source identities are rejected, original bytes go to `data/raw`, extracted text and page metadata go to `data/processed`, and one document-level record goes to `data/source_manifest.csv`. CSV replacement is atomic; the complete multi-file save is not transactional. Storage is designed for one trusted local user. Planned topics are a separate CSV to avoid representing uncollected sources as evidence.

Current flow: permitted documents → reviewed text → fixed-size and section-aware chunks → configured embeddings → cosine Pinecone namespaces → retrieval/context expansion → LangGraph evidence checks → Groq-selected verified excerpts and citations → human review and explicit workflow validation. Reranking and quality benchmarking remain future work. Index dimensions and namespaces remain unchanged. Preserve units, test identity and qualifiers; do not infer missing clinical requirements.

## Day 2 local processing

`src/ingestion/test_metadata.py` detects explicit name/code labels, bounded Test ID/Definition title patterns, and labeled table columns from extracted text. It records exact evidence and entry offsets, computes single-test versus catalog summaries, and leaves unresolved values blank. Document-level test names/codes are optional, with a backward-compatible `test_scope` CSV column. During both fixed and section chunking, detected test entry boundaries are honored before text splitting, and per-entry name/code/evidence metadata is attached. Validation compares these values to the source entry context independently of document-level provider/source attribution. Unknown catalog pages and introduction text remain unassigned rather than inheriting a neighboring test. No catalog lookup, LLM, embedding, or clinical inference is used.

Saved extracted JSON + latest manifest metadata → conservative cleaner with original extraction retained → independent fixed/section chunking → validation → atomic strategy-specific JSONL snapshots → UI preview/comparison/export/reload. The pipeline shares existing loaders and metadata services; it never concatenates different documents or pages. Section continuation prefixes are recorded separately from exact source offsets. Tables are conservatively flagged and oversized table sections remain intact for review. Every chunk includes source/document/test/provider provenance, sequence, token count, source snapshot hash, content hash, and deterministic identity.

Validation compares chunk spans with cleaned source text, verifies coverage, token budgets, attribution, IDs, and recognized section boundaries. It does not claim semantic equivalence or clinical correctness. Loading chunk JSONL verifies schema, counts, and identities; UI reload additionally validates against the current cleaned source. Reprocessing replaces the document/strategy snapshot rather than appending. Changed bytes normally create a new Day 1 document ID, so revision retirement needs explicit corpus management before Day 3.

Day 2 does not call the future embedding/retrieval/answer flow. Earlier `labrag/` modules remain inactive, preserved from the initial broader request. Retrieval evaluation and API integration require separate implementation review and live testing in later days.

## Day 3 embeddings and retrieval

Explicit Knowledge Base actions → existing strategy JSONL streams → structural metadata validation and exact embedding-token preflight → local CPU HuggingFaceEmbeddings → bounded, metadata-size-checked Pinecone upserts → atomic local ownership/checkpoint manifest. No original PDF is read by this service. Input text is not truncated. Normalized document and question vectors use the same model/dimension.

`PineconeConnection` constructs no SDK client at import/startup; connection validates an existing dense cosine index on explicit use. `IndexManager` records both attempted ownership (to account for ambiguous timeouts) and acknowledged checkpoints, making retries idempotent through deterministic vector IDs. Source changes retain obsolete owned IDs until explicit scoped cleanup. Public and synthetic corpora use separate strategy-specific namespaces. Source identity/page ranges block known original/split overlaps. A local removal intentionally leaves cloud ownership available for explicit administration.

Ask LabRAG → validated question/filter inputs → local query embedding → `PineconeVectorStore` by-vector similarity query → Document text from the `text` metadata field plus source metadata/score → evidence cards. Comparison reuses one question embedding and queries both namespaces. It warns about differing source collections/revisions; labels/clinical relevance remain for future evaluation. No LLM, reranking, or agent graph runs in the active Day 3 path.

Model caches and vector manifests are local; cloud text uploads require a user action. The manifest is a single-writer checkpoint, not proof that remotely edited vectors still exist. The earlier `labrag/` modules remain inactive. Active Groq configuration is used lazily by `src/llm/`.

## Day 4 workflow and evaluation

`src/rag/` orchestrates question validation, compact user-only follow-up context, fresh
retrieval, evidence validation, generation/abstention and exact citation validation.
Test switching uses the new question's test identity, not the prior test. Treatment
requests are blocked before retrieval. Missing requested values are distinguished from
retrieval failure; related citations are labeled context-only. Source text is untrusted.
See [RAG architecture](rag_architecture.md) for source boundaries and quote validation.

`evaluation/mayo_day4_cases.json` holds source-verified reference facts and expected
behavior separately for TC01–TC20. `src/evaluation/day4_runner.py` runs real graph paths
with injected fixtures by default. Explicit opt-in live mode reuses existing services
read-only; TC20 is never uploaded. The optional TC17 audit lists/fetches an entire bounded
namespace; local snapshots or top-k results alone cannot verify cloud absence.

Reports contain actual per-turn queries, evidence/metadata/chunk IDs, answers, citations,
expectation checks and perf-counter timings. Mock results and live results are labeled.
The Evaluation tab reads reports without clients or live tests on startup, supports
filters/inspection/downloads, and preserves the earlier retrieval comparison tools.
Day 4 behavior is retained; Day 5 adds the separate optional evaluation layer below.

## Day 5 evaluation and optional reranking

`src/retrieval/reranker.py` lazily caches the pinned local CPU cross-encoder. Normal chat
defaults to the existing context-expanded path. Enabling reranking selects raw candidates,
runs the optional graph node, and sends only the selected top-k through existing evidence
and citation guards. Source IDs remain stable through sorting. Model failures do not fall
back to a different protocol. Vectors and source metadata are never rewritten.

`src/evaluation/dataset_validator.py` re-verifies local catalog facts and generates separate
positive relevance pools for fixed and section namespaces. `retrieval_comparator.py` uses
the same raw query protocol for A/B/C/D, without chat expansion or gold-label leakage.
`evaluation_runner.py` executes retrieval or the full graph, records model/prompt/pipeline
and corpus fingerprints, and resumes only compatible runs. Negative/safety cases are
separate from ranking averages. Exact Recall is unavailable for incomplete relevance labels.

Faithfulness and citation evaluators distinguish claim support, numerical contradictions,
unknown evidence, citation structure and citation support. Optional Groq judging produces
estimates only; ambiguity/disagreement is reviewable. The dashboard reads historical
artifacts and creates Plotly charts from available computed values only. Execution requires
an explicit Start action/consent or CLI live flag. No live evaluation runs on startup.

The real local cross-encoder and synthetic A/B/C/D execution are verified. Live Mayo quality
metrics remain unmeasured until explicitly initiated. See [evaluation methodology](evaluation_methodology.md).
