# LabRAG — Clinical Laboratory Knowledge Assistant

Week 2 project for The Gen Academy Agentic AI Master Class. **Current scope: Day 5 evaluation and optional local reranking.**

Laboratory information is scattered across provider catalogs and documents. LabRAG helps healthcare professionals locate source-grounded passages about tests, collection, preparation, and handling. Days 1–2 prepare local documents and reviewable chunks. Day 3 provides embeddings and Pinecone retrieval. Day 4 selects verified source excerpts through a conversational LangGraph workflow with Groq.

## Features and status

Optional privacy-preserving LangSmith tracing is available for the existing RAG
workflow. Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and
`LANGSMITH_PROJECT=LabRAG` in the local `.env`, then restart the application.
Tracing defaults to disabled; only sanitized stage summaries are uploaded.
Optional `LANGSMITH_CAPTURE_FULL_IO=true` enables question/answer/citation capture
for approved public/synthetic course-demo questions; it defaults to false. Restart
the dashboard/API after changing it. Unknown questions remain filtered.
See [LangSmith setup, privacy rules, manual verification and checkpoint caveat](docs/langsmith_tracing.md).

- Three Streamlit tabs: Ask LabRAG, Knowledge Base, Evaluation.
- PDF, HTML, and UTF-8 TXT validation, extraction, and text previews.
- PDF page references, HTML boilerplate removal, conservative whitespace cleaning.
- Configurable 20 MB file limit, content hashing, duplicate detection, validated metadata.
- Local raw files, processed JSON, CSV manifest, search, filters, counts.
- Three clearly synthetic samples and 15 planned laboratory collection topics.
- Fixed-size and section-aware token chunking with deterministic IDs and source-span validation.
- Side-by-side chunk comparison, actual token statistics, metadata inspection, JSONL save/export/reload.
- Preserved original extracted text, conservative cleaning, and oversized table review flags.
- Twenty-question evaluation template with unverified references and expected answers left blank.
- No API keys required for local ingestion/chunking; no startup cloud calls, model loading, indexing, or scraping.
- Lazy CPU embeddings, exact model-token validation, explicit checkpointed Pinecone indexing, semantic search, and retrieval comparison.

The earlier full-pipeline modules in `labrag/` are retained for continuity with the initial request. They are not imported by `app.py` or the active UI and are outside the current supported workflow. Their notes are archived in `docs/previous_pipeline_notes.md`.

## Install and run

Python 3.11+ is required. In PowerShell from this project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-day5.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
streamlit run app.py
```

On Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate`, and copy `.env.example` only if `.env` does not exist. If Python is not on PATH, use `uv venv --python 3.11` and `uv pip install -r requirements-day4.txt`. Preserve an existing `.env` and its credentials.

## Day 5 evaluation

Install `requirements-day5.txt`, then prepare source labels from the project root:

```powershell
.venv\Scripts\python.exe scripts/prepare_day5_ground_truth.py
.venv\Scripts\python.exe scripts/prepare_day5_report.py
.venv\Scripts\python.exe -m pytest -q
```

The ground truth re-verifies the local Mayo PDF and labels 26 turns, including 22
comparable answerable turns. Fixed and section namespaces have separate actual chunk-ID
sets. Labels are positive judged pools: exact Recall@5 is not claimed; judged-pool recall
and known-positive Hit/MRR/Precision are explicitly distinguished.

The **Evaluation → Day 5** dashboard configures and inspects four protocols:

| Key | Retrieval | Reranking |
|---|---|---|
| A | Fixed top 5 | Off |
| B | Section top 5 | Off |
| C | Fixed top 15 | Local cross-encoder → 5 |
| D | Section top 15 | Local cross-encoder → 5 |

Normal chat retains its existing working retrieval unless **Enable local reranking** is
checked. The cross-encoder runs on CPU, loads lazily, caches its weights and preserves
original similarity scores, source IDs and metadata. It uses the canonical
`cross-encoder/ms-marco-MiniLM-L6-v2` at a recorded pinned revision. It never changes
Pinecone vectors. Reranker scores are not clinical confidence.

Live comparison must be explicitly initiated using the dashboard's checkbox and Start
button, or this CLI opt-in. No Day 5 live evaluation was run during implementation:

```powershell
$env:LABRAG_RUN_DAY5_LIVE = "1"
# Four-configuration demonstration: minimum volume for Calcium, Total, Serum.
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --questions TC07.1
# Optional full graph/Groq generation, same four configurations:
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --questions TC07.1 --mode rag
# Optional evidence-only Groq judge adds calls; scores remain estimates:
.venv\Scripts\python.exe scripts/run_day5_evaluation.py --questions TC07.1 --mode rag --judge
Remove-Item Env:LABRAG_RUN_DAY5_LIVE
```

Use `--configurations A B C D`, `--category CATEGORY`, `--top-k 5`, `--candidates 15`,
`--questions TC14.1 TC14.2`, or `--resume RUN_ID` to reproduce a subset/run. Keep all
parameters identical for resume; changed dataset, model/prompt/pipeline, namespace or
question selection requires a new run. Ctrl+C saves completed work. Judge use is optional;
Pinecone reads, configured query embeddings and Groq calls incur normal provider usage.

Unique JSON/CSV artifacts are in `evaluation/results/`, with Markdown reports in
`evaluation/reports/`. Historical runs are preserved. The dashboard offers measured
summaries, Plotly charts, question-level failure/evidence inspection, and exports.
Source text in raw run files is gitignored. Preparation-only reports have no executed
rows and blank quality values. Synthetic test results are never presented as Mayo scores.

Faithfulness classifies cited factual units as Supported, Unsupported, Contradicted or
Unverifiable; labeled numeric fields preserve requested/minimum volume and units.
Citation validity and citation support are independent. Ambiguous claims need human
review. These checks and optional judging are not definitive clinical validation.

Default regression suite: **235 passed, 4 optional integrations skipped**. A separate
real cached local cross-encoder smoke/long-input test passed. Enable that test explicitly:

```powershell
$env:LABRAG_RUN_LOCAL_RERANKER = "1"
.venv\Scripts\python.exe -m pytest tests/test_reranker.py -m integration -q
Remove-Item Env:LABRAG_RUN_LOCAL_RERANKER
```

No live quality comparison has established a best configuration or reranking benefit.
See [methodology](docs/evaluation_methodology.md), [Day 5 progress](docs/day5_progress.md)
and generated preparation reports. Day 6 implementation has not begun.

## Day 4 chat (preserved)

Use the **existing populated index**. Do not recreate it or reindex for Day 4. Preserve
the Day 3 embedding provider, model, dimensions and namespaces in `.env`. Set
`GROQ_API_KEY` and leave these configurable defaults unless your account needs another model:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1200
RETRIEVAL_TOP_K=5
RAG_DEFAULT_STRATEGY=section
RAG_MAX_HISTORY_TURNS=5
RAG_ENABLE_TRACING=false
```

In **Ask LabRAG**, submit “What specimen is required for BUN?”, then “How long can it
be frozen?”. Review the test name/code and **Sources and retrieved evidence**. The model
selects exact source excerpts that are validated before display. Missing or ambiguous
evidence produces an abstention scoped to the partial catalog. The sidebar controls model,
strategy and top-k; **Clear conversation** removes session history. Retrieval/LLM/total
timings and counts appear under each answer. Enable tracing for measured node timings.

Each follow-up retrieves new evidence. The prior assistant answer is never a source.
Read-only indexed page neighbors can supply missing test identity context, with no change
to existing vectors. Printed catalog pages and one-based PDF ordinals are shown separately;
unknown page metadata is not inferred. **Semantic retrieval explorer** preserves Day 3 search.

In **Semantic retrieval explorer**, **Retrieve test context** defaults on so section
results include related instructions beyond small test-heading chunks. Uncheck it for
raw similarity search. Context mode preserves filters/corpus, shows at most the selected
top-k verified instruction passages, and exposes counts and identity checks under
**Search diagnostics**. It does not invoke Groq or change indexed data.

See [RAG architecture](docs/rag_architecture.md) and [Day 4 progress](docs/day4_progress.md).
The [reference cases](evaluation/mayo_day4_cases.json) contain TC01–TC20, including the
supplied continuation's follow-ups, switching, insufficient-evidence and safety cases.
Expected clinical excerpts were verified within the correct local PDF test entries.
TC17's real test (DHVD) was located in the original catalog and was absent from exhaustive
local snapshot scans; current cloud absence is not assumed. Run
`python scripts/validate_day4_corpus.py` to regenerate local source verification. These
reference cases are not live accuracy scores.

### Day 4 validation and reports

```powershell
.venv\Scripts\python.exe -m pytest tests/test_day4_corpus_cases.py -v
.venv\Scripts\python.exe scripts/run_day4_evaluation.py
.venv\Scripts\python.exe -m pytest -q
```

Default execution is credential-free mocked workflow validation. The measured mocked
report contains **19 passed, 0 failed, 1 skipped** (TC17's current cloud absence is not
verified). JSON and CSV exports are written to `evaluation/day4_results.json` and
`evaluation/day4_results.csv`. Generated reports are gitignored because live reports can
contain source text. Source cases remain separate and are not assigned invented chunk IDs.

Open **Evaluation → Day 4 — RAG Chatbot Validation** for measured counts, citation status,
follow-up/insufficient-evidence summaries, category/code/result filters, and turn-by-turn
resolved questions, retrieved evidence, answers, citations, expected facts and failures.
The **Run mocked Day 4 validation** button never connects to Pinecone or Groq. Page load
only reads existing reports and shows unexecuted cases as Not Tested.

Live tests are optional, read-only and **must be explicitly enabled**:

```powershell
$env:LABRAG_RUN_DAY4_LIVE = "1"
.venv\Scripts\python.exe scripts/run_day4_evaluation.py --live
# Optional exhaustive namespace listing/fetching to establish TC17's absence precondition:
.venv\Scripts\python.exe scripts/run_day4_evaluation.py --live --audit-absence
Remove-Item Env:LABRAG_RUN_DAY4_LIVE
```

These commands use the existing index, configured namespace/embedding model and Groq
model, and incur normal provider usage. They never upsert, delete or recreate vectors.
The optional absence audit reads at most 100,000 vectors, rechecks ID-list consistency,
and records its scope/method and observed matches; incomplete or unsupported audits
cannot certify absence. TC20 always remains mock-only. Live case failures are reported
honestly; no live suite was enabled for this continuation. To run through pytest instead,
set the same explicit flag and execute
`python -m pytest tests/test_day4_corpus_cases.py -m integration -v`.

The current full regression result is **194 passed, 2 optional integration tests skipped**.
Mock passes verify mechanics, not clinical accuracy, retrieval recall or model faithfulness.
The framework is ready for Day 5 evaluation work; reranking and quality benchmarking have
not been implemented.

The lightweight Day 1/2 requirements remain usable for local ingestion and chunking because cloud/model imports are lazy. The complete `requirements.txt` additionally includes future pipeline dependencies. Pinecone SDK 7.x is the compatible modern range required by `langchain-pinecone` 0.2.x; do not add `pinecone-client`. No Pinecone index needs to exist for local processing. First token counting downloads the public `cl100k_base` vocabulary into tiktoken's cache. Day 3 model actions download MiniLM to `.model-cache/hub`; startup does not download it.

## Day 3 workflow

### Optional paid OpenAI embeddings

To use OpenAI instead of local MiniLM, update the existing `.env` (preserve your Pinecone key):

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=384
OPENAI_API_KEY=your-private-openai-api-key
```

Keep `PINECONE_INDEX_NAME=lab-rag` if it has dimension 384. The app explicitly requests 384 output dimensions and automatically uses separate namespaces such as `section-chunks-openai-text-embedding-3-small-384`; changing models/dimensions never reuses MiniLM vectors. Re-index the approved Day 2 chunks using OpenAI and use the same provider for questions. Existing local vectors are preserved. No change is made to your real `.env` by this implementation.

OpenAI embeddings send text to the external API and incur usage charges. Validation, indexing, and uncached query embeddings make paid requests only on explicit actions; startup never does. Configure API billing and access in your account. The adapter uses the official SDK, validates dimensions/finite values, caches repeated inputs in memory, and batches below the API's per-request token limit. It rejects inputs above 8192 tokens rather than truncating them. Unlike MiniLM's shorter input window, the previously reported 366-token passage fits this model's input limit, but table/source review still applies. See the [official OpenAI embeddings API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

To switch back, set `EMBEDDING_PROVIDER=local` and restore `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` with dimension 384. OpenAI integration is verified with mocked requests; no paid API test or reference upload was performed during implementation.

1. For a new Day 3 installation only, follow [Pinecone setup](docs/pinecone_setup.md) to create a **384-dimensional dense cosine index** named `lab-rag`, then set `PINECONE_API_KEY` in `.env`. Existing populated installations should keep their index. Groq is required only for Day 4 answer generation.
2. In Knowledge Base, save permitted public references with correct HTTPS source attribution and process the same source collection with both chunking strategies if comparing them.
3. Open **Pinecone Vector Indexing** → **Validate Local Embedding Model**, then **Validate Pinecone Connection**. Neither validation uploads references.
4. Select strategy, public/synthetic corpus, and processed documents. Review tables against originals and approve non-oversized flagged table chunks if appropriate. Confirm permitted upload and click **Index Processed Documents**. Chunks stream from JSONL; the PDF is not read by the Day 3 service.
5. In **Ask LabRAG → Semantic retrieval explorer**, choose strategy/top-k and optional exact provider/test/document/revision filters. Search to inspect source passages, similarity and metadata without invoking Groq.
6. In **Evaluation**, use **Compare Strategies** to see both namespaces side by side. Mismatched source collections/revisions are warned about. Scores alone do not establish which strategy is better; no labeled quality metrics are fabricated.

The general-purpose model is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), accessed using `langchain_huggingface.HuggingFaceEmbeddings`. Document and question vectors use the same local CPU model with normalization. The model is not specialized for medicine. **Actual model input length is checked before embedding**, including special tokens; it is limited to 256 word pieces by default. Existing Day 2 defaults of 500 approximate tiktoken tokens may be too long. Reprocess oversized chunks with a smaller size (for example 150–180 approximate tokens and overlap 25), then verify again; this is not a guaranteed conversion between tokenizers. Oversized unsplittable tables are blocked, not truncated.

Indexing uses deterministic IDs, bounded embedding/upsert batches, size-checked scalar metadata, and three-attempt exponential retries for transient API failures. `data/indexing_manifest.json` checkpoints successful and attempted vector ownership plus source snapshot/model/index/namespace identity. Retrying skips acknowledged IDs. This file is local operational state, not an evaluation dataset; keep it with the data and inspect before sharing. Metadata-only/content changes produce new IDs. Obsolete IDs remain identifiable and are removed only when explicitly requested after successful reindexing. Back up the manifest; losing it loses local ownership knowledge. Only one indexing writer should run at once.

Synthetic fixtures use `fixed-chunks-synthetic` and `section-chunks-synthetic`, separate from public-reference namespaces. The default public path requires HTTPS attribution and rejects explicitly labeled synthetic fixtures. The human upload declaration is still required; software cannot detect all PHI or mislabeling.

The project's PDF splitter filenames preserve explicit source page ranges. For those filenames, Day 3 records original filename, original page offset, split-file identity, and a stable source identity from provider/URL/original filename; custom provenance can also be present in chunk metadata. Overlapping pages in original/split or revised sources within a namespace are blocked using the manifest. Do not rename split files arbitrarily, and review lineage before uploading. Changed-byte revisions with new document IDs must be explicitly retired before replacements are indexed. Duplicate detection cannot establish lineage for unrelated filenames/URLs without supplied provenance.

Local **Remove document** does not delete Pinecone records. Use the explicit, scoped admin command from [setup guidance](docs/pinecone_setup.md) to retire remote document IDs. Search reflects Pinecone eventual consistency, not the local source table alone.

Open Knowledge Base, upload any file under `data/samples/`, inspect text, enter its document title and provider, confirm content permission, and save. Records appear immediately in the table. Real permitted files should include their original HTTPS source URL, revision date when known, and access date.

### Automatic test names and codes

To remove a saved upload, expand **Remove a saved document** at the top of Knowledge Base, select it, and click **Remove document**. This deletes its local uploaded copy, extracted JSON, fixed/section chunk snapshots, and manifest record, then refreshes the UI and clears the upload/chunk preview. Your source file outside managed upload storage is preserved. There is no undo; re-upload the original to add it again. Missing derived files do not prevent removal; a failed manifest update restores staged files.

Uploaded documents now display **Detected test names and codes**, with page numbers, text offsets, and source evidence. Supported layouts include `Test Name:` with `Test Code:`, `Test ID:` or `Test Definition:` followed by a nearby title and recognizable laboratory section, and clearly labeled pipe/tab/aligned-space test-name/code tables. This is deterministic local parsing; it does not call an LLM, use a test dictionary, or infer an identifier from a CPT code.

The upload form contains document-level fields only: title, laboratory/provider, source URL, revision date, and access date. Test-name/code inputs and the Multiple-test catalog checkbox have been removed. New UI uploads are treated as catalogs, with blank document-level test identity. Detected test names/codes remain visible for review and are assigned to their corresponding chunks; unknown entries remain unresolved. Existing saved records are unchanged.

Both chunkers separate detected test entries and attach the corresponding `test_name`, `test_code`, evidence text, and `test_metadata_source` to each chunk. A catalog introduction, an unlabeled continuation page, or an ambiguous span can remain unresolved; it does not inherit the last catalog test. Directory rows apply only to their row. Chunk preview tables show test names/codes directly. Unknown layouts, wrapped titles, or lost PDF table structure may require a parser extension or source review; extraction is not guaranteed for every Mayo catalog layout.

Already-saved source documents can be selected and processed again without re-uploading: detection runs on their extracted text. Existing JSONL snapshots keep their old metadata until **Process Document** is run again. Their document-level CSV summaries are not automatically rewritten. Old manifests without `test_scope` remain readable.

In **Document Chunking**, select the saved document and choose fixed-size, section-aware, or compare both. Set the token size and overlap, then **Process Document**. Valid results are saved automatically as one replaceable snapshot per document/strategy under `data/processed/fixed/` and `data/processed/section/`. Inspect validation messages, preview rows, select individual chunk metadata, compare original/cleaned text, and export UTF-8 JSONL. **Load saved chunks** reloads those snapshots and validates them against the current source. Loaded metrics use the saved settings shown in chunk metadata; input changes take effect only when reprocessing.

Oversized table sections are preserved with `unsplittable=true` and `table_review_required=true`. They can be exported for review but are blocked from Day 3 indexing until reprocessed appropriately. Fixed chunks may divide table context and are flagged for review. The chunking path itself performs no embedding or upsert. No Groq, reranking, or active LangGraph workflow is added on Day 3.

## Configuration

### Split large PDFs before uploading

Use the standalone script with the project's installed PyMuPDF dependency:

```powershell
.venv\Scripts\python.exe scripts/split_pdf.py "C:\path\catalog.pdf" --output-dir "C:\path\catalog_parts"
```

Each output contains **up to 500 pages** by default (`--max-pages` changes this) and is strictly below **200 MB (200,000,000 bytes)**. The last part contains the remaining pages; parts may be smaller if the size limit requires it. It preserves page order and content, uses lossless compression, verifies serialized sizes, and refuses overwrites. The original remains unchanged. A single oversized page produces an error rather than silently reducing image quality. Output parts are independent PDFs; original document-level bookmarks/signatures are not retained. Splits use page boundaries and may separate a test across files.

For LabRAG uploads, the application's default limit is **20 MiB per file**, so use `--max-mb 20` instead of the script's 200 MB default unless you have raised `MAX_FILE_BYTES` and Streamlit's upload ceiling.

`.env.example` provides the requested embedding model, dimension 384, Groq model, index `lab-rag`, and namespaces `fixed-chunks` and `section-chunks`. Secrets are optional and hidden from UI/log representations. Invalid integer settings display a warning and the UI uses development defaults. `CHUNK_SIZE=500`, `CHUNK_OVERLAP=75`, and `SECTION_MAX_TOKENS=500` configure local defaults; overlap must be nonnegative and smaller than both limits. Counts use tiktoken `cl100k_base`, an approximation for Sentence Transformers. Day 3 must check the actual embedding tokenizer and context window to avoid truncation. The known default model dimension is validated without downloading a model; custom models require verification before future indexing. `MAX_FILE_BYTES` controls upload validation. Streamlit also has its own upload ceiling.

## Structure and technology

`config/settings.py` loads python-dotenv configuration. `src/ingestion/` and `src/chunking/` retain document processing. `src/embeddings/` provides lazy configured embeddings. `src/vectorstore/` owns explicit connections and indexing. `src/retrieval/semantic_retriever.py` retrieves Pinecone passages, with bounded read-only context expansion for chat. `src/llm/` supplies lazy Groq structured selection; `src/rag/` contains the workflow, evidence and citation checks. `src/ui_day3.py` preserves indexing/search/comparison; `src/ui_day4.py` adds session-only chat. Autonomous agents remain future work.

`data/source_manifest.csv` contains only ingested documents. `data/planned_topics.csv` is the unverified collection plan; it contains no invented source URLs or clinical instructions. Uploaded raw files and processed JSON are ignored by Git. Review the manifest before committing it, since local saves intentionally update this metadata artifact.

## Verification

```powershell
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py --server.headless true
```

Tests cover Day 1 ingestion, configuration, metadata, and local persistence plus units/ranges/tables, source coverage, token budgets, overlap, section fallback, stable IDs, corrupted JSONL, replacement, validation, comparison statistics, and the preserved pipeline modules. Streamlit AppTest checks startup and all three tabs without keys, and processing/comparison/reload in an isolated temporary workspace. CI installs complete requirements and runs pytest on Python 3.11. The development requirements add LangGraph only for regression tests of the inactive earlier pipeline.

Day 3 tests fake the model backend/Pinecone to verify dimensions, finite numbers, batch/cached embeddings, index configuration, namespace filters, source preservation, idempotency, checkpoint recovery, scoped deletion, source-overlap prevention, search, and UI controls. Run `python scripts/check_embeddings.py` for an explicit real local model smoke check. The live integration test is skipped unless `LABRAG_RUN_INTEGRATION=1`; it uploads labeled synthetic text to unique test namespaces and removes only its own recorded IDs afterward. Enable it only when intentionally testing your account: `python -m pytest -q -m integration` with that environment flag set. It does not create an index.

## Seven-day roadmap

| Day | Deliverable |
| --- | --- |
| 1 | Local ingestion, metadata, samples, UI, tests |
| 2 | Fixed-size and section-aware chunking comparisons |
| 3 | Local embeddings and explicit Pinecone indexing |
| 4 | Retrieval, Groq generation, LangGraph, citations |
| 5 | Reranking, abstention, source conflict handling |
| 6 | Recall@5, MRR, faithfulness, latency evaluation |
| 7 | Integration testing, documentation, course demonstration |

## Limitations and privacy

Use public permitted documents or synthetic fixtures only. Do not upload PHI, real patient records, or secrets. Chat selects source-verified excerpts through Groq; it is not a validated clinical decision-support system. Answers must identify laboratory and document sources and avoid unsupported clinical decisions. Allowlisted chunk metadata excludes arbitrary credential fields; this cannot detect sensitive text that a user uploads.

LabRAG is an educational laboratory reference assistant. Verify specimen collection and handling instructions against the current official laboratory test catalog before clinical use.

PDF tables are best-effort text; scanned PDFs need external OCR. HTML/TXT must be UTF-8. Section detection uses configurable full-line patterns; unfamiliar headings fall back to fixed splitting per page. Pages are never merged, so cross-page tables need human review. Multi-space/tab/pipe tables trigger conservative review flags; plain PDF text may lose table structure before chunking. Repeated clinical headings are retained. Only repeated, explicit boilerplate/page counters at page edges are removed confidently. Exact coverage checks refer to cleaned source text, not clinical equivalence with the original layout.

CSV persistence is for a trusted single-user development process, without authentication or concurrent-writer guarantees. Raw and processed file writes and the CSV update are not a cross-file transaction; a disk failure may leave orphaned files. Duplicate source identity is URL + provider + test; content hashes also prevent duplicate bytes. Metadata updates are available through `MetadataManager.update`, not through a UI editor. Reprocessing the same document/strategy atomically replaces its JSONL snapshot; changed content updates chunk IDs. Re-uploaded changed raw documents have new document IDs, so manually retire obsolete document records before Day 3 rather than mixing revisions.

See [architecture](docs/architecture.md), [Pinecone setup/troubleshooting](docs/pinecone_setup.md), [Day 3 progress](docs/day3_progress.md), [chunking strategy](docs/chunking_strategy.md), and [source collection guidance](docs/data_sources.md).
## Grounded conversational v2: verified evidence coverage

`grounded_conversational_v2` is the default for fresh chat sessions and the local
API. In **Chat Settings → Answer generation mode**, select **Grounded conversational
v2 (evidence coverage)** explicitly if an existing session still has v1 selected.
`grounded_conversational_v1` and `extractive_v1` remain available; historical v1
behavior is preserved. Day 5 still defaults to extractive mode, with v2 available
as a separate explicit evaluation choice.

V2 preserves the LLM's original selection and exact-quote validation. Before
conversational rendering, it checks required typed facts per test. Missing facts
can be supplemented only from already retrieved, identity-scoped public Mayo
catalog or explicitly synthetic passages. The application adds exact existing
source spans, verifies their mapping again, and cites the sources actually used.
There is no extra retrieval, provider call, source reassignment, or inference from
test titles/tube names. Missing facts, conflicting explicit values, unsupported
layouts, and untrusted supplemental sources remain unanswerable. Empty or invalid
original selections are never repaired.

For example, an LLM-selected BUN tube/volume passage can be supplemented with the
already retrieved table explicitly identifying Serum. Stability requirements keep
refrigerated and frozen conditions distinct. The parser remains conservative;
long sources are scanned in bounded application-owned spans, so split/unrecognized
tables may still lead to abstention. Original selections remain available for audit.

Privacy-safe diagnostics appear under **Supporting Documents → View Retrieval
Details → Answer synthesis validation → evidence_coverage** and in existing saved
synthesis diagnostics. They contain missing fact kinds, hashed test references,
supplemental source IDs, offsets, quote hashes, and conflict/status flags; no new
raw quote or question text is logged by this step.

Generation mode is already included in evaluation signatures and reports. The
pipeline fingerprint now also includes the coverage module. Historical reports,
datasets, and migration allowlists are unchanged. An older checkpoint can reject
resume after this code change; do not bypass validation or reinterpret its scores
as v2 results. Compare answer quality only within a stated generation version.
No evaluation is started automatically. Restart the API to load the new default.

Offline coverage tests: `python -m pytest tests/test_evidence_coverage.py -q`.

V2 also supports the verified Calprotectin fecal stability-table layout and its
three explicit fresh/no-preservative, immediate refrigerated-send, and delayed
frozen-send instructions. Collection-and-storage questions require both kinds of
evidence. Bordetella respiratory specimen fields retain the explicit Preferred/
Acceptable headings and the `not throat` exclusion. These alternative roles are
not competing values; conflicting values within one role still abstain. Plural
accepted-specimen questions require both roles when this supported layout is
present; missing alternatives are not invented. These narrowly supported layouts
are enabled only in v2. V1 parsing remains unchanged. Coverage diagnostics use
`verified_selection_coverage_v2`, and the existing source-code fingerprint records
the implementation revision without rewriting historical results.

Offline saved-passage replays:
`python -m pytest tests/test_calprotectin_bordetella_layouts.py -q`.

# Local API for Lab Agent

### Server answer provider and request diagnostics

The answer provider defaults to **OpenAI**, model **gpt-4.1-mini**. The existing
Groq integration remains available; Groq is not Grok/xAI. The API route calls
`LocalRAGService.query` → `Settings.from_env()` → `create_llm` → the shared RAG
graph. No Lab Agent request field, intent-router choice, or Streamlit session
setting overrides this server-side provider.

Use these server `.env` settings (keep your existing key private):

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your-private-server-key
```

Alternatively remove `LLM_PROVIDER` and `LLM_MODEL` to use defaults.
`OPENAI_LLM_MODEL` supplies the OpenAI model fallback when `LLM_MODEL` is absent;
an explicit `LLM_MODEL` wins. Remove an old Groq `LLM_MODEL` when relying on the
OpenAI fallback. This repository's loader intentionally lets `.env` values override
same-named process environment values. Set `PYTHON_DOTENV_DISABLED=1` only if you
want process environment configuration alone. Do not remove embedding settings.

Explicit Groq override: `LLM_PROVIDER=groq`,
`LLM_MODEL=openai/gpt-oss-120b`, and `GROQ_API_KEY` on the server. There is no silent
provider fallback. Missing selected-provider credentials return HTTP 503 with a
fixed actionable configuration message before connecting to Pinecone.

The embedding provider/model/dimension and Pinecone namespaces are unchanged.
OpenAI answers do not imply OpenAI embeddings. The separate Lab Agent intent
router is not modified. Existing evaluation provider selections and signatures
are not bypassed; historical reports are unchanged.

Restart the API from the project root after saving `.env` (stop its current process
with Ctrl+C first):

```powershell
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Successful query responses retain the existing JSON contract and now include
diagnostic headers: `X-Request-ID`, `X-LabRAG-Provider`, `X-LabRAG-Model`,
`X-LabRAG-Retrieval-Seconds`, `X-LabRAG-Generation-Seconds`,
`X-LabRAG-Generation-Executed`, and `X-LabRAG-API-Seconds`. Unexecuted/unmeasured
stages can omit their timing headers; generation is zero/false for evidence-gate
abstentions. Provider/model identify the selected server service, not proof that
generation executed. The server also writes one `labrag_request` JSON log event
with request ID, HTTP status, provider, model, measured times and execution flag.
These logs contain no question, answer, evidence, credential or exception text.
Existing optional LangSmith demo-content capture remains a separate configuration.

Retrieval time is the graph's retrieval node (embedding/search/context preparation),
generation time is its LLM excerpt-selection node. Citation validation and grounded
rendering are included in total API processing time. Total API time also includes
client initialization, queue wait and response adaptation; it excludes client
network transfer. Failed requests may have only total API time and selected-service
identity because completed stage timings are unavailable. No live latency comparison
has been performed; mocked test durations are not performance measurements.

After an explicitly authorized test request, inspect the headers in PowerShell:

```powershell
$result = Invoke-WebRequest -UseBasicParsing -Method Post `
  -Uri 'http://127.0.0.1:8000/api/v1/query' -ContentType 'application/json' `
  -Body (@{question='What specimen is required for BUN?'} | ConvertTo-Json)
$result.Headers
$result.Content | ConvertFrom-Json
```

Use `X-Request-ID` to correlate the response with its server log event. Check
`X-LabRAG-Provider: openai`, the chosen model, and the generation-executed flag.
`GET /health` remains a liveness check and does not verify provider credentials.

The API reuses the same UI-free chat service and LangGraph pipeline as Ask
LabRAG. It does not ingest, index, or execute evaluations. Streamlit still runs
separately with `streamlit run app.py`.

Install dependencies from the project root (Python 3.11+):

```powershell
python -m pip install -r requirements.txt
# For this project's uv-managed environment without pip:
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Keep credentials in the existing server-side `.env`. The API uses `Settings.from_env()`:
the existing LLM provider/model (Groq by default), embedding model, Pinecone index,
strategy-specific namespace, score threshold, and top-k. Runtime selections made
inside Streamlit are session-specific and do not change the API configuration.
Restart the API after changing `.env`.

Start from the LabRAG project root:

```powershell
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Base URL: `http://127.0.0.1:8000`. Interactive schema and testing:
`http://127.0.0.1:8000/docs`. Keep this process running while Lab Agent uses it.
This is a local demo: no authentication, CORS, or public hosting is configured.

- `GET /health`: liveness only. Returns
  `{"status":"ok","external_connectivity":"not_checked"}` without provider calls.
- `POST /api/v1/query`: independent, stateless question; no client conversation
  history or provider override. JSON request contains only `question`, a trimmed
  nonempty string of at most 3000 characters. Extra fields are rejected.

```powershell
$body = @{ question = 'What specimen is required for BUN?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/query' `
  -ContentType 'application/json' -Body $body
```

Illustrative response (source fields depend on actual retrieved evidence):

```json
{
  "status": "answered",
  "answer": "The specimen type is serum.",
  "citations": [{
    "source_id": "S1",
    "document_title": "Mayo laboratory catalog",
    "test_code": "BUN",
    "test_name": "Blood Urea Nitrogen (BUN), Serum",
    "printed_catalog_page": null,
    "pdf_page_1_based": null,
    "source_url": null
  }],
  "matched_tests": [{"test_code": "BUN", "test_name": "Blood Urea Nitrogen (BUN), Serum"}],
  "request_id": "server-generated-uuid"
}
```

The authoritative complete schema is available in `/docs` and `/openapi.json`.
Response status is `answered`, `needs_clarification` (only when explicitly supplied
by the underlying pipeline), or `insufficient_evidence`. Existing graph abstentions,
including unresolved identity and policy refusals, remain conservative insufficient
evidence responses; the adapter does not invent a clarification or a medical answer.
`answer` uses the validated conversational display text, with the original graph
answer as fallback. Citations come only from graph citations; context-only citations
retain `citation_role`. Matched tests come from citations and scoped evidence,
so they are evidence identities, not proof that an answer was established.

Every citation field is a nullable string: `source_id`, `document_id`,
`document_title`, `provider`, `test_code`, `test_name`, `printed_catalog_page`,
`pdf_page_1_based`, `original_pdf_page_1_based`, `section`, `chunk_id`,
`source_url`, `citation_role`. Printed catalog pages are printed labels;
PDF pages are one-based positions in the uploaded PDF, with original PDF positions
provided separately when preserved by ingestion. Unknown values remain null;
the API never infers a conversion between printed and PDF pages.

Invalid input returns HTTP 422. Provider/configuration/retrieval failures return
503 (including rate limits); identifiable timeouts return 504. Upstream errors
that the existing pipeline combines as connectivity/timeout remain 503. Errors
contain only `error`, a fixed safe `message`, and `request_id`; no provider bodies,
credentials, or tracebacks are returned. `X-Request-ID` is also included in responses.
Provider retries/timeouts remain those of the existing pipeline. Synchronous
execution runs in worker threads; one service lock serializes requests and protects
reused clients and mutable provider usage state. Initialization is lazy, so the first
query may load the embedding model. There is no new whole-request timeout or queue
limit; local callers should allow sufficient time for model loading and existing retries.

Offline checks: `python -m pytest tests/test_api.py tests/test_day4_ui.py tests/test_conversational_answers.py -q`.

Integration handoff for a separate Lab Agent project:

> Use the already-running LabRAG service at http://127.0.0.1:8000. POST
> /api/v1/query with JSON {"question":"<laboratory question>"}. Consume the
> typed status, answer, citations, matched_tests and request_id; preserve source
> references. Treat insufficient_evidence/needs_clarification as unresolved, not
> as a medical answer. Handle 422 input errors and 503/504 upstream failures
> separately. Do not automatically retry chargeable queries; let the user retry.
> GET /health verifies liveness only. Inspect /openapi.json for the full schema.
> LabRAG must remain running. Do not access its .env or duplicate its RAG pipeline.


### Service endpoint configuration

Set `OPENAI_BASE_URL=https://api.openai.com/v1` and
`LANGSMITH_ENDPOINT=https://api.smith.langchain.com` in `.env`.
These configure OpenAI answer generation and LangSmith tracing respectively;
missing or blank values retain the official service defaults. Public catalog
source URLs remain document metadata. Restart Streamlit and the API after edits.
Only use trusted service endpoints: the corresponding API credential is sent
to the configured service. This does not change the embedding endpoint.
