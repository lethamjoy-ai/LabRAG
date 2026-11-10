# LabRAG — Clinical Laboratory Knowledge Assistant

Gen Academy Agentic AI Master Class · Week 2. A modular Streamlit RAG prototype for asking evidence-grounded questions of laboratory reference documents.

## Quick start

Python 3.11 or newer is required. From this directory in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with your credentials
streamlit run app.py
```

On macOS/Linux use `python3 -m venv .venv`, `source .venv/bin/activate`, and `cp .env.example .env`. Alternatively use `uv venv --python 3.11`, then `uv pip install -r requirements.txt`.

Create a **dense cosine Pinecone index** in your account before connecting. Its dimension must match the embedding model: the default `sentence-transformers/all-MiniLM-L6-v2` produces 384 dimensions. Configure `PINECONE_INDEX_NAME` and both API keys in `.env`. Enable the sidebar connection. The application validates dimension, readiness, and metric. It does not create or delete cloud resources automatically.

First launch downloads the embedding and cross-encoder models from Hugging Face; allow disk space and outbound network access. Ingestion sends text embeddings and source text to Pinecone. Questions and retrieved excerpts go to the selected LLM provider. Pinecone and inference may incur charges. Use trusted public reference content, without patient information.

## Demo walkthrough

1. Open **Ingest documents** and upload `data/source_manifest.json` as the manifest.
2. Upload `data/demo_lab_reference.txt` and choose **Parse, embed, and index**.
3. Wait briefly for Pinecone consistency, then ask which container the synthetic test requires.
4. Inspect numbered source excerpts, page numbers, retrieval scores, and reranking scores.
5. In **Evaluate**, upload `data/evaluation_cases.json`, run evaluation, inspect charts, and download CSV.

The included fixture is entirely synthetic and contains no clinical guidance. Add your own licensed, reviewed laboratory references for meaningful domain evaluation.

## Configuration

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Required with default Groq provider |
| `PINECONE_API_KEY` | Required for retrieval and ingestion |
| `PINECONE_INDEX_NAME` | Existing dense cosine index; default `labrag` |
| `EMBEDDING_MODEL` | Sentence Transformer model identifier |
| `LLM_MODEL` | Provider-specific model identifier |
| `LLM_PROVIDER` | `groq` (default) or `openai` |
| `OPENAI_API_KEY` | Required for `openai` provider |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible API endpoint |
| `PINECONE_NAMESPACE` | Corpus namespace; default `lab-reference-v1` |
| `RERANKER_MODEL` | Sentence Transformers cross-encoder identifier |

Choose an available model in your provider account. For OpenAI change both provider and model. Credentials are excluded from dataclass representations and UI errors; `.env` is gitignored. Never commit secrets.

## Architecture

```text
Uploaded PDF / HTML / TXT + optional manifest
  → parse with provenance → conservative cleaning → overlapping chunks
  → Sentence Transformer embeddings → batched Pinecone upserts

Question → LangGraph retrieve → cross-encoder rerank → prompt + chat model
         → citation-label validation → answer and inspectable evidence
```

`labrag/ingestion.py` uses PyMuPDF (one-based PDF pages), BeautifulSoup (heading lines and table rows), and LangChain recursive splitting. Cleaning removes NULs, excess horizontal whitespace, and excessive blank lines while preserving signs, units, decimals, and paragraph structure. PDF tables are best-effort text; scanned images need external OCR. TXT and HTML require UTF-8. Uploads are capped at 20 MB per document. Header/footer heuristics are deliberately limited to HTML navigation and footer elements to avoid deleting clinically important text.

`labrag/retrieval.py` embeds normalized vectors, stores chunk text and scalar metadata, and reranks the top 12 candidates into 4 excerpts. Stable chunk hashes make identical reingestion idempotent. Changed text, provenance, or chunk settings creates new records: use a **new namespace for corpus revisions** to avoid stale records. Metadata includes the embedding model and retrieval filters enforce model identity. Nested manifest metadata is not stored in Pinecone. Pinecone is eventually consistent.

`labrag/graph.py` uses current `StateGraph`, `ChatPromptTemplate`, and model `invoke` APIs rather than deprecated chains. It abstains on empty retrieval or absent/out-of-range citation labels. Evidence is explicitly treated as untrusted data. Citation labels alone do not establish factual support, and relevant retrieval is not guaranteed; review original documents. There is no calibrated relevance threshold. Queries are independent; previous questions are not sent as conversation context.

`labrag/evaluation.py` reports source recall@4, reciprocal rank, nDCG@4, expected-term coverage, citation-format validity, abstention, and latency. These deterministic proxies do not assess medical correctness or claim entailment. Expected sources should match the stored `source` URL or filename exactly. Evaluation needs nonempty source and term lists. Review generated answers manually and extend the dataset with real laboratory reference questions and negative cases before practical use.

`labrag/ui.py` supplies upload, question, evidence inspection, and evaluation views. Cache lifetime is the Streamlit process; uploaded originals are not written to disk by the app. Indexed chunks persist remotely. The prototype has no authentication or tenant isolation; host only for trusted users, with service permissions controlled outside the app.

## Source manifest

The manifest associates uploaded local filenames with curated provenance; URLs are **references**, not automatically fetched. Download documents lawfully and upload them alongside their manifest. No crawler runs.

```json
[
  {
    "filename": "reference.pdf",
    "test_name": "Your laboratory test",
    "provider": "Source laboratory",
    "source_url": "https://your-laboratory.example/reference",
    "metadata": {"version": "2026-01", "population": "adults"}
  }
]
```

Each filename must be unique. Entries require all four string fields and HTTPS source URLs. Without a matching entry, the uploaded filename becomes the source. Test filters match `test_name` exactly.

## Development and verification

```powershell
pytest -q
python -m compileall -q labrag app.py
```

Tests cover parsing (including PDF pages), chunk identity, manifest validation, secret redaction, workflow generation, abstention, citation checks, and evaluation using fake chat responses. They need no API credentials or downloaded embedding models. GitHub Actions runs the suite on Python 3.11. Live Pinecone, model downloads, and Groq/OpenAI require an independently configured integration smoke test using the demo walkthrough above. Dependency ranges constrain major upgrades; create a lockfile for your deployment after resolving the complete stack.

## API references

- [LangGraph graph API](https://reference.langchain.com/python/langgraph/graph)
- [LangChain Groq integration](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [Pinecone Python SDK](https://sdk.pinecone.io/python/)

This educational application is not a validated clinical decision support system. Reference ranges vary with laboratory, population, method, and units; verify the exact source and local policy.
