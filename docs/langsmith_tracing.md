# Optional LangSmith tracing

LabRAG uses the supported Python SDK `traceable` decorators and scoped
`tracing_context` integration. No changes were made to retrieval, prompts,
providers, relevance labels, scoring, or indexed data.

## Configuration

Install the updated requirements, then add these settings to the existing local
`.env` (never commit the real key):

```dotenv
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_CAPTURE_FULL_IO=false
LANGSMITH_PROJECT=LabRAG
```

Tracing defaults to false. A missing key disables remote tracing. Invalid boolean
values produce an actionable settings error. Configuration is read through the
existing `Settings.from_env()` loader, not copied into global environment variables.
Do not export global LangChain/LangSmith tracing flags for this demo; the application
sets its own scoped tracing context. `RAG_ENABLE_TRACING` remains the separate
existing local node-timing display setting.

Restart Streamlit and/or the API after changing configuration. No LangSmith account
or project was created and no remote tracing was performed during implementation.

## Trace layout

Both Streamlit and FastAPI invoke the same graph. Day 4/Day 5 RAG evaluations also
use that graph and therefore receive these spans when explicitly run:

| Span | Existing operation |
| --- | --- |
| LabRAG workflow | Question input and final response summary |
| validate_question | Input and policy handling |
| resolve_follow_up | Existing conversation-context resolution |
| retrieve_evidence | Pinecone retrieval and existing neighboring context expansion |
| rerank_evidence | Optional existing reranking, when enabled |
| validate_evidence | Scoped context preparation and evidence gate |
| generate_answer | Groq/OpenAI selection, including existing retries; LLM span |
| validate_citations | Strict existing quote/citation validation |
| synthesize_answer | Existing grounded conversational rendering, when enabled |
| abstain | Existing insufficient-evidence/policy path, when taken |

Standalone retrieval-only evaluation, optional judge calls outside the RAG graph,
and ingestion are not newly instrumented. The current application uses synchronous
`invoke`; delegated streaming/async graph entry points are not instrumented by this
wrapper. This feature does not start evaluations or make additional LLM calls.

## Privacy boundaries (default: capture full I/O is false)

Only explicit custom sanitized spans are sent. Automatic nested LangChain,
LangGraph, embedding and provider tracing is suppressed during execution to prevent
raw state, prompts, messages, passages or provider bodies from being uploaded.
The only question text permitted in traces is the exact preapproved public-catalog
demo question `What specimen is required for BUN?` (TC01.1 in the unchanged v3
dataset). All other questions are represented by `[withheld]` and character count;
history is never uploaded. No patient information should be entered in LabRAG.

Trace outputs contain status, answer length (not answer text), citation count,
selected-excerpt count, evidence sufficiency and allowlisted quote-validation outcome.
The configured provider (`groq` or `openai`) is recorded as safe trace metadata.
Document metadata contains SHA-256 references to document/chunk identifiers and
numeric page positions only. Names, filenames, source URLs, arbitrary metadata,
test names, passages and selected quotes are withheld. Settings and credential-bearing
objects are never decorator arguments. SDK runtime environment metadata is disabled.
Project names must be non-sensitive. The endpoint is fixed to the standard US
LangSmith API; regional/self-hosted endpoint support is not included.

## Authorized development/demo I/O

The prior BUN-only exception explains why the specimen question was visible while
the BUN stability question was withheld. This was the same filter for every
provider, generation mode and RAG entry point; it was not an OpenAI or v2 issue.

For an explicitly authorized development/course demo using only public reference
or synthetic data, set:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_CAPTURE_FULL_IO=true
LANGSMITH_PROJECT=LabRAG-Demo
```

This is an opt-in authorization to upload demo content to LangSmith. Keep the key
only in `.env`. Restart both dashboard and API processes after changing these settings.
The setting does not enable tracing by itself and does not run any query.

Full capture uses an exact question allowlist loaded from the repository-owned
`evaluation/mayo_day4_cases.json` and `evaluation/mayo_day5_v3_overlap15_ground_truth.json`
without changing either file. Both **What specimen is required for BUN?** and
**How long can BUN serum be refrigerated or frozen?** are approved. Unknown questions,
and requests with unknown prior user questions in their history, retain default
filtering even when this flag is true. The cached bank reloads at process restart.
This deliberate restriction prevents arbitrary patient narratives from becoming
eligible merely by toggling an environment variable.

For approved demo requests, custom traces contain complete question and final
answer text, conversational display/extractive versions when available, structured
citations and allowlisted source references, generation mode, configured provider,
node timings, quote-validation metadata and v2 evidence-coverage diagnostics.
No raw history, prompts, provider bodies, credentials, arbitrary document metadata,
or additional raw retrieved passages are uploaded. Actual extractive answers can
contain the validated source quotations because those are the generated answer.
Automatic native LangChain/provider tracing remains disabled in both modes.

Only public Mayo catalog or explicitly synthetic source metadata is eligible;
private/unclassified sources and recognizable patient metadata/PHI markers suppress
full output capture. The same check is applied to stage inputs and outputs so an
answer-only child span cannot bypass a sensitive source decision. Exact configured
credential values and common token formats are redacted, citation fields are
allowlisted, and URL userinfo/query strings/fragments are removed. Errors continue
to record only fixed failure flags, never exception bodies or tracebacks.

These safeguards are **not a comprehensive clinical de-identification system**.
Use only genuinely non-sensitive public/synthetic material, keep the repository
question bank reviewed, and never mark patient documents as synthetic. Unrecognized
PHI in incorrectly classified content cannot be guaranteed detectable. Set the flag
back to false outside the approved demo. Existing traces are not rewritten or
deleted when the setting changes.

Offline tests intercept SDK create/update transport calls and verify that both BUN
questions and their answers would be submitted across Groq/OpenAI and all three
generation modes. They do not prove server-side ingestion, permissions, or account
connectivity. Use the manual verification below for an authorized live check.

Application exceptions are recorded only as a fixed `execution: failed` flag;
the original exception is re-raised locally. Thus traces do not include exception
text/tracebacks or identify provider failure categories. Existing safe evaluation
diagnostics remain available locally. Tracing failures do not retry application
operations. Tracing uses the SDK's normal background delivery, so an abrupt process
exit can lose pending telemetry. No delivery/connectivity claim is made by `/health`.

## Manual tracing verification (explicit external calls)

1. Set the three settings above and restart the relevant application.
2. In Ask LabRAG, keep the desired existing Groq/OpenAI provider and model. Submit
   **What specimen is required for BUN?** This is an existing verified catalog
   question; do not use patient names, results or history.
3. Alternatively start the API with the README command and run:

   ```powershell
   $body = @{ question = 'What specimen is required for BUN?' } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/query' `
     -ContentType 'application/json' -Body $body
   ```

4. Open https://smith.langchain.com, select the configured workspace, open
   **Tracing Projects**, select **LabRAG** (or your configured project), and open
   the newest **LabRAG workflow** trace. Allow background delivery a moment.
5. Check child retrieval/context/generation/citation spans, parent links, timings,
   source counts and page summaries. Confirm inputs/outputs contain no raw source
   text or secrets. A successful answer depends on existing credentials, quota and
   available evidence; an abstention should show only the stages actually executed.
6. Set `LANGSMITH_TRACING=false` and restart to disable uploads.

## Evaluation checkpoint compatibility

The existing checkpoint signature hashes the graph source. Adding instrumentation
changes that fingerprint even though retrieval/scoring behavior is unchanged.
Historical reports and datasets are untouched and still load. Existing checkpoint
resume validation is deliberately unchanged: an old run may reject resumption
because of the pipeline fingerprint. Do not bypass it or check recovery migration
without a separate compatibility investigation. No migration allowlist was expanded
and no evaluation was resumed by this change.

## Offline verification

```powershell
.venv\Scripts\python.exe -m pytest tests/test_langsmith_tracing.py -q
.venv\Scripts\python.exe -m pytest -q -m 'not integration'
```

Tests use real SDK decorators with mocked transport, mocked retrievers and mocked
generation. They check configuration, disabled/missing-key behavior, nested spans,
redaction, unchanged outcomes, exception propagation and no duplicate provider calls
when tracing fails. No live LangSmith, Pinecone or LLM calls are required.

Supported SDK references:
[traceable](https://reference.langchain.com/python/langsmith/run_helpers/traceable),
[tracing_context](https://reference.langchain.com/python/langsmith/run_helpers/tracing_context).
