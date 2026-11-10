# Runtime Groq and OpenAI providers

Groq remains the default in both chat and evaluation controls. OpenAI is an
explicit alternative, never an automatic fallback. Selecting a provider or
editing a model initializes no clients and makes no requests. Existing embedding
provider, Pinecone namespaces, retrieval, reranking, prompts and citation checks
are unchanged.

## Configuration

Existing `.env` keys `GROQ_API_KEY` and `OPENAI_API_KEY` are reused. `LLM_MODEL`
retains the configured Groq model; `OPENAI_LLM_MODEL` provides the OpenAI selector
default (`gpt-4.1-mini`). No key is displayed or written to reports. Provider
selection does not change these defaults or require restarting the app. Active
services receive immutable settings with explicit provider/model values.

OpenAI API usage may incur charges and is billed separately from ChatGPT.
No prices, costs or quota availability are guessed. Model availability, lifecycle,
account permissions, billing and quotas are not verified offline. The configured
model may require changing based on your account's current supported access.

## Service design and strict source grounding

`src/llm/provider_factory.py` exposes `create_llm(settings, backend=None)` and
lazy `build_backend`. `OpenAIService` reuses the existing Groq service's selection,
verified-excerpt lookup, bounded transport handling and sanitized schema failures.
The same AnswerSelection reaches the unchanged graph and citation validator.

Groq uses LangChain function calling. OpenAI uses `langchain-openai` with strict
`json_schema` output. Its schema is independently normalized: all object
properties are required, objects forbid extra fields, and schema defaults are
removed. Pydantic and per-request excerpt-ID verification still run locally.
No unknown ID, rewritten quote or cross-source quotation is accepted. Malformed
output fails without changing provider or relaxing the schema.

A conservative OpenAI model registry enables tested-capability GPT-4.1/GPT-4o
aliases and selected snapshots. Unsupported model IDs fail locally; account
availability is still enforced by the provider. No automatic JSON-mode fallback
or model substitution occurs. Requests use the standard OpenAI endpoint, not an
embedding-specific/custom OPENAI_BASE_URL.

Official references: [strict function schemas](https://developers.openai.com/api/docs/guides/function-calling)
and [GPT-4.1 mini capabilities and availability](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

## Chat

Use LLM Provider above the conversation, then Chat model inside collapsed Advanced
Chat Settings. Model edits are remembered separately per provider across reruns.
Switching preserves conversation and does not regenerate earlier turns. Each new
answer/error records its selected provider and model. Earlier turns without that
metadata say it was not recorded. Samples fill an editable draft; only Send
submits it. Follow-ups use the existing user-question context resolver.

Groq 429 messages suggest waiting or explicitly selecting OpenAI. OpenAI rate
limits, authentication, model/context errors, connection failures and billing
restrictions receive fixed provider-specific diagnostics. Raw errors, headers,
provider bodies, prompts and keys are never copied into error diagnostics.

## Evaluation and judges

Advanced Evaluation Settings > Configure Day 5 evaluation exposes independent
answer provider/model and optional judge provider/model controls. Selectors are
outside the execution form so changes rerender safely before submission. The judge
is off by default. When enabled, its provider/model must be explicit; it never
silently inherits answer settings. Claim/rubric prompts and deterministic metrics
are unchanged. Both providers support optional judges through the same factory.

Provider/model already participate in the signed evaluation models configuration.
Judge identity is signed when enabled. They are fixed throughout the invocation;
changing them on resume fails validation, even with recovery migration enabled.
Historical Groq-only reports remain readable and unchanged. New reports and CSV
exports include answer provider/model, and judge results identify their provider.
Existing backup, writer lock, explicit initiation and cooldown checks remain.
Different provider benchmarks must use separate runs; never edit a signature.

Historical pipeline fingerprints can fail closed after this provider-layer change.
Only existing explicitly allowlisted recovery migrations remain available, with
every other signed field matching. An unrecognized pipeline requires separate
review, not a bypass. Installation does not resume or rewrite historical runs.

Available response usage metadata is reduced to nonnegative integer input/output/
total token counts plus provider/model. Raw model messages are discarded by the
service. Answer usage and supplemental judge usage are separate; missing counts
stay null. These are successful-response counts, not a total across unsuccessful
requests or a cost estimate.

## Errors and pacing

Transient HTTP/connection errors retain at most three attempts and the existing
run-wide pacing/cooldown handling. Authentication, invalid request/model, schema
and OpenAI insufficient-quota/billing errors are not automatically retried.
HTTP 429 rate limits pause the run. OpenAI billing-related 429 pauses as Paused —
Provider Configuration before another pair. Fix billing/credits/access before an
explicit retry. The system never switches providers automatically. Quota limits
are not assumed and paid calls never start merely by rendering this UI.

## Manual validation (requests require your explicit action)

1. Reload the idle app. Verify Groq is selected and its existing model is retained.
2. Switch to OpenAI without submitting; inspect the model and cost notice. Confirm
   history persists. Edit each provider's model and switch back to verify retention.
3. Select a verified catalog sample, edit if desired, then press Send when ready
   to authorize a request. Review provider/model, exact quotations and citations.
4. Switch explicitly and ask a follow-up. Earlier answers must remain unchanged.
5. For evaluation, configure a separate new run for a different provider. Keep
   A/B/C/D and retrieval parameters identical for a deliberate comparison. Leave
   the judge off unless separately selected and authorized.
6. Review provider/model and token metadata in exports. Do not submit or resume
   evaluation merely to check the selector. Never change provider on an old run.

Offline tests mock all external APIs. They verify defaults, switching, persistence,
key errors, strict schemas, quote/citation preservation, safe provider failures,
usage extraction, judge independence, resume locks/signatures and historical reads.
