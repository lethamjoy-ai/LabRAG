# Day 4 progress

Implemented lazy LangChain ChatGroq structured selection, the seven requested LangGraph
nodes and their conditional abstention route, retrieved-source checks, citations, bounded
follow-up history, Streamlit chat, clear conversation, model/strategy/top-k controls,
latency/count panels and optional workflow timing trace.

Pinecone dimensions, names, namespaces, existing vectors, local chunks and indexing
manifest were preserved. Only read-only neighbor retrieval was added for chat. Missing
Day 4 settings were appended to the existing .env without replacing any existing values.
No credentials were displayed or placed in documentation.

`scripts/validate_day4_corpus.py` verified the 13 supplied complete reference cases against
the accessible 500-page local PDF, including requested/minimum volumes, preparation,
storage, non-orderability and total/ionized distinctions. It records the PDF SHA-256,
literal supporting excerpts, printed labels and separately verified PDF ordinals in
`evaluation/mayo_day4_cases.json`. The attached prompt ends at TC14 'User'; no missing
TC14 content was inferred. These are reference cases, not measured live RAG scores.

Validation uses pytest, injected Groq responses, fake retrieval, Streamlit AppTest and
optional local PDF verification. No live Groq inference, live Pinecone retrieval, cloud
mutation, paid embedding request or reindexing was performed during implementation.
Live model access/availability and end-to-end retrieval accuracy remain to be checked
with the configured account. An invalid/unavailable model returns an explicit error.

Final regression result: **149 passed, 1 optional integration test skipped**. All 13
reference questions also passed the identity/facet gates when supplied complete relevant
page neighborhoods from the existing local section JSONL snapshot. That offline check
assumes the relevant pages were retrieved; it does not measure Pinecone recall or Groq
answer selection. Chat submission/citation rendering and clearing history passed AppTest.

For a manual read-only smoke test, open Ask LabRAG, leave section/top-k 5 selected, ask
“What specimen is required for BUN?”, then “How long can it be frozen?”. Inspect the
source excerpts, correct test code, printed versus PDF pages, and timings. Try an unknown
test to check abstention, then clear the conversation. Do not use indexing controls for
this Day 4 test. Compare real outputs against the reference cases rather than assuming
that the PDF's presence guarantees those passages exist in the selected namespace.

Known limitations: deterministic identity/facet gates favor abstention and are not a
clinical entailment engine. Incomplete indexed page context, unusual PDF layouts or
unresolved metadata can prevent answering. Groq's excerpt selection may omit qualifiers;
review cited instructions. Conflict reporting is best-effort. There is no reranker,
autonomous browsing, persistent conversation store or quality benchmark in Day 4.

## Small-chunk retrieval correction

The reported question “What Specimen type is required for Comprehensive metabolic panel ?”
exposed overly strict identity-word filtering and heading pages falling outside the three
page-seed expansion limit. Chat now separately searches for the test name/code, prioritizes
matching heading pages, and excludes generic instruction words from the identity check.
No chunk-size change or reindex is required for this correction.

A regression fixture using 100-token section chunks with overlap 10 covers a lower-ranked
heading, requirements in separate chunks, next-page minimum volume and exclusion of the
next test. Full regression result after correction: **151 passed, 1 skipped**.

A live read-only Pinecone check of the exact question returned 5 instruction matches,
5 identity matches and 81 expanded evidence views, narrowed to 8 CMAMA sources. The
retrieved text includes preferred serum gel, acceptable red top, amber submission vial,
requested volume 0.6 mL (printed 926), and minimum 0.5 mL (printed 927). Both identity and
instruction gates passed. This verifies retrieval for this query; Groq answer selection
was not invoked. Existing cloud vectors, namespaces and local indexing state were unchanged.

## Semantic retrieval explorer correction

The explorer previously retained raw similarity-only search while chat used test context.
It now defaults to **Retrieve test context**, applying identity search, indexed-page
expansion, test identity scoping and requested-instruction filtering. Results are limited
to the selected top-k and sorted by full-question cosine similarity. Exact user filters
and the public/synthetic corpus remain in every search. Unchecking the option restores
raw search for inspection. No Groq call or indexing is performed by this control.

Live verification for the same Comprehensive Metabolic Panel question in section mode
returned four CMAMA instruction chunks on printed pages 926–927. The earlier raw search
returned generic specimen headings from unrelated pages. The corrected results cover
collection requirements, minimum volume, specimen type/stability and handling, while
withholding unrelated test regions. **153 tests passed, 1 skipped**, including explorer
UI defaults, raw-mode regression and corpus/filter preservation.

## Groq model availability correction

A live model-list check and synthetic request showed that the configured
`llama-3.3-70b-versatile` was unavailable to this account (`404 model_not_found`).
`openai/gpt-oss-120b` was available and passed a live structured-output test with an
exact synthetic supporting quote. The .env LLM_MODEL, example and default were updated
to that verified replacement. This model is called through Groq using GROQ_API_KEY;
the embedding provider and Pinecone configuration were preserved.

The sidebar updates a retained retired/default model value when configuration changes.
Provider errors now distinguish unavailable models, tool-generation failures, context
limits, invalid parameters, authentication and rate limits without exposing response
bodies or credentials. Regression result: **161 passed, 1 skipped**.

The live answer test also found that Groq removed PDF indentation in an otherwise
identical quote. Generation now restores the actual indexed substring only if all
non-whitespace tokens (including case, punctuation, values and units) remain identical.
Exact citation validation still runs afterward. Tests reject changed volumes, units,
preferred specimens and case changes. A final live Pinecone → Groq → citation-validation
run answered the CMAMA specimen question with two verified citations. Final regression
result: **162 passed, 1 skipped**. Cloud vectors and embeddings were unchanged.

## Submission-container question correction

The question “What is the Submission container for Comprehensive Metabolic Panel,
Serum ?” failed identity validation because “submission” was treated as a test-name
word. Instruction vocabulary is now excluded, and a literal complete test name can
establish identity without requiring every question word to occur in the name. Explicit
test-code checks and ambiguity rejection remain in place. The prompt distinguishes
collection and submission containers and requests the source field label in the quote.

The exact question passed the live Pinecone/Groq workflow and returned
“Submission Container/Tube: Amber vial” with a verified CMAMA citation on printed page
926. Regression cases include the exact question, the name without the serum qualifier,
and an unfamiliar instruction word with the complete test name. **165 passed, 1 skipped**.
No vectors, chunk files, embedding configuration or namespaces were changed.

## TC14–TC20 continuation and evaluation deliverable

Implemented the supplied complete continuation, with TC01–TC20 reference cases, separate
expected behavior/clinical facts, preserved codes/pages, empty expected source IDs,
per-turn expectations and explicit source-verification flags. Clinical excerpts were
verified within the correct local Mayo PDF test entries. Only a catalog subset is indexed.

Conversation resolution now maintains CPR across a storage follow-up, switches CA → CAI,
and maintains BUN through four fresh retrieval turns without using assistant answers as
evidence. A storage question no longer incorrectly requires a collection passage solely
because it mentions “specimen.” Treatment is blocked before provider calls; missing exact
temperature is distinguished from retrieval failure and gets context-only citations.
Injection tests are mock-only and reject even verbatim attack instructions as answers.

The reusable runner exports `evaluation/day4_results.json` and `.csv`, including actual
resolved questions, evidence, observed chunk IDs/codes/metadata, answers, citations,
failure reasons and timings. Reports are labeled mock/live and gitignored. The Evaluation
tab preserves existing tools and adds counts, filters, turn inspection, downloads and a
credential-free mock-run button. It makes no startup API requests. Dataset status remains
not_tested; actual results live in the report rather than overwriting reference facts.

**Measured mocked report: 19 Passed, 0 Failed, 1 Skipped, 0 Not Tested.** TC17 is skipped:
DHVD / 1,25-Dihydroxyvitamin D, Serum is verified on original PDF ordinal 10 / printed 9,
and exhaustive local snapshot scans found zero literal code/name matches. This does not
prove current cloud absence. An explicitly enabled --live --audit-absence command can
list/fetch the bounded selected namespace and record audit scope, IDs/count and matches.
No such cloud audit or live suite was enabled in this continuation. TC20 never runs live.

**Full regression result: 194 passed, 2 skipped** (optional Day 3 and Day 4 integrations).
AppTest verified all existing tabs plus evaluation filters, report inspection, no-report
Not Tested states and no automatic live calls. Existing embedding model, dimensions,
namespaces, vectors, chunk snapshots, indexing manifest and .env were preserved.

Completion checks verified with mocks: graph routing, history, switching, source/citation
validation, abstention, safety behavior, error handling, corpus cases and Days 1–3
regressions. Earlier live single-question model/retrieval checks are documented above;
new live corpus/follow-up validation remains optional and unexecuted. Current model is
the previously verified Groq-hosted openai/gpt-oss-120b. No configuration change was
needed for this continuation.

Known limits: literal identity/facet checks favor conservative abstention; conflict and
omission detection are not clinical entailment checks. Mock passes are not clinical
accuracy or live retrieval scores. TC17 is pending a current read-only namespace audit.
The implementation is ready for Day 5 evaluation experiments; no reranking, recall/MRR,
faithfulness scoring, autonomous browsing or full quality benchmark was added.
