# Pinecone setup and operating guide

## Account and index

Create an account at [Pinecone](https://www.pinecone.io/) and obtain a project API key from its console. Keep it only in your local `.env` as `PINECONE_API_KEY`. Choose permissions allowing index description, vector upsert/query, and explicit scoped vector removal. Do not commit or paste the key into source, logs, or screenshots.

Create an existing **dense, bring-your-own-vectors index**, dimension **384**, metric **cosine**, default name **lab-rag**. Choose a cloud/region actually available in your account; `aws/us-east-1` is a configurable example, not a guarantee of availability. Provisioning/cloud usage can incur charges. Existing index dimension/metric cannot be silently changed by this app.

You can create the index manually in the console or explicitly use the provided CLI:

```powershell
.venv\Scripts\python.exe scripts/pinecone_admin.py create --confirm-create
.venv\Scripts\python.exe scripts/pinecone_admin.py validate
```

The CLI refuses to alter an existing index. Wait for readiness before validation. App startup never invokes creation or deletion, and connection validation never uploads references. No automatic cloud resources were created while implementing Day 3.

## Environment

Copy `.env.example` to `.env` only if you do not already have one. Keep existing keys private. Set:

The app rereads the project-root `.env` on each Streamlit rerun, using UTF-8/BOM-compatible decoding. Values present in that file take precedence over inherited process settings; environment-only deployments work when the file is absent. Save changes in the editor, then refresh the app. Credentials are not displayed during configuration checks.

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
PINECONE_API_KEY=your-private-key
PINECONE_INDEX_NAME=lab-rag
PINECONE_NAMESPACE_FIXED=fixed-chunks
PINECONE_NAMESPACE_SECTION=section-chunks
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_BATCH_SIZE=32
PINECONE_UPSERT_BATCH_SIZE=50
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.0
```

The cloud/region settings apply only to explicit index creation. Namespaces must differ. Groq variables are retained for Day 4 but unused. Install `requirements-day3.txt`; its Pinecone 7.x range matches the requested LangChain integration's constraints and uses the modern `pinecone` package. Do not install `pinecone-client` alongside it.

## Prepare, approve, and index

OpenAI is an optional embedding provider: set `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small`, `EMBEDDING_DIMENSION=384`, and a private `OPENAI_API_KEY`. The same existing 384-dimensional index can be used because the request explicitly sets output dimensions. The app automatically isolates OpenAI vectors in model/dimension-specific namespaces such as `fixed-chunks-openai-text-embedding-3-small-384` and `section-chunks-openai-text-embedding-3-small-384`. Re-index documents with the chosen provider; questions use the same provider. Existing local vectors remain unchanged. Explicit OpenAI validation/indexing/search actions send text externally and may incur API charges; startup makes no requests. The provider adapter has mocked test coverage, not a paid live verification. See the README's optional OpenAI setup and [official API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

Keep original source attribution and permitted reference usage. Process saved documents in Knowledge Base. Day 3 reads JSONL incrementally, never the complete original catalog PDF. Validate the local embedding model and Pinecone connection explicitly. Approve the source collection, select processed files and strategy, review flagged tables, and click Index Processed Documents. Observe progress and failed-batch details. A successful checkpoint is written after each acknowledged group; re-running skips those IDs. Synthetic references must use the synthetic corpus, whose namespaces append `-synthetic`.

Use the same source documents and revisions in both namespaces for meaningful comparisons. A namespace is an isolation convention, not a security boundary. Comparison warns when locally recorded complete source collections differ. Actual retrieval quality requires labeled evaluation, not comparison of raw cosine scores alone.

## Original/split provenance and revisions

The `scripts/split_pdf.py` filename convention (`..._part_001_pages_1-500.pdf`) permits reconstructing original page numbers from local page offsets. Day 3 preserves source URL/provider, original filename/source identity, split-file identity, and original page where supported by that convention or existing metadata. Overlapping source pages already registered in the same namespace are blocked, preventing accidental simultaneous indexing of the original and split copies. A renamed or unfamiliar split requires explicit lineage metadata before reliable overlap detection is possible.

Reprocessing a stable document ID tracks new deterministic IDs and obsolete owned IDs. The indexing UI's explicit obsolete-vector option removes only those stale IDs after the replacement is fully acknowledged. A revision producing a different raw-document ID needs explicit retirement of the prior ID:

```powershell
.venv\Scripts\python.exe scripts/pinecone_admin.py remove-document --document-id YOUR_DOCUMENT_ID --strategy section --corpus public --confirm-remove
```

Repeat for fixed strategy if applicable. This uses exact owned IDs from `data/indexing_manifest.json`; it never deletes an entire namespace or unrelated vectors. Removing the local upload in the UI does **not** automatically remove remote vectors. Keep the indexing manifest to retain cleanup ownership, including after local removal. The manifest assumes one writer and no external mutation of the namespace; it is not a distributed transaction log.

## Verify and search

Validate Pinecone Connection shows actual namespace counts from `describe_index_stats()`. Use the console or CLI validate command for independent counts. A recent upsert/delete may take time to appear due to eventual consistency. Local manifest counts record acknowledged writes and can diverge if someone changes Pinecone outside the app.

Ask LabRAG returns top-k source passages with cosine scores, source links, page numbers, and chunk/test metadata. Optional filters use exact strings and supported equality operators for provider, test name, document ID, or revision date. Scores are not clinical confidence probabilities. Evaluation searches both namespaces with the same question embedding and reports actual results/latency. Per-namespace comparison timing excludes the shared query-embedding step.

## Troubleshooting

- Missing API key: local ingestion/chunking still works; add the key only when connecting.
- Missing/not-ready index: manually create the correct dense cosine index or wait for provisioning. Validation does not recreate anything.
- Dimension/metric mismatch: point to a compatible index. Do not delete a populated index to resolve a configuration typo.
- Model-token overflow: MiniLM's default limit is 256 word pieces including special tokens in this implementation's check. Day 2 tiktoken counts differ. Reprocess smaller chunks (try 150–180 approximate tokens, overlap 25), then verify; values are not truncated or changed.
- Oversized/ambiguous tables: resolve layout issues against the original; unsplittable records are blocked. Approve other flagged table chunks only after review.
- Partial batches/quota/network errors: transient errors retry at most three times with exponential delays. Reduce batch sizes if necessary and rerun to resume from checkpoints.
- Empty search: confirm the correct corpus/namespace, completed indexing, filters, model, and consistency delay. No evidence means no generated answer.
- Lost/corrupt manifest: restore a backup. Do not discard ownership history and upload blindly into a populated namespace.
- Model changes: use fresh namespaces and verify dimensions. The default public/synthetic selectors prevent intentionally mixed model spaces recorded by the local manifest.

Primary API references: [Hugging Face local embeddings in LangChain](https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Pinecone SDK](https://sdk.pinecone.io/python/), and [Pinecone upsert guidance](https://docs.pinecone.io/guides/index-data/upsert-data). Metadata/text is limited conservatively to 38 KB and request batches to 1.8 MB; fields with null values are omitted.
