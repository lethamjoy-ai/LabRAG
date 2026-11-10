# Chunking laboratory references

Chunking determines the context available to a future retriever and answer generator. A fragment that separates a value from its unit, table column, specimen qualifier, or laboratory attribution can mislead even when each token is copied correctly. Days 1–2 perform no clinical inference or indexing.

## Fixed-size

`chunk_document_fixed(documents, chunk_size=500, overlap=75)` uses LangChain `RecursiveCharacterTextSplitter` with a tiktoken length function. Paragraphs, lines, spaces, then characters provide recursive boundaries. It processes each Document independently and never joins laboratories, source documents, or pages. The target is a maximum token budget, not an exact size. LangChain overlap is bounded and best-effort at natural boundaries; it is not a guarantee of exactly 75 shared tokens between all chunks.

`cl100k_base` is documented by [tiktoken](https://github.com/openai/tiktoken). The [LangChain token splitting guide](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token) describes using recursive splitting to enforce a token budget while retaining valid Unicode. This encoding only approximates Sentence Transformer tokens. Day 3 must measure the actual embedding tokenizer/context length; 500 approximate tokens may exceed that model's usable window and lead to truncation.

## Section-aware

`chunk_document_by_section(documents, max_tokens=500, overlap=75, heading_patterns=None)` recognizes full-line laboratory headings, with optional Markdown markers, numbering, colons, and case differences. Default headings include specimen requirements, collection instructions, preparation, stability, rejection criteria, interpretation, reference values, and cautions. Callers may pass alternate regular-expression heading patterns. A title/preamble is retained without inventing a heading. No recognized heading triggers fixed-size fallback and a warning.

Sections are not merged. Oversized text sections split with overlap and repeat the actual heading on continuation chunks. Heading prefix tokens are budgeted; a limit too small for the heading and overlap produces an actionable error. Prefix text is identified separately from exact source spans. Page boundaries remain intact; section metadata does not claim to reconstruct a cross-page section.

## Tables and clinical values

The cleaner preserves signs, ranges, decimals, special characters, tabs, multi-space columns, and row breaks. HTML table rows become separate pipe-delimited lines. Original extracted text remains available for comparison. No value is interpreted or corrected. Section chunking isolates contiguous pipe/tab/aligned-space table blocks from surrounding prose; indentation alone is not a table. Prose is split normally while detected table blocks stay intact with their source rows/labels and repeated section heading where available. Only a table block that exceeds the limit is marked `unsplittable`, rather than blocking the entire surrounding section. It is suitable for review/export, **not automatic indexing**. Smaller detected tables retain review flags. Blank lines and lost PDF layout can obscure table relationships, so human review remains necessary; no missing column labels are inferred.

Fixed-size splitting may split table context and flags affected documents for manual review. Detection cannot reliably recover tables already flattened by PDF extraction, and cross-page tables are never combined automatically. Review source PDFs whenever row/column relationships are ambiguous. A numeric/unit phrase crossing a chunk boundary generates a warning even when exact coverage passes.

## Provenance and validation

Every chunk carries document/test/provider/source metadata, page when present, section when detected, strategy, sequence, token count, and deterministic ID. `content_hash` retains the original document's SHA-256; `source_content_hash` hashes the cleaned document parts; `chunk_content_hash` hashes the chunk. IDs incorporate source snapshot, document identity, source part/offsets, strategy, size/overlap, position, and text. Changed content therefore invalidates old chunk identities even when a caller retains the same document ID.

Validation checks metadata, identity/hash consistency, actual token counts, limits, duplicate IDs/spans, source attribution equality, exact source-span content, cleaned source coverage, detected section boundaries, and selected numeric/unit context. Repeated text at distinct genuine positions is retained with a warning; it is not deleted silently. Coverage ignores whitespace-only gaps. An empty source can yield zero chunks, but empty chunk records are invalid. Structural success does not establish complete semantic preservation, clinical truth, or faithfulness to PDF layout.

## Compare and persist

Save an uploaded source in Knowledge Base, then use Document Chunking → Compare both strategies → Process Document. Counts, mean/min/max tokens, empty chunks, and warning/error counts are calculated from the actual results. Side-by-side previews and metadata show what caused differences. Fixed-size often gives more uniform sizes; section-aware generally gives more coherent context with smaller sections and conservative oversized-table exceptions. Neither strategy wins solely by producing fewer warnings or larger chunks.

Valid outputs save as UTF-8 JSONL under `data/processed/{fixed,section}/`. One file per document/strategy is replaced atomically; repeated processing does not append duplicates. Export/download and loading use the same record format: `{"text": "...", "metadata": {...}}`. Reload verifies schema and identity; the UI also compares snapshots against the current cleaned source. Corrupt files produce actionable errors and are not silently skipped. Chunk IDs are not used as paths; hashed document keys prevent directory traversal.

Day 3 should load reviewed chunks, reject unresolved validation errors and oversized/table-review flags until assessed, check actual embedding token length, and only then explicitly embed/index into the configured cosine index and strategy-specific namespace. No API keys or embeddings are stored in Day 2 JSONL. Use permitted reference material or labeled synthetic fixtures only.
