# Day 1 progress

Completed: modular Day 1 structure, optional-key configuration with integer/dimension validation, PDF/HTML/TXT validation and extraction, conservative cleaning, PDF provenance, configurable file size, local persistence, metadata add/list/update/validation, duplicate detection, CSV manifest, topic plan, three synthetic references, three-tab UI, counts/search/filtering, future metric placeholders, architecture and source guidance, tests and CI configuration.

The default app makes no LLM or Pinecone calls and performs no embedding, chunking, reranking, or model download. Earlier broader-scope modules were preserved separately; they are inactive in the supported Day 1 entry point.

Pending: real permitted collection and provenance review, OCR if needed, robust section segmentation, concurrent storage, fixed/section chunk comparison, embeddings, explicit indexing, retrieval, LangGraph answer generation, source citations, and clinical evaluation in subsequent days.

Verification: the final executed pytest run passed **31 tests in 7.29 seconds**, including Streamlit AppTest startup and all three tabs without credentials. Day 1 runtime/test dependencies were installed through uv using Python 3.11. Tests include synthetic PDFs generated in memory rather than real laboratory or patient content. No paid API integration or browser upload interaction was tested.

Lessons: keep planned sources distinct from collected evidence; validate bytes as well as extensions; retain page/heading provenance; allow local foundation work without secrets; use content identity to prevent accidental duplicate ingestion. Citation formatting and term overlap in any future evaluation are not equivalent to factual or clinical correctness.
