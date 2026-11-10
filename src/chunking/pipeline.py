"""Reusable local processing used by UI, tests, and future indexing preparation."""
from src.ingestion.text_cleaner import clean_documents
from src.chunking.fixed_chunker import chunk_document_fixed
from src.chunking.section_chunker import chunk_document_by_section
from src.chunking.chunk_validator import validate_chunks


def process_document(documents, strategy, size=500, overlap=75):
    if strategy not in {"fixed", "section"}:
        raise ValueError("Strategy must be fixed or section.")
    cleaned = clean_documents(list(documents))
    chunks = (chunk_document_fixed(cleaned, size, overlap) if strategy == "fixed"
              else chunk_document_by_section(cleaned, size, overlap))
    validation = validate_chunks(chunks, cleaned, size)
    return cleaned, chunks, validation
