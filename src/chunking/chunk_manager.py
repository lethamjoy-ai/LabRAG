"""Atomic UTF-8 JSONL snapshots per document/strategy; no external services."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from src.chunking.chunk_validator import validate_chunks

def export_jsonl(chunks):
    records = []
    for doc in chunks:
        # Allowlisted provenance and processing metadata, not arbitrary environment/custom fields.
        metadata = {k: v for k, v in doc.metadata.items() if k in STORAGE_KEYS}
        records.append(json.dumps({"text": doc.page_content, "metadata": metadata}, ensure_ascii=False, allow_nan=False))
    return "\n".join(records) + ("\n" if records else "")


STORAGE_KEYS = {"chunk_id", "document_id", "document_title", "test_name", "test_code", "laboratory_provider",
    "source_url", "source_revision_date", "accessed_date", "page_number", "section_heading", "chunk_index",
    "chunking_strategy", "token_count", "content_hash", "source_content_hash", "chunk_content_hash", "source_part",
    "start_char", "end_char", "tokenizer", "chunk_size", "chunk_overlap", "table_review_required", "unsplittable",
    "fallback_to_fixed", "section_start", "section_end", "heading_prefix", "filename", "document_type", "ingestion_timestamp",
    "test_scope", "test_metadata_source", "test_name_evidence", "test_code_evidence"}
STORAGE_KEYS.update({"original_document_id", "original_page_number", "split_file_identity", "original_filename", "synthetic"})


def iter_chunk_file(path):
    """Stream validated records without loading a catalog or complete JSONL into memory."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not isinstance(record.get("metadata"), dict):
                        raise ValueError
                    if set(record["metadata"]) - STORAGE_KEYS:
                        raise ValueError
                    chunk = Document(page_content=record["text"], metadata=record["metadata"])
                    result = validate_chunks([chunk], max_tokens=chunk.metadata.get("chunk_size", 500))
                    if result.errors:
                        raise ValueError
                except (ValueError, TypeError, KeyError):
                    raise ChunkStorageError(f"Invalid chunk on line {number}; reprocess the source document.") from None
                yield chunk
    except OSError:
        raise ChunkStorageError("Chunk JSONL cannot be read. Check its location and permissions.") from None


class ChunkStorageError(ValueError):
    pass


class ChunkManager:
    def __init__(self, root="data/processed"):
        self.root = Path(root)

    def path_for(self, document_id, strategy):
        if strategy not in {"fixed", "section"}:
            raise ValueError("Strategy must be fixed or section.")
        identity = hashlib.sha256(document_id.encode()).hexdigest()
        return self.root / strategy / f"{identity}.jsonl"

    def save(self, chunks):
        chunks = list(chunks)
        if not chunks:
            raise ValueError("Cannot save an empty chunk collection.")
        doc_ids = {c.metadata.get("document_id") for c in chunks}
        strategies = {c.metadata.get("chunking_strategy") for c in chunks}
        if len(doc_ids) != 1 or len(strategies) != 1 or not next(iter(doc_ids)):
            raise ValueError("Save one document and strategy at a time.")
        limit = chunks[0].metadata.get("chunk_size", 500)
        validation = validate_chunks(chunks, max_tokens=limit)
        if validation.errors:
            raise ValueError("Cannot save invalid chunks: " + "; ".join(validation.errors[:3]))
        path = self.path_for(next(iter(doc_ids)), next(iter(strategies)))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(export_jsonl(chunks))
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return path

    def load(self, document_id, strategy):
        path = self.path_for(document_id, strategy)
        try:
            rows = path.read_text(encoding="utf-8").splitlines()
            chunks = []
            for number, line in enumerate(rows, 1):
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not isinstance(record.get("metadata"), dict):
                    raise ValueError(f"Invalid record on line {number}.")
                if set(record["metadata"]) - STORAGE_KEYS:
                    raise ValueError(f"Unexpected metadata on line {number}.")
                chunks.append(Document(page_content=record["text"], metadata=record["metadata"]))
            if not chunks:
                raise ValueError("File contains no chunks.")
            if any(c.metadata.get("document_id") != document_id or c.metadata.get("chunking_strategy") != strategy for c in chunks):
                raise ValueError("Stored identity or strategy mismatch.")
            result = validate_chunks(chunks, max_tokens=chunks[0].metadata.get("chunk_size", 500))
            if result.errors:
                raise ValueError("; ".join(result.errors[:3]))
            return chunks
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ChunkStorageError("Could not load chunk JSONL; file is missing, corrupt, or invalid. Reprocess the original document.") from exc


def load_processed_document(document_id, root="data/processed", metadata=None):
    # Day 1 IDs are SHA-256; never use an arbitrary UI/manifest value as a path.
    if len(document_id) != 64 or any(c not in "0123456789abcdef" for c in document_id):
        raise ValueError("Invalid document identifier.")
    try:
        records = json.loads((Path(root) / f"{document_id}.json").read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("No extracted documents.")
        documents = []
        for record in records:
            if not isinstance(record["text"], str) or not isinstance(record["metadata"], dict):
                raise ValueError("Malformed processed document.")
            meta = {**record["metadata"], **(metadata or {})}
            # Manifest is document-level; page/heading provenance belongs to each extracted part.
            for key in ("page_number", "section_heading"):
                meta[key] = record["metadata"].get(key)
            meta["original_text"] = record.get("original_text", record["metadata"].get("original_text", record["text"]))
            if meta.get("document_id") != document_id:
                raise ValueError("Document identity mismatch.")
            documents.append(Document(page_content=record["text"], metadata=meta))
        return documents
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Could not load extracted document; re-upload it or inspect the local JSON.") from exc
