"""Streaming, checkpointed indexing of Day 2 JSONL; explicit vector cleanup only."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from src.chunking.chunk_manager import iter_chunk_file, STORAGE_KEYS
from src.vectorstore.indexing_manifest import IndexingManifest
from src.vectorstore.pinecone_client import namespace_for, retry, field


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def provenance(metadata):
    meta = dict(metadata)
    filename = meta.get("filename", "")
    match = re.fullmatch(r"(.+)_part_\d+_pages_(\d+)-(\d+)\.pdf", filename, re.I)
    original_name = match[1] + ".pdf" if match else filename
    meta.setdefault("original_filename", original_name)
    if not meta.get("original_document_id"):
        meta["original_document_id"] = digest([meta.get("laboratory_provider"), meta.get("source_url"), original_name or meta["document_id"]])
    if match:
        meta["split_file_identity"] = meta["document_id"]
    page = meta.get("page_number")
    if type(page) is int and page > 0:
        if match and int(match[2]) + page - 1 > int(match[3]):
            raise ValueError("Split PDF page exceeds its filename's declared page range.")
        meta.setdefault("original_page_number", page + int(match[2]) - 1 if match else page)
    return meta


def vector_metadata(chunk, settings, corpus):
    meta = provenance(chunk.metadata)
    if meta.get("unsplittable"):
        page = meta.get("original_page_number", meta.get("page_number", "unknown"))
        raise ValueError(f"Oversized table block on page {page}, chunk {meta.get('chunk_index', 'unknown')} cannot be indexed. "
                         "Reprocess with the updated section chunker or review that table against the source.")
    if corpus == "public":
        if not meta.get("source_url", "").startswith("https://"):
            raise ValueError("Public references require an HTTPS source URL. Use synthetic namespaces for teaching fixtures.")
        fixture = meta.get("synthetic") or re.search(r"synthetic|fictional|teaching fixture", " ".join(
            str(meta.get(k, "")) for k in ("document_title", "laboratory_provider")), re.I)
        if fixture or "SYNTHETIC DEVELOPMENT FIXTURE" in chunk.page_content.upper():
            raise ValueError("Synthetic fixtures must use the separate synthetic corpus namespaces.")
    meta = {k: v for k, v in meta.items() if k in STORAGE_KEYS and v is not None
            and isinstance(v, (str, int, float, bool))}
    meta.update(text=chunk.page_content, embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension, embedding_provider=settings.embedding_provider, corpus_type=corpus)
    if len(json.dumps(meta, ensure_ascii=False, allow_nan=False).encode()) > 38_000:
        raise ValueError("Chunk metadata/text exceeds the safe Pinecone metadata limit. Reprocess with smaller chunks.")
    return meta


def vector_id(chunk, metadata):
    return digest([chunk.metadata["chunk_id"], metadata])


class IndexManager:
    def __init__(self, settings, embeddings, connection, manifest_path="data/indexing_manifest.json"):
        self.settings, self.embeddings, self.connection = settings, embeddings, connection
        self.manifest = IndexingManifest(manifest_path)

    def _key(self, document_id, strategy, corpus):
        return digest([self.settings.pinecone_index_name, namespace_for(self.settings, strategy, corpus),
                       self.settings.embedding_model, document_id])

    def index_file(self, path, strategy, corpus="public", approve_tables=False, remove_obsolete=False, progress=None):
        namespace = namespace_for(self.settings, strategy, corpus)
        source_hash = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                source_hash.update(block)
        file_hash = source_hash.hexdigest()
        desired, document_id, source_content_hash, original_id = set(), None, None, None
        pages = set()
        # Preflight the entire file before the first upsert, using bounded text batches.
        for chunk in iter_chunk_file(path):
            meta = vector_metadata(chunk, self.settings, corpus)
            if meta["chunking_strategy"] != strategy:
                raise ValueError("Chunk strategy does not match the selected namespace.")
            if chunk.metadata.get("table_review_required") and not approve_tables:
                raise ValueError("Table-review flags require explicit review approval before indexing.")
            self.embeddings.validate_text(chunk.page_content)
            if document_id is not None and (meta["document_id"] != document_id or meta["source_content_hash"] != source_content_hash):
                raise ValueError("A chunk file must contain exactly one document and source snapshot.")
            document_id, source_content_hash, original_id = meta["document_id"], meta["source_content_hash"], meta["original_document_id"]
            identifier = vector_id(chunk, meta)
            if identifier in desired:
                raise ValueError("Duplicate vector/chunk in the JSONL file.")
            desired.add(identifier)
            if meta.get("original_page_number"):
                pages.add(meta["original_page_number"])
        if not desired:
            raise ValueError("No chunks found. Process a source document first.")
        state = self.manifest.read()
        key = self._key(document_id, strategy, corpus)
        for other_key, other in state["entries"].items():
            if other_key != key and other.get("index_name") == self.settings.pinecone_index_name and other.get("namespace") == namespace:
                if other.get("embedding_model") != self.settings.embedding_model:
                    raise ValueError("Namespace already belongs to a different embedding model. Use fresh namespaces.")
                if other.get("document_id") != document_id and other.get("original_document_id") == original_id:
                    overlap = pages.intersection(other.get("original_pages", []))
                    if overlap or not pages or not other.get("original_pages"):
                        raise ValueError("Original/split or revised documents overlap an indexed source. Explicitly remove the previous document's vectors before replacing it.")
        entry = state["entries"].get(key, {})
        completed = set(entry.get("current_vector_ids", [])).intersection(desired) if entry.get("file_hash") == file_hash else set()
        owned = set(entry.get("vector_ids", []))
        entry.update(document_id=document_id, original_document_id=original_id, original_pages=sorted(pages),
            embedding_provider=self.settings.embedding_provider, embedding_dimension=self.settings.embedding_dimension,
            source_content_hash=source_content_hash, chunking_strategy=strategy, embedding_model=self.settings.embedding_model,
            namespace=namespace, index_name=self.settings.pinecone_index_name, file_hash=file_hash, status="indexing",
            current_vector_ids=sorted(completed), vector_ids=sorted(owned), corpus_type=corpus)
        state["entries"][key] = entry
        self.manifest.write(state)
        if self.connection.index is None:
            self.connection.connect()
        summary = {"document_id": document_id, "namespace": namespace, "total": len(desired),
                   "uploaded": 0, "skipped": len(completed), "failed_batches": []}

        def upload(batch):
            vectors = self.embeddings.embed_documents([c.page_content for c, _, _ in batch])
            records = [{"id": identifier, "values": vector, "metadata": meta}
                       for (_, meta, identifier), vector in zip(batch, vectors, strict=True)]
            groups, group, size = [], [], 0
            for record in records:
                record_size = len(json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
                if group and (len(group) >= self.settings.pinecone_upsert_batch_size or size + record_size > 1_800_000):
                    groups.append(group)
                    group, size = [], 0
                group.append(record)
                size += record_size
            if group:
                groups.append(group)
            for group in groups:
                # Record attempted ownership before I/O: a timeout may occur after server acceptance.
                owned.update(r["id"] for r in group)
                entry["vector_ids"] = sorted(owned)
                self.manifest.write(state)
                retry(lambda: self.connection.index.upsert(vectors=group, namespace=namespace))
                identifiers = [r["id"] for r in group]
                owned.update(identifiers)
                completed.update(identifiers)
                entry.update(vector_ids=sorted(owned), current_vector_ids=sorted(completed))
                self.manifest.write(state)
                summary["uploaded"] += len(group)
                if progress:
                    progress(len(completed), len(desired))
        try:
            batch = []
            for chunk in iter_chunk_file(path):
                meta = vector_metadata(chunk, self.settings, corpus)
                identifier = vector_id(chunk, meta)
                if identifier not in desired:
                    raise ValueError("Chunk file changed during indexing. Stop concurrent processing and retry.")
                if identifier in completed:
                    continue
                batch.append((chunk, meta, identifier))
                if len(batch) >= self.settings.embedding_batch_size:
                    upload(batch)
                    batch = []
            if batch:
                upload(batch)
            if completed != desired:
                raise ValueError("Chunk file changed or lost records during indexing. Retry with a stable snapshot.")
            obsolete = owned - desired
            entry["obsolete_vector_ids"] = sorted(obsolete)
            if remove_obsolete:
                for start in range(0, len(obsolete), 500):
                    ids = sorted(obsolete)[start:start + 500]
                    retry(lambda: self.connection.index.delete(ids=ids, namespace=namespace))
                entry["vector_ids"] = sorted(desired)
                entry["obsolete_vector_ids"] = []
            entry.update(status="complete", successful_indexing_timestamp=datetime.now(timezone.utc).isoformat())
            self.manifest.write(state)
        except Exception:
            entry["status"] = "failed"
            self.manifest.write(state)
            summary["failed_batches"].append({"completed": len(completed), "remaining": len(desired - completed),
                                             "error": "Batch failed. Check model limits, Pinecone quota/connectivity, and retry to resume."})
        summary["obsolete_vectors"] = len(entry.get("obsolete_vector_ids", []))
        summary["status"] = entry["status"]
        return summary

    def remove_indexed_document(self, document_id, strategy, corpus="public"):
        state = self.manifest.read()
        key = self._key(document_id, strategy, corpus)
        entry = state["entries"].get(key)
        if not entry:
            raise ValueError("No owned vectors recorded for this document/model/index/namespace.")
        if self.connection.index is None:
            self.connection.connect()
        ids = entry["vector_ids"]
        for start in range(0, len(ids), 500):
            retry(lambda: self.connection.index.delete(ids=ids[start:start + 500], namespace=entry["namespace"]))
        del state["entries"][key]
        self.manifest.write(state)
