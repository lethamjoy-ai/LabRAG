import json
from dataclasses import replace
import pytest
from config.settings import Settings
from src.chunking.fixed_chunker import chunk_document_fixed
from src.chunking.chunk_manager import export_jsonl
from src.vectorstore.index_manager import IndexManager, vector_metadata, vector_id
from tests.day3_fakes import embedding_service, connection


def source_file(tmp_path, laboratory_document, text=None, filename="chunks.jsonl", **meta):
    doc = laboratory_document(text or " ".join(f"Synthetic{i}" for i in range(70)), **meta)
    chunks = chunk_document_fixed([doc], 40, 5)
    path = tmp_path / filename
    path.write_text(export_jsonl(chunks), encoding="utf-8")
    return path, chunks


def manager(tmp_path):
    settings = replace(Settings(), embedding_batch_size=2, pinecone_upsert_batch_size=1)
    conn = connection()
    return IndexManager(settings, embedding_service(), conn, tmp_path / "manifest.json"), conn


def test_batches_metadata_idempotency_and_manifest(tmp_path, laboratory_document):
    path, chunks = source_file(tmp_path, laboratory_document)
    service, conn = manager(tmp_path)
    result = service.index_file(path, "fixed", "synthetic")
    assert result["status"] == "complete" and result["uploaded"] == len(chunks)
    calls = conn.index.upsert.call_args_list
    assert len(calls) == len(chunks)
    record = calls[0].kwargs["vectors"][0]
    assert record["metadata"]["text"] == chunks[0].page_content
    assert record["metadata"]["laboratory_provider"] == "Fictional laboratory"
    assert len(record["values"]) == 384
    assert record["id"] == vector_id(chunks[0], vector_metadata(chunks[0], service.settings, "synthetic"))
    again = service.index_file(path, "fixed", "synthetic")
    assert again["uploaded"] == 0 and again["skipped"] == len(chunks)
    assert len(conn.index.upsert.call_args_list) == len(chunks)


def test_interrupted_resume_and_scoped_obsolete_removal(tmp_path, laboratory_document):
    path, chunks = source_file(tmp_path, laboratory_document)
    service, conn = manager(tmp_path)
    conn.index.upsert.side_effect = [None, RuntimeError("failure")]
    result = service.index_file(path, "fixed", "synthetic")
    assert result["status"] == "failed" and result["uploaded"] == 1
    conn.index.upsert.side_effect = None
    result = service.index_file(path, "fixed", "synthetic")
    assert result["skipped"] == 1 and result["status"] == "complete"
    old_ids = set(next(iter(service.manifest.read()["entries"].values()))["vector_ids"])
    path, _ = source_file(tmp_path, laboratory_document, "Changed synthetic text")
    result = service.index_file(path, "fixed", "synthetic", remove_obsolete=True)
    removed = {identifier for call in conn.index.delete.call_args_list for identifier in call.kwargs["ids"]}
    assert old_ids == removed
    assert all(call.kwargs["namespace"] == "fixed-chunks-synthetic" for call in conn.index.delete.call_args_list)


def test_invalid_preflight_no_upserts(tmp_path, laboratory_document):
    path, _ = source_file(tmp_path, laboratory_document)
    service, conn = manager(tmp_path)
    with pytest.raises(ValueError, match="Synthetic fixtures"):
        service.index_file(path, "fixed")
    with pytest.raises(ValueError, match="strategy"):
        service.index_file(path, "section", "synthetic")
    conn.index.upsert.assert_not_called()
    path.write_text("{corrupt}", encoding="utf-8")
    with pytest.raises(ValueError):
        service.index_file(path, "fixed", "synthetic")


def test_original_split_overlap_is_blocked(tmp_path, laboratory_document):
    service, conn = manager(tmp_path)
    path, _ = source_file(tmp_path, laboratory_document, "Original", filename="original.jsonl")
    # Source filenames carry the splitter's explicit original page range convention.
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    for row in raw:
        row["metadata"]["filename"] = "catalog.pdf"
    path.write_text("\n".join(json.dumps(r) for r in raw) + "\n")
    assert service.index_file(path, "fixed", "synthetic")["status"] == "complete"
    split, _ = source_file(tmp_path, laboratory_document, "Split", filename="split.jsonl")
    raw = [json.loads(line) for line in split.read_text().splitlines()]
    for row in raw:
        row["metadata"].update(document_id="other-document", filename="catalog_part_001_pages_1-500.pdf")
        from src.chunking.common import chunk_identity
        row["metadata"]["chunk_id"] = chunk_identity(row["metadata"], row["text"])
    split.write_text("\n".join(json.dumps(r) for r in raw) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        service.index_file(split, "fixed", "synthetic")


def test_scoped_purge_only_owned_ids(tmp_path, laboratory_document):
    path, _ = source_file(tmp_path, laboratory_document)
    service, conn = manager(tmp_path)
    service.index_file(path, "fixed", "synthetic")
    entry = next(iter(service.manifest.read()["entries"].values()))
    service.remove_indexed_document("synthetic-1", "fixed", "synthetic")
    conn.index.delete.assert_called_once_with(ids=entry["vector_ids"], namespace="fixed-chunks-synthetic")
    assert not service.manifest.read()["entries"]
