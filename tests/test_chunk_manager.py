import json
import pytest
from src.chunking.chunk_manager import ChunkManager, ChunkStorageError, export_jsonl, load_processed_document
from src.chunking.fixed_chunker import chunk_document_fixed
from src.ingestion.document_loader import load_document
from src.ingestion.metadata_manager import save_document


def test_store_reload_replace_and_allowlist(tmp_path, laboratory_document):
    manager = ChunkManager(tmp_path)
    doc = laboratory_document("Synthetic reference body", groq_api_key="never-persist")
    chunks = chunk_document_fixed([doc], 40, 5)
    path = manager.save(chunks)
    manager.save(chunks)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(chunks)
    assert manager.load("synthetic-1", "fixed")[0].page_content == doc.page_content
    assert "never-persist" not in export_jsonl(chunks)
    changed = chunk_document_fixed([laboratory_document("Changed reference")], 40, 5)
    assert manager.save(changed) == path
    assert manager.load("synthetic-1", "fixed")[0].metadata["chunk_id"] != chunks[0].metadata["chunk_id"]


def test_duplicate_rejected(tmp_path, laboratory_document):
    chunks = chunk_document_fixed([laboratory_document("Body")], 40, 5)
    with pytest.raises(ValueError, match="invalid chunks"):
        ChunkManager(tmp_path).save(chunks + chunks)


@pytest.mark.parametrize("content", ["{broken}", "", '{"text": 3, "metadata": {}}', '{}', '[1]'])
def test_corrupt_jsonl(tmp_path, content):
    manager = ChunkManager(tmp_path)
    path = manager.path_for("synthetic-1", "fixed")
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ChunkStorageError):
        manager.load("synthetic-1", "fixed")


def test_day1_saved_source_roundtrip(tmp_path):
    data = b"Synthetic title\n\n\nSpecimen Required\nPlaceholder"
    docs = load_document(data, "fixture.txt")
    record = save_document(data, docs, {"document_title": "Title", "test_name": "Demo", "laboratory_provider": "Fictional",
                                      "source_url": "", "accessed_date": "2026-10-17"}, tmp_path)
    loaded = load_processed_document(record["document_id"], tmp_path / "processed", record)
    assert loaded[0].metadata["original_text"] == data.decode()
    assert loaded[0].metadata["document_title"] == "Title"
    with pytest.raises(ValueError):
        load_processed_document("../invalid", tmp_path)
