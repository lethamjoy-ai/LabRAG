import pytest
from src.ingestion.document_loader import load_document
from src.ingestion.metadata_manager import save_document, MetadataManager
from src.ingestion.document_store import remove_document
from src.chunking.pipeline import process_document
from src.chunking.chunk_manager import ChunkManager, load_processed_document


def store(root, title):
    data = f"Synthetic {title}\nSpecimen Required\nPlaceholder".encode()
    docs = load_document(data, "demo.txt")
    record = save_document(data, docs, {"document_title": title, "laboratory_provider": "Fictional",
        "source_url": "", "accessed_date": "2026-10-19"}, root)
    for strategy in ("fixed", "section"):
        loaded = load_processed_document(record["document_id"], root / "processed", record)
        _, chunks, _ = process_document(loaded, strategy)
        ChunkManager(root / "processed").save(chunks)
    return record


def test_removes_only_selected_document_and_chunks(tmp_path):
    selected = store(tmp_path, "First")
    kept = store(tmp_path, "Second")
    original = tmp_path / "original.txt"
    original.write_text("Original source")
    remove_document(selected["document_id"], tmp_path)
    assert MetadataManager(tmp_path / "source_manifest.csv").list_records().document_id.tolist() == [kept["document_id"]]
    assert not (tmp_path / "raw" / f"{selected['document_id']}.txt").exists()
    assert not (tmp_path / "processed" / f"{selected['document_id']}.json").exists()
    for strategy in ("fixed", "section"):
        assert not ChunkManager(tmp_path / "processed").path_for(selected["document_id"], strategy).exists()
        assert ChunkManager(tmp_path / "processed").path_for(kept["document_id"], strategy).exists()
    assert original.read_text() == "Original source"


def test_missing_derived_files_and_last_record(tmp_path):
    selected = store(tmp_path, "Only")
    ChunkManager(tmp_path / "processed").path_for(selected["document_id"], "fixed").unlink()
    remove_document(selected["document_id"], tmp_path)
    assert MetadataManager(tmp_path / "source_manifest.csv").list_records().empty


def test_failure_rolls_back_files(tmp_path, monkeypatch):
    selected = store(tmp_path, "Only")
    def fail(*args):
        raise OSError("Simulated manifest write failure")
    monkeypatch.setattr(MetadataManager, "_write", fail)
    with pytest.raises(OSError):
        remove_document(selected["document_id"], tmp_path)
    assert (tmp_path / "raw" / f"{selected['document_id']}.txt").exists()
    assert len(MetadataManager(tmp_path / "source_manifest.csv").list_records()) == 1


@pytest.mark.parametrize("identity", ["../outside", "", "a" * 64])
def test_invalid_or_unknown_ids(tmp_path, identity):
    with pytest.raises(ValueError):
        remove_document(identity, tmp_path)
