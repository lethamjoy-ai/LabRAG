import pytest
from src.ingestion.metadata_manager import MetadataManager, validate_metadata, save_document
from src.ingestion.document_loader import load_document


def record():
    return dict(document_id="demo", test_name="Synthetic", laboratory_provider="Fictional",
                document_title="Demo", source_url="", document_type="txt", content_hash="a" * 64,
                ingestion_timestamp="2026-10-16T00:00:00+00:00", accessed_date="2026-10-16")


def test_manifest_create_duplicate_and_update(tmp_path):
    manager = MetadataManager(tmp_path / "source_manifest.csv")
    assert manager.list_records().empty
    manager.add(record())
    assert len(manager.list_records()) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        manager.add(record())
    manager.update("demo", {"document_title": "Updated"})
    assert manager.list_records().iloc[0].document_title == "Updated"


def test_metadata_validation():
    assert validate_metadata({**record(), "test_name": ""})["test_name"] == ""
    with pytest.raises(ValueError):
        validate_metadata({**record(), "test_name": None})
    with pytest.raises(ValueError):
        validate_metadata({**record(), "source_url": "http://example.com"})


def test_local_save(tmp_path):
    data = b"Synthetic title\nPlaceholder"
    saved = save_document(data, load_document(data, "demo.txt"), record(), tmp_path)
    assert (tmp_path / "raw" / f"{saved['document_id']}.txt").read_bytes() == data
    assert len(MetadataManager(tmp_path / "source_manifest.csv").list_records()) == 1
