from copy import deepcopy
import pytest
from langchain_core.documents import Document
from src.ingestion.test_metadata import detect_test_entries, summarize_tests
from src.ingestion.document_loader import load_document
from src.ingestion.metadata_manager import save_document, MetadataManager
from src.chunking.pipeline import process_document
from src.chunking.chunk_validator import validate_chunks
from src.chunking.chunk_manager import ChunkManager, load_processed_document


CATALOG = """Synthetic multiple-test catalog
Test Name: Fictional Alpha
Test Code: DEMO-A
Specimen Required
Alpha placeholder only.
Interpretation
No clinical guidance for Alpha.
Test Name: Fictional Beta
Test Code: DEMO-B
Specimen Required
Beta placeholder only.
Interpretation
No clinical guidance for Beta.
"""


def test_labeled_fields_and_mayo_style_headers():
    entries = detect_test_entries(CATALOG)
    assert [(e.test_name, e.test_code) for e in entries] == [("Fictional Alpha", "DEMO-A"), ("Fictional Beta", "DEMO-B")]
    mayo = detect_test_entries("Test Definition: DEMO-A\nFictional Alpha, Serum\nSpecimen Required\nPlaceholder")
    assert (mayo[0].test_name, mayo[0].test_code) == ("Fictional Alpha, Serum", "DEMO-A")
    assert mayo[0].name_evidence == "Fictional Alpha, Serum"


def test_labels_on_separate_lines_and_no_cpt_inference():
    entries = detect_test_entries("Test ID:\nDEMO1\nTest Name:\nFictional Alpha\nSpecimen Required\nPlaceholder")
    assert len(entries) == 1
    assert (entries[0].test_code, entries[0].test_name) == ("DEMO1", "Fictional Alpha")
    assert not detect_test_entries("CPT Code: 12345\nSpecimen Required\nValue 2 mg/L")
    code_only = detect_test_entries("Test ID: DEMO\nSpecimen Required\nPlaceholder")[0]
    assert code_only.test_name == "" and code_only.test_code == "DEMO"


def test_labeled_table_only():
    text = "Test Code | Test Name\n--- | ---\nDEMO-A | Fictional Alpha\nDEMO-B | Fictional Beta"
    entries = detect_test_entries(text)
    assert [(e.test_name, e.test_code) for e in entries] == [("Fictional Alpha", "DEMO-A"), ("Fictional Beta", "DEMO-B")]
    assert not detect_test_entries("Specimen | Value\nSerum | 1.0")
    aligned = detect_test_entries("Test ID    Test Name\nDEMO-A    Fictional Alpha\nDEMO-B    Fictional Beta")
    assert [entry.test_code for entry in aligned] == ["DEMO-A", "DEMO-B"]


def test_instructions_not_inferred_as_test_titles():
    entry = detect_test_entries("Test ID: DEMO\nStore at 4 °C\nSpecimen Required\nPlaceholder")[0]
    assert entry.test_name == ""
    assert detect_test_entries("Test ID: DEMO\nKeep frozen\nSpecimen Required\nPlaceholder")[0].test_name == ""


def test_directory_rows_do_not_claim_following_prose(laboratory_document):
    text = "Test Code | Test Name\nDEMO-A | Alpha\nDEMO-B | Beta\n\nUnrelated catalog introduction."
    doc = laboratory_document(text, test_scope="catalog")
    cleaned, chunks, validation = process_document([doc], "fixed", 100, 10)
    assert not validation.errors
    assert chunks[-1].metadata["test_name"] == ""
    assert "Unrelated" in chunks[-1].page_content


@pytest.mark.parametrize("strategy", ["fixed", "section"])
def test_catalog_chunks_have_correct_test_context_and_reload(tmp_path, strategy):
    data = CATALOG.encode()
    documents = load_document(data, "catalog.txt")
    summary = summarize_tests(documents)
    assert summary == {"test_name": "", "test_code": "", "test_scope": "catalog"}
    record = save_document(data, documents, {"document_title": "Catalog", "laboratory_provider": "Fictional",
                         "source_url": "", "accessed_date": "2026-10-17"}, tmp_path)
    assert record["test_name"] == "" and record["test_scope"] == "catalog"
    loaded = load_processed_document(record["document_id"], tmp_path / "processed", record)
    cleaned, chunks, validation = process_document(loaded, strategy, 50, 5)
    assert not validation.errors
    for chunk in chunks:
        text, meta = chunk.page_content, chunk.metadata
        if "Beta" in text:
            assert meta["test_name"] == "Fictional Beta" and meta["test_code"] == "DEMO-B"
            assert "Alpha" not in text
        if "Alpha" in text:
            assert meta["test_name"] == "Fictional Alpha" and meta["test_code"] == "DEMO-A"
        if meta["test_name"]:
            assert meta["test_name_evidence"] in cleaned[0].page_content
    manager = ChunkManager(tmp_path / "chunks")
    manager.save(chunks)
    restored = manager.load(record["document_id"], strategy)
    assert [c.metadata for c in restored] == [c.metadata for c in chunks]
    broken = deepcopy(chunks)
    broken[-1].metadata["test_code"] = "WRONG"
    assert validate_chunks(broken, cleaned, 50).errors


def test_single_test_auto_summary_and_unknown_catalog_save(tmp_path):
    data = b"Test Name: Fictional Alpha\nTest Code: DEMO-A\nSpecimen Required\nPlaceholder"
    docs = load_document(data, "single.txt")
    assert docs[0].metadata["test_name"] == "Fictional Alpha"
    saved = save_document(data, docs, {"document_title": "Single", "laboratory_provider": "Fictional",
        "source_url": "", "accessed_date": "2026-10-17"}, tmp_path)
    assert saved["test_code"] == "DEMO-A"
    unknown = load_document(b"Unknown catalog format\nNo test labels", "unknown.txt")
    cleaned, chunks, result = process_document(unknown, "fixed", 50, 5)
    assert chunks[0].metadata["test_metadata_source"] == "unresolved"
    assert chunks[0].metadata["test_name"] == ""


def test_no_identity_carried_to_unknown_catalog_page(laboratory_document):
    docs = [laboratory_document("Test Name: Fictional Alpha\nTest Code: DEMO-A", test_scope="catalog"),
            laboratory_document("Unlabeled continuation page", test_scope="catalog", page_number=2)]
    cleaned, chunks, validation = process_document(docs, "fixed", 50, 5)
    assert chunks[-1].metadata["test_name"] == ""
    assert not validation.errors


def test_legacy_manifest_loads_without_scope_column(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("document_id,test_name\nold,Legacy single\n", encoding="utf-8")
    frame = MetadataManager(path).list_records()
    assert frame.iloc[0].test_scope == "single"
