from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_day1_ui_launches_without_keys(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Ask LabRAG", "Evaluation Results", "About the Project"]


def test_chunk_comparison_process_and_reload(tmp_path, monkeypatch):
    from src.ingestion.document_loader import load_document
    from src.ingestion.metadata_manager import save_document
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.chdir(tmp_path)
    data = b"Synthetic fixture\nSpecimen Required\nPlaceholder only.\nInterpretation\nNo clinical guidance."
    save_document(data, load_document(data, "demo.txt"), {"document_title": "Fixture", "test_name": "Demo",
        "laboratory_provider": "Fictional", "source_url": "", "accessed_date": "2026-10-17"})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    strategy = next(s for s in app.selectbox if s.label == "Chunking strategy")
    strategy.select("Compare both strategies").run(timeout=30)
    next(b for b in app.button if b.label == "Process Document").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(list((tmp_path / "data/processed").glob("*/*.jsonl"))) == 2
    assert set(app.session_state["chunk_outputs"][1]) == {"fixed", "section"}
    next(b for b in app.button if b.label == "Load saved chunks").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) >= 4


def test_catalog_ui_displays_per_test_chunks_without_manual_names(tmp_path, monkeypatch):
    from src.ingestion.document_loader import load_document
    from src.ingestion.metadata_manager import save_document
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.chdir(tmp_path)
    data = b"Test Name: Alpha\nTest Code: DEMO-A\nSpecimen Required\nPlaceholder A\nTest Name: Beta\nTest Code: DEMO-B\nSpecimen Required\nPlaceholder B"
    save_document(data, load_document(data, "catalog.txt"), {"document_title": "Catalog", "laboratory_provider": "Fictional",
        "source_url": "", "accessed_date": "2026-10-17"})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    next(b for b in app.button if b.label == "Process Document").click().run(timeout=30)
    assert not app.exception and not app.error
    chunks = app.session_state["chunk_outputs"][1]["fixed"]["chunks"]
    assert {c.metadata["test_code"] for c in chunks} == {"DEMO-A", "DEMO-B"}
    assert any("test_code" in table.value.columns for table in app.dataframe)


def test_remove_document_refreshes_counts_and_clears_chunk_preview(tmp_path, monkeypatch):
    from src.ingestion.document_loader import load_document
    from src.ingestion.metadata_manager import save_document, MetadataManager
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.chdir(tmp_path)
    data = b"Synthetic fixture\nSpecimen Required\nPlaceholder"
    save_document(data, load_document(data, "demo.txt"), {"document_title": "Fixture", "laboratory_provider": "Fictional",
        "source_url": "", "accessed_date": "2026-10-19"})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    next(b for b in app.button if b.label == "Process Document").click().run(timeout=30)
    assert "chunk_outputs" in app.session_state
    next(b for b in app.button if b.label == "Remove document").click().run(timeout=30)
    assert not app.exception and not app.error
    assert MetadataManager().list_records().empty
    assert next(m for m in app.metric if m.label == "Documents").value == "0"
    assert "chunk_outputs" not in app.session_state
    assert app.session_state["upload_generation"] == 1
    assert any("Knowledge Base refreshed" in message.value for message in app.success)
    assert not any(s.label == "Document to process" for s in app.selectbox)


def test_inspect_chunk_updates_text_and_metadata_for_each_strategy(tmp_path, monkeypatch):
    import json
    from src.ingestion.document_loader import load_document
    from src.ingestion.metadata_manager import save_document
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.chdir(tmp_path)
    data = b"Test Name: Alpha\nTest Code: DEMO-A\nSpecimen Required\nAlpha placeholder.\nTest Name: Beta\nTest Code: DEMO-B\nSpecimen Required\nBeta placeholder."
    save_document(data, load_document(data, "catalog.txt"), {"document_title": "Catalog", "laboratory_provider": "Fictional",
        "source_url": "", "accessed_date": "2026-10-19"})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    next(s for s in app.selectbox if s.label == "Chunking strategy").select("Compare both strategies").run(timeout=30)
    next(b for b in app.button if b.label == "Process Document").click().run(timeout=30)
    assert not app.exception and not app.error
    outputs = app.session_state["chunk_outputs"][1]
    for strategy in ("fixed", "section"):
        chunks = outputs[strategy]["chunks"]
        for index in (len(chunks) - 1, 0):
            next(s for s in app.selectbox if (s.key or "").startswith(f"inspect_{strategy}_")).select(index).run(timeout=30)
            assert not app.exception and not app.error
            expected = chunks[index]
            assert expected.page_content.strip() in [text.value.strip() for text in app.text]
            displayed = [json.loads(element.value) for element in app.json]
            assert expected.metadata in displayed
            other = chunks[0 if index else -1]
            assert other.metadata not in displayed
