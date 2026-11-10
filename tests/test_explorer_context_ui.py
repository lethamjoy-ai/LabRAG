from pathlib import Path
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
from tests.day3_fakes import connection, embedding_service


def test_explorer_defaults_to_test_context(tmp_path, monkeypatch):
    import src.ui_day3 as ui
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    conn = connection()
    doc = Document(page_content="Specimen required: serum.", metadata={
        "test_name": "Blood Urea Nitrogen, Serum", "test_code": "BUN", "chunk_id": "bun-specimen"})
    conn.vector_store.return_value.similarity_search_by_vector_with_score.return_value = [(doc, .7)]
    monkeypatch.setattr(ui, "services", lambda settings: (conn, embedding_service()))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert next(w for w in app.checkbox if w.label == "Retrieve test context").value
    next(w for w in app.text_area if w.label == "Your laboratory question").input("What specimen is required for BUN?")
    next(b for b in app.button if b.label == "Search laboratory references").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["semantic_result"].diagnostics["mode"] == "test_context"
    assert doc.page_content in [item.value for item in app.text]


def test_previous_search_results_are_cleared_after_upgrade(tmp_path, monkeypatch):
    from src.retrieval.semantic_retriever import SearchResult
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["semantic_search_version"] = 1
    app.session_state["semantic_result"] = SearchResult("old query", "section", "section-chunks", [
        {"document": Document(page_content="Unrelated old result"), "score": .9}], .1)
    app.run(timeout=30)
    assert not app.exception
    assert "semantic_result" not in app.session_state
    assert "Unrelated old result" not in [item.value for item in app.text]
