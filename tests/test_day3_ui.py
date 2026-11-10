from pathlib import Path
from unittest.mock import Mock
from streamlit.testing.v1 import AppTest
from langchain_core.documents import Document
from tests.day3_fakes import embedding_service, connection

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_startup_has_no_service_calls(tmp_path, monkeypatch):
    from src.vectorstore.pinecone_client import PineconeConnection
    from src.embeddings.embedding_service import EmbeddingService
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    def forbidden(*args, **kwargs):
        raise AssertionError("Startup must not connect or load a model")
    monkeypatch.setattr(PineconeConnection, "connect", forbidden)
    monkeypatch.setattr(EmbeddingService, "embed_query", forbidden)
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == ["Ask LabRAG", "Evaluation Results", "About the Project"]
    assert any(b.label == "Validate Pinecone Connection" for b in app.button)


def test_search_and_comparison_display_returned_evidence(tmp_path, monkeypatch):
    import src.ui_day3 as ui
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    conn = connection()
    doc = Document(page_content="Returned synthetic evidence for UI testing.", metadata={
        "document_title": "Test source", "source_url": "https://example.org/reference", "page_number": 3,
        "laboratory_provider": "Fictional", "chunk_id": "chunk-1"})
    conn.vector_store.return_value.similarity_search_by_vector_with_score.return_value = [(doc, 0.75)]
    monkeypatch.setattr(ui, "services", lambda settings: (conn, embedding_service()))
    app = AppTest.from_file(APP).run(timeout=30)
    next(w for w in app.checkbox if w.label == "Retrieve test context").uncheck()
    next(w for w in app.text_area if w.label == "Your laboratory question").input("Find a reference")
    next(b for b in app.button if b.label == "Search laboratory references").click().run(timeout=30)
    assert not app.exception and not app.error
    assert doc.page_content in [element.value for element in app.text]
    assert any("0.7500" in element.value for element in app.markdown)
    next(w for w in app.text_area if w.label == "Comparison question").input("Compare the references")
    next(b for b in app.button if b.label == "Compare Strategies").click().run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert set(app.session_state["retrieval_comparison"]) == {"fixed", "section"}
