from pathlib import Path
from unittest.mock import Mock
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
from tests.day3_fakes import embedding_service, connection

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_retired_model_in_session_updates_to_configured_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    app = AppTest.from_file(APP)
    app.session_state["chat_model"] = "llama-3.3-70b-versatile"
    app.run(timeout=30)
    assert not app.exception
    assert next(w for w in app.text_input if w.label == "Chat model").value == "openai/gpt-oss-120b"


def test_chat_submit_citations_and_clear(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','groq')
    import src.ui_day3 as services
    import src.ui_day4 as chat
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    conn = connection()
    doc = Document(page_content="BUN Serum specimen. Frozen 365 days.", metadata={"chunk_id": "chunk-1"})
    conn.vector_store.return_value.similarity_search_by_vector_with_score.return_value = [(doc, .8)]
    monkeypatch.setattr(services, "services", lambda settings: (conn, embedding_service()))
    model = Mock()
    from src.llm.groq_service import AnswerSelection
    model.select.return_value = AnswerSelection(excerpts=[{"source_id": "S1", "quote": "Frozen 365 days."}])
    monkeypatch.setattr(chat, "GroqService", lambda settings: model)
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception and not app.error
    next(w for w in app.selectbox if w.label=='Answer generation mode').select('extractive_v1').run(timeout=30)
    next(w for w in app.text_input if w.label=="Laboratory question").set_value("How long can BUN be frozen?")
    next(b for b in app.button if b.label=="Send").click().run(timeout=30)
    assert not app.exception and not app.error
    assert "Frozen 365 days. [S1]" in app.session_state["rag_history"][0]["answer"]
    assert model.select.call_count == 1
    next(b for b in app.button if b.label == "Clear Conversation").click().run(timeout=30)
    assert "rag_history" not in app.session_state
