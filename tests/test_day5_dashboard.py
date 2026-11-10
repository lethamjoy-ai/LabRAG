from pathlib import Path
from unittest.mock import Mock
from streamlit.testing.v1 import AppTest
from config.settings import Settings
from tests.day5_helpers import synthetic_dataset,SyntheticRetriever,SyntheticCrossEncoder,SyntheticGenerator
from src.evaluation.evaluation_runner import EvaluationRunner
from src.retrieval.reranker import Reranker


def test_dashboard_reads_measured_run_without_startup_services(tmp_path,monkeypatch):
    import json
    import src.ui_day5 as ui
    from src.vectorstore.pinecone_client import PineconeConnection
    from src.llm.groq_service import GroqService
    from src.retrieval import reranker
    root=tmp_path
    (root/'evaluation').mkdir()
    (root/'evaluation/synthetic_ground_truth.json').write_text(json.dumps(synthetic_dataset()),encoding='utf-8')
    run=EvaluationRunner(synthetic_dataset(),Settings(),SyntheticRetriever(),SyntheticGenerator(),Reranker(backend=SyntheticCrossEncoder()),'mock',root/'evaluation').run(mode='rag')
    monkeypatch.setattr(ui,'ROOT',root)
    forbidden=Mock(side_effect=AssertionError('No API or model load on page render'))
    monkeypatch.setattr(PineconeConnection,'connect',forbidden)
    monkeypatch.setattr(GroqService,'select',forbidden)
    monkeypatch.setattr(reranker,'load_cross_encoder',forbidden)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception and not app.error
    assert any('SYNTHETIC FIXTURE RUN' in w.value for w in app.warning)
    assert next(s for s in app.selectbox if s.label=='Day 5 historical run').options
    protocol=next(s for s in app.selectbox if s.label=='Day 5 retrieval protocol')
    assert protocol.value=='raw'
    assert protocol.options==['Raw Similarity Retrieval','Test-Context Retrieval']
    assert any(b.label=='Export Day 5 Markdown' for b in app.get('download_button'))
    next(s for s in app.selectbox if s.label=='Day 5 configuration filter').select('C').run(timeout=30)
    assert next(s for s in app.selectbox if s.label=='Inspect Day 5 question').options==['SYN01.1 / C']
    next(b for b in app.button if b.label=='Start Day 5 Evaluation').click().run(timeout=30)
    assert any('No API calls' in w.value for w in app.warning)
    forbidden.assert_not_called()


def test_chat_reranking_remains_disabled_by_default(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception
    assert not next(c for c in app.checkbox if c.label=='Enable local reranking').value
