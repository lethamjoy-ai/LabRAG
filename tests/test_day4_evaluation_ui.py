import json
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_day4_report_viewer_and_filters_never_run_live(tmp_path,monkeypatch):
    import src.ui_day4_evaluation as ui
    from src.evaluation.day4_runner import load_cases,run_cases,export_report
    from src.vectorstore.pinecone_client import PineconeConnection
    from src.llm.groq_service import GroqService
    from unittest.mock import Mock
    report=run_cases(mode="mock")
    path=tmp_path/"day4_results.json"
    export_report(report,path)
    monkeypatch.setattr(ui,"RESULTS",path)
    cloud=Mock(side_effect=AssertionError("No live calls from Evaluation UI"))
    monkeypatch.setattr(PineconeConnection,"connect",cloud)
    monkeypatch.setattr(GroqService,"select",cloud)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py").run(timeout=30)
    assert not app.exception and not app.error
    assert next(m for m in app.metric if m.label=="Total test cases").value=="20"
    assert next(m for m in app.metric if m.label=="Passed").value=="19"
    next(w for w in app.selectbox if w.label=="Day 4 test code").select("CPR").run(timeout=30)
    inspector=next(w for w in app.selectbox if w.label=="Inspect Day 4 case")
    assert "TC14" in inspector.options and "TC16" not in inspector.options
    inspector.select("TC14").run(timeout=30)
    assert any("Regarding c-peptide serum" in str(m.value) for m in app.markdown)
    next(b for b in app.button if b.label=="Run mocked Day 4 validation").click().run(timeout=30)
    assert not app.exception and not app.error
    assert json.loads(path.read_text(encoding="utf-8"))["mode"]=="mock"
    cloud.assert_not_called()


def test_no_report_shows_not_tested_without_creating_files(tmp_path,monkeypatch):
    import src.ui_day4_evaluation as ui
    path=tmp_path/"no-report.json"
    monkeypatch.setattr(ui,"RESULTS",path)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py").run(timeout=30)
    assert not app.exception
    assert next(m for m in app.metric if m.label=="Not Tested").value=="20"
    assert not path.exists()
