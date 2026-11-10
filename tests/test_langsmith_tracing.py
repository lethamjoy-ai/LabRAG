import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langsmith import Client
from langsmith.run_helpers import get_tracing_context

from config.settings import Settings
from src.observability import langsmith_tracing as tracing
from src.rag.rag_graph import build_rag_graph
from src.retrieval.semantic_retriever import SearchResult
from src.llm.groq_service import AnswerSelection


@pytest.fixture
def captured(monkeypatch):
    # Exercise supported SDK decorators, but replace every transport operation.
    client = Client(api_key='test-only', auto_batch_tracing=False, info={})
    created, updated = [], []
    monkeypatch.setattr(Client, 'create_run', lambda *a, **kw: created.append(kw))
    monkeypatch.setattr(Client, 'update_run', lambda *a, **kw: updated.append(kw))
    monkeypatch.setattr(tracing, '_client', lambda key, endpoint: client)
    return created, updated


def graph(settings, fail=False):
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = SearchResult('q', 'section', 'section-chunks', [
        {'document': Document(page_content='BUN Serum specimen.', metadata={
            'document_id': 'private-file', 'chunk_id': 'chunk1', 'page_number': 590,
            'patient_name': 'Private Patient', 'api_key': 'secret-canary'}), 'score': .9}], .01)
    if fail:
        llm.select.side_effect = RuntimeError('secret-canary private prompt')
    else:
        llm.select.return_value = AnswerSelection(excerpts=[{'source_id': 'S1', 'quote': 'Serum specimen.'}])
    return build_rag_graph(retriever, llm, settings), retriever, llm


def test_settings_reads_project_dotenv(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('LANGSMITH_TRACING=true\nLANGSMITH_PROJECT=demo\nLANGSMITH_API_KEY=secret-canary\n')
    monkeypatch.setattr('config.settings.ENV_FILE', path)
    settings = Settings.from_env()
    assert settings.langsmith_tracing and settings.langsmith_project == 'demo'
    assert settings.langsmith_api_key == 'secret-canary'
    assert 'secret-canary' not in repr(settings)


def test_invalid_boolean(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('LANGSMITH_TRACING=maybe')
    monkeypatch.setattr('config.settings.ENV_FILE', path)
    with pytest.raises(ValueError, match='LANGSMITH_TRACING'):
        Settings.from_env()


def test_disabled_overrides_ambient_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    monkeypatch.setattr(tracing, '_client', Mock(side_effect=AssertionError('must not initialize')))
    workflow, _, llm = graph(Settings())
    def select(messages):
        assert get_tracing_context()['enabled'] is False
        return llm.select.return_value
    llm.select.side_effect = select
    assert workflow.invoke({'question': tracing.DEMO_QUESTION})['status'] == 'answered'
    llm.select.assert_called_once()


def test_complete_trace_and_redaction(captured):
    workflow, _, llm = graph(Settings(langsmith_tracing=True, langsmith_api_key='secret-canary'))
    result = workflow.invoke({'question': tracing.DEMO_QUESTION,
                             'history': [{'question': 'Private Patient secret-canary'}]})
    assert result['status'] == 'answered'
    llm.select.assert_called_once()
    created, updated = captured
    names = {row['name'] for row in created}
    assert {'LabRAG workflow', 'validate_question', 'resolve_follow_up', 'retrieve_evidence',
            'validate_evidence', 'generate_answer', 'validate_citations'} <= names
    assert any(row.get('parent_run_id') for row in created)
    payload = json.dumps([created, updated], default=str)
    for forbidden in ('secret-canary', 'Private Patient', 'private-file', 'Serum specimen.', 'patient_name'):
        assert forbidden not in payload
    assert 'source_count' in payload and '590' in payload
    assert tracing.DEMO_QUESTION in payload


def test_arbitrary_question_redacted(captured):
    workflow, _, _ = graph(Settings(langsmith_tracing=True, langsmith_api_key='test'))
    workflow.invoke({'question': 'Private Patient needs BUN specimen'})
    assert 'Private Patient' not in json.dumps(captured, default=str)


def test_error_preserved_but_not_uploaded(captured):
    workflow, _, llm = graph(Settings(langsmith_tracing=True, langsmith_api_key='test'), fail=True)
    with pytest.raises(RuntimeError, match='secret-canary'):
        workflow.invoke({'question': tracing.DEMO_QUESTION})
    llm.select.assert_called_once()
    payload = json.dumps(captured, default=str)
    assert 'secret-canary' not in payload and 'private prompt' not in payload
    assert 'failed' in payload


def test_telemetry_failure_never_repeats_provider(monkeypatch):
    monkeypatch.setattr(tracing, '_client', lambda key: object())
    def broken(*args, **kwargs):
        tracing._operation.get()()
        raise RuntimeError('telemetry unavailable')
    monkeypatch.setattr(tracing, '_record', broken)
    monkeypatch.setattr(tracing, '_record_llm', broken)
    workflow, retriever, llm = graph(Settings(langsmith_tracing=True, langsmith_api_key='test'))
    assert workflow.invoke({'question': tracing.DEMO_QUESTION})['status'] == 'answered'
    retriever.search.assert_called_once()
    llm.select.assert_called_once()


def test_missing_key_never_constructs_client(monkeypatch):
    monkeypatch.setattr(tracing, '_client', Mock(side_effect=AssertionError('must not initialize')))
    workflow, _, _ = graph(Settings(langsmith_tracing=True))
    assert workflow.invoke({'question': tracing.DEMO_QUESTION})['status'] == 'answered'


def test_telemetry_failure_before_execution_falls_back_once(monkeypatch):
    monkeypatch.setattr(tracing, '_client', lambda key: object())
    def broken(*args, **kwargs):
        raise RuntimeError('telemetry unavailable')
    monkeypatch.setattr(tracing, '_record', broken)
    monkeypatch.setattr(tracing, '_record_llm', broken)
    workflow, retriever, llm = graph(Settings(langsmith_tracing=True, langsmith_api_key='test'))
    assert workflow.invoke({'question': tracing.DEMO_QUESTION})['status'] == 'answered'
    retriever.search.assert_called_once()
    llm.select.assert_called_once()


def test_selected_demo_exists_in_unchanged_dataset():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'evaluation/mayo_day5_v3_overlap15_ground_truth.json'
    questions = json.loads(path.read_text(encoding='utf-8'))['questions']
    assert any(q['question'] == tracing.DEMO_QUESTION and q['expected_test_codes'] == ['BUN'] for q in questions)


def test_tracing_preserves_the_complete_final_response(captured):
    outcomes = []
    for enabled in (False, True):
        workflow, retriever, llm = graph(Settings(langsmith_tracing=enabled, langsmith_api_key='test'))
        result = workflow.invoke({'question': tracing.DEMO_QUESTION})
        outcomes.append({key: value for key, value in result.items() if key not in {'metrics', 'trace'}})
        retriever.search.assert_called_once()
        llm.select.assert_called_once()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('question', ['What specimen is required for BUN?',
                                    'How long can BUN serum be refrigerated or frozen?'])
@pytest.mark.parametrize('provider', ['groq','openai'])
@pytest.mark.parametrize('mode', ['extractive_v1','grounded_conversational_v1','grounded_conversational_v2'])
def test_full_demo_capture_consistent_for_bun_questions_providers_and_modes(captured, question, provider, mode):
    from pathlib import Path
    data=json.loads((Path(__file__).parent/'fixtures/bun_selected_evidence.json').read_text())['cases'][0]
    documents=[Document(page_content=s['text'],metadata=s['metadata']) for s in data['sources'].values()]
    retriever,llm=Mock(),Mock()
    retriever.search.return_value=SearchResult(question,'fixed','fixed-chunks',
        [{'document':doc,'score':.8} for doc in documents],.01)
    llm.select.return_value=AnswerSelection(excerpts=data['selected_previous_extractive_excerpts'])
    settings=Settings(llm_provider=provider,langsmith_tracing=True,langsmith_api_key='test-only',
                      langsmith_capture_full_io=True)
    workflow=build_rag_graph(retriever,llm,settings,raw_retrieval=True,answer_generation_mode=mode)
    result=workflow.invoke({'question':question})
    assert result['status']=='answered'
    created,updated=captured
    outputs=[r['outputs'] for r in updated if isinstance(r.get('outputs'),dict)]
    final=next(o for o in reversed(outputs) if o.get('status')=='answered' and o.get('question')==question)
    assert final['answer']==result['answer'] and final['answer_generation_mode']==mode
    assert final['citations'] and final['source_references'] and final['timings']
    assert final['validation']['quote_validation']=='passed'
    assert any(r.get('extra',{}).get('metadata',{}).get('labrag_provider')==provider for r in created)
    if mode=='grounded_conversational_v2':assert 'evidence_coverage' in final
    llm.select.assert_called_once();retriever.search.assert_called_once()


def test_capture_boolean_loaded_from_env_and_defaults_false(tmp_path,monkeypatch):
    assert not Settings().langsmith_capture_full_io
    path=tmp_path/'.env';monkeypatch.setattr('config.settings.ENV_FILE',path)
    path.write_text('LANGSMITH_CAPTURE_FULL_IO=true')
    assert Settings.from_env().langsmith_capture_full_io
    path.write_text('LANGSMITH_CAPTURE_FULL_IO=maybe')
    with pytest.raises(ValueError,match='LANGSMITH_CAPTURE_FULL_IO'):Settings.from_env()


def test_full_capture_still_withholds_unknown_patient_questions(captured):
    workflow,_,_=graph(Settings(langsmith_tracing=True,langsmith_api_key='test',langsmith_capture_full_io=True))
    workflow.invoke({'question':'What specimen for BUN? Patient name: Jane Doe DOB: 1/1/2000'})
    payload=json.dumps(captured,default=str)
    assert 'Jane Doe' not in payload and '1/1/2000' not in payload
    assert 'Serum specimen.' not in payload


def test_full_capture_does_not_upload_sensitive_source_answers(captured):
    workflow,_,_=graph(Settings(langsmith_tracing=True,langsmith_api_key='test',langsmith_capture_full_io=True))
    workflow.invoke({'question':tracing.DEMO_QUESTION})
    payload=json.dumps(captured,default=str)
    assert 'Private Patient' not in payload and 'secret-canary' not in payload
    assert 'Serum specimen.' not in payload  # Sensitive source blocks downstream full output.


def test_full_capture_redacts_credentials_and_reference_query_tokens():
    token=tracing._active.set((None,'demo','openai',True,('private-credential',)))
    try:
        value=tracing.summary({'question':tracing.DEMO_QUESTION,
            'answer':'Serum private-credential sk-testtoken Bearer hidden-token',
            'citations':[{'source_id':'S1','api_key':'private-credential',
                'source_url':'https://user:password@example.org/catalog?token=private-credential#secret'}]})
    finally:tracing._active.reset(token)
    payload=json.dumps(value)
    for secret in ('private-credential','sk-testtoken','hidden-token','password','api_key','?token='):
        assert secret not in payload
    assert value['citations'][0]['source_url']=='https://example.org/catalog'
