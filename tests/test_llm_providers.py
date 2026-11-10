import copy
import json
from dataclasses import replace
from unittest.mock import Mock
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from config.settings import Settings
from src.llm.groq_service import GroqService,LLMError,InvalidStructuredGeneration,ExcerptIdSelection
from src.llm.provider_factory import create_llm,build_backend,strict_schema,OpenAIService
from src.rag.answer_generator import generate_answer
from src.rag.citation_manager import validate_selection
from tests.test_error_recovery import HttpFailure


@pytest.mark.parametrize('provider',['groq','openai'])
def test_shared_exact_selection_validation_and_usage(provider):
    settings=Settings(llm_provider=provider,llm_model='gpt-4.1-mini' if provider=='openai' else 'openai/gpt-oss-120b')
    backend=Mock();backend.invoke.return_value={'parsed':{'selected_excerpt_ids':['S1_E1']},'parsing_error':None,
        'raw':AIMessage(content='SECRET untrusted text',usage_metadata={'input_tokens':12,'output_tokens':4,'total_tokens':16})}
    service=create_llm(settings,backend)
    text='Minimum volume:\n   0.25 mL.';sources={'S1':Document(page_content=text)}
    selection=generate_answer(service,'Minimum volume?',sources)
    assert selection.excerpts[0].quote==text and validate_selection(selection,sources)
    assert service.last_usage=={'input_tokens':12,'output_tokens':4,'total_tokens':16,'provider':provider,'model':settings.llm_model}
    assert 'SECRET' not in json.dumps(selection.model_dump())
    backend.invoke.return_value={'selected_excerpt_ids':['S999_E1']}
    with pytest.raises(InvalidStructuredGeneration):generate_answer(service,'q',sources)
    assert service.last_usage is None


def test_provider_specific_backend_settings_and_closed_schema(monkeypatch):
    import langchain_openai,langchain_groq
    groq=Mock();openai=Mock()
    monkeypatch.setattr(langchain_groq,'ChatGroq',groq)
    monkeypatch.setattr(langchain_openai,'ChatOpenAI',openai)
    settings=Settings(llm_provider='groq',llm_model='openai/gpt-oss-120b',groq_api_key='fake-groq',openai_api_key='fake-openai')
    build_backend(settings,ExcerptIdSelection)
    assert groq.return_value.with_structured_output.call_args.kwargs['method']=='function_calling'
    build_backend(replace(settings,llm_provider='openai',llm_model='gpt-4.1-mini'),ExcerptIdSelection)
    args=openai.return_value.with_structured_output.call_args
    assert args.kwargs=={'method':'json_schema','strict':True,'include_raw':True}
    schema=args.args[0]
    assert set(schema['required'])==set(schema['properties']) and not schema['additionalProperties']
    assert all('default' not in v for v in schema['properties'].values())
    assert openai.call_args.kwargs['max_retries']==0
    assert openai.call_args.kwargs['base_url']=='https://api.openai.com/v1'
    assert openai.call_args.kwargs['max_completion_tokens']==1200


@pytest.mark.parametrize('provider,key',[('groq','GROQ_API_KEY'),('openai','OPENAI_API_KEY')])
def test_missing_keys_are_lazy_actionable_and_safe(provider,key):
    settings=Settings(llm_provider=provider,llm_model='gpt-4.1-mini' if provider=='openai' else 'openai/gpt-oss-120b')
    service=create_llm(settings)
    assert service.backend is None
    with pytest.raises(LLMError,match=key):service.select([])


def test_unknown_openai_model_rejected_without_api_call(monkeypatch):
    import langchain_openai
    forbidden=Mock(side_effect=AssertionError('No provider call'))
    monkeypatch.setattr(langchain_openai,'ChatOpenAI',forbidden)
    with pytest.raises(LLMError,match='strict structured output'):
        create_llm(Settings(llm_provider='openai',llm_model='unsupported-model',openai_api_key='fake')).select([])
    forbidden.assert_not_called()


@pytest.mark.parametrize('status,code,category,retries',[
    (429,'insufficient_quota','billing_or_quota',0),(401,'invalid_api_key','authentication_or_permission',0),
    (400,'context_length_exceeded','invalid_request',0),(404,'model_not_found','invalid_request',0),
    (429,'rate_limit_exceeded','rate_limit',2)])
def test_openai_errors_are_sanitized_and_never_fall_back(status,code,category,retries):
    error=HttpFailure(status);error.body={'error':{'code':code,'message':'SECRET patient Alice'}}
    backend=Mock();backend.invoke.side_effect=error
    service=OpenAIService(Settings(llm_provider='openai',llm_model='gpt-4.1-mini'),backend,sleep=Mock())
    with pytest.raises(LLMError) as caught:service.select([])
    diagnostic=caught.value.error_diagnostics
    assert diagnostic['provider']=='openai' and diagnostic['provider_error_category']==category
    assert diagnostic['retry_count']==retries and backend.invoke.call_count==retries+1
    assert 'OpenAI' in diagnostic['sanitized_message']
    assert 'SECRET' not in json.dumps(diagnostic) and 'Alice' not in str(caught.value)


def test_evaluation_provider_signature_lock_usage_and_historical_exports(tmp_path):
    from tests.test_evaluation_runner import runner
    from src.evaluation.report_generator import csv_text
    engine=runner(tmp_path)
    engine.settings=replace(engine.settings,llm_provider='openai',llm_model='gpt-4.1-mini')
    backend=Mock();backend.invoke.return_value={'parsed':{'selected_excerpt_ids':['S2_E1']},'parsing_error':None,
        'raw':AIMessage(content='',usage_metadata={'input_tokens':10,'output_tokens':2,'total_tokens':12})}
    engine.service=create_llm(engine.settings,backend)
    run=engine.run(mode='rag',configurations=['A'])
    assert run['records'][0]['status']=='Completed'
    assert run['models']['llm_provider']=='openai' and run['records'][0]['llm_model']=='gpt-4.1-mini'
    assert run['records'][0]['token_usage']['input_tokens']==10
    assert 'token_usage' in csv_text(run) and 'openai' in csv_text(run)
    path=tmp_path/'results'/(run['run_id']+'.json');before=path.read_bytes()
    engine.settings=replace(engine.settings,llm_provider='groq',llm_model='openai/gpt-oss-120b')
    with pytest.raises(ValueError,match='changed'):engine.run(mode='rag',configurations=['A'],resume=run['run_id'])
    assert path.read_bytes()==before and backend.invoke.call_count==1


def test_judge_is_separate_explicit_and_default_off(tmp_path):
    from tests.test_evaluation_runner import runner
    engine=runner(tmp_path)
    engine.source='live'
    with pytest.raises(ValueError,match='explicit judge'):
        engine.run(mode='rag',enable_judge=True,allow_live=True)
    assert not (tmp_path/'results').exists()


def test_billing_failure_pauses_without_retrying_other_pairs(tmp_path):
    from tests.test_evaluation_runner import runner
    engine=runner(tmp_path);engine.settings=replace(engine.settings,llm_provider='openai',llm_model='gpt-4.1-mini')
    error=HttpFailure(429);error.body={'error':{'code':'insufficient_quota'}}
    backend=Mock();backend.invoke.side_effect=error
    engine.service=OpenAIService(engine.settings,backend,sleep=Mock())
    run=engine.run(mode='rag')
    assert run['status']=='Paused — Provider Configuration' and len(run['records'])==1
    assert backend.invoke.call_count==1


def test_runtime_provider_and_models_persist_without_startup_calls(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_presentation as presentation
    import src.rag.rag_graph as graph_module
    import src.ui_day3 as ui_services
    monkeypatch.setattr(presentation,'sample_questions',lambda root:[])
    forbidden=Mock(side_effect=AssertionError('No request when switching'))
    monkeypatch.setattr(ui_services,'services',forbidden)
    app=AppTest.from_string('from src.ui_day4 import render_chat\nfrom config.settings import Settings\nrender_chat(Settings())').run(timeout=30)
    selector=lambda:next(w for w in app.selectbox if w.label=='LLM Provider')
    assert selector().value=='openai'
    selector().select('groq').run(timeout=30)
    next(w for w in app.text_input if w.label=='Chat model').set_value('groq-fixture-model').run(timeout=30)
    app.session_state['rag_history']=[{'question':'Fixture question','answer':'Fixture answer','llm_provider':'groq','llm_model':'groq-fixture-model'}]
    selector().select('openai').run(timeout=30)
    assert not app.exception and next(w for w in app.text_input if w.label=='Chat model').value=='gpt-4.1-mini'
    assert len(app.session_state['rag_history'])==1
    next(w for w in app.text_input if w.label=='Chat model').set_value('gpt-4.1').run(timeout=30)
    selector().select('groq').run(timeout=30)
    assert next(w for w in app.text_input if w.label=='Chat model').value=='groq-fixture-model'
    selector().select('openai').run(timeout=30)
    assert next(w for w in app.text_input if w.label=='Chat model').value=='gpt-4.1'
    assert any('may incur charges' in c.value for c in app.caption)
    forbidden.assert_not_called()


def test_historical_groq_reports_remain_unchanged():
    from src.evaluation.report_generator import csv_text,summarize
    path=Path('evaluation/results/20261106T075303_624a0c1d52.json')
    if not path.exists():pytest.skip('Optional saved report')
    before=path.read_bytes();run=json.loads(before)
    summarize(run);csv_text(run)
    assert path.read_bytes()==before


def test_selected_provider_serves_followup_without_regenerating_history(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_presentation as presentation
    import src.ui_day3 as ui_services
    import src.rag.rag_graph as graph_module
    monkeypatch.setattr(presentation,'sample_questions',lambda root:[])
    monkeypatch.setattr(ui_services,'services',Mock(return_value=(Mock(),Mock())))
    graph=Mock()
    graph.invoke.side_effect=lambda payload:{'question':payload['question'],'answer':'OpenAI fixture answer','status':'abstained'}
    build=Mock(return_value=graph);monkeypatch.setattr(graph_module,'build_rag_graph',build)
    app=AppTest.from_string('from src.ui_day4 import render_chat\nfrom config.settings import Settings\nrender_chat(Settings())').run(timeout=30)
    old={'question':'What specimen for BUN?','answer':'Groq fixture answer','llm_provider':'groq','llm_model':'groq-fixture'}
    app.session_state['rag_history']=[old]
    next(w for w in app.selectbox if w.label=='LLM Provider').select('openai').run(timeout=30)
    graph.invoke.assert_not_called()
    next(w for w in app.text_input if w.label=='Laboratory question').set_value('How long can it be frozen?')
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert not app.exception and isinstance(build.call_args.args[1],OpenAIService)
    assert build.call_args.args[2].llm_provider=='openai'
    assert graph.invoke.call_args.args[0]['history']==[old]
    assert app.session_state['rag_history'][0]==old
    assert app.session_state['rag_history'][1]['llm_provider']=='openai'
    app.run(timeout=30);assert graph.invoke.call_count==1


def test_judge_identity_is_independent_signed_and_not_inherited(tmp_path,monkeypatch):
    from tests.test_evaluation_runner import runner
    import src.evaluation.evaluation_runner as module
    from src.evaluation.answer_judge import GroqAnswerQualityJudge
    engine=runner(tmp_path);engine.source='live'
    engine.settings=replace(engine.settings,llm_provider='openai',llm_model='gpt-4.1-mini')
    observed=[];original=module.evaluate_faithfulness
    def faithfulness(q,a,s,codes,abstained,judge):
        observed.append((judge.settings.llm_provider,judge.settings.llm_model))
        return original(q,a,s,codes,abstained,None)
    monkeypatch.setattr(module,'evaluate_faithfulness',faithfulness)
    monkeypatch.setattr(GroqAnswerQualityJudge,'assess',lambda self,*args:{'provider':self.settings.llm_provider,'model':self.settings.llm_model,'estimate_only':True})
    result=engine.run(mode='rag',configurations=['A'],enable_judge=True,judge_provider='groq',judge_model='groq-judge-fixture',allow_live=True)
    assert result['records'][0]['status']=='Completed'
    assert observed==[('groq','groq-judge-fixture')]
    assert result['models']['judge_provider']=='groq' and result['models']['llm_provider']=='openai'
    with pytest.raises(ValueError,match='changed'):
        engine.run(mode='rag',configurations=['A'],enable_judge=True,judge_provider='openai',judge_model='gpt-4.1-mini',resume=result['run_id'],allow_live=True)
