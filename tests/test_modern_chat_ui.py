import copy
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest
from src.llm.groq_service import LLMError

APP='from src.ui_day4 import render_chat\nfrom config.settings import Settings\nrender_chat(Settings(llm_provider="groq",llm_model="openai/gpt-oss-120b"))'


@pytest.fixture
def chat(monkeypatch):
    import src.ui_day3 as services
    import src.ui_presentation as presentation
    import src.rag.rag_graph as graph_module
    monkeypatch.setattr(presentation,'sample_questions',lambda root:['What specimen is required for BUN?','What is the minimum volume for Calcium?'])
    service=Mock(return_value=(Mock(),Mock()))
    monkeypatch.setattr(services,'services',service)
    graph=Mock();inputs=[]
    def invoke(payload):
        inputs.append(copy.deepcopy(payload))
        return {'question':payload['question'],'query':payload['question'],'answer':'Verified fixture answer [S1]',
            'status':'answered','citations':[{'source_id':'S1','test_name':'Synthetic BUN','test_code':'BUN',
                'printed_catalog_page':'1','source_url':'https://example.org/catalog'}],
            'sources':{},'matches':[],'metrics':{}}
    graph.invoke.side_effect=invoke
    monkeypatch.setattr(graph_module,'build_rag_graph',Mock(return_value=graph))
    app=AppTest.from_string(APP).run(timeout=30)
    assert not app.exception
    service.assert_not_called()
    return app,graph,inputs,service


def submit(app,text):
    next(w for w in app.text_input if w.label=='Laboratory question').set_value(text)
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert not app.exception and not app.error


def test_native_input_order_persistence_and_followup_context(chat):
    app,graph,inputs,_=chat
    assert next(w for w in app.text_input if w.label=='Laboratory question').placeholder=='Ask a laboratory question...'
    assert not app.chat_input
    assert not app.text_area
    submit(app,'What specimen is required for BUN?')
    submit(app,'How long can it be frozen?')
    assert [m.proto.name for m in app.chat_message]==['user','assistant','user','assistant']
    assert inputs[0]['history']==[]
    assert inputs[1]['history'][0]['question']=='What specimen is required for BUN?'
    assert len(app.session_state['rag_history'])==2
    app.run(timeout=30)
    assert graph.invoke.call_count==2 and len(app.session_state['rag_history'])==2
    assert any(c.value=='Synthetic BUN · BUN' for c in app.caption)
    assert any(b.label=='Open source S1' for b in app.get('link_button'))
    assert any(e.proto.label=='View Retrieval Details' and not e.proto.expanded for e in app.get('expander'))
    assert any(e.proto.label=='Copy answer' and not e.proto.expanded for e in app.get('expander'))


@pytest.mark.parametrize('button',['New Chat','Clear Conversation'])
def test_chat_reset_preserves_evaluation_and_resets_followup(chat,button):
    app,_,inputs,_=chat
    submit(app,'What specimen is required for BUN?')
    app.session_state['last_day5_run']='keep-evaluation'
    app.session_state['chunk_outputs']={'keep':'chunks'}
    next(b for b in app.button if b.label==button).click().run(timeout=30)
    assert not app.exception and 'rag_history' not in app.session_state
    assert not app.chat_message
    assert app.session_state['last_day5_run']=='keep-evaluation'
    assert app.session_state['chunk_outputs']=={'keep':'chunks'}
    submit(app,'What specimen is required for BUN?')
    assert inputs[-1]['history']==[]


def test_sample_uses_one_submission_flow_settings_do_not_resubmit(chat):
    app,graph,inputs,_=chat
    next(w for w in app.selectbox if w.label=='Sample Questions').select('What specimen is required for BUN?').run(timeout=30)
    graph.invoke.assert_not_called()
    assert next(w for w in app.text_input if w.label=='Laboratory question').value=='What specimen is required for BUN?'
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert not app.exception
    assert inputs[0]['question']=='What specimen is required for BUN?'
    app.run(timeout=30)
    next(s for s in app.selectbox if s.label=='Chat retrieval strategy').select('fixed').run(timeout=30)
    assert graph.invoke.call_count==1 and len(app.session_state['rag_history'])==1
    # Explicit repetition is allowed; reruns alone never repeat a request.
    submit(app,'What specimen is required for BUN?')
    assert graph.invoke.call_count==2


def test_rate_limit_is_friendly_preserves_question_and_allows_later_submission(chat):
    app,graph,_,_=chat
    submit(app,'What specimen is required for BUN?')
    error=LLMError('SECRET patient Alice provider payload',{'http_status':429})
    graph.invoke.side_effect=error
    submit(app,'How long can it be frozen?')
    turns=app.session_state['rag_history']
    assert len(turns)==2 and turns[-1]['question']=='How long can it be frozen?'
    assert turns[-1]['answer']=="Groq's API usage limit has been reached. Please try again after the quota resets or select OpenAI."
    assert turns[-1]['error'] and 'citations' not in turns[-1]
    assert not any('SECRET' in m.value or 'Alice' in m.value for m in app.markdown)
    assert graph.invoke.call_count==2
    app.run(timeout=30);assert graph.invoke.call_count==2
    graph.invoke.side_effect=lambda payload:{'question':payload['question'],'answer':'Later fixture response','status':'abstained'}
    submit(app,'What specimen is required for BUN?')
    assert graph.invoke.call_count==3 and len(app.session_state['rag_history'])==3


def test_arbitrary_exception_text_never_reaches_ui(chat):
    app,graph,_,_=chat
    graph.invoke.side_effect=RuntimeError('SECRET patient Alice raw request')
    submit(app,'What specimen is required for BUN?')
    assert 'SECRET' not in app.session_state['rag_history'][-1]['answer']
    assert not any('Alice' in m.value for m in app.markdown)


def test_samples_populate_editable_draft_and_remain_after_responses(chat):
    app,graph,inputs,_=chat
    sample=lambda:next(w for w in app.selectbox if w.label=='Sample Questions')
    draft=lambda:next(w for w in app.text_input if w.label=='Laboratory question')
    sample().select('What specimen is required for BUN?').run(timeout=30)
    assert draft().value=='What specimen is required for BUN?'
    graph.invoke.assert_not_called()
    draft().set_value('What specimen and container are required for BUN?').run(timeout=30)
    graph.invoke.assert_not_called()
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert inputs[0]['question']=='What specimen and container are required for BUN?'
    assert draft().value=='' and sample().value not in ('What specimen is required for BUN?','What is the minimum volume for Calcium?')
    sample().select('What is the minimum volume for Calcium?').run(timeout=30)
    assert draft().value=='What is the minimum volume for Calcium?'
    assert graph.invoke.call_count==1 and len(app.session_state['rag_history'])==1
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert inputs[-1]['history'][0]['question']==inputs[0]['question']
    assert len(app.session_state['rag_history'])==2 and draft().value==''
    # Reset selection permits selecting exactly the same sample again, deliberately.
    sample().select('What is the minimum volume for Calcium?').run(timeout=30)
    assert draft().value=='What is the minimum volume for Calcium?'
    assert graph.invoke.call_count==2


def test_unsent_draft_survives_provider_changes_and_empty_send_does_not_call(chat):
    app,graph,_,_=chat
    draft=lambda:next(w for w in app.text_input if w.label=='Laboratory question')
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    graph.invoke.assert_not_called()
    assert any('Enter a laboratory question' in w.value for w in app.warning)
    draft().set_value('Unsent custom laboratory question').run(timeout=30)
    next(w for w in app.selectbox if w.label=='LLM Provider').select('openai').run(timeout=30)
    assert draft().value=='Unsent custom laboratory question'
    app.run(timeout=30)
    assert draft().value=='Unsent custom laboratory question' and not app.chat_message
    graph.invoke.assert_not_called()


def test_error_restores_draft_without_automatically_repeating_request(chat):
    app,graph,_,_=chat
    graph.invoke.side_effect=RuntimeError('SECRET')
    submit(app,'Custom laboratory question')
    assert next(w for w in app.text_input if w.label=='Laboratory question').value=='Custom laboratory question'
    app.run(timeout=30)
    assert graph.invoke.call_count==1 and len(app.session_state['rag_history'])==1


def test_chat_settings_live_only_in_sidebar_and_persist(chat):
    app,graph,_,_=chat
    settings=next(e for e in app.sidebar.get('expander') if e.proto.label=='Chat Settings')
    assert not settings.proto.expanded
    assert len([s for s in app.sidebar.selectbox if s.label=='LLM Provider'])==1
    assert not any(s.label in {'LLM Provider','Chat retrieval strategy','Answer generation mode'} for s in app.main.selectbox)
    next(s for s in app.sidebar.selectbox if s.label=='LLM Provider').select('openai').run(timeout=30)
    next(s for s in app.sidebar.selectbox if s.label=='Chat retrieval strategy').select('fixed').run(timeout=30)
    next(n for n in app.sidebar.number_input if n.label=='Chat top-k').set_value(7).run(timeout=30)
    app.run(timeout=30)
    assert next(s for s in app.sidebar.selectbox if s.label=='LLM Provider').value=='openai'
    assert next(s for s in app.sidebar.selectbox if s.label=='Chat retrieval strategy').value=='fixed'
    assert next(n for n in app.sidebar.number_input if n.label=='Chat top-k').value==7
    graph.invoke.assert_not_called()
    submit(app,'What specimen is required for BUN?')
    before=copy.deepcopy(app.session_state['rag_history'][0])
    next(s for s in app.sidebar.selectbox if s.label=='LLM Provider').select('groq').run(timeout=30)
    assert app.session_state['rag_history'][0]==before
    assert graph.invoke.call_count==1


def test_supporting_documents_group_all_existing_reference_tools(chat):
    app,graph,_,_=chat
    submit(app,'What specimen is required for BUN?')
    groups=[e for e in app.main.get('expander') if e.proto.label=='Supporting Documents']
    assert len(groups)==1 and not groups[0].proto.expanded
    group=groups[0]
    assert any(b.label=='Open source S1' for b in group.get('link_button'))
    assert any(c.value=='Verified fixture answer [S1]' for c in group.code)
    assert any('Synthetic BUN' in c.value for c in group.caption)
    assert any(e.proto.label=='View Retrieval Details' for e in group.get('expander'))
    assert any('Supporting source citations' in m.value for m in group.markdown)
    app.run(timeout=30)
    assert graph.invoke.call_count==1
