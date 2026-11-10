import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from config.settings import Settings
from src.llm.groq_service import GroqService,LLMError
from src.llm.error_diagnostics import classify,retry_delay
from src.llm.request_pacing import RequestPacer,ACTIVE
from src.evaluation.checkpoint_safety import ResumeDeferred,run_lock
from tests.test_evaluation_runner import runner
from tests.test_error_recovery import HttpFailure


def limited_service(delay='900'):
    backend=Mock();backend.invoke.side_effect=HttpFailure(429,{'retry-after':delay})
    return GroqService(Settings(),backend,sleep=Mock()),backend


@pytest.mark.parametrize('header,seconds,inline',[
    ('5',5,5),('900',900,None),('invalid',None,None),('-2',None,None),('inf',None,None),
])
def test_safe_retry_after_diagnostics(header,seconds,inline):
    error=HttpFailure(429,{'retry-after':header})
    detail=classify(error)
    assert detail['retry_after_seconds']==seconds
    assert retry_delay(error,detail,0)==inline
    assert bool(detail['resume_after'])==(seconds is not None)
    assert 'SECRET' not in json.dumps(detail)


def test_quota_headers_are_allowlisted_and_not_assumed_per_minute():
    error=HttpFailure(429,{'retry-after':'90','x-ratelimit-remaining-tokens':'0',
        'x-ratelimit-reset-tokens':'3m','Authorization':'SECRET','x-ratelimit-remaining-requests':'8'})
    detail=classify(error)
    assert detail['quota_type']=='tokens' and detail['quota_reset_seconds']==180
    assert detail['backoff_reason']=='provider_delay_exceeds_inline_budget'
    assert datetime.fromisoformat(detail['resume_after'])>datetime.now(timezone.utc)+timedelta(seconds=178)
    assert 'SECRET' not in json.dumps(detail) and 'Authorization' not in json.dumps(detail)
    assert retry_delay(error,detail,0) is None


def test_pause_is_atomic_no_later_requests_or_pairs(tmp_path):
    engine=runner(tmp_path);engine.service,backend=limited_service()
    run=engine.run(mode='rag')
    assert run['status']=='Paused — Rate Limited'
    assert len(run['records'])==1 and run['records'][0]['status']=='Failed'
    assert backend.invoke.call_count==1 and len(engine.retriever.calls)==1
    assert run['rate_limit']['http_status']==429 and run['rate_limit']['resume_after']
    saved=json.loads((tmp_path/'results'/(run['run_id']+'.json')).read_text(encoding='utf-8'))
    assert saved==run and not list((tmp_path/'locks').glob('*.lock'))
    assert not list((tmp_path/'results').glob('*.tmp'))


def test_retry_exhaustion_also_pauses_missing_headers(tmp_path):
    engine=runner(tmp_path);engine.service,backend=limited_service(None)
    run=engine.run(mode='rag')
    assert run['status']=='Paused — Rate Limited' and len(run['records'])==1
    assert backend.invoke.call_count==3
    assert run['rate_limit']['retry_count']==2
    assert run['rate_limit']['resume_after'] is None


def test_resume_cooldown_then_skip_success_and_preserve_history(tmp_path,monkeypatch):
    engine=runner(tmp_path)
    backend=Mock();backend.invoke.side_effect=[{'selected_excerpt_ids':['S2_E1']},HttpFailure(429,{'retry-after':'900'})]
    engine.service=GroqService(Settings(),backend,sleep=Mock())
    paused=engine.run(mode='rag',configurations=['A','B'])
    assert [r['status'] for r in paused['records']]==['Completed','Failed']
    path=tmp_path/'results'/(paused['run_id']+'.json');before=path.read_bytes()
    with pytest.raises(ResumeDeferred):engine.run(mode='rag',configurations=['A','B'],resume=paused['run_id'])
    assert path.read_bytes()==before and backend.invoke.call_count==2
    with run_lock(tmp_path,paused['run_id']):
        with pytest.raises(ValueError,match='locked'):
            engine.run(mode='rag',configurations=['A','B'],resume=paused['run_id'])
    with pytest.raises(ValueError,match='changed'):
        engine.run(mode='rag',configurations=['A','B'],top_k=3,resume=paused['run_id'])
    class Later(datetime):
        @classmethod
        def now(cls,tz=None):return datetime.now(tz)+timedelta(seconds=1000)
    monkeypatch.setattr('src.evaluation.evaluation_runner.datetime',Later)
    backend.invoke.side_effect=None;backend.invoke.return_value={'selected_excerpt_ids':['S2_E1']}
    recovered=engine.run(mode='rag',configurations=['A','B'],resume=paused['run_id'])
    assert recovered['status']=='Completed' and backend.invoke.call_count==3
    assert recovered['records'][0]==paused['records'][0]
    assert recovered['events'][-1]['previous_error_diagnostics']['http_status']==429
    assert next((tmp_path/'backups').glob('**/*.json')).read_bytes()==before


def test_run_wide_pacing_across_services_and_retries(tmp_path):
    ticks=[0.0];waits=[]
    def sleep(seconds):waits.append(seconds);ticks[0]+=seconds
    pacer=RequestPacer(1,30,clock=lambda:ticks[0],sleep=sleep)
    token=ACTIVE.set(pacer)
    try:
        for _ in range(3):
            backend=Mock();backend.invoke.return_value={'excerpts':[]}
            GroqService(Settings(),backend).select([])
        assert waits==[2,2]
    finally:ACTIVE.reset(token)
    assert ACTIVE.get() is None


def test_optional_judge_429_is_not_swallowed():
    from src.evaluation.answer_judge import GroqAnswerQualityJudge
    from src.evaluation.faithfulness_evaluator import GroqFaithfulnessJudge,evaluate_faithfulness
    from langchain_core.documents import Document
    for kind in ('quality','faithfulness'):
        backend=Mock();backend.invoke.side_effect=HttpFailure(429,{'retry-after':'900'})
        with pytest.raises(LLMError):
            if kind=='quality':GroqAnswerQualityJudge(Settings(),backend).assess('q','a',{},[])
            else:evaluate_faithfulness('q','Serum volume is 0.25 mL [S1]',
                {'S1':Document(page_content='Serum volume is 0.25 mL')},judge=GroqFaithfulnessJudge(Settings(),backend))
        assert backend.invoke.call_count==1


def test_graph_requests_share_run_pacer(tmp_path,monkeypatch):
    ticks=[0.0];seen=[];waits=[]
    def sleep(seconds):waits.append(seconds);ticks[0]+=seconds
    cls=RequestPacer
    monkeypatch.setattr('src.llm.request_pacing.RequestPacer',
        lambda minimum,requests_per_minute:cls(minimum,requests_per_minute,clock=lambda:ticks[0],sleep=sleep))
    backend=Mock()
    def invoke(messages):
        seen.append(ACTIVE.get())
        return {'selected_excerpt_ids':['S2_E1']}
    backend.invoke.side_effect=invoke
    engine=runner(tmp_path);engine.service=GroqService(Settings(),backend,sleep=Mock())
    result=engine.run(mode='rag',configurations=['A','B'],min_request_interval=2)
    assert all(r['status']=='Completed' for r in result['records'])
    assert len(seen)==2 and seen[0] is seen[1] and seen[0] is not None
    assert waits==[2]
    assert result['execution_controls']['min_request_interval']==2


def test_paused_dashboard_is_read_only(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    root=tmp_path/'evaluation'
    engine=runner(root);engine.service,_=limited_service();engine.run(mode='rag')
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No live calls'))
    monkeypatch.setattr(GroqService,'select',forbidden)
    app=AppTest.from_string('from src.ui_day5 import render_day5\nfrom config.settings import Settings\nrender_day5(Settings())').run(timeout=30)
    assert not app.exception
    assert any('Paused — Rate Limited' in w.value for w in app.warning)
    forbidden.assert_not_called()


def test_historical_results_remain_byte_identical():
    from src.evaluation.report_generator import summarize,csv_text
    path=Path('evaluation/results/20261106T075303_624a0c1d52.json')
    if not path.exists():pytest.skip('Optional historical run')
    before=path.read_bytes();run=json.loads(before)
    summarize(run);csv_text(run)
    assert path.read_bytes()==before
