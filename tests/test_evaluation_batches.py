import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_evaluation_runner import runner
from tests.day5_helpers import synthetic_dataset,SyntheticGenerator
from src.evaluation.dataset_validator import fingerprint,validate_ground_truth


def engine_for(tmp_path,count=7):
    engine=runner(tmp_path)
    data=synthetic_dataset();original=data['questions'][0]
    data['questions']=[{**copy.deepcopy(original),'question_id':f'BATCH{i:02}.1','test_id':f'BATCH{i:02}'} for i in range(1,count+1)]
    data.pop('fingerprint');data['fingerprint']=fingerprint(data)
    engine.dataset=validate_ground_truth(data)
    return engine


def saved(path):return json.loads(path.read_text(encoding='utf-8'))


def test_default_five_questions_all_variants_repeated_resume_and_signature(tmp_path):
    engine=engine_for(tmp_path)
    first=engine.run()
    assert first['status']=='Batch Completed — Ready to Resume'
    assert len(first['records'])==20 and len(engine.retriever.calls)==20
    assert len(first['batch_session']['question_ids'])==5
    for q in first['batch_session']['question_ids']:
        assert {r['configuration'] for r in first['records'] if r['question_id']==q}==set('ABCD')
    assert len(first['parameters']['question_ids'])==7
    signature=first['signature'];parameters=copy.deepcopy(first['parameters'])
    path=tmp_path/'results'/(first['run_id']+'.json');before=path.read_bytes()
    assert saved(path)==first
    second=engine.run(resume=first['run_id'],max_new_questions_per_session=1)
    assert second['status']=='Batch Completed — Ready to Resume' and len(second['records'])==24
    assert second['signature']==signature and second['parameters']==parameters
    assert next((tmp_path/'backups').glob('**/*.json')).read_bytes()==before
    third=engine.run(resume=first['run_id'])
    assert third['status']=='Completed' and len(third['records'])==28
    assert len(engine.retriever.calls)==28
    assert third['run_id']==first['run_id'] and third['signature']==signature
    assert len({(r['question_id'],r['configuration']) for r in third['records']})==28
    assert not list((tmp_path/'locks').glob('*.lock'))


def test_partial_question_consumes_one_slot_finishes_pending_variants(tmp_path):
    engine=engine_for(tmp_path,3)
    interrupted=engine.run(cancel=lambda:len(engine.retriever.calls)>=2)
    assert interrupted['status']=='Interrupted' and len(interrupted['records'])==2
    previous=copy.deepcopy(interrupted['records'])
    result=engine.run(resume=interrupted['run_id'],max_new_questions_per_session=1)
    assert result['status']=='Batch Completed — Ready to Resume'
    assert len(result['records'])==4 and len(engine.retriever.calls)==4
    assert result['records'][:2]==previous
    assert result['batch_session']['question_ids']==['BATCH01.1']


def test_failed_pair_retry_counts_question_and_preserves_success_history(tmp_path):
    from src.llm.groq_service import LLMError
    engine=engine_for(tmp_path,4)
    generator=SyntheticGenerator();calls=[]
    class FailOnce:
        def select(self,messages):
            calls.append(1)
            if len(calls)==1:raise LLMError('Synthetic non-rate failure')
            return generator.select(messages)
    engine.service=FailOnce()
    first=engine.run(mode='rag',max_new_questions_per_session=2)
    assert first['status']=='Batch Completed — Ready to Resume'
    assert len(first['records'])==8 and sum(r['status']=='Failed' for r in first['records'])==1
    completed={(r['question_id'],r['configuration']):copy.deepcopy(r) for r in first['records'] if r['status']=='Completed'}
    resumed=engine.run(mode='rag',resume=first['run_id'],max_new_questions_per_session=2)
    assert resumed['batch_session']['question_ids']==['BATCH01.1','BATCH03.1']
    assert len(calls)==13  # Eight initial pairs, one retry, four variants for question 3.
    assert len(resumed['records'])==12 and resumed['status']=='Batch Completed — Ready to Resume'
    assert any(e.get('previous_status')=='Failed' for e in resumed['events'])
    for r in resumed['records']:
        key=(r['question_id'],r['configuration'])
        if key in completed:assert r==completed[key]


def test_previously_excluded_questions_do_not_consume_batch_slots(tmp_path):
    engine=engine_for(tmp_path,3)
    data=copy.deepcopy(engine.dataset)
    data['questions'][0].update(source_verified=False,score_ranking=False)
    data.pop('fingerprint');data['fingerprint']=fingerprint(data)
    engine.dataset=validate_ground_truth(data)
    first=engine.run(max_new_questions_per_session=1)
    assert all(r['status']=='Excluded' for r in first['records'])
    assert not engine.retriever.calls
    second=engine.run(resume=first['run_id'],max_new_questions_per_session=1)
    assert second['batch_session']['question_ids']==['BATCH02.1']
    assert len(second['records'])==8 and len(engine.retriever.calls)==4


def test_rate_limit_takes_precedence_over_batch_completion(tmp_path):
    from tests.test_rate_limit_pause import limited_service
    engine=engine_for(tmp_path)
    engine.service,backend=limited_service()
    result=engine.run(mode='rag',max_new_questions_per_session=1)
    assert result['status']=='Paused — Rate Limited'
    assert len(result['records'])==1 and backend.invoke.call_count==1
    assert not any(e.get('event')=='batch_completed' for e in result['events'])


@pytest.mark.parametrize('limit',[0,-1,True,1.5,1001])
def test_invalid_batch_limits_do_not_execute(tmp_path,limit):
    engine=engine_for(tmp_path)
    with pytest.raises(ValueError,match='Maximum new questions'):
        engine.run(max_new_questions_per_session=limit)
    assert not engine.retriever.calls and not (tmp_path/'results').exists()


def test_ui_control_defaults_to_five_and_never_executes(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    from src.evaluation.evaluation_runner import EvaluationRunner
    root=tmp_path/'evaluation';root.mkdir()
    (root/'synthetic_ground_truth.json').write_text(json.dumps(synthetic_dataset()),encoding='utf-8')
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No automatic evaluation'))
    monkeypatch.setattr(EvaluationRunner,'run',forbidden)
    app=AppTest.from_string('from src.ui_day5 import render_day5\nfrom config.settings import Settings\nrender_day5(Settings())').run(timeout=30)
    assert not app.exception
    assert next(w for w in app.number_input if w.label=='Maximum new questions per session').value==5
    forbidden.assert_not_called()


def test_existing_checkpoint_remains_unchanged():
    from src.evaluation.report_generator import summarize,csv_text
    path=Path('evaluation/results/20261106T075303_624a0c1d52.json')
    if not path.exists():pytest.skip('Optional historical checkpoint')
    before=path.read_bytes();run=json.loads(before)
    signature=run['signature'];summarize(run);csv_text(run)
    assert path.read_bytes()==before and run['signature']==signature
