import json
from unittest.mock import Mock
import pytest
from config.settings import Settings
from tests.day5_helpers import synthetic_dataset,SyntheticRetriever,SyntheticCrossEncoder,SyntheticGenerator
from src.evaluation.evaluation_runner import EvaluationRunner
from src.evaluation.dataset_validator import validate_ground_truth,fingerprint
from src.retrieval.reranker import Reranker


def runner(tmp_path,retriever=None):
    return EvaluationRunner(synthetic_dataset(),Settings(),retriever or SyntheticRetriever(),SyntheticGenerator(),Reranker(backend=SyntheticCrossEncoder()),'mock',tmp_path)


def test_four_configurations_and_reports_preserve_history(tmp_path):
    engine=runner(tmp_path)
    result=engine.run()
    assert len(result['records'])==4 and all(r['status']=='Completed' for r in result['records'])
    assert [c[2] for c in engine.retriever.calls]==[5,5,15,15]
    assert {r['strategy'] for r in result['records']}=={'fixed','section'}
    assert result['summary'][0]['mrr']==.5 and result['summary'][2]['mrr']==1
    assert result['summary'][0]['recall']==1
    path=tmp_path/'results'/f"{result['run_id']}.json"
    before=path.read_bytes()
    another=engine.run()
    assert another['run_id']!=result['run_id'] and path.read_bytes()==before
    assert path.with_suffix('.csv').exists()
    assert (tmp_path/'reports'/f"{result['run_id']}_summary.md").exists()


def test_interrupt_resume_and_configuration_guard(tmp_path):
    engine=runner(tmp_path);calls=[]
    result=engine.run(cancel=lambda:len(calls)>=1,progress=lambda *args:calls.append(args))
    assert result['status']=='Interrupted' and len(result['records'])==1
    resumed=engine.run(resume=result['run_id'])
    assert len(resumed['records'])==4 and len(engine.retriever.calls)==4
    with pytest.raises(ValueError,match='changed'):engine.run(resume=result['run_id'],top_k=3)


def test_rag_claims_citations_and_selected_evidence(tmp_path):
    result=runner(tmp_path).run(mode='rag')
    assert all(r['status']=='Completed' for r in result['records'])
    for record in result['records']:
        assert record['faithfulness']['supported_claim_rate']==1
        assert record['citation_evaluation']['validity_rate']==record['citation_evaluation']['support_rate']==1
        assert '0.25' in record['answer']
        assert len(record['retrieved_chunk_ids'])<=5
        if record['reranking_applied']:
            assert any(t['node']=='rerank_evidence' for t in record['node_timings'])
            assert record['citations'][0]['source_id']=='S2'


@pytest.mark.integration
def test_optional_day5_live_four_configuration_demo():
    import os
    from pathlib import Path
    if os.getenv('LABRAG_RUN_DAY5_LIVE')!='1':pytest.skip('Explicit Day 5 live initiation required')
    data=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    run=EvaluationRunner(data).run(question_ids=['TC07.1'])
    assert len(run['records'])==4
    assert all(r['status']=='Completed' for r in run['records'])


def test_errors_are_recorded_safely_without_fallback(tmp_path):
    retriever=Mock();retriever.search.side_effect=RuntimeError('SECRET API BODY')
    result=runner(tmp_path,retriever).run()
    assert all(r['status']=='Failed' for r in result['records'])
    assert 'SECRET' not in json.dumps(result)


def test_source_guards_and_unverified_exclusion(tmp_path,monkeypatch):
    monkeypatch.delenv('LABRAG_RUN_DAY5_LIVE',raising=False)
    with pytest.raises(ValueError,match='opt-in'):EvaluationRunner(synthetic_dataset(),root=tmp_path).run()
    data=synthetic_dataset();data['questions'][0].update(source_verified=False,score_ranking=False)
    data['fingerprint']=fingerprint({k:v for k,v in data.items() if k!='fingerprint'})
    retriever=SyntheticRetriever()
    result=EvaluationRunner(data,Settings(),retriever,source='mock',root=tmp_path).run()
    assert not retriever.calls and all(r['status']=='Excluded' for r in result['records'])
    assert all(s['hit_rate'] is None for s in result['summary'])
    data['questions'][0]['score_ranking']=True
    data['fingerprint']=fingerprint({k:v for k,v in data.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='Unverified'):validate_ground_truth(data)


def test_judged_pool_recall_and_regressions_reported(tmp_path):
    from src.evaluation.report_generator import reranking_regressions
    result=runner(tmp_path).run()
    # Simulate a measured degradation to validate reporting, not production scores.
    record=next(r for r in result['records'] if r['configuration']=='C')
    record['retrieval_metrics'].update(hit_rate=0,mrr=0,precision=0)
    assert reranking_regressions(result)[0]['reranked']=='C'
