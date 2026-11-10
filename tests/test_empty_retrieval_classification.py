"""Offline classification regressions; empty evidence never becomes a success."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from src.evaluation.chunk_compatibility import check_compatibility
from src.evaluation.dashboard_reporting import common_question_metrics,comparison_rows
from src.evaluation.dataset_validator import fingerprint
from src.retrieval.semantic_retriever import SearchResult
from tests.test_evaluation_runner import runner


def evidence(overlap=15,document='verified-fixture'):
    return {'chunk_id':'fixture-heading','document_id':document,'source_content_hash':'fixture-source',
        'chunking_strategy':'fixed','chunk_size':100,'chunk_overlap':overlap}


def engine(tmp_path):
    instance=runner(tmp_path)
    instance.dataset.pop('synthetic')
    instance.dataset['source_sha256']='verified-fixture'
    instance.dataset['chunk_configurations']={s:{'chunking_strategy':s,'chunk_size':100,'chunk_overlap':15} for s in ('fixed','section')}
    q=instance.dataset['questions'][0]
    q.update(question_id='TC06.1',question='What specimen and minimum volume are required for CRP serum?',
        resolved_question='What specimen and minimum volume are required for CRP serum?',expected_test_codes=['CRP'])
    q['labels']['fixed']['chunk_ids']=['fixture-positive']
    instance.dataset['fingerprint']=fingerprint({k:v for k,v in instance.dataset.items() if k!='fingerprint'})
    return instance


@pytest.mark.parametrize('intermediate,status,state',[
    ([evidence()],'Completed','compatible'),
    (None,'Unverified','unverified'),
    ([evidence(10)],'Incompatible','mismatched'),
    ([evidence(document='other')],'Unverified','unverified'),
    ([{'chunk_id':'id-only'}],'Unverified','unverified'),
])
def test_empty_result_classification(tmp_path,intermediate,status,state):
    instance=engine(tmp_path)
    instance.retriever=Mock()
    instance.retriever.search_with_context.return_value=SearchResult('q','fixed','fixed-chunks',[],0,
        {'configuration_evidence':intermediate} if intermediate is not None else {})
    result=instance.run(configurations=['A'],protocol='test_context')
    row=result['records'][0]
    assert row['status']==status and row['compatibility']['state']==state
    metrics=row['retrieval_metrics']
    if status=='Completed':
        assert metrics['scored'] and all(metrics[k]==0 for k in ('hit_rate','mrr','precision','judged_pool_recall') if metrics[k] is not None)
        assert metrics['recall']==0  # This fixture's positive pool is exhaustive.
        assert row['retrieval_outcome']=='Retrieval miss'
        assert result['summary'][0]['retrieval_misses']==1
    else:
        assert not metrics['scored'] and metrics['hit_rate'] is None
    assert not instance.retriever.search.called


def test_partial_labels_empty_compatible_pool_has_zero_judged_recall(tmp_path):
    instance=engine(tmp_path);instance.dataset['questions'][0]['labels']['fixed']['complete']=False
    instance.retriever=Mock()
    instance.retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[],0,{'configuration_evidence':[evidence()]})
    row=instance.run(configurations=['A'])['records'][0]
    assert row['retrieval_metrics']['judged_pool_recall']==0 and row['retrieval_metrics']['recall'] is None


def test_tc06_a_misses_c_finds_before_reranking_without_depth_change(tmp_path):
    instance=engine(tmp_path)
    positive={'document':Document(page_content='Minimum specimen volume: 0.25 mL.',metadata={
        **evidence(),'chunk_id':'fixture-positive','test_code':'CRP','test_name':'Synthetic CRP'}),'score':.7}
    depths=[]
    class Retriever:
        def search_with_context(self,query,strategy,top_k,filters=None):
            depths.append(top_k)
            return SearchResult(query,strategy,'fixed-chunks',[positive] if top_k==15 else [],0,{'configuration_evidence':[evidence()]})
    instance.retriever=Retriever()
    result=instance.run(configurations=['A','C'],protocol='test_context')
    assert depths==[5,15]
    a,c=result['records']
    assert a['status']==c['status']=='Completed'
    assert a['retrieval_metrics']['hit_rate']==0 and c['retrieval_metrics']['hit_rate']==1
    assert c['retrieved'][0]['metadata']==positive['document'].metadata
    assert a['compatibility']['state']==c['compatibility']['state']=='compatible'


def test_missing_config_is_unverified_not_mismatched():
    data={'source_sha256':'verified-fixture','chunk_configurations':{'fixed':{'chunking_strategy':'fixed','chunk_size':100,'chunk_overlap':15}}}
    incomplete=evidence();incomplete.pop('chunk_overlap')
    assert check_compatibility(data,'fixed',[],[incomplete])['state']=='unverified'
    assert check_compatibility(data,'fixed',[],[evidence(),evidence(10)])['state']=='mismatched'


def test_common_set_metrics_and_historical_immutability():
    run={'parameters':{'question_ids':['one','two'],'configurations':['A','B']},'records':[]}
    for config in ('A','B'):
        run['records'].append({'question_id':'one','configuration':config,'status':'Completed',
            'retrieval_metrics':{'scored':True,'hit_rate':0,'mrr':0,'precision':0,'judged_pool_recall':0}})
    run['records'] += [{'question_id':'two','configuration':'A','status':'Incompatible'},
        {'question_id':'two','configuration':'B','status':'Completed','retrieval_metrics':{'scored':True,'hit_rate':1,'mrr':1,'precision':1,'judged_pool_recall':1}}]
    before=copy.deepcopy(run);common=common_question_metrics(run)
    assert common['question_ids']==['one'] and common['coverage_percentage']==50
    assert all(r['Common scored questions']==1 and r['hit_rate']==0 for r in common['summaries'])
    assert run==before


def test_saved_tc06_report_remains_unmodified():
    path=Path('evaluation/results/20261105T022431_313eb14e27.json')
    if not path.exists():pytest.skip('Optional saved report')
    before=path.read_bytes();run=json.loads(before)
    common=common_question_metrics(run)
    assert len(common['question_ids'])==20 and 'TC06.1' not in common['question_ids']
    assert common['summaries'][0]['mrr']==pytest.approx(.8016666667)
    rows=comparison_rows(run)
    assert next(r for r in rows if r['Question ID']=='TC06.1' and r['Configuration']=='A')['Status']=='Incompatible'
    assert path.read_bytes()==before


def test_common_question_dashboard_and_unverified_filter(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    root=tmp_path/'evaluation'
    run=runner(root).run()
    run['parameters']['question_ids'].append('second')
    for config in ('A','B','C','D'):
        row=copy.deepcopy(run['records'][0]);row.update(question_id='second',configuration=config)
        if config=='A':row.update(status='Unverified',retrieval_metrics={'scored':False,'hit_rate':None})
        run['records'].append(row)
    path=root/'results'/f"{run['run_id']}.json";path.write_text(json.dumps(run),encoding='utf-8')
    before=path.read_bytes()
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No provider calls'))
    monkeypatch.setattr('src.vectorstore.pinecone_client.PineconeConnection.connect',forbidden)
    app=AppTest.from_string('from src.ui_day5 import render_day5\nfrom config.settings import Settings\nrender_day5(Settings())').run(timeout=30)
    assert not app.exception
    assert any('Common-question comparison' in s.value for s in app.subheader)
    assert next(m for m in app.metric if m.label=='Unverified questions').value=='1'
    next(s for s in app.selectbox if s.label=='Day 5 result filter').select('Unverified').run(timeout=30)
    assert next(s for s in app.selectbox if s.label=='Inspect Day 5 question').options==['second / A']
    assert path.read_bytes()==before
    forbidden.assert_not_called()
