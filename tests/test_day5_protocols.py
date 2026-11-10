"""Offline protocol integration; no model downloads or provider requests."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from config.settings import Settings
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.retrieval_comparator import RetrievalComparator
from src.retrieval.reranker import Reranker
from src.evaluation.evaluation_runner import failure_analysis
from src.evaluation.report_generator import csv_text, markdown_report
from tests.day3_fakes import embedding_service, connection
from tests.test_evaluation_runner import runner


@pytest.mark.parametrize('strategy', ['fixed','section'])
@pytest.mark.parametrize('rerank', [False,True])
def test_calcium_context_recovers_minimum_and_respects_next_test(strategy,rerank):
    # Simulate Pinecone's indexed spans, with the real context/enrichment code.
    text='CA    Calcium, Total, Serum\nSpecimen Volume: 0.5 mL\nSpecimen Minimum Volume: 0.25 mL\nCAI    Calcium, Ionized, Serum\nSpecimen Minimum Volume: 1 mL\nPage 590'
    boundaries=[0,text.index('Specimen Minimum'),text.index('CAI    '),len(text)]
    docs=[Document(page_content=text[a:b],metadata={'document_id':'synthetic','source_content_hash':'hash',
        'page_number':91,'chunk_id':f'{strategy}-{i}','start_char':a,'end_char':b,'test_name':'','test_code':''})
        for i,(a,b) in enumerate(zip(boundaries,boundaries[1:]))]
    conn=connection()
    def search(vector,k,filter):
        page=next((c['page_number']['$eq'] for c in filter['$and'] if 'page_number' in c),None)
        return [(d,.8) for d in docs] if page==91 else ([] if page else [(docs[0],.8)])
    conn.vector_store.return_value.similarity_search_by_vector_with_score.side_effect=search
    retriever=SemanticRetriever(Settings(),embedding_service(),conn)
    backend=Mock();backend.predict.side_effect=lambda pairs,**kw:[1 if '0.25' in t else 0 for _,t in pairs]
    key={('fixed',False):'A',('section',False):'B',('fixed',True):'C',('section',True):'D'}[strategy,rerank]
    result=RetrievalComparator(retriever,Reranker(backend=backend)).run(
        'What is the minimum specimen volume for Calcium, Total, Serum?',key,protocol='test_context')
    matches=result['matches']
    diagnostics=result['diagnostics']
    assert 'final returned context candidates' in diagnostics['candidate_stage']
    assert diagnostics['identity_search_ids'] and diagnostics['intermediate_expanded_ids']
    assert diagnostics['returned_context_ids']==[m['document'].metadata['chunk_id'] for m in result['candidate_matches']]
    assert matches and any('0.25 mL' in m['document'].page_content for m in matches)
    assert all(m['document'].metadata['test_code']=='CA' for m in matches)
    assert all(m['document'].metadata['test_name']=='Calcium, Total, Serum' for m in matches)
    assert all('Ionized' not in m['document'].page_content and '1 mL' not in m['document'].page_content for m in matches)
    for match in matches:
        candidate=next(c for c in result['candidate_matches'] if c['document'].metadata['chunk_id']==match['document'].metadata['chunk_id'])
        assert match['document'] is candidate['document'] and match['score']==candidate['score']
    conn.index.upsert.assert_not_called();conn.index.delete.assert_not_called()


@pytest.mark.parametrize('mode',['retrieval','rag'])
def test_runner_protocol_routing_signature_and_reports(tmp_path,mode):
    engine=runner(tmp_path)
    original=engine.retriever.search
    engine.retriever.search_with_context=Mock(side_effect=original)
    result=engine.run(protocol='test_context',mode=mode)
    assert all(r['status']=='Completed' for r in result['records'])
    assert result['parameters']['protocol']=='test_context'
    assert engine.retriever.search_with_context.call_count==4
    assert [c[2] for c in engine.retriever.calls]==[5,5,15,15]
    assert all(r['retrieval_metrics']['hit_rate']==1 for r in result['records'])
    assert 'test_context' in csv_text(result) and 'test_context' in markdown_report(result)
    if mode=='retrieval':
        assert all(r['answer']=='' for r in result['records'])
        assert 'Not generated — retrieval-only mode' in csv_text(result)
    with pytest.raises(ValueError,match='changed'):
        engine.run(resume=result['run_id'],protocol='raw',mode=mode)


def test_raw_default_and_historical_report_fallback(tmp_path):
    engine=runner(tmp_path)
    engine.retriever.search_with_context=Mock(side_effect=AssertionError('raw must not expand'))
    result=engine.run()
    engine.retriever.search_with_context.assert_not_called()
    assert [c[2] for c in engine.retriever.calls]==[5,5,15,15]
    del result['parameters']['protocol']
    for r in result['records']:del r['protocol']
    assert '**raw**' in markdown_report(result)
    assert 'raw' in csv_text(result)


def test_failure_causes_are_independent_of_identity_metadata():
    question={'labels':{'fixed':{'chunk_ids':['gold']}},'expected_test_codes':['CA']}
    row={'strategy':'fixed','retrieval_metrics':{'scored':True,'hit_rate':0},'candidate_chunk_ids':['other'],
        'retrieved':[{'metadata':{'test_code':'','test_name':''}}]}
    findings=failure_analysis(row,question)
    assert {f['category'] for f in findings}=={'Candidate retrieval issue','Metadata issue'}
    row['candidate_chunk_ids']=['gold']
    assert failure_analysis(row,question)[0]['category']=='Ranking issue'
    row['retrieval_metrics']['hit_rate']=1
    assert [f['category'] for f in failure_analysis(row,question)]==['Metadata issue']


def test_tc07_historical_labels_remain_documented():
    data=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    q=next(q for q in data['questions'] if q['question_id']=='TC07.1')
    assert q['expected_facts']==['Specimen Volume: 0.5 mL','Specimen Minimum Volume: 0.25 mL']


def test_retrieval_dashboard_displays_explanation_without_calls(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    from tests.day5_helpers import synthetic_dataset
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'evaluation/synthetic_ground_truth.json').write_text(json.dumps(synthetic_dataset()),encoding='utf-8')
    runner(tmp_path/'evaluation').run()
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No live calls'))
    monkeypatch.setattr('src.vectorstore.pinecone_client.PineconeConnection.connect',forbidden)
    monkeypatch.setattr('src.retrieval.reranker.load_cross_encoder',forbidden)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception
    assert any('Not generated — retrieval-only mode' in e.value for e in app.markdown)
    forbidden.assert_not_called()
