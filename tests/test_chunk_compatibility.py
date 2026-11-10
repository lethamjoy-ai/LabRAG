import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from src.evaluation.chunk_compatibility import check_compatibility,label_configurations
from src.evaluation.dataset_validator import fingerprint,validate_ground_truth
from src.evaluation.retrieval_metrics import ranking_metrics
from tests.test_evaluation_runner import runner


def match(overlap=15):
    return {'document':Document(page_content='Specimen Minimum Volume: 0.25 mL',metadata={
        'chunking_strategy':'fixed','chunk_size':100,'chunk_overlap':overlap,'chunk_id':'new'}),'score':.8}


def test_configuration_mismatch_never_remaps_ids():
    dataset={'chunk_configurations':{'fixed':{'chunking_strategy':'fixed','chunk_size':100,'chunk_overlap':10}}}
    details=check_compatibility(dataset,'fixed',[match()])
    assert details['status']=='Incompatible ground-truth labels' and not details['compatible']
    assert check_compatibility(dataset,'fixed',[match(10)])['compatible']
    assert not check_compatibility(dataset,'fixed',[])['compatible']
    assert ranking_metrics(['new'],['old'])['hit_rate']==0  # Metric remains strictly ID-based.


@pytest.mark.parametrize('mode',['retrieval','rag'])
def test_runner_withholds_metrics_and_generation_for_mismatched_labels(tmp_path,mode):
    engine=runner(tmp_path)
    engine.dataset.pop('synthetic')
    engine.dataset['chunk_configurations']={'fixed':{'chunking_strategy':'fixed','chunk_size':100,'chunk_overlap':10}}
    from src.retrieval.semantic_retriever import SearchResult
    engine.retriever=Mock()
    engine.retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[match()],0)
    engine.service=Mock(side_effect=AssertionError('No generation on incompatible evidence'))
    result=engine.run(configurations=['A'],mode=mode)
    row=result['records'][0]
    assert row['status']=='Incompatible'
    assert row['retrieval_metrics']['hit_rate'] is None and not row['retrieval_metrics']['scored']
    assert row['failure_analysis'][0]['category']=='Incompatible ground-truth labels'
    engine.service.select.assert_not_called()
    assert result['summary'][0]['scored_questions']==0


def test_versioned_calcium_labels_and_historical_parent():
    old=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    new=validate_ground_truth(json.loads(Path('evaluation/mayo_day5_v2_overlap15_ground_truth.json').read_text(encoding='utf-8')))
    assert new['parent_fingerprint']==old['fingerprint']
    assert all(c['chunk_overlap']==10 for c in label_configurations(old).values())
    assert all(c['chunk_overlap']==15 for c in label_configurations(new).values())
    q=next(q for q in new['questions'] if q['question_id']=='TC07.1')
    assert q['score_ranking'] and q['expected_facts']==['Specimen Minimum Volume: 0.25 mL']
    for s,prefix in [('fixed','d8221455'),('section','b880ceb2')]:
        label=q['labels'][s]
        assert len(label['chunk_ids'])==1 and label['chunk_ids'][0].startswith(prefix)
        passage=label['verified_passages'][0]
        assert passage['test_code']=='CA' and passage['printed_catalog_page']=='590'
        assert 'Specimen Minimum Volume: 0.25 mL' in passage['text']
        assert not label['complete']
    assert all(not q['score_ranking'] for q in new['questions'] if q['reconciliation_status'].startswith('Unscorable'))
    for evidence in new['reconciliation_evidence']:
        if Path(evidence['path']).exists():assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest()==evidence['sha256']


@pytest.mark.parametrize('tamper',['source_content_hash','test_name','start_char','chunk_content_hash','document_id'])
def test_reconciliation_rejects_unverified_identity_spans_and_hashes(tmp_path,tamper):
    from scripts.reconcile_day5_ground_truth import reconcile
    old=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    if not Path('evaluation/results/20261104T071713_458becbc74.json').exists() or any(not Path(s['path']).exists() for s in old['snapshots']):
        pytest.skip('Optional saved captures/local source snapshots are not distributed')
    old['questions']=[next(q for q in old['questions'] if q['question_id']=='TC07.1')]
    old.pop('fingerprint');old['fingerprint']=fingerprint(old)
    capture=json.loads(Path('evaluation/results/20261104T071713_458becbc74.json').read_text(encoding='utf-8'))
    capture['records']=[r for r in capture['records'] if r['question_id']=='TC07.1']
    for row in capture['records']:
        for item in row['candidates']:item['metadata'][tamper]=-99 if tamper=='start_char' else 'unverified'
    path=tmp_path/'capture.json';path.write_text(json.dumps(capture),encoding='utf-8')
    result=reconcile(old,[path])
    assert not result['questions'][0]['score_ranking']
    assert all(not label['chunk_ids'] for label in result['questions'][0]['labels'].values())
