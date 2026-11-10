import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.improve_day5_coverage import TARGETS
from scripts.reconcile_day5_ground_truth import reconcile
from src.evaluation.dataset_validator import validate_ground_truth,fingerprint


def load(name):return json.loads(Path('evaluation',name).read_text(encoding='utf-8'))


def test_v3_eligibility_and_historical_preservation():
    old=validate_ground_truth(load('mayo_day5_v2_overlap15_ground_truth.json'))
    new=validate_ground_truth(load('mayo_day5_v3_overlap15_ground_truth.json'))
    assert new['parent_fingerprint']==old['fingerprint']
    assert sum(q['score_ranking'] for q in old['questions'])==14
    assert sum(q['score_ranking'] for q in new['questions'])==21
    original={q['question_id']:q for q in old['questions']}
    for q in new['questions']:
        if q['question_id'] not in TARGETS:assert q==original[q['question_id']]
        if q['question_id'] in TARGETS-{'TC11.1'}:
            assert q['score_ranking'] and all(q['labels'][s]['chunk_ids'] for s in ('fixed','section'))
            for strategy in ('fixed','section'):
                passages=[' '.join(p['text'].split()).casefold() for p in q['labels'][strategy]['verified_passages']]
                assert all(any(' '.join(f.split()).casefold() in text for text in passages) for f in q['expected_facts'])
        if q['question_id']=='TC11.1':
            assert not q['score_ranking'] and q['labels']['section']['chunk_ids']==[]
            assert 'no saved overlap-15 section candidate ID' in q['reconciliation_status']
    assert new['chunk_configurations']==old['chunk_configurations']


def test_new_section_labels_are_captured_and_question_specific():
    new=load('mayo_day5_v3_overlap15_ground_truth.json')
    captures={}
    for evidence in new['reconciliation_evidence']:
        path=Path(evidence['path'])
        if not path.exists():continue
        assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence['sha256']
        for row in json.loads(path.read_text(encoding='utf-8')).get('records',[]):
            if row.get('strategy')!='section':continue
            for item in row.get('candidates',[]):captures.setdefault(item['metadata']['chunk_id'],[]).append(item)
    if not captures:pytest.skip('Saved captures are optional local artifacts')
    for q in new['questions']:
        if q['question_id'] not in TARGETS:continue
        for passage in q['labels']['section']['verified_passages']:
            assert passage['chunk_id'] in captures
            assert passage['test_code'] in q['expected_test_codes']
            assert passage['printed_catalog_page'] in q['verified_catalog_pages']
            assert any(' '.join(f.split()).casefold() in ' '.join(passage['text'].split()).casefold() for f in q['expected_facts'])
            matching=[c for c in captures[passage['chunk_id']] if c['metadata'].get('chunk_overlap')==15
                and c['metadata'].get('chunk_size')==100 and c['metadata'].get('document_id')==passage['document_id']
                and c['metadata'].get('source_content_hash')==passage['source_content_hash']
                and c['metadata'].get('start_char')==passage['start_char'] and c['metadata'].get('end_char')==passage['end_char']]
            assert matching
    by_id={q['question_id']:q for q in new['questions']}
    assert by_id['TC14.2']['labels']['section']['chunk_ids']==['688b43d90495aa541a0243b8947565b169d4d6febc2f521e016893ce550dcee2']
    assert '602d282688ace8ce86228995de11294e9ea6638bdc8b5d71d21941f92772b66c' not in by_id['TC14.2']['labels']['section']['chunk_ids']


def synthetic_sources(tmp_path,conflict=False):
    text='BAP    Bone Alkaline Phosphatase, Serum\nSpecimen Volume: 0.6 mL\nPage 505'
    document='verified-document';source='same-source-hash';start=text.index('Specimen Volume')
    snapshots=[]
    for strategy in ('fixed','section'):
        chunks=[(0,len(text))] if strategy=='fixed' else [(0,start-1),(start,len(text))]
        rows=[]
        for i,(a,b) in enumerate(chunks):
            value=text[a:b]
            if conflict and strategy=='section' and a==0:value='X'+value[1:]
            rows.append({'text':value,'metadata':{'document_id':document,'source_content_hash':source,'page_number':6,
                'start_char':a,'end_char':b,'chunk_id':f'old-{strategy}-{i}'}})
        path=tmp_path/f'{strategy}.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
        snapshots.append({'strategy':strategy,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    q={'question_id':'TC03.1','source_verified':True,'score_ranking':True,'answerability':'answerable',
        'expected_test_codes':['BAP'],'expected_facts':['Specimen Volume: 0.6 mL'],
        'labels':{s:{'namespace':s+'-chunks','chunk_ids':['old'],'complete':False} for s in ('fixed','section')}}
    original={'snapshots':snapshots,'source_sha256':document,'questions':[q]};original['fingerprint']=fingerprint(original)
    captures=[]
    for strategy in ('fixed','section'):
        raw=text[start:];m={'chunk_id':'saved-'+strategy,'document_id':document,'source_content_hash':source,
            'page_number':6,'start_char':start,'end_char':len(text),'chunk_content_hash':hashlib.sha256(raw.encode()).hexdigest(),
            'test_code':'BAP','test_name':'Bone Alkaline Phosphatase, Serum','chunking_strategy':strategy,'chunk_size':100,'chunk_overlap':15}
        captures.append({'strategy':strategy,'namespace':strategy+'-chunks','candidates':[{'text':raw,'metadata':m}]})
    run={'source':'live','index_name':'lab-rag','run_id':'saved','records':captures}
    path=tmp_path/'saved.json';path.write_text(json.dumps(run),encoding='utf-8')
    return original,path


def test_cross_strategy_source_requires_exact_agreement(tmp_path):
    original,path=synthetic_sources(tmp_path)
    assert not reconcile(original,[path])['questions'][0]['score_ranking']
    before=copy.deepcopy(original)
    result=reconcile(original,[path],cross_strategy_source=True)
    assert result['questions'][0]['score_ranking']
    assert original==before


def test_conflicting_section_source_is_not_filled(tmp_path):
    original,path=synthetic_sources(tmp_path,conflict=True)
    result=reconcile(original,[path],cross_strategy_source=True)
    assert not result['questions'][0]['score_ranking']
    assert not result['questions'][0]['labels']['section']['chunk_ids']
