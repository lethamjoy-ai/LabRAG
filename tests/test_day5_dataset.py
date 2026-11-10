import json
from pathlib import Path
import pytest
from src.evaluation.dataset_validator import validate_ground_truth


def test_ground_truth_has_distinct_namespace_pools_and_unscored_negatives():
    data=validate_ground_truth(json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8')))
    assert len(data['questions'])==26
    assert {q['test_id'] for q in data['questions']}=={f'TC{i:02d}' for i in range(1,21)}
    assert all(q['source_verified'] and q['answerability']=='answerable' for q in data['questions'] if q['score_ranking'])
    ca=next(q for q in data['questions'] if q['question_id']=='TC07.1')
    assert ca['relevant_document_ids'] and 590 in ca['source_catalog_pages']
    assert set(ca['labels']['fixed']['chunk_ids']).isdisjoint(ca['labels']['section']['chunk_ids'])
    assert not ca['labels']['fixed']['complete'] and not ca['labels']['section']['complete']
    data['questions'][0]['expected_facts']=['unverified changed value']
    with pytest.raises(ValueError,match='fingerprint'):validate_ground_truth(data)


def test_relevance_ids_exist_in_local_snapshots_when_available():
    data=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    for strategy in ['fixed','section']:
        files=list(Path('data/processed',strategy).glob('*.jsonl'))
        if not files:pytest.skip('Local snapshots are not distributed with code')
        observed={json.loads(line)['metadata']['chunk_id'] for path in files for line in path.open(encoding='utf-8')}
        expected={identifier for q in data['questions'] for identifier in q['labels'][strategy]['chunk_ids']}
        assert expected<=observed
