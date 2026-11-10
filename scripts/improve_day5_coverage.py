"""Offline v3 label reconciliation; never overwrite historical artifacts."""
import copy
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.reconcile_day5_ground_truth import reconcile
from src.evaluation.dataset_validator import fingerprint,validate_ground_truth

TARGETS={'TC03.1','TC04.1','TC05.1','TC06.1','TC10.1','TC11.1','TC14.1','TC14.2'}


def improve(historical,current,runs,snapshot_root=Path('.')):
    validate_ground_truth(current)
    checked=reconcile(historical,runs,snapshot_root,cross_strategy_source=True)
    result=copy.deepcopy(current)
    result.update(dataset_version='v3-overlap15-source-page-reconciliation',parent_fingerprint=current['fingerprint'],
        reconciliation_evidence=checked['reconciliation_evidence'],
        reconciliation_method='Section source spans independently verified against complete fixed source pages with matching document/source hash/page and all observed section characters')
    lookup={q['question_id']:q for q in checked['questions']}
    for q in result['questions']:
        if q['question_id'] not in TARGETS:continue
        verified=lookup[q['question_id']]
        q['labels']=verified['labels']
        q['score_ranking']=verified['score_ranking']
        q['reconciliation_status']='Independently verified overlap-15 labels for both strategies' if q['score_ranking'] else 'Excluded: no saved overlap-15 section candidate ID with verified test-specific supporting text; local overlap-10 IDs cannot establish indexed overlap-15 identity'
    result.pop('fingerprint',None);result['fingerprint']=fingerprint(result)
    return validate_ground_truth(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('evaluation/mayo_day5_v3_overlap15_ground_truth.json'))
    args=parser.parse_args()
    historical=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    current=json.loads(Path('evaluation/mayo_day5_v2_overlap15_ground_truth.json').read_text(encoding='utf-8'))
    result=improve(historical,current,sorted(Path('evaluation/results').glob('*.json')))
    output=args.output
    with output.open('x',encoding='utf-8') as handle:handle.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for q in result['questions']:
        if q['question_id'] in TARGETS:print(q['question_id'],q['score_ranking'],q['labels']['section']['chunk_ids'])
