"""Explicit live evaluation only; source verification/report preparation is offline."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluation_runner import EvaluationRunner

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('evaluation/mayo_day5_ground_truth.json'))
    parser.add_argument('--configurations',nargs='+',default=['A','B','C','D'])
    parser.add_argument('--questions',nargs='+',help='Per-turn IDs, for example TC07.1')
    parser.add_argument('--category')
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--candidates',type=int,default=15)
    parser.add_argument('--mode',choices=['retrieval','rag'],default='retrieval')
    parser.add_argument('--protocol',choices=['raw','test_context'],default='raw')
    parser.add_argument('--resume')
    parser.add_argument('--judge',action='store_true')
    args=parser.parse_args()
    dataset=json.loads(args.dataset.read_text(encoding='utf-8'))
    run=EvaluationRunner(dataset).run(configurations=args.configurations,question_ids=args.questions,category=args.category,
        top_k=args.top_k,candidate_count=args.candidates,mode=args.mode,resume=args.resume,enable_judge=args.judge,protocol=args.protocol,
        progress=lambda current,total,row:print(f"{current}/{total}: {row['question_id']} {row['configuration']} {row['status']}",flush=True))
    print('Run:',run['run_id'],'Status:',run['status'])
    raise SystemExit(1 if any(r['status'] in {'Failed','Incompatible','Unverified'} for r in run['records']) else 0)
