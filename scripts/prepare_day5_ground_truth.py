import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.settings import Settings
from src.evaluation.dataset_validator import build_ground_truth,validate_ground_truth,GROUND_TRUTH

if __name__=="__main__":
    dataset=validate_ground_truth(build_ground_truth(Settings.from_env()))
    GROUND_TRUTH.write_text(json.dumps(dataset,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print('Questions:',len(dataset['questions']),'Comparable ranking questions:',sum(q['score_ranking'] for q in dataset['questions']))
