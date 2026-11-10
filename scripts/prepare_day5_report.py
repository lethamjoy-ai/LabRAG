"""Create a zero-execution readiness report. Makes no model or API calls."""
import hashlib
import json
import sys
import uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.settings import Settings
from src.evaluation.report_generator import save_run
from src.rag.prompt_templates import SYSTEM_PROMPT
from src.retrieval.reranker import MODEL,MODEL_REVISION

if __name__=="__main__":
    dataset=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    settings=Settings.from_env()
    run={'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_prepared_'+uuid.uuid4().hex[:8],
        'timestamp':datetime.now(timezone.utc).isoformat(),'status':'Not Tested','source':'not_run',
        'dataset_fingerprint':dataset['fingerprint'],'corpus_identifier':dataset['source_sha256'],
        'index_name':settings.pinecone_index_name,'parameters':{'configurations':['A','B','C','D'],
            'question_ids':[q['question_id'] for q in dataset['questions']],'mode':'retrieval','top_k':5,'candidate_count':15,'filters':{},'judge':False},
        'models':{'embedding_model':settings.embedding_model,'embedding_dimension':settings.embedding_dimension,
            'llm_model':settings.llm_model,'reranker_model':MODEL,'reranker_revision':MODEL_REVISION,'temperature':settings.llm_temperature,
            'max_output_tokens':settings.llm_max_tokens,'prompt_version':hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()},
        'records':[],'events':[{'note':'Readiness report only. No queries executed and no quality scores measured. Start an explicitly enabled run for metrics.'}]}
    print(save_run(run))
