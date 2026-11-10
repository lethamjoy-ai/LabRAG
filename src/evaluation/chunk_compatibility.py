"""Offline configuration checks; never remap IDs during scoring."""
import hashlib
import json
from pathlib import Path


def configuration(metadata):
    keys=('chunking_strategy','chunk_size','chunk_overlap')
    return {k:int(metadata[k]) if k!='chunking_strategy' else metadata[k] for k in keys if k in metadata}


def label_configurations(dataset):
    if dataset.get('chunk_configurations'):return dataset['chunk_configurations']
    configs={}
    for snapshot in dataset.get('snapshots',[]):
        path=Path(snapshot['path'])
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot['sha256']:continue
        observed={json.dumps(configuration(json.loads(line)['metadata']),sort_keys=True) for line in path.read_text(encoding='utf-8').splitlines()}
        if len(observed)==1:configs[snapshot['strategy']]=json.loads(observed.pop())
    return configs


def check_compatibility(dataset,strategy,candidates,intermediate=None):
    if dataset.get('synthetic'):return {'status':'Synthetic fixture','compatible':True,'state':'compatible'}
    expected=label_configurations(dataset).get(strategy)
    metadata=[m['document'].metadata for m in candidates]
    independent=[m for m in (intermediate or []) if m.get('chunk_id') and m.get('source_content_hash')
        and m.get('document_id')==dataset.get('source_sha256') and dataset.get('source_sha256')]
    metadata.extend(independent)
    observed=[configuration(m) for m in metadata]
    observed=list({json.dumps(c,sort_keys=True):c for c in observed}.values())
    mismatched=bool(expected and any(any(k in c and c[k]!=v for k,v in expected.items()) for c in observed))
    if mismatched:
        return {'status':'Incompatible ground-truth labels','state':'mismatched','compatible':False,
            'expected':expected,'observed':observed,'reason':'Observed chunk configuration differs from verified labels; IDs were not remapped.'}
    if expected and observed and all(c==expected for c in observed):
        return {'status':'Compatible observed chunk configuration','state':'compatible','compatible':True,'expected':expected,'observed':observed,
            'evidence_stage':'final candidates and/or pre-scoping retrieved metadata','independent_evidence_count':len(independent)}
    return {'status':'Unverified dataset/index compatibility','state':'unverified',
        'compatible':False,'expected':expected,'observed':observed,
        'reason':'Insufficient configuration evidence; an empty pool alone establishes neither compatibility nor mismatch.'}


def saved_index_configuration(root=Path('evaluation/results')):
    """Latest saved live observation, not an assertion about current cloud state."""
    for path in sorted(Path(root).glob('*.json'),reverse=True):
        run=json.loads(path.read_text(encoding='utf-8'))
        if run.get('source')!='live':continue
        configs={}
        for row in run.get('records',[]):
            for item in row.get('candidates',[]):
                c=configuration(item['metadata'])
                if len(c)==3:configs.setdefault(row['strategy'],[]).append(c)
        if configs:
            return {'run_id':run['run_id'],'index':run['index_name'],'configurations':{
                s:list({json.dumps(c,sort_keys=True):c for c in values}.values()) for s,values in configs.items()},
                'status':'Saved observation only; live state not checked'}
    return {'status':'No saved indexed configuration evidence','configurations':{}}
