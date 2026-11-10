"""Create versioned labels from independently verified local spans and saved live IDs.

No network, embeddings or PDF reingestion. Existing outputs are never overwritten.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.dataset_validator import fingerprint,validate_ground_truth
from src.evaluation.chunk_compatibility import configuration
from src.rag.indexed_context import headers


def reconcile(original, runs, snapshot_root=Path('.'), cross_strategy_source=False):
    validate_ground_truth(original)
    # Reconstruct original pages independently of retrieved identity annotations.
    pages={};provenance=[]
    for snapshot in original['snapshots']:
        path=Path(snapshot_root)/snapshot['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot['sha256']:
            raise ValueError('Historical snapshot fingerprint changed')
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);m=row['metadata']
            key=(snapshot['strategy'],m['document_id'],m['source_content_hash'],int(m['page_number']))
            raw=row['text'];prefix=m.get('heading_prefix','')
            if prefix and raw.startswith(prefix):raw=raw[len(prefix):]
            a,b=int(m['start_char']),int(m['end_char'])
            if len(raw)!=b-a:raise ValueError('Invalid local source span')
            buffer=pages.setdefault(key,[])
            buffer.extend([None]*max(0,b-len(buffer)))
            for i,char in enumerate(raw,a):
                if buffer[i] is not None and buffer[i]!=char:raise ValueError('Conflicting local source spans')
                buffer[i]=char
    buffers=pages
    pages={k:''.join(v) for k,v in buffers.items() if None not in v}
    if cross_strategy_source:
        # Section snapshots can omit inter-section whitespace. A complete fixed
        # source page is usable only with the same document/hash/page and exact
        # agreement at every position actually present in the section snapshot.
        for key,buffer in buffers.items():
            if key[0]!='section' or key in pages:continue
            fixed=pages.get(('fixed',*key[1:]))
            if fixed is not None and len(buffer)<=len(fixed) and all(c is None or c==fixed[i] for i,c in enumerate(buffer)):
                pages[key]=fixed
    captures=[]
    for path in runs:
        raw=Path(path).read_bytes();run=json.loads(raw)
        if run.get('source')!='live' or run.get('index_name')!='lab-rag':continue
        provenance.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'run_id':run['run_id']})
        for row in run.get('records',[]):
            for item in row.get('candidates',[]):
                captures.append((row['strategy'],row['namespace'],item))
    result=copy.deepcopy(original)
    result.update(dataset_version='v2-overlap15-saved-evidence',parent_fingerprint=original['fingerprint'],
        reconciliation_evidence=provenance,chunk_configurations={s:{'chunking_strategy':s,'chunk_size':100,'chunk_overlap':15} for s in ('fixed','section')},
        label_policy='Versioned positive pools reconciled against local page spans and saved live captures; incomplete and not a cloud audit.')
    for q in result['questions']:
        if q['question_id']=='TC07.1':q['expected_facts']=['Specimen Minimum Volume: 0.25 mL']
        for strategy in ('fixed','section'):
            positive={};sections=[]
            for s,namespace,item in captures:
                m=item['metadata']
                if s!=strategy or namespace!=q['labels'][strategy]['namespace'] or configuration(m)!=result['chunk_configurations'][strategy]:continue
                if m.get('document_id')!=original['source_sha256']:continue
                key=(s,m.get('document_id'),m.get('source_content_hash'),int(m.get('page_number',0)))
                text=pages.get(key)
                if text is None:continue
                a,b=int(m.get('start_char',-1)),int(m.get('end_char',-1))
                if not 0<=a<b<=len(text):continue
                raw=m.get('heading_prefix','')+text[a:b]
                if hashlib.sha256(raw.encode()).hexdigest()!=m.get('chunk_content_hash'):continue
                entries=headers(text)
                previous=pages.get((*key[:3],key[3]-1),'')
                prior=headers(previous)
                if prior and prior[-1][1] and (not entries or entries[0][0]>0):entries=[(0,prior[-1][1],prior[-1][2]),*entries]
                for i,(start,name,code) in enumerate(entries):
                    end=entries[i+1][0] if i+1<len(entries) else len(text)
                    if code not in q['expected_test_codes'] or not name:continue
                    if m.get('test_code')!=code or m.get('test_name')!=name:continue
                    region=text[max(a,start):min(b,end)] if min(b,end)>max(a,start) else ''
                    normalized=' '.join(region.split()).casefold()
                    if not region or item['text'] not in raw:continue
                    if not any(' '.join(f.split()).casefold() in normalized for f in q['expected_facts']):continue
                    identifier=m.get('chunk_id')
                    if not identifier:continue
                    footer=re.search(r'\bPage\s+(\d+)\b',text)
                    positive[identifier]={'chunk_id':identifier,'document_id':m['document_id'],'source_content_hash':m['source_content_hash'],
                        'chunk_content_hash':m['chunk_content_hash'],'start_char':a,'end_char':b,'page_number':key[3],
                        'test_code':code,'test_name':name,'text':region,
                        'printed_catalog_page':footer[1] if footer else None,'section':m.get('section_heading','')}
            q['labels'][strategy].update(chunk_ids=sorted(positive),complete=False,verified_passages=list(positive.values()),
                sections=[{k:v for k,v in p.items() if k in ('document_id','test_code','printed_catalog_page','section')} for p in positive.values()],
                method='Saved live IDs verified against local source hash, raw text hash, span boundaries and independent test region')
        q['score_ranking']=bool(q['source_verified'] and q['answerability']=='answerable' and not q.get('mock_only') and all(q['labels'][s]['chunk_ids'] for s in ('fixed','section')))
        q['reconciliation_status']='Reconciled positive pools' if q['score_ranking'] else 'Unscorable: insufficient verified overlap-15 evidence or non-ranking case'
    result.pop('fingerprint',None);result['fingerprint']=fingerprint(result)
    return validate_ground_truth(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('evaluation/mayo_day5_v2_overlap15_ground_truth.json'))
    args=parser.parse_args()
    original=json.loads(Path('evaluation/mayo_day5_ground_truth.json').read_text(encoding='utf-8'))
    data=reconcile(original,sorted(Path('evaluation/results').glob('*.json')))
    with args.output.open('x',encoding='utf-8') as handle:handle.write(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print('Saved',args.output,'; scorable questions:',sum(q['score_ranking'] for q in data['questions']))
