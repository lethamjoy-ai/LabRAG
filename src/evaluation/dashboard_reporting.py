"""Read-only views and separate aggregates of stored per-question metrics."""
import json
from pathlib import Path


def matching_questions(run, paths):
    for path in paths:
        try:
            dataset=json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if dataset.get('fingerprint')==run.get('dataset_fingerprint'):
            from src.evaluation.dataset_validator import fingerprint
            if dataset['fingerprint']==fingerprint({k:v for k,v in dataset.items() if k!='fingerprint'}):
                return {q['question_id']:q for q in dataset.get('questions',[])}
    return {}


def coverage(run):
    rows=run.get('records',[])
    ids=run.get('parameters',{}).get('question_ids')
    inferred=ids is None
    selected=set(ids if ids is not None else [r['question_id'] for r in rows])
    scored={r['question_id'] for r in rows if r.get('status')=='Completed' and r.get('retrieval_metrics',{}).get('scored')}
    counts={'Selected questions':len(selected),'Scored questions':len(scored),
        **{label:len({r['question_id'] for r in rows if r.get('status')==status}) for label,status in
           [('Excluded questions','Excluded'),('Incompatible questions','Incompatible'),('Unverified questions','Unverified'),('Failed questions','Failed')]}}
    counts['Coverage percentage']=100*len(scored)/len(selected) if selected else None
    configs=run.get('parameters',{}).get('configurations') or sorted({r['configuration'] for r in rows})
    sets=[{r['question_id'] for r in rows if r['configuration']==c and r.get('status')=='Completed' and r.get('retrieval_metrics',{}).get('scored')} for c in configs]
    per_config=[]
    for c,scored_set in zip(configs,sets):
        records=[r for r in rows if r['configuration']==c]
        per_config.append({'Configuration':c,'Selected':len(selected),'Scored':len(scored_set),
            **{s:sum(r.get('status')==s for r in records) for s in ('Completed','Excluded','Incompatible','Unverified','Failed')},
            'Records':len(records),'Coverage %':100*len(scored_set)/len(selected) if selected else None})
    return counts,per_config,{'selection_inferred':inferred,'same_scored_set':all(s==sets[0] for s in sets) if sets else True,
        'record_count':len(rows),'planned_records':len(selected)*len(configs)}


def common_question_metrics(run):
    """Separate aggregate view of stored scores; never relabel historical outcomes."""
    from statistics import mean
    rows=run.get('records',[])
    configs=run.get('parameters',{}).get('configurations') or sorted({r['configuration'] for r in rows})
    selected=set(run.get('parameters',{}).get('question_ids') or [r['question_id'] for r in rows])
    lookups=[{r['question_id']:r for r in rows if r['configuration']==c and r.get('status')=='Completed' and r.get('retrieval_metrics',{}).get('scored')} for c in configs]
    common=set.intersection(*(set(lookup) for lookup in lookups)) if lookups else set()
    common &= selected
    summaries=[]
    for config,lookup in zip(configs,lookups):
        summary={'Configuration':config,'Common scored questions':len(common)}
        for metric in ('hit_rate','mrr','precision','judged_pool_recall'):
            values=[lookup[q]['retrieval_metrics'].get(metric) for q in sorted(common)]
            summary[metric]=mean(values) if values and all(v is not None for v in values) else None
        summaries.append(summary)
    return {'question_ids':sorted(common),'coverage_percentage':100*len(common)/len(selected) if selected else None,
        'selected_questions':len(selected),'summaries':summaries}


def eligibility(row, question=None):
    if row.get('ground_truth_eligibility') is not None:return row['ground_truth_eligibility']
    if question is None:return {}
    return {'eligible':question.get('score_ranking'),'source_verified':question.get('source_verified'),
        'mock_only':question.get('mock_only',False),'category':question.get('category'),
        'answerability':question.get('answerability'),
        'fixed_labels':len(question.get('labels',{}).get('fixed',{}).get('chunk_ids',[])),
        'section_labels':len(question.get('labels',{}).get('section',{}).get('chunk_ids',[]))}


def exclusion_reason(row, details):
    if row.get('status')!='Excluded':return ''
    if details.get('mock_only'):return 'Mock-only prompt-injection case' if details.get('category')=='prompt_injection' else 'Mock-only evaluation case'
    if details.get('source_verified') is False:return 'Unverified source evidence'
    if details.get('category')=='safety':return 'Safety/refusal evaluation case'
    if details.get('category')=='missing_information':return 'Missing-information/abstention case'
    if details.get('section_labels')==0 and details.get('answerability')=='answerable':return 'Missing section labels'
    if details.get('fixed_labels')==0 and details.get('answerability')=='answerable':return 'Missing fixed labels'
    return row.get('exclusion_reason') or 'Unknown — historical metadata unavailable'


def comparison_rows(run, questions=None):
    questions=questions or {};output=[]
    for row in run.get('records',[]):
        details=eligibility(row,questions.get(row['question_id']))
        scored=row.get('status')=='Completed' and row.get('retrieval_metrics',{}).get('scored',False)
        metrics=row.get('retrieval_metrics',{}) if scored else {}
        output.append({'Question ID':row['question_id'],'Question':row.get('question','Unknown'),
            'Configuration':row['configuration'],'Status':row.get('status','Unknown'),
            'Retrieval outcome':('Successful retrieval' if metrics.get('hit_rate') else 'Retrieval miss') if scored else 'Not scored',
            'Hit Rate':metrics.get('hit_rate'),'MRR':metrics.get('mrr'),'Precision':metrics.get('precision'),
            'First relevant chunk rank':metrics.get('first_relevant_rank'),
            'Retrieved test code':'Not executed — excluded' if row.get('status')=='Excluded' else ', '.join(row.get('retrieved_test_codes') or []) or 'Not recorded',
            'Exclusion reason':exclusion_reason(row,details),'Ground-truth eligibility':details.get('eligible'),
            'Fixed-strategy labels':details.get('fixed_labels'),'Section-strategy labels':details.get('section_labels')})
    return output


def reciprocal_rank_heatmap(rows, configurations):
    import plotly.graph_objects as go
    ids=list(dict.fromkeys(r['Question ID'] for r in rows))
    lookup={(r['Question ID'],r['Configuration']):r for r in rows}
    z=[];text=[]
    for identifier in ids:
        values=[];labels=[]
        for config in configurations:
            row=lookup.get((identifier,config))
            value=row.get('MRR') if row else None
            values.append(value)
            labels.append(f'{value:.2f}' if value is not None else (row['Status'] if row else 'Not recorded'))
        z.append(values);text.append(labels)
    figure=go.Figure(go.Heatmap(z=z,x=configurations,y=ids,zmin=0,zmax=1,colorscale='Blues',
        colorbar={'title':'Reciprocal rank'},hoverongaps=False))
    # An independent text layer labels null cells; they are never encoded as zero.
    figure.add_trace(go.Scatter(x=[c for _ in ids for c in configurations],y=[i for i in ids for _ in configurations],
        text=[t for line in text for t in line],mode='text',textfont={'color':'#d46a00'},hoverinfo='text',showlegend=False))
    figure.update_layout(title='Question-level reciprocal rank',height=max(320,32*len(ids)),
        yaxis={'autorange':'reversed'},plot_bgcolor='#999999')
    return figure
