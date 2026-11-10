"""Day 5 dashboard: data reads on load; execution only after explicit submission."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]


def _render_configuration(settings,datasets):
    if not datasets:
        st.info('No ground-truth dataset is available for new execution. Saved evaluation reports remain available below.')
        return
    preferred=next((i for i,p in enumerate(datasets) if p.name=='mayo_day5_v2_overlap15_ground_truth.json'),0)
    dataset_path=st.selectbox('Day 5 dataset',datasets,index=preferred,format_func=lambda p:p.name)
    try:dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        st.warning('Could not read selected ground truth.');return
    questions=dataset['questions']
    from src.evaluation.chunk_compatibility import label_configurations,saved_index_configuration
    labels=label_configurations(dataset)
    indexed=saved_index_configuration(ROOT/'evaluation/results')
    compatible=bool(labels and all(indexed['configurations'].get(s)==[c] for s,c in labels.items()))
    mismatch=any(indexed['configurations'].get(s) and indexed['configurations'][s]!=[c] for s,c in labels.items())
    st.write({'Ground-truth version':dataset.get('dataset_version','historical-v1'),
        'Label chunk configuration':labels,'Indexed chunk configuration (saved observation)':indexed,
        'Compatibility status':'Compatible with saved observation; checked again on retrieved candidates' if compatible else 'Incompatible ground-truth labels' if mismatch else 'Unverified dataset/index compatibility'})
    st.write({'Total questions':len(questions),'Source verified':sum(q['source_verified'] for q in questions),
        'Unverified':sum(not q['source_verified'] for q in questions),'Comparable ranking questions':sum(q['score_ranking'] for q in questions)})
    with st.expander('Configure Day 5 evaluation'):
        preset=st.selectbox('Day 5 question preset',['Manual selection','Generated-answer pilot (TC07.1, TC04.1, TC12.1)'])
        pilot=preset!='Manual selection'
        from src.ui_llm import provider_control,model_control
        provider=provider_control('Evaluation LLM Provider','evaluation')
        model=model_control(settings,provider,'evaluation','Evaluation model')
        configured=replace(settings,llm_provider=provider,llm_model=model)
        st.caption(f'Answer provider fixed for execution: {provider.title()} · {model}')
        judge=st.checkbox('Day 5 optional LLM judge',value=False)
        judge_provider=judge_model=None
        if judge:
            judge_provider=provider_control('Judge LLM Provider','judge')
            judge_model=model_control(settings,judge_provider,'judge','Judge model')
            st.caption('Independent supplemental judge; existing scoring rules are unchanged.')
        with st.form('day5_configuration'):
            configurations=st.multiselect('Day 5 configurations',['A','B','C','D'],default=['A','B','C','D'])
            category=st.selectbox('Day 5 category',['All']+sorted({q['category'] for q in questions}))
            ids=st.multiselect('Day 5 questions (empty means all)',[q['question_id'] for q in questions],
                default=[q for q in ['TC07.1','TC04.1','TC12.1'] if q in {x['question_id'] for x in questions}] if pilot else [],key='day5_pilot_ids' if pilot else 'day5_manual_ids')
            top_k=st.number_input('Day 5 final top-k',1,100,5)
            candidates=st.number_input('Day 5 initial candidates',int(top_k),100,max(15,int(top_k)))
            mode=st.selectbox('Day 5 evaluation mode',['retrieval','rag'],index=1 if pilot else 0,key='day5_pilot_mode' if pilot else 'day5_manual_mode')
            generation_mode=st.selectbox('Evaluation answer-generation mode',['extractive_v1','grounded_conversational_v1','grounded_conversational_v2'],
                help='Extractive is the historical default. Conversational uses verified typed facts and constrained templates. Never compare answer scores across modes without identifying the difference.')
            protocol=st.selectbox('Day 5 retrieval protocol',['raw','test_context'],index=1 if pilot else 0,key='day5_pilot_protocol' if pilot else 'day5_manual_protocol',format_func=lambda p: {'raw':'Raw Similarity Retrieval','test_context':'Test-Context Retrieval'}[p])
            resume=st.text_input('Resume Day 5 run ID (optional)')
            recovery=st.checkbox('Allow recovery-only migration from the investigated pre-recovery pipeline fingerprint',value=False)
            interval=st.number_input('Day 5 minimum API request interval (seconds)',0.0,60.0,0.0)
            rpm=st.number_input('Account requests per minute (0 = unknown)',0,100000,0)
            batch_limit=st.number_input('Maximum new questions per session',1,1000,5,
                help='Counts distinct remaining questions, including partial questions and failed-pair retries. All pending selected variants for each question are attempted unless interrupted or rate-limited. The original signed question selection stays unchanged.')
            st.caption('Enter your verified account/model quota; no quota is assumed. Pacing is shared across generation, retries and optional judges. A terminal 429 pauses the run; resumption is always explicit.')
            st.caption('Resume validates the stored signature and matching settings, takes a read-only backup, and locks the run. Completed/excluded pairs are skipped. A stale lock requires manual verification; do not start two resumes.')
            permitted=st.checkbox('I explicitly initiate read-only Pinecone evaluation; RAG/judge modes call the selected API providers and may incur charges. Reranking may download local model weights.',value=False)
            start=st.form_submit_button('Start Day 5 Evaluation')
        if start:
            if pilot and not {'TC07.1','TC04.1','TC12.1'}<={q['question_id'] for q in questions}:
                st.warning('Selected dataset does not contain all three pilot questions. Choose the v3 dataset; no API calls were made.')
            elif not permitted:
                st.warning('Explicit initiation checkbox is required. No API calls were made.')
            else:
                from src.evaluation.evaluation_runner import EvaluationRunner
                bar=st.progress(0.0)
                try:
                    run=EvaluationRunner(dataset,settings=configured,root=ROOT/'evaluation').run(
                        configurations=configurations,question_ids=ids or None,category=None if category=='All' else category,
                        top_k=int(top_k),candidate_count=int(candidates),mode=mode,resume=resume or None,
                        enable_judge=judge,judge_provider=judge_provider,judge_model=judge_model,allow_live=True,protocol=protocol,allow_recovery_migration=recovery,
                        min_request_interval=interval,account_requests_per_minute=rpm or None,
                        max_new_questions_per_session=int(batch_limit),answer_generation_mode=generation_mode,progress=lambda done,total,row:bar.progress(done/total))
                    st.session_state['last_day5_run']=run['run_id']
                    st.success(f"Saved {run['run_id']}: {run['status']}")
                except Exception as exc:
                    from src.evaluation.checkpoint_safety import ResumeDeferred
                    if isinstance(exc,ResumeDeferred):st.warning(str(exc))
                    else:
                        st.warning('Evaluation could not start or resume. Check dataset/configuration, counts, mode and run ID. Provider bodies are not displayed.')


def render_day5(settings):
    st.subheader('Day 5 — Retrieval and Answer Quality')
    st.caption('A: fixed top-k · B: section top-k · C: fixed candidates reranked · D: section candidates reranked. No automatic evaluations or model loading.')
    datasets=sorted((ROOT/'evaluation').glob('*ground_truth.json'))
    runs=sorted((ROOT/'evaluation/results').glob('*.json'),reverse=True)
    if not runs:
        st.info('No saved evaluation is available. Results appear only after explicit execution.')
        with st.expander('Advanced Evaluation Settings',expanded=False):
            _render_configuration(settings,datasets)
        return
    selected=st.selectbox('Day 5 historical run',runs,format_func=lambda p:p.stem)
    try:run=json.loads(selected.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        st.warning('Report cannot be read.');return
    if run['status']=='Running':
        st.info('This checkpoint was saved as Running; that does not prove execution is still active. Verify the original execution has stopped before resuming. Stale checkpoints are never treated as completed.')
    if run['status']=='Paused — Rate Limited':
        st.warning('Paused — Rate Limited. No further pairs were submitted. Wait until resume-after (if provided), verify your account quota, then explicitly submit the same run ID and signed settings. Completed/excluded pairs will be skipped; a backup and writer lock protect recovery.')
        with st.expander('Provider rate-limit diagnostics',expanded=False):
            st.write(run.get('rate_limit',{}))
    if run['status']=='Paused — Provider Configuration':
        st.warning('Paused — Provider Configuration. Check API billing, credits and model access before explicitly retrying. No provider fallback occurred.')
        with st.expander('Provider diagnostics',expanded=False):st.write(run.get('rate_limit',{}))
    if run['status']=='Batch Completed — Ready to Resume':
        st.success('Batch Completed — Ready to Resume. Explicitly resume this same run ID with the original full question selection and signed settings. Completed/excluded pairs are skipped. Keep account pacing enabled; rate-limit cooldowns still apply.')
        with st.expander('Batch checkpoint details',expanded=False):
            st.write(run.get('batch_session',{}))
    from src.ui_presentation import render_overview,render_questions
    render_overview(run)
    st.caption('Answer-generation mode: '+run.get('answer_generation_mode',run['parameters'].get('answer_generation_mode','extractive_v1'))+' · Compare answer-quality scores only within the same generation mode.')
    with st.expander('Run configuration & provenance',expanded=False):
        st.write({'Run':run['run_id'],'Source':run['source'],'Mode':run['parameters']['mode'],'Status':run['status'],
            'Dataset fingerprint':run['dataset_fingerprint'],'Ground-truth version':run.get('dataset_version','historical-v1'),
            'Label chunk configurations':run.get('label_chunk_configurations',{}),'Index':run['index_name'],'Models':run['models']})
    if run['source']=='mock':st.warning('SYNTHETIC FIXTURE RUN: these computed values are not Mayo retrieval performance or clinical validation.')
    from src.evaluation.dashboard_reporting import coverage,matching_questions,comparison_rows,reciprocal_rank_heatmap,common_question_metrics
    counts,coverage_rows,coverage_details=coverage(run)
    with st.expander('Coverage & matched-question comparison',expanded=False):
        st.subheader('Selected run coverage')
        cards=st.columns(3)
        for i,(label,value) in enumerate(counts.items()):
            cards[i%3].metric(label,('Not available' if value is None else f'{value:.2f}%') if label=='Coverage percentage' else value)
        st.caption(f"Saved records: {coverage_details['record_count']} / planned question/configuration pairs: {coverage_details['planned_records']}. Counts above are unique questions; if statuses differ by configuration, counts can overlap. Coverage counts questions scored in at least one configuration.")
        if coverage_details['selection_inferred']:st.info('Historical selection was not recorded; selected count is inferred from existing records and may omit unrecorded questions.')
        if not coverage_details['same_scored_set']:st.warning('Configurations have different scored question sets. Aggregate metrics are not a matched-set comparison.')
        st.dataframe(pd.DataFrame(coverage_rows),use_container_width=True,hide_index=True)
        if not coverage_details['same_scored_set']:
            common=common_question_metrics(run)
            st.subheader('Common-question comparison — separate from original aggregates')
            st.write({'Common scored questions':len(common['question_ids']),'Selected questions':common['selected_questions'],
                'Common-set coverage %':common['coverage_percentage'],'Common question IDs':common['question_ids']})
            st.dataframe(pd.DataFrame(common['summaries']),use_container_width=True,hide_index=True)
            st.caption('Only the intersection of scored question IDs is used here. Original charts and saved metrics remain unchanged. Candidate retrieval depths still differ between A/B and C/D; this does not isolate reranking alone.')
    report_questions=matching_questions(run,datasets)
    comparison=comparison_rows(run,report_questions)
    with st.expander('Exclusion details',expanded=False):
        excluded=[r for r in comparison if r['Status']=='Excluded']
        if excluded:
            st.subheader('Excluded questions — not retrieval failures')
            exclusion_columns=['Question ID','Question','Exclusion reason','Ground-truth eligibility','Fixed-strategy labels','Section-strategy labels','Retrieved test code']
            st.dataframe(pd.DataFrame(excluded)[exclusion_columns].drop_duplicates(),use_container_width=True,hide_index=True)
            st.caption('Blank eligibility or label availability means historical metadata is unavailable. Dataset details are used only when its fingerprint matches this saved run.')
    summary=pd.DataFrame(run.get('summary',[]))
    with st.expander('Metric definitions & aggregate table',expanded=False):
        st.caption('Retrieval protocol: '+run['parameters'].get('protocol','raw')+' (compare results only within the same protocol).')
        st.caption(f"Metrics use top-{run['parameters']['top_k']}; partial labels yield known-positive Hit/MRR/Precision and judged-pool recall. Exact Recall is separate. Unmeasured values remain blank.")
        st.caption('Hit Rate is averaged only over scored questions. Excluded questions count as neither successes nor failures. Precision divides by the actual number of unique returned results, capped at Top K. Judged-pool recall is not exhaustive recall. Aggregate charts describe the full selected run and do not change with question-table filters.')
        st.caption('Completed records distinguish successful retrieval from retrieval misses in the question table. Verified misses score zero; Unverified, Incompatible, Excluded and Failed records remain unscored. Historical statuses are preserved as recorded.')
        st.dataframe(summary,use_container_width=True,hide_index=True)
    st.subheader('Evaluation Graphs')
    if run['parameters']['mode']=='rag':
        st.subheader('Generated-answer quality — separate from retrieval ranking')
        quality_rows=[]
        for record in run.get('records',[]):
            quality=record.get('answer_quality',{})
            quality_rows.append({'Question ID':record['question_id'],'Configuration':record['configuration'],'Status':record['status'],
                'Category':record.get('evaluation_category','Historical — not recorded'),'Generated answer':record.get('answer'),
                'Validation stage':(record.get('selection_diagnostics') or {}).get('stage','Historical — not recorded'),
                'Selection failure reason':(record.get('selection_diagnostics') or {}).get('failure_category'),
                'Verified required facts':record.get('verified_required_facts'),
                **{d:quality.get(d,{}).get('score') for d in ('correctness','faithfulness','completeness')},
                'Citation validity':record.get('citation_evaluation',{}).get('verified_validity_rate'),
                'Citation support':record.get('citation_evaluation',{}).get('support_rate'),
                'Abstention behavior':quality.get('abstention',{}).get('classification'),
                'Generation seconds':record.get('latency',{}).get('generation_seconds'),
                'RAG end-to-end seconds':record.get('latency',{}).get('rag_end_to_end_seconds')})
        with st.expander('Answer quality table & rubric',expanded=False):
            st.dataframe(pd.DataFrame(quality_rows),use_container_width=True,hide_index=True)
        st.caption('Correctness and completeness use verified requested facts only after generation. Faithfulness denominator excludes Unverifiable claims; unknown counts remain visible in details. Empty/abstained answers are not perfect faithfulness. Exact-text checks leave paraphrases for human review. Unmeasured values stay blank. TC20 uses the existing mocked Day 4 validation, never cloud ingestion; TC17 is unscorable without verified references.')
        import plotly.express as px
        for metric,label in [('answer_correctness','Verified-reference correctness'),('answer_faithfulness','Evaluable-claim faithfulness'),('answer_completeness','Required-fact completeness'),('answer_citation_validity','Verified citation validity')]:
            if metric in summary.columns:
                measured=summary.dropna(subset=[metric])
                if not measured.empty:st.plotly_chart(px.bar(measured,x='configuration',y=metric,title=label),use_container_width=True)
    if not summary.empty:
        import plotly.express as px
        for metric,label in [('hit_rate','Hit Rate'),('mrr','MRR'),('precision','Precision'),('pipeline_seconds','Average retrieval pipeline seconds'),
                             ('supported_claim_rate','Supported-claim rate'),('citation_support_rate','Citation support rate')]:
            if metric not in summary.columns:continue
            values=summary.dropna(subset=[metric])
            if not values.empty:
                figure=px.bar(values,x='configuration',y=metric,title=('Synthetic fixture: ' if run['source']=='mock' else '')+label)
                if metric in {'pipeline_seconds','supported_claim_rate','citation_support_rate'}:
                    with st.expander(label,expanded=False):
                        st.plotly_chart(figure,use_container_width=True)
                else:st.plotly_chart(figure,use_container_width=True)
    render_questions(run,report_questions)
    with st.expander('Detailed comparison & validation',expanded=False):
        rows=run['records']
        state_filter=st.selectbox('Day 5 result filter',['All']+list(dict.fromkeys(['Completed','Failed','Excluded','Incompatible','Unverified']+[r.get('status','Unknown') for r in rows])))
        config_filter=st.selectbox('Day 5 configuration filter',['All','A','B','C','D'])
        question_filter=st.multiselect('Day 5 question filter',list(dict.fromkeys(r['question_id'] for r in rows)))
        visible=[r for r in rows if (state_filter=='All' or r['status']==state_filter) and (config_filter=='All' or r['configuration']==config_filter) and (not question_filter or r['question_id'] in question_filter)]
        selected_pairs={(r['question_id'],r['configuration']) for r in visible}
        compared=[r for r in comparison if (r['Question ID'],r['Configuration']) in selected_pairs]
        st.subheader(f"Question-level comparison (Top K = {run['parameters']['top_k']})")
        st.dataframe(pd.DataFrame(compared).rename(columns={'Hit Rate':f"Hit Rate@{run['parameters']['top_k']}",'MRR':f"MRR@{run['parameters']['top_k']}"}),use_container_width=True,hide_index=True)
        if compared:
            columns=[config_filter] if config_filter!='All' else run['parameters'].get('configurations',['A','B','C','D'])
            st.plotly_chart(reciprocal_rank_heatmap(compared,columns),use_container_width=True)
            st.caption('Numbers and blue shading show scored reciprocal rank (including genuine zero). Gray cells carry explicit status labels for Excluded, Incompatible, Failed or otherwise unscored records; they are not zero scores.')
        if visible:
            lookup={f"{r['question_id']} / {r['configuration']}":r for r in visible}
            identifier=st.selectbox('Inspect Day 5 question',list(lookup))
            row=lookup[identifier]
            st.write('Failure analysis',row.get('failure_analysis') or row.get('exclusion_reason') or 'No failure classified')
            with st.expander('Day 5 evidence and answer details',expanded=False):
                st.write('Execution error diagnostics',row.get('error_diagnostics') or 'Not recorded in this result')
                st.write('Selection validation diagnostics',row.get('selection_diagnostics') or 'Not recorded in this historical result')
                mismatch_rows=[]
                for label,key in [('Raw selection','raw_selection_diagnostics'),('After restoration','selection_diagnostics')]:
                    diagnostic=row.get(key) or {}
                    for excerpt in diagnostic.get('excerpts',[]):
                        detail=excerpt.get('mismatch_diagnostics')
                        if detail:
                            mismatch_rows.append({'Selection':label,'Source ID':excerpt['source_id'],'Mismatch category':detail['category'],
                                'Alternate matching sources':detail['alternate_source_ids'],'Boundary sources':detail['cross_chunk_source_ids'],
                                'Inserted tokens':detail['token_insertions'],'Deleted tokens':detail['token_deletions'],
                                'Substituted tokens':detail['token_substitutions'],'Whitespace restoration succeeded':detail.get('whitespace_restoration_succeeded'),
                                'Validation stage':diagnostic.get('stage'),'Final answer status':(row.get('selection_diagnostics') or {}).get('final_answer_status')})
                if mismatch_rows:st.dataframe(pd.DataFrame(mismatch_rows),use_container_width=True,hide_index=True)
                else:st.caption('Detailed token mismatch diagnostics were not recorded in this historical run.')
                st.write('Answer', 'Not generated — retrieval-only mode' if run['parameters']['mode']=='retrieval' else row.get('answer',''))
                with st.expander("Raw record JSON",expanded=False):
                    st.json(row)
        if run.get('reranking_regressions'):
            st.warning('Reranking regressions were measured for some questions. Inspect the baseline/reranked records.')
            st.json(run['reranking_regressions'])
    from src.evaluation.report_generator import csv_text,markdown_report
    st.download_button('Export Day 5 JSON',json.dumps(run,indent=2),file_name=run['run_id']+'.json',mime='application/json')
    st.download_button('Export Day 5 CSV',csv_text(run),file_name=run['run_id']+'.csv',mime='text/csv')
    st.download_button('Export Day 5 Markdown',markdown_report(run),file_name=run['run_id']+'_summary.md',mime='text/markdown')

    with st.expander('Advanced Evaluation Settings',expanded=False):
        _render_configuration(settings,datasets)
