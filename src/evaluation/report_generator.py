"""Unique run artifacts and transparent metric summaries."""
import csv
import io
import json
import re
from pathlib import Path
from statistics import mean


def summarize(run):
    summaries=[]
    for config in run["parameters"]["configurations"]:
        rows=[r for r in run["records"] if r["configuration"]==config and r["status"]=="Completed"]
        ranked=[r for r in rows if r.get("retrieval_metrics",{}).get("scored")]
        summary={"configuration":config,"scored_questions":len(ranked),"completed_records":len(rows),
            "incompatible_records":sum(r['configuration']==config and r['status']=='Incompatible' for r in run['records']),
            "unverified_records":sum(r['configuration']==config and r['status']=='Unverified' for r in run['records']),
            "retrieval_misses":sum(r['retrieval_metrics'].get('hit_rate')==0 for r in ranked),
            "failed_records":sum(r["configuration"]==config and r["status"]=="Failed" for r in run["records"])}
        for metric in ["hit_rate","mrr","precision","recall","judged_pool_recall"]:
            values=[r["retrieval_metrics"][metric] for r in ranked if r["retrieval_metrics"].get(metric) is not None]
            summary[metric]=mean(values) if values else None
        for key in ["retrieval_seconds","reranking_seconds","pipeline_seconds","generation_seconds","rag_end_to_end_seconds","total_seconds"]:
            values=[r["latency"][key] for r in ranked if r["latency"].get(key) is not None]
            summary[key]=mean(values) if values else None
        latencies=sorted(r["latency"]["pipeline_seconds"] for r in ranked)
        if len(latencies)>=20:
            import numpy as np
            summary.update(p50_seconds=float(np.percentile(latencies,50)),p95_seconds=float(np.percentile(latencies,95)))
        else:summary.update(p50_seconds=None,p95_seconds=None)
        claims=[c for r in rows for c in r.get("faithfulness",{}).get("claims",[])]
        citations=[c for r in rows for c in r.get("citation_evaluation",{}).get("citations",[])]
        supports=[c for r in rows for c in r.get("citation_evaluation",{}).get("claim_support",[])]
        summary.update(supported_claim_rate=sum(c["classification"]=="Supported" for c in claims)/len(claims) if claims else None,
            contradicted_claims=sum(c["classification"]=="Contradicted" for c in claims),
            citation_validity_rate=sum(c["valid"] for c in citations)/len(citations) if citations else None,
            citation_support_rate=sum(c["supported"] for c in supports)/len(supports) if supports else None,
            claim_count=len(claims),citation_count=len(citations))
        for dimension in ('correctness','faithfulness','completeness'):
            measured=[r['answer_quality'][dimension]['score'] for r in rows if r.get('answer_quality',{}).get(dimension,{}).get('score') is not None]
            summary['answer_'+dimension]=mean(measured) if measured else None
            summary['answer_'+dimension+'_questions']=len(measured)
        summary['answer_quality_records']=sum('answer_quality' in r for r in rows)
        validity=[r['citation_evaluation']['verified_validity_rate'] for r in rows if r.get('citation_evaluation',{}).get('verified_validity_rate') is not None]
        summary['answer_citation_validity']=mean(validity) if validity else None
        summary['answer_citation_validity_questions']=len(validity)
        summaries.append(summary)
    return summaries


def reranking_regressions(run):
    lookup={(r["question_id"],r["configuration"]):r for r in run["records"] if r.get("retrieval_metrics",{}).get("scored")}
    regressions=[]
    for (question,baseline),row in lookup.items():
        reranked={"A":"C","B":"D"}.get(baseline)
        other=lookup.get((question,reranked))
        if other and any(other["retrieval_metrics"][m]<row["retrieval_metrics"][m] for m in ["hit_rate","mrr","precision"]):
            regressions.append({"question_id":question,"baseline":baseline,"reranked":reranked,
                "before":row["retrieval_metrics"],"after":other["retrieval_metrics"]})
    return regressions


def csv_text(run):
    buffer=io.StringIO()
    fields=["run_id","dataset_version","protocol","question_id","test_id","question","configuration","strategy","reranking_applied","namespace",
        "expected_test_codes","retrieved_test_codes","retrieved_chunk_ids","status","retrieval_metrics","faithfulness",
        "citation_evaluation","latency","answer","errors","failure_analysis","compatibility","retrieval_diagnostics","retrieval_outcome",
        "evaluation_category","verified_required_facts","answer_quality","answer_quality_judge","generation_evidence","source_views",
        "selection_diagnostics","raw_selection_diagnostics","error_diagnostics","llm_provider","llm_model","token_usage",
        'answer_generation_mode','extractive_answer','display_answer','synthesis_diagnostics']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for row in run["records"]:
        flat={"run_id":run["run_id"],**row}
        flat['dataset_version']=run.get('dataset_version','historical-v1')
        flat['protocol']=row.get('protocol',run['parameters'].get('protocol','raw'))
        flat['llm_provider']=row.get('llm_provider',run.get('models',{}).get('llm_provider','groq'))
        flat['llm_model']=row.get('llm_model',run.get('models',{}).get('llm_model'))
        flat['answer_generation_mode']=row.get('answer_generation_mode',run.get('answer_generation_mode',run['parameters'].get('answer_generation_mode','extractive_v1')))
        if run['parameters']['mode']=='retrieval':flat['answer']='Not generated — retrieval-only mode'
        writer.writerow({k:json.dumps(flat.get(k),ensure_ascii=False) if isinstance(flat.get(k),(dict,list)) else flat.get(k) for k in fields})
    return buffer.getvalue()


def markdown_report(run):
    from src.evaluation.dashboard_reporting import coverage,common_question_metrics
    coverage_counts,_,coverage_details=coverage(run)
    lines=["# LabRAG — RAG Retrieval and Answer Quality Evaluation","",
        f"Run: `{run['run_id']}` · Mode: **{run['source']} / {run['parameters']['mode']}** · Status: {run['status']}","",
        "Objective: compare A fixed top-k, B section top-k, C fixed candidates reranked to k, and D section candidates reranked to k under identical rules.","",
        "Corpus: the specified partial Mayo catalog, not the complete Mayo catalog. Clinical facts are source-verified separately from workflow tests. Mock runs measure only synthetic fixture behavior.","",
        f"Dataset fingerprint: `{run['dataset_fingerprint']}`. Question subset: `{run['parameters']['question_ids']}`.","",
        "Ground truth: verified positive chunk pools per namespace. Local snapshot labels do not prove cloud completeness. Whole-catalog document hits are not used as a relevance shortcut.","",
        "Metrics: Hit Rate = any known relevant item in top-k; MRR = reciprocal first relevant rank; Precision = known relevant / unique actual returned count. Exact Recall requires exhaustive labels. Judged-pool recall uses only known positives and is not exact Recall. Unjudged results may also be relevant.","",
        f"Parameters: `{json.dumps(run['parameters'],sort_keys=True)}`.","",
        f"Models/prompt: `{json.dumps(run['models'],sort_keys=True)}`.","",
        "| Configuration | Scored questions | Hit Rate | MRR | Precision | Exact Recall | Judged-pool Recall | Pipeline seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.insert(4, 'Retrieval protocol: **'+run['parameters'].get('protocol','raw')+'**. Compare configurations within the same protocol. Raw benchmarks exclude context enrichment; test_context reuses the Explorer pipeline.\n')
    lines.insert(4,'Answer-generation mode: **'+run.get('answer_generation_mode',run['parameters'].get('answer_generation_mode','extractive_v1'))+'**. Compare answer quality only within the same generation mode. Historical reports remain unchanged.\n')
    lines.insert(4,'Ground-truth version: '+run.get('dataset_version','historical-v1')+'; label chunk configurations: '+json.dumps(run.get('label_chunk_configurations',{}),sort_keys=True)+'\n')
    lines.insert(4,'Selected-run coverage (unique questions, scored in at least one configuration): '+json.dumps(coverage_counts,sort_keys=True)+'. Status counts may overlap across configurations. Historical selection inferred: '+str(coverage_details['selection_inferred'])+'. Same scored set across configurations: '+str(coverage_details['same_scored_set'])+'.\n')
    if not coverage_details['same_scored_set']:
        lines.insert(4,'Separate common-question comparison (original aggregates unchanged): '+json.dumps(common_question_metrics(run),sort_keys=True)+'.\n')
    if run['parameters']['mode']=='retrieval':lines.insert(5,'Answer: Not generated — retrieval-only mode.\n')
    for row in summarize(run):
        def value(key):return "Not measured" if row.get(key) is None else f"{row[key]:.4f}"
        lines.append(f"| {row['configuration']} | {row['scored_questions']} | {value('hit_rate')} | {value('mrr')} | {value('precision')} | {value('recall')} | {value('judged_pool_recall')} | {value('pipeline_seconds')} |")
    lines.extend(["","Faithfulness uses conservative claim checks against cited passages. Numeric contradictions require the same labeled field; missing support is not proof of falsehood. Ambiguous claims need human review. Optional Groq judgments are recorded as estimates and do not replace deterministic scores.","",
        "Citation validity checks source/chunk/document/test/page identity; citation support independently checks associated claims. Abstention and empty answers are not counted as perfect faithfulness. Mock functional passes, retrieval quality, faithfulness and citation support are distinct.","",
        "Latency includes initial search through reranking for retrieval pipeline time; full RAG additionally records generation and end-to-end time. Model cold load is included when it occurs. Execution order, caches and hardware affect latency; temperature zero does not guarantee deterministic generation. p50/p95 require at least 20 scored queries per configuration.","",
        "Charts: the Streamlit Day 5 dashboard renders Plotly comparisons from this run's computed summary; unavailable metrics remain absent.","",
        "## Failure analysis",""])
    failures=[r for r in run["records"] if r.get("failure_analysis") or r["status"]=="Failed"]
    for row in failures:lines.append(f"- {row['question_id']} / {row['configuration']}: {json.dumps(row.get('failure_analysis') or row.get('errors'),ensure_ascii=False)}")
    if not failures:lines.append("No failures observed in executed records; unexecuted cases are not implied successful.")
    lines.extend(["",f"Reranking regressions: {json.dumps(reranking_regressions(run),ensure_ascii=False)}.","",
        "Recommended configuration: no automatic winner is asserted. Compare the same scored question set, quality differences and measured latency; synthetic runs cannot justify a Mayo production recommendation. Reranking remains off by default.","",
        "Known limitations: positive-pool labels can miss relevant paraphrases; metadata may be unresolved; lexical checks are not clinical entailment; cloud corpus state can diverge from local snapshots; safety/negative cases are excluded from ranking averages.","",
        "Day 6: review failures with the current official catalog, explicitly run live comparisons if desired, obtain human labels for ambiguous claims, and prepare final tests/documentation/demo. No Day 6 implementation is included.","",
        "LabRAG is an educational laboratory reference assistant. Verify specimen collection and handling instructions against the current official laboratory test catalog before clinical use."])
    if run['parameters']['mode']=='rag':
        lines.extend(['','## Generated-answer quality','',
            'Correctness: supported / evaluable verified-reference claims. Faithfulness: supported / evaluable retrieved-evidence claims. Unverifiable claims are excluded from these new score denominators and counted explicitly. Completeness: present / measurable requested verified facts. Unknown values remain unmeasured; lexical checks are not clinical entailment.',
            'Answer-quality summary scores are means over measured questions, with measured-question counts recorded. Strict citation validity reports its own verified denominator and unresolved-metadata count. Legacy supported-claim and citation rates are retained separately for historical compatibility.',
            'Abstention behavior is independent of retrieval success and completeness. Appropriate abstention after retrieval failure is not an answer-quality success. Supplemental judge estimates never replace deterministic checks. See JSON/CSV for actual generation inputs, answers, per-claim checks, fact omissions and judge details.',''])
        for row in summarize(run):
            lines.append(f"- {row['configuration']}: "+json.dumps({k:v for k,v in row.items() if k.startswith('answer_')},sort_keys=True))
    return "\n".join(lines)+"\n"


def save_run(run,root=Path("evaluation")):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run["run_id"]):raise ValueError("Invalid run identifier")
    root=Path(root);results=root/"results";reports=root/"reports"
    results.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    run["summary"]=summarize(run);run["reranking_regressions"]=reranking_regressions(run)
    path=results/f"{run['run_id']}.json"
    temporary=path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");temporary.replace(path)
    path.with_suffix(".csv").write_text(csv_text(run),encoding="utf-8")
    (reports/f"{run['run_id']}_summary.md").write_text(markdown_report(run),encoding="utf-8")
    return path
