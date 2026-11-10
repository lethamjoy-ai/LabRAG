"""Explicit, resumable four-configuration evaluation. Never mutates vector storage."""
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime,timezone
from dataclasses import replace
from pathlib import Path
from config.settings import Settings
from src.evaluation.dataset_validator import validate_ground_truth,fingerprint
from src.evaluation.retrieval_metrics import ranking_metrics
from src.evaluation.faithfulness_evaluator import evaluate_faithfulness,GroqFaithfulnessJudge
from src.evaluation.citation_evaluator import evaluate_citations
from src.evaluation.report_generator import save_run
from src.retrieval.retrieval_comparator import RetrievalComparator,CONFIGURATIONS
from src.retrieval.reranker import Reranker,MODEL
from src.rag.rag_graph import build_rag_graph
from src.rag.prompt_templates import SYSTEM_PROMPT
from src.evaluation.answer_quality import evaluate_answer_quality,required_facts
from src.evaluation.chunk_compatibility import check_compatibility,label_configurations
from src.evaluation.checkpoint_safety import guarded_run,backup_checkpoint


class IncompatibleLabelsError(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__(details['status'])


def serialize(matches):
    return [{"text":m["document"].page_content,"metadata":m["document"].metadata,"pinecone_score":m["score"],
             "reranker_score":m.get("reranker_score")} for m in matches]


def failure_analysis(row,question):
    findings=[]
    def add(kind,observation,remedy):findings.append({"category":kind,"observation":observation,"possible_remediation":remedy})
    metrics=row.get("retrieval_metrics",{})
    if metrics.get("scored") and not metrics["hit_rate"]:
        relevant=set(question["labels"][row["strategy"]]["chunk_ids"])
        if relevant & set(row.get("candidate_chunk_ids",[])):
            add("Ranking issue","Known relevant evidence was in candidates but outside final top-k","Review reranker behavior and candidate count without changing the index")
        else:
            add("Candidate retrieval issue","Known relevant evidence absent from candidate pool","Review retrieval protocol, identity search and candidate coverage")
        if row.get("retrieved_test_codes") and not set(row["retrieved_test_codes"]) & set(question["expected_test_codes"]):
            add("Wrong test retrieved","Returned test codes do not match the expected test","Inspect identity and specimen distinctions")
    if row.get("retrieved") and any(not item["metadata"].get("test_code") or not item["metadata"].get("test_name") for item in row["retrieved"]):
        add("Metadata issue","Returned chunks have missing or unresolved test identity metadata","Inspect source context and unresolved labels; this does not establish the cause of a retrieval miss")
    if row.get("faithfulness",{}).get("claims") and any(c["classification"]!="Supported" for c in row["faithfulness"]["claims"]):
        add("Answer generation issue","Unsupported, contradicted or ambiguous claim","Review cited text and preserve field qualifiers/units")
    if row.get("missing_expected_facts"):
        add("Answer generation issue","Expected source facts omitted from the observed answer","Review selected evidence and generation; faithfulness is not answer completeness")
    citations=row.get("citation_evaluation",{})
    if citations.get("missing_citations") or any(not c["valid"] for c in citations.get("citations",[])):
        add("Citation issue","Missing or invalid citation metadata","Review source IDs, test/document identity and page provenance")
    return findings


class EvaluationRunner:
    def __init__(self,dataset,settings=None,retriever=None,service=None,reranker=None,source="live",root=Path("evaluation")):
        self.dataset=validate_ground_truth(dataset)
        if source not in {"mock","live"}:raise ValueError("source must be mock or live")
        if source=="mock" and not dataset.get("synthetic"):
            raise ValueError("Mock quality measurements require an explicitly synthetic dataset; they cannot simulate Mayo retrieval accuracy")
        self.settings=settings or (Settings() if source=="mock" else None)
        self.retriever,self.service,self.reranker=retriever,service,reranker or Reranker()
        self.source,self.root=source,Path(root)

    @guarded_run
    def run(self,configurations=("A","B","C","D"),question_ids=None,category=None,top_k=5,candidate_count=15,
            mode="retrieval",filters=None,resume=None,progress=None,cancel=None,allow_live=False,enable_judge=False,protocol="raw",
            allow_recovery_migration=False,_checkpoint_id=None,min_request_interval=0,account_requests_per_minute=None,
            max_new_questions_per_session=5,judge_provider=None,judge_model=None,answer_generation_mode='extractive_v1'):
        from src.rag.conversational_answer import MODES
        if answer_generation_mode not in MODES:raise ValueError('Unknown answer-generation mode')
        if type(max_new_questions_per_session) is not int or not 1<=max_new_questions_per_session<=1000:
            raise ValueError('Maximum new questions per session must be an integer from 1 to 1000')
        if protocol not in {"raw","test_context"}:raise ValueError("Unknown retrieval protocol")
        if self.source=="live" and not (allow_live or os.getenv("LABRAG_RUN_DAY5_LIVE")=="1"):
            raise ValueError("Live Day 5 evaluation requires explicit initiation/opt-in")
        if mode not in {"retrieval","rag"} or not configurations or any(c not in CONFIGURATIONS for c in configurations):raise ValueError("Invalid evaluation mode/configurations")
        if len(configurations)!=len(set(configurations)) or not 1<=top_k<=candidate_count<=100:raise ValueError("Invalid counts or duplicate configurations")
        if enable_judge and (self.source!="live" or mode!="rag"):raise ValueError("Groq judge requires explicitly initiated live RAG mode")
        settings=self.settings or Settings.from_env()
        judge_settings=None
        if enable_judge:
            if judge_provider not in {'groq','openai'} or not isinstance(judge_model,str) or not judge_model.strip():
                raise ValueError('Choose an explicit judge provider and model; answer-provider settings are not inherited')
            judge_settings=replace(settings,llm_provider=judge_provider,llm_model=judge_model)
        compatibility_dataset={**self.dataset,'chunk_configurations':label_configurations(self.dataset)}
        if self.dataset.get("embedding_model")!=settings.embedding_model or self.dataset.get("embedding_dimension")!=settings.embedding_dimension:
            raise ValueError("Ground-truth embedding configuration differs; preserve the index/model and review labels")
        selected=[q for q in self.dataset["questions"] if (not question_ids or q["question_id"] in question_ids) and (not category or q["category"]==category)]
        if question_ids and set(question_ids)-{q["question_id"] for q in selected}:raise ValueError("Unknown question ID or incompatible category filter")
        if not selected:raise ValueError("No selected questions")
        safe_models={"embedding_model":settings.embedding_model,"embedding_dimension":settings.embedding_dimension,
            "embedding_provider":settings.embedding_provider,"namespace_fixed":settings.namespace_fixed,"namespace_section":settings.namespace_section,
            "retrieval_score_threshold":settings.retrieval_score_threshold,
            "llm_model":settings.llm_model,"llm_provider":settings.llm_provider,"temperature":settings.llm_temperature,
            "max_output_tokens":settings.llm_max_tokens,"reranker_model":self.reranker.model_name,
            "reranker_revision":self.reranker.model_revision if self.source=="live" else "injected synthetic backend",
            "prompt_version":hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
        if enable_judge:safe_models.update(judge_provider=judge_provider,judge_model=judge_model)
        code_root=Path(__file__).resolve().parents[2]
        safe_models["pipeline_fingerprint"]=fingerprint({p:hashlib.sha256((code_root/p).read_bytes()).hexdigest() for p in
            ["src/rag/rag_graph.py","src/rag/evidence_coverage.py","src/retrieval/reranker.py","src/retrieval/retrieval_comparator.py",
             "src/retrieval/semantic_retriever.py","src/rag/indexed_context.py","src/evaluation/evaluation_runner.py","src/evaluation/retrieval_metrics.py",
             "src/evaluation/chunk_compatibility.py","src/evaluation/answer_quality.py","src/evaluation/answer_judge.py","src/evaluation/faithfulness_evaluator.py","src/evaluation/citation_evaluator.py",
             "src/llm/provider_factory.py","src/llm/groq_service.py","src/llm/error_diagnostics.py","src/rag/answer_generator.py","src/rag/excerpt_spans.py","src/rag/conversational_answer.py"]})
        parameters={"configurations":list(configurations),"question_ids":[q["question_id"] for q in selected],
            "top_k":top_k,"candidate_count":candidate_count,"mode":mode,"filters":filters or {},"judge":enable_judge,"protocol":protocol,
            'answer_generation_mode':answer_generation_mode}
        signature=fingerprint({"dataset":self.dataset["fingerprint"],"parameters":parameters,"models":safe_models,"source":self.source,"index":settings.pinecone_index_name})
        if resume:
            if not re.fullmatch(r"[A-Za-z0-9_-]+",resume):raise ValueError("Invalid run ID")
            run=json.loads((self.root/"results"/f"{resume}.json").read_text(encoding="utf-8"))
            original_signature=fingerprint({'dataset':run['dataset_fingerprint'],'parameters':run['parameters'],
                'models':run['models'],'source':run['source'],'index':run['index_name']})
            if run.get('signature')!=original_signature:
                raise ValueError('Stored resume signature is invalid; checkpoint was not changed')
            if run.get("signature")!=signature:
                # Explicit, narrow migration for the investigated pre-recovery checkpoint.
                legacy={'e39863e7a927d249068780b6fed870cb06d917ecb7f0638fcfbc050987575875',
                    '1e98cfbcc29f6ac5a1b82536fb93a4720a366de3e6a65139539abb73f094369f',
                    '2aa75a26206722e11477bd34bb45b33f60999a7e8c81197105dbd3886fdf36f5'}
                previous_models={k:v for k,v in run['models'].items() if k!='pipeline_fingerprint'}
                current_models={k:v for k,v in safe_models.items() if k!='pipeline_fingerprint'}
                permitted=allow_recovery_migration and run['models'].get('pipeline_fingerprint') in legacy and \
                    previous_models==current_models and run['parameters']==parameters and \
                    run['dataset_fingerprint']==self.dataset['fingerprint'] and run['source']==self.source and run['index_name']==settings.pinecone_index_name
                if not permitted:raise ValueError("Resume configuration/dataset changed or this is a preparation-only report; start a new run")
            resume_after=(run.get('rate_limit') or {}).get('resume_after')
            if resume_after:
                try:future=datetime.fromisoformat(resume_after)>datetime.now(timezone.utc)
                except (ValueError,TypeError):raise ValueError('Invalid saved resume-after time; manual review required') from None
                if future:
                    from src.evaluation.checkpoint_safety import ResumeDeferred
                    raise ResumeDeferred('Rate-limit cooldown is still active. Wait until the saved resume-after time and submit explicitly again.')
            backup=backup_checkpoint(self.root,resume)
            run.setdefault('events',[]).append({'event':'resume','timestamp':datetime.now(timezone.utc).isoformat(),
                'backup':backup,'original_signature':run['signature'],'execution_signature':signature,
                'recovery_migration':run['signature']!=signature})
        else:
            run={"run_id":_checkpoint_id,
                "timestamp":datetime.now(timezone.utc).isoformat(),"dataset_fingerprint":self.dataset["fingerprint"],
                "corpus_identifier":self.dataset.get("source_sha256","synthetic"),"index_name":settings.pinecone_index_name,
                "source":self.source,"parameters":parameters,"models":safe_models,"signature":signature,"records":[],"events":[]}
        run['dataset_version']=self.dataset.get('dataset_version','historical-v1' if not self.dataset.get('synthetic') else 'synthetic')
        run['answer_generation_mode']=answer_generation_mode
        run['label_chunk_configurations']=compatibility_dataset.get('chunk_configurations',{})
        run['execution_controls']={'min_request_interval':min_request_interval,'account_requests_per_minute':account_requests_per_minute,
            'max_new_questions_per_session':max_new_questions_per_session}
        run['batch_session']={'limit':max_new_questions_per_session,'question_ids':[]}
        if resume:run['events'][-1]['execution_controls']=dict(run['execution_controls'])
        run.pop('termination',None)
        run.pop('rate_limit',None)
        run["status"]="Running";save_run(run,self.root)
        plan=[(q,c) for q in selected for c in configurations]
        try:
            for question,key in plan:
                previous=next((r for r in run["records"] if r["question_id"]==question["question_id"] and r["configuration"]==key),None)
                if previous and previous["status"] in {"Completed","Excluded"}:continue
                batch_questions=run['batch_session']['question_ids']
                if question['question_id'] not in batch_questions:
                    if len(batch_questions)>=max_new_questions_per_session:
                        run['status']='Batch Completed — Ready to Resume'
                        run['events'].append({'event':'batch_completed','timestamp':datetime.now(timezone.utc).isoformat(),
                            'limit':max_new_questions_per_session,'question_ids':list(batch_questions)})
                        return run  # finally atomically saves before releasing the writer lock.
                    batch_questions.append(question['question_id'])
                if cancel and cancel():raise KeyboardInterrupt
                config=CONFIGURATIONS[key]
                row={"question_id":question["question_id"],"test_id":question["test_id"],"question":question["question"],
                    "resolved_question":question["resolved_question"],"history":question["history"],"category":question["category"],
                    "expected_test_codes":question["expected_test_codes"],"configuration":key,"strategy":config.strategy,
                    "namespace":question["labels"][config.strategy]["namespace"],"reranking_applied":config.rerank,
                    "status":"Excluded","errors":[],"failure_analysis":[],"latency":{},"answer":"","protocol":protocol,
                    'llm_provider':settings.llm_provider,'llm_model':settings.llm_model,'answer_generation_mode':answer_generation_mode}
                from src.evaluation.dashboard_reporting import eligibility
                row['ground_truth_eligibility']=eligibility(row,question)
                row['evaluation_category']='positive_answer' if question['score_ranking'] else 'unverified_reference' if not question['source_verified'] else 'mock_only_injection' if question.get('mock_only') else question['category'] if question['answerability']!='answerable' else 'verified_reference_without_ranking_labels'
                started=time.perf_counter()
                execution_stage='retrieval'
                try:
                    if not question["source_verified"] or question.get("mock_only") or (mode=="retrieval" and not question["score_ranking"]):
                        row["exclusion_reason"]=question.get('reconciliation_status',"Unverified/negative/safety or incomparable label pool; not scored as a ranking question")
                    else:
                        if self.retriever is None and not (mode=='rag' and question.get('category')=='safety'):
                            from src.ui_day3 import services
                            from src.retrieval.semantic_retriever import SemanticRetriever
                            connection,embeddings=services(settings)
                            self.retriever=SemanticRetriever(settings,embeddings,connection)
                        from src.vectorstore.pinecone_client import namespace_for
                        if self.source=="live" and row["namespace"]!=namespace_for(settings,config.strategy):raise ValueError("Label namespace differs from selected namespace")
                        if mode=="retrieval":
                            observed=RetrievalComparator(self.retriever,self.reranker).run(question["resolved_question"],key,top_k,candidate_count,filters,protocol)
                            candidates,matches=observed["candidate_matches"],observed["matches"]
                            row['retrieval_diagnostics']=observed['diagnostics']
                            row["namespace"]=observed["namespace"]
                            row["latency"]={k:observed[k] for k in ["embedding_seconds","search_seconds","retrieval_seconds","reranking_seconds","pipeline_seconds"]}
                        else:
                            from src.llm.provider_factory import create_llm
                            original_retriever=self.retriever
                            class FilteredRetriever:
                                def search(_,query,strategy,top_k):
                                    search=original_retriever.search if protocol=="raw" else original_retriever.search_with_context
                                    found=search(query,strategy=strategy,top_k=top_k,filters=filters)
                                    details=check_compatibility(compatibility_dataset,strategy,found.matches,found.diagnostics.get('configuration_evidence'))
                                    row['compatibility']=details
                                    row['retrieval_diagnostics']=found.diagnostics
                                    if not details['compatible']:raise IncompatibleLabelsError(details)
                                    return found
                            retriever=FilteredRetriever()
                            base_service=self.service or create_llm(settings)
                            generation_inputs=[]
                            raw_selection=[]
                            generation_usage=[]
                            class RecordedGenerator:
                                def select(_,messages):
                                    nonlocal execution_stage
                                    execution_stage='generate_answer'
                                    generation_inputs.append(json.loads(messages[-1].content)['retrieved_evidence'])
                                    selected=base_service.select(messages)
                                    usage=getattr(base_service,'last_usage',None)
                                    if isinstance(usage,dict):generation_usage.append(dict(usage))
                                    from src.llm.groq_service import AnswerSelection
                                    if isinstance(selected,AnswerSelection):
                                        from src.rag.citation_manager import selection_diagnostics
                                        from langchain_core.documents import Document
                                        actual_sources={e['source_id']:Document(page_content=e['text'],metadata=e['metadata']) for e in generation_inputs[-1]}
                                        raw_selection.append(selection_diagnostics(selected,actual_sources))
                                    return selected
                            result=build_rag_graph(retriever,RecordedGenerator(),settings,
                                strategy=config.strategy,top_k=top_k,reranker=self.reranker if config.rerank else None,
                                candidate_count=candidate_count,raw_retrieval=True,answer_generation_mode=answer_generation_mode).invoke({"question":question["question"],"history":question["history"]})
                            rag_seconds=time.perf_counter()-started
                            evaluation_started=time.perf_counter()
                            execution_stage='answer_evaluation'
                            candidates,matches=result.get("candidate_matches",[]),result.get("matches",[])
                            row['retrieval_diagnostics']=result.get('retrieval_details',{})
                            row.update(answer=result["answer"],citations=result["citations"],answer_status=result["status"],
                                extractive_answer=result.get('extractive_answer'),display_answer=result.get('display_answer'),
                                synthesis_diagnostics=result.get('synthesis_diagnostics'),
                                selection_diagnostics=result.get('selection_diagnostics'),
                                raw_selection_diagnostics=raw_selection[-1] if raw_selection else None,
                                generation_evidence=generation_inputs[-1] if generation_inputs else [],
                                evidence_sufficient=result.get('sufficient',False),
                                resolved_question=result.get("query",question["resolved_question"]),
                                source_views={s:{"text":doc.page_content,"metadata":doc.metadata} for s,doc in result.get("sources",{}).items()},
                                node_timings=result.get("trace",[]),behavior_matches_expectation=result["status"]==question["expected_outcome"])
                            if raw_selection and row.get('selection_diagnostics'):
                                for raw,restored in zip(raw_selection[-1].get('excerpts',[]),row['selection_diagnostics'].get('excerpts',[])):
                                    restored['mismatch_diagnostics']['whitespace_restoration_succeeded']=bool(
                                        raw.get('failure_category') and not restored.get('failure_category') and
                                        raw['mismatch_diagnostics'].get('whitespace_restoration_possible'))
                            row["faithfulness"]=evaluate_faithfulness(row["resolved_question"],row["answer"],result.get("sources",{}),
                                question["expected_test_codes"],result["status"]=="abstained",
                                GroqFaithfulnessJudge(judge_settings) if enable_judge else None)
                            row["citation_evaluation"]=evaluate_citations(row["answer"],row["citations"],result.get("sources",{}),
                                question["expected_test_codes"],result["status"]=="abstained",recorded_matches=matches)
                            row['answer_quality']=evaluate_answer_quality(question,row['answer'],result.get('sources',{}),row['faithfulness'],
                                row['citation_evaluation'],result['status']=='abstained',result.get('sufficient',False),result.get('selection_diagnostics'))
                            row['verified_required_facts']=required_facts(question) if question['source_verified'] else []
                            row['missing_expected_facts']=row['answer_quality']['completeness']['missing_required_facts']
                            if enable_judge:
                                from src.evaluation.answer_judge import GroqAnswerQualityJudge
                                row['answer_quality_judge']=GroqAnswerQualityJudge(judge_settings).assess(row['resolved_question'],row['answer'],result.get('sources',{}),row['verified_required_facts'])
                            times=result.get("metrics",{});details=result.get("retrieval_details",{})
                            row["latency"]={"embedding_seconds":details.get("embedding_seconds"),"search_seconds":details.get("search_seconds"),
                                "rag_end_to_end_seconds":rag_seconds,
                                "retrieval_seconds":times.get("retrieve_evidence_seconds",0),"reranking_seconds":times.get("rerank_evidence_seconds",0),
                                "generation_seconds":times.get("generate_answer_seconds"),
                                "answer_evaluation_seconds":time.perf_counter()-evaluation_started,
                                "pipeline_seconds":times.get("retrieve_evidence_seconds",0)+times.get("rerank_evidence_seconds",0)}
                            row["llm_chunks"]=len(result.get("sources",{})) if "generate_answer_seconds" in times else 0
                            row['token_usage']=generation_usage[-1] if generation_usage else None
                        row.update(status="Completed",retrieved=serialize(matches),candidates=serialize(candidates),
                            retrieved_chunk_ids=[m["document"].metadata.get("chunk_id") for m in matches],
                            candidate_chunk_ids=[m["document"].metadata.get("chunk_id") for m in candidates],
                            retrieved_test_codes=sorted({m["document"].metadata.get("test_code") for m in matches if m["document"].metadata.get("test_code")}))
                        labels=question["labels"][config.strategy]
                        row['compatibility']=check_compatibility(compatibility_dataset,config.strategy,candidates,row.get('retrieval_diagnostics',{}).get('configuration_evidence'))
                        if question['score_ranking'] and not row['compatibility']['compatible']:raise IncompatibleLabelsError(row['compatibility'])
                        row["retrieval_metrics"]=ranking_metrics(row["retrieved_chunk_ids"],labels["chunk_ids"],top_k,
                            question["source_verified"] and question["score_ranking"],labels["complete"])
                        row['retrieval_outcome']=('Successful retrieval' if row['retrieval_metrics']['hit_rate'] else 'Retrieval miss') if row['retrieval_metrics']['scored'] else 'Not scored'
                        row["failure_analysis"]=failure_analysis(row,question)
                except IncompatibleLabelsError as exc:
                    row.update(status='Incompatible' if exc.details['state']=='mismatched' else 'Unverified',compatibility=exc.details,
                        retrieval_metrics={**ranking_metrics([],None),'reason':exc.details['status']},
                        failure_analysis=[{'category':exc.details['status'],'observation':exc.details,
                            'possible_remediation':'Select a separately verified dataset for this chunk configuration; IDs were not remapped.'}])
                except Exception as exc:
                    from src.llm.groq_service import InvalidStructuredGeneration,LLMError
                    from src.llm.error_diagnostics import details
                    row['error_diagnostics']=dict(exc.error_diagnostics) if isinstance(exc,LLMError) else details(
                        'execution_error','Evaluation failed; exception text withheld.',stage=execution_stage,retries=None)
                    row['error_diagnostics']['execution_stage']=getattr(exc,'labrag_execution_stage',execution_stage)
                    if isinstance(exc,InvalidStructuredGeneration):row['selection_diagnostics']=dict(exc.diagnostics)
                    row.update(status="Failed",errors=[f"{type(exc).__name__}: evaluation failed; inspect model/namespace/configuration. Raw provider bodies withheld."],
                        failure_analysis=[{"category":"Retrieval or execution error","observation":type(exc).__name__,"possible_remediation":"Check model availability, dependencies, network and metadata; no fallback configuration was used"}])
                row["latency"]["total_seconds"]=time.perf_counter()-started
                if previous:
                    run["events"].append({"retry":question["question_id"],"configuration":key,"previous_errors":previous.get("errors",[]),
                        'previous_error_diagnostics':previous.get('error_diagnostics'),'previous_status':previous['status'],
                        'timestamp':datetime.now(timezone.utc).isoformat()})
                    run["records"].remove(previous)
                run["records"].append(row)
                failure=row.get('error_diagnostics') or {}
                if failure.get('http_status')==429:
                    run['status']='Paused — Provider Configuration' if failure.get('provider_error_category')=='billing_or_quota' else 'Paused — Rate Limited'
                    run['rate_limit']=dict(row['error_diagnostics'])
                    if run['rate_limit'].get('backoff_reason')=='provider_delay':
                        run['rate_limit']['backoff_reason']='bounded_retries_exhausted'
                    run['events'].append({'event':'rate_limit_pause','question_id':row['question_id'],
                        'configuration':key,**run['rate_limit']})
                    save_run(run,self.root)
                    return run
                save_run(run,self.root)
                if progress:progress(len(run["records"]),len(plan),row)
            run["status"]="Completed with failures" if any(r['status']=='Failed' for r in run['records']) else "Completed"
        except KeyboardInterrupt:
            run["status"]="Interrupted"
            run['termination']={'reason':'keyboard_interrupt','timestamp':datetime.now(timezone.utc).isoformat()}
        except Exception:
            run['status']='Failed'
            run['termination']={'reason':'execution_or_checkpoint_failure','timestamp':datetime.now(timezone.utc).isoformat()}
            raise
        except BaseException:
            # Streamlit rerun/stop and process-level exits may bypass Exception.
            run['status']='Interrupted'
            run['termination']={'reason':'runtime_stop_or_rerun','timestamp':datetime.now(timezone.utc).isoformat()}
            raise
        finally:save_run(run,self.root)
        return run
