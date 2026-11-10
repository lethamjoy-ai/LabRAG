"""A fresh graph invocation retrieves evidence for every submitted turn."""
import time
from src.observability.langsmith_tracing import observe, TracedGraph
from langgraph.graph import StateGraph, START, END
from src.rag.rag_state import RAGState
from src.rag.citation_manager import make_sources, citation, validate_selection,selection_diagnostics
from src.rag.evidence_validator import validate_evidence as check_evidence
from src.rag.evidence_validator import scope_sources
from src.rag.evidence_validator import policy_blocked
from src.rag.answer_generator import generate_answer as select_answer
from src.utils.conversation_manager import resolve_question

ABSTENTION = "I cannot answer this from the retrieved passages in the indexed partial catalog. This does not establish absence from the full Mayo catalog."


def build_rag_graph(retriever, service, settings, strategy=None, top_k=None, reranker=None, candidate_count=15, raw_retrieval=False,
                    answer_generation_mode='extractive_v1'):
    from src.rag.conversational_answer import MODES,CONVERSATIONAL,CONVERSATIONAL_V2,synthesize
    conversational_modes=(CONVERSATIONAL,CONVERSATIONAL_V2)
    if answer_generation_mode not in MODES:raise ValueError('Unknown answer-generation mode')
    final_k=top_k or settings.retrieval_top_k
    if type(candidate_count) is not int or not 1<=candidate_count<=100 or (reranker and candidate_count<final_k):
        raise ValueError("Candidate count must be 1–100 and at least final top-k when reranking.")
    graph = StateGraph(RAGState)

    def timed(name, function):
        def run(state):
            start = time.perf_counter()
            try:
                result = observe(name, lambda: function(state), state)
            except Exception as exc:
                exc.labrag_execution_stage = name
                raise
            elapsed = time.perf_counter() - start
            metrics = dict(state.get("metrics", {}))
            metrics[name + "_seconds"] = elapsed
            result["metrics"] = metrics
            result["trace"] = [*state.get("trace", []), {"node": name, "seconds": elapsed}]
            return result
        graph.add_node(name, run)

    def validate_question(state):
        q = state["question"]
        if not isinstance(q, str) or not q.strip() or len(q) > 3000:
            raise ValueError("Enter a question between 1 and 3000 characters.")
        blocked = policy_blocked(q)
        return {"question": q.strip(), "trace": [], "metrics": {}, "policy_blocked": blocked,
                'answer_generation_mode':answer_generation_mode,
                "sources": {}, "evidence_status": "policy_blocked" if blocked else "pending"}

    def retrieve(state):
        contextual=not raw_retrieval and reranker is None
        search = retriever.search_for_chat if contextual and hasattr(type(retriever), "search_for_chat") else retriever.search
        result = search(state["query"], strategy=strategy or settings.rag_default_strategy,
                                  top_k=candidate_count if reranker else final_k)
        matches = retriever.expand_context(result) if contextual and hasattr(type(retriever), "expand_context") else result.matches
        sources=make_sources(matches)
        ids={(doc.metadata.get("chunk_id"),doc.page_content):key for key,doc in sources.items()}
        matches=[{**m,"source_id":ids[(m["document"].metadata.get("chunk_id"),m["document"].page_content)]} for m in matches]
        return {"matches": matches,"candidate_matches":matches,"sources":sources, "namespace": result.namespace,
                "retrieval_details":{**result.diagnostics,"strategy":strategy or settings.rag_default_strategy,
                    "reranking_applied":False,"initial_candidates":len(matches),"final_chunks":len(matches)}}

    def rerank(state):
        ranked=reranker.rerank(state["query"],state["matches"],final_k)
        return {"matches":ranked.matches,"sources":{m["source_id"]:m["document"] for m in ranked.matches},
            "retrieval_details":{**state["retrieval_details"],"reranking_applied":True,"reranker_model":reranker.model_name,
                "reranking_seconds":ranked.latency_seconds,"final_chunks":len(ranked.matches)}}

    def validate(state):
        scoped, identity_reason = scope_sources(state["query"], state["sources"])
        sufficient, reason = check_evidence(state["query"], scoped)
        return {"sufficient": sufficient, "reason": reason if scoped else identity_reason,
                "sources": scoped, "evidence_status": "sufficient" if sufficient else "missing_information" if scoped else "retrieval_failure"}

    def citations(state):
        selection = state.get("selection")
        diagnostics=selection_diagnostics(selection,state['sources'])
        diagnostics.update(generation_executed=selection is not None,evidence_gate_passed=state.get('sufficient',False))
        for excerpt in diagnostics.get('excerpts',[]):
            excerpt['mismatch_diagnostics']['whitespace_restoration_succeeded']=None
        if state.get('policy_blocked'):diagnostics.update(stage='policy_gate',failure_category='Safety/refusal policy')
        if selection is None or not validate_selection(selection, state["sources"]):
            limitation = state.get("reason", "No verifiable supporting excerpts.") if selection is None else "The model did not select verifiable supporting excerpts for this question."
            answer, context_citations = ABSTENTION, []
            if state.get("policy_blocked"):
                answer = "LabRAG is an educational laboratory reference assistant and cannot recommend patient-specific diagnosis or treatment. I can help retrieve laboratory reference information instead."
                limitation = "Patient-specific treatment is outside the scope of this assistant."
            elif state.get("evidence_status") == "missing_information":
                answer = "Relevant test passages were retrieved, but the requested information is not specified in them. The citations provide context only; they do not establish the missing value."
                context_citations = [{**citation(key, doc), "citation_role": "context_only"}
                                    for key, doc in list(state["sources"].items())[:2]]
            return {"answer": answer, "status": "abstained", "citations": context_citations,
                'selection_diagnostics':{**diagnostics,'final_answer_status':'abstained'},
                "missing_information": [limitation], "conflicting_evidence": []}
        ids = list(dict.fromkeys(e.source_id for e in selection.excerpts))
        return {"answer": "\n\n".join(f'{e.quote} [{e.source_id}]' for e in selection.excerpts),
            'selection_diagnostics':{**diagnostics,'final_answer_status':'answered'},
            "status": "answered", "citations": [citation(key, state["sources"][key]) for key in ids],
            "missing_information": ["The model flagged incomplete evidence; verify the source excerpts."] if selection.missing_information else [],
            "conflicting_evidence": ["The model flagged a possible conflict. Review the cited passages before using these instructions."] if selection.conflicting_evidence else []}

    def conversational_answer(state):
        import re
        base={'answer_generation_mode':answer_generation_mode}
        if answer_generation_mode not in conversational_modes or state.get('status')!='answered':return base
        effective=state['selection']
        coverage=None;coverage_valid=True
        if answer_generation_mode==CONVERSATIONAL_V2:
            from src.rag.evidence_coverage import supplement_selection
            effective,coverage,coverage_valid=supplement_selection(state['query'],effective,state['sources'])
        diagnostic=synthesize(state['query'],effective,state['sources'],
            extended_layouts=answer_generation_mode==CONVERSATIONAL_V2)
        if coverage is not None:
            diagnostic.update(version='typed_catalog_templates_v2',evidence_coverage=coverage)
            if not coverage_valid:
                diagnostic.update(validated=False,answer='',reason='Required selected evidence is missing, conflicting, or unsupported.')
        audit={'extractive_answer':state['answer'],'synthesis_diagnostics':diagnostic}
        if not diagnostic['validated']:
            return {**base,**audit,'answer':ABSTENTION,'display_answer':ABSTENTION,'status':'abstained','citations':[],
                'selection_diagnostics':{**state['selection_diagnostics'],'stage':'conversational_validation',
                    'failure_category':'Synthesis facts unresolved','final_answer_status':'abstained'},
                'missing_information':[diagnostic['reason']]}
        used={c['source_id'] for c in diagnostic['claims']}
        return {**base,**audit,'answer':diagnostic['answer'],
            'display_answer':re.sub(r'\s*\[S\d+(?:_\d+)?\]','',diagnostic['answer']),
            'missing_information':[*state.get('missing_information',[]),*diagnostic.get('limitations',[])],
            'citations':([citation(sid,doc) for sid,doc in state['sources'].items() if sid in used]
                if answer_generation_mode==CONVERSATIONAL_V2 else
                [c for c in state['citations'] if c['source_id'] in used])}

    timed("validate_question", validate_question)
    timed("resolve_follow_up", lambda s: {"query": resolve_question(s["question"], s.get("history", []), settings.rag_max_history_turns)})
    timed("retrieve_evidence", retrieve)
    if reranker:timed("rerank_evidence",rerank)
    timed("validate_evidence", validate)
    timed("generate_answer", lambda s: {"selection": select_answer(service, s["query"], s["sources"])})
    timed("abstain", lambda s: {"selection": None})
    timed("validate_citations", citations)
    if answer_generation_mode in conversational_modes:timed('synthesize_answer',conversational_answer)
    graph.add_edge(START, "validate_question")
    graph.add_conditional_edges("validate_question", lambda s: "abstain" if s["policy_blocked"] else "resolve_follow_up")
    for left, right in [("resolve_follow_up", "retrieve_evidence")]:
        graph.add_edge(left, right)
    if reranker:
        graph.add_edge("retrieve_evidence","rerank_evidence")
        graph.add_edge("rerank_evidence","validate_evidence")
    else:graph.add_edge("retrieve_evidence","validate_evidence")
    graph.add_conditional_edges("validate_evidence", lambda s: "generate_answer" if s["sufficient"] else "abstain")
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_edge("abstain", "validate_citations")
    if answer_generation_mode in conversational_modes:
        graph.add_edge('validate_citations','synthesize_answer')
        graph.add_edge('synthesize_answer',END)
    else:graph.add_edge('validate_citations',END)
    return TracedGraph(graph.compile(), settings)
