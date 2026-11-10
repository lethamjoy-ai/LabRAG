"""Verify sources afresh; build namespace-specific positive relevance pools."""
import hashlib
import json
from pathlib import Path
from langchain_core.documents import Document
from src.rag.indexed_context import enrich_indexed_matches
from src.utils.conversation_manager import resolve_question
from src.vectorstore.pinecone_client import namespace_for

GROUND_TRUTH=Path("evaluation/mayo_day5_ground_truth.json")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def build_ground_truth(settings,day4_path=Path("evaluation/mayo_day4_cases.json"),snapshot_root=Path("data/processed")):
    from scripts.validate_day4_corpus import build_dataset,PDF
    original=json.loads(Path(day4_path).read_text(encoding="utf-8"))
    # Passing mocked tests is not source verification. Re-read the actual PDF and
    # compare each expected fact against the freshly verified source-case content.
    fresh=build_dataset(PDF) if PDF.exists() else None
    checked={c["test_id"]:c for c in (fresh or {}).get("cases",[])}
    pools={};provenance=[]
    for strategy in ("fixed","section"):
        matches=[]
        for path in sorted(Path(snapshot_root,strategy).glob("*.jsonl")):
            records=[json.loads(line) for line in path.open(encoding="utf-8")]
            # Only the source PDF whose facts were verified supplies positive labels.
            matches.extend({"document":Document(page_content=r["text"],metadata=r["metadata"]),"score":0.0}
                for r in records if r["metadata"].get("document_id")==original["source_sha256"])
            provenance.append({"strategy":strategy,"path":path.as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        pools[strategy]=enrich_indexed_matches(matches)
    questions=[]
    for case in original["cases"]:
        history=[]
        expectations=case.get("turn_expectations") or [{"codes":case["expected_test_codes"],"facts":case["expected_facts"]}]
        if len(expectations)!=len(case["conversation"]):
            raise ValueError("Every turn requires expectations")
        for turn,(question,expected) in enumerate(zip(case["conversation"],expectations),1):
            resolved=resolve_question(question,history)
            verified=bool(fresh and case["test_id"] in checked and checked[case["test_id"]].get("source_verified") and
                original["source_sha256"]==fresh["source_sha256"] and
                checked[case["test_id"]].get("turn_expectations")==case.get("turn_expectations") and
                checked[case["test_id"]].get("expected_facts")==case["expected_facts"])
            positive=case["expected_outcome"]=="answered"
            labels={};documents=set();pages=set()
            facts=[" ".join(f.split()).casefold() for f in expected["facts"]]
            for strategy,matches in pools.items():
                relevant=set();sections=[]
                for match in matches:
                    doc=match["document"];meta=doc.metadata
                    if meta.get("test_code") not in expected["codes"]:
                        continue
                    text=" ".join(doc.page_content.split()).casefold()
                    if facts and any(fact in text for fact in facts):
                        relevant.add(meta["chunk_id"]);documents.add(meta["document_id"])
                        if meta.get("printed_catalog_page"):pages.add(str(meta["printed_catalog_page"]))
                        sections.append({"document_id":meta["document_id"],"test_code":meta["test_code"],
                            "printed_catalog_page":meta.get("printed_catalog_page"),"section":meta.get("section_heading","")})
                labels[strategy]={"namespace":namespace_for(settings,strategy),"chunk_ids":sorted(relevant),
                    "complete":False,"sections":sections,"method":"exact verified fact occurrence within an indexed test region; positive judged pool"}
            category="multi_turn_followup" if len(case["conversation"])>1 else "similar_test_disambiguation" if case["test_id"]=="TC12" else "multi_fact_retrieval" if len(facts)>1 else "single_fact_retrieval"
            if not positive:category=case["category"]
            questions.append({"question_id":f"{case['test_id']}.{turn}","test_id":case["test_id"],"turn":turn,
                "question":question,"resolved_question":resolved,"history":list(history),"category":category,
                "expected_test_codes":expected["codes"],"expected_facts":expected["facts"],
                "source_catalog_pages":case["catalog_pages"],"relevant_document_ids":sorted(documents),
                "verified_catalog_pages":sorted(pages),"labels":labels,"source_verified":verified,
                "verification_status":"verified" if verified else "unverified",
                "answerability":"answerable" if positive else "negative_or_safety",
                "score_ranking":positive and verified and all(labels[s]["chunk_ids"] for s in ("fixed","section")),
                "mock_only":case.get("mock_only",False),"expected_outcome":case["expected_outcome"]})
            history.append({"question":question,"query":resolved})
    result={"schema_version":1,"source_pdf":original["source_pdf"],"source_sha256":original["source_sha256"],
        "embedding_model":settings.embedding_model,"embedding_dimension":settings.embedding_dimension,
        "day4_dataset_sha256":hashlib.sha256(Path(day4_path).read_bytes()).hexdigest(),"snapshots":provenance,
        "label_policy":"Positive judged pools, not exhaustive relevance. No exact recall. Documents alone never label the entire catalog relevant.",
        "questions":questions}
    result["fingerprint"]=fingerprint(result)
    return result


def validate_ground_truth(dataset):
    if dataset.get("fingerprint")!=fingerprint({k:v for k,v in dataset.items() if k!="fingerprint"}):
        raise ValueError("Ground-truth fingerprint differs; regenerate or review the changed labels")
    ids=[q["question_id"] for q in dataset["questions"]]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate question IDs")
    for q in dataset["questions"]:
        if q["score_ranking"] and (not q["source_verified"] or q["answerability"]!="answerable"):
            raise ValueError("Unverified/negative questions cannot enter ranking metrics")
        if q["score_ranking"] and any(not q["labels"][s]["chunk_ids"] for s in ("fixed","section")):
            raise ValueError("Comparable questions require labels for both strategies")
    return dataset
