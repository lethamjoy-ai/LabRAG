"""Citation structure and claim support are measured independently."""
from src.evaluation.faithfulness_evaluator import atomic_claims,classify_claim
from src.rag.citation_manager import citation as source_citation


def evaluate_citations(answer,citations,sources,expected_test_codes=None,abstained=False,recorded_matches=None):
    details=[];seen=set()
    for item in citations:
        identifier=item.get("source_id");issues=[];missing=[]
        if identifier in seen:continue
        seen.add(identifier)
        if identifier not in sources:issues.append("Fabricated source ID")
        else:
            if recorded_matches is not None:
                source=sources[identifier]
                if not any(m['document'].metadata.get('chunk_id')==source.metadata.get('chunk_id')
                    and m['document'].metadata.get('document_id')==source.metadata.get('document_id')
                    and source.page_content in m['document'].page_content for m in recorded_matches):
                    issues.append('Cited source is not part of recorded retrieval evidence')
            expected=source_citation(identifier,sources[identifier])
            missing=[key for key in ("document_id","test_code","printed_catalog_page","pdf_page_1_based")
                     if expected.get(key) in {None,"","Unknown","Unresolved in indexed metadata"}]
            for key in ("chunk_id","document_id","test_code","printed_catalog_page","pdf_page_1_based"):
                same=str(item.get(key))==str(expected.get(key))
                if key in {"printed_catalog_page","pdf_page_1_based"}:
                    try:same=float(item.get(key))==float(expected.get(key))
                    except (TypeError,ValueError):pass
                if expected.get(key) not in {None,"","Unknown","Unresolved in indexed metadata"} and not same:
                    issues.append(f"Incorrect or missing {key}")
            if expected_test_codes and expected["test_code"] not in expected_test_codes:
                issues.append("Citation refers to the wrong test")
        details.append({"source_id":identifier,"valid":not issues,"issues":issues,"missing_metadata":missing})
    claims=[] if abstained else atomic_claims(answer)
    supports=[]
    valid_ids={d["source_id"] for d in details if d["valid"]}
    for claim in claims:
        docs=[sources[s] for s in claim["source_ids"] if s in sources and s in valid_ids]
        status,reason=classify_claim(claim["text"],docs,expected_test_codes)
        supports.append({**claim,"classification":status,"reason":reason,"supported":status=="Supported"})
    # Referenced IDs absent from the metadata citation list are invalid too.
    for identifier in sorted({s for c in claims for s in c["source_ids"]}-seen):
        details.append({"source_id":identifier,"valid":False,"issues":["Missing citation metadata"]})
    verified=[d for d in details if not d.get('missing_metadata') or not d['valid']]
    return {"citations":details,"claim_support":supports,
        "verified_validity_rate":sum(d['valid'] for d in verified)/len(verified) if verified else None,
        "verified_validity_denominator":len(verified),"unverified_metadata_citations":len(details)-len(verified),
        "validity_rate":sum(d["valid"] for d in details)/len(details) if details else None,
        "support_rate":sum(c["supported"] for c in supports)/len(supports) if supports else None,
        "missing_citations":sum(not c["source_ids"] for c in claims),
        "human_review":any(c["classification"]=="Unverifiable" for c in supports) or any(c.get("missing_metadata") for c in details)}
