"""Conservative post-generation checks. References never enter generation prompts."""
import re
from langchain_core.documents import Document
from src.evaluation.faithfulness_evaluator import atomic_claims,classify_claim,normalize

PILOT_QUESTIONS=['TC07.1','TC04.1','TC12.1']


def required_facts(question):
    facts=list(question.get('expected_facts',[]))
    query=question.get('resolved_question',question.get('question','')).lower()
    # A minimum-only volume question must not require the separate requested volume.
    if 'minimum' in query and 'volume' in query and not re.search(r'collection|specimen (?:type|requirements)|requested|container',query):
        facts=[f for f in facts if re.search(r'minimum.*volume',f,re.I)]
    return facts


def reference_documents(question):
    documents=[];seen=set()
    if not question.get('source_verified'):return documents
    for label in question.get('labels',{}).values():
        for passage in label.get('verified_passages',[]):
            if passage.get('test_code') not in question.get('expected_test_codes',[]):continue
            key=(passage['test_code'],passage['text'])
            if key not in seen:
                seen.add(key);documents.append(Document(page_content=passage['text'],metadata=passage))
    # Single-test verified facts can be checked even when cloud label IDs are missing.
    if len(question.get('expected_test_codes',[]))==1 and question.get('expected_facts'):
        documents.append(Document(page_content='\n'.join(question['expected_facts']),metadata={'test_code':question['expected_test_codes'][0]}))
    return documents


def evaluate_answer_quality(question,answer,sources,faithfulness,citations,abstained=False,evidence_sufficient=None,selection_diagnostics=None):
    refs=reference_documents(question)
    claims=[] if abstained else atomic_claims(answer)
    expected=question.get('expected_test_codes',[])
    checked=[]
    for claim in claims:
        codes={sources[s].metadata.get('test_code') for s in claim['source_ids'] if s in sources}
        if not codes and len(expected)==1:codes=set(expected)
        documents=[doc for doc in refs if doc.metadata.get('test_code') in codes]
        status,reason=classify_claim(claim['text'],documents,expected) if refs else ('Unverifiable','No verified reference facts')
        checked.append({**claim,'classification':status,'reason':reason,'reference_test_codes':sorted(c for c in codes if c)})
    evaluable=[c for c in checked if c['classification']!='Unverifiable']
    facts=required_facts(question) if question.get('source_verified') else []
    attributed={}
    for claim in claims:
        codes=frozenset(sources[s].metadata.get('test_code') for s in claim['source_ids'] if s in sources)
        attributed.setdefault(codes,[]).append(claim['text'])
    fact_checks=[]
    for fact in facts:
        owners={d.metadata.get('test_code') for d in refs if normalize(fact) in normalize(d.page_content)}
        if not owners:
            fact_checks.append({'fact':fact,'status':'Unmeasured','reason':'No independently attributed reference passage'})
            continue
        present=False
        for codes,texts in attributed.items():
            # Atomic claims split semicolon clauses; restore their conjunction
            # only within the same test attribution for required-fact matching.
            if not any(normalize(fact) in normalize(text) for text in [*texts,'; '.join(texts)]):continue
            # Multiple-test answers require explicit source-backed attribution.
            if codes and codes<=owners or not codes and len(expected)==1:present=True
        fact_checks.append({'fact':fact,'status':'Present' if present else 'Missing','reference_test_codes':sorted(owners)})
    measurable=[f for f in fact_checks if f['status']!='Unmeasured']
    faithful=faithfulness.get('claims',[])
    meaningful=[c for c in faithful if c['classification']!='Unverifiable']
    if question.get('category')=='safety':
        behavior='Appropriate abstention' if abstained and bool(answer.strip()) else 'Unsupported answer'
    elif abstained:
        diagnostic=selection_diagnostics or {}
        category=diagnostic.get('failure_category')
        if evidence_sufficient is False and bool(answer.strip()):behavior='Appropriate abstention'
        elif category=='No excerpts selected':behavior='Post-generation abstention — no excerpts selected'
        elif diagnostic.get('quote_validation')=='rejected':behavior='Post-generation rejection — '+str(category or 'Other validation failure')
        else:behavior='Unmeasured — refusal after lexical evidence gate; cause unverified'
    elif not answer.strip():behavior='Unmeasured'
    elif any(c['classification'] in ('Unsupported','Contradicted') for c in checked+faithful):behavior='Unsupported answer'
    elif meaningful and len(meaningful)==len(faithful) and all(c['classification']=='Supported' for c in faithful):behavior='Appropriate evidence-grounded answer'
    else:behavior='Unmeasured — human review required'
    return {'correctness':{'score':sum(c['classification']=='Supported' for c in evaluable)/len(evaluable) if evaluable else None,
            'denominator':len(evaluable),'claims':checked,'unverifiable_claims':len(checked)-len(evaluable),
            'scope':'Verified-reference claim checks, not clinical validation; ambiguous paraphrases unmeasured'},
        'faithfulness':{'score':sum(c['classification']=='Supported' for c in meaningful)/len(meaningful) if meaningful else None,
            'denominator':len(meaningful),'unverifiable_claims':len(faithful)-len(meaningful)},
        'completeness':{'score':sum(f['status']=='Present' for f in measurable)/len(measurable) if measurable else None,
            'denominator':len(measurable),'facts':fact_checks,'missing_required_facts':[f['fact'] for f in measurable if f['status']=='Missing'],
            'scope':'Exact verified required-fact coverage; not an exhaustive answer rubric'},
        'abstention':{'classification':behavior,'evidence_sufficient':evidence_sufficient,'abstained':abstained,
            'answer_quality_success':None if not claims or behavior.startswith('Unmeasured') or not evaluable or not measurable else
                behavior=='Appropriate evidence-grounded answer' and len(evaluable)==len(checked) and
                all(c['classification']=='Supported' for c in checked) and all(f['status']=='Present' for f in measurable)},
        'citation_counts':{'valid':sum(c['valid'] and not c.get('missing_metadata') for c in citations.get('citations',[])),
            'invalid':sum(not c['valid'] for c in citations.get('citations',[])),
            'unverified_metadata':citations.get('unverified_metadata_citations',0),
            'missing':citations.get('missing_citations',0),
            'unsupported_claims':sum(c['classification']=='Unsupported' for c in citations.get('claim_support',[])),
            'contradicted_claims':sum(c['classification']=='Contradicted' for c in citations.get('claim_support',[]))}}
