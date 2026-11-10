"""V2 deterministic supplements from already retrieved, scoped public evidence."""
import hashlib
import json
import re
from urllib.parse import urlparse

from src.llm.groq_service import AnswerSelection, SupportedExcerpt
from src.rag.citation_manager import validate_selection
from src.rag.conversational_answer import extract_facts, required_kinds
from src.rag.evidence_validator import scope_sources
from src.rag.excerpt_spans import excerpt_spans

VERSION = 'verified_selection_coverage_v2'


def trusted(doc):
    meta = doc.metadata
    return meta.get('corpus_type') == 'synthetic' or (
        meta.get('corpus_type') == 'public' and meta.get('test_scope') == 'catalog' and
        urlparse(meta.get('source_url', '')).hostname in {
            'mml-content.s3.amazonaws.com', 'www.mayocliniclabs.com', 'mayocliniclabs.com'})


def requirements(query, owners):
    kinds = required_kinds(query)
    if re.search('prepar', query, re.I):kinds.update({'fasting', 'biotin'})
    if re.search('volume', query, re.I) and not re.search('minimum', query, re.I):kinds.add('specimen_volume')
    if re.search('preferred.*tube|preferred.*container', query, re.I):kinds.add('preferred_tube')
    if re.search('acceptable.*tube|acceptable.*container', query, re.I):kinds.add('acceptable_tube')
    if re.search('collect',query,re.I) and re.search('stor',query,re.I):kinds.add('collection_instruction')
    pairs = set()
    for owner in owners:
        for kind in kinds:
            qualifiers = ['']
            if kind == 'storage':
                qualifiers = [word for word, pattern in [('frozen', 'frozen'), ('refrigerated', 'refrigerat')]
                              if re.search(pattern, query, re.I)] or ['']
            pairs.update((owner, kind, qualifier) for qualifier in qualifiers)
    return pairs


def supports(item, requirement):
    code, kind, qualifier = requirement
    return item.test_code == code and item.kind == kind and (
        not qualifier or qualifier == item.data.get('specimen_role') or qualifier in item.data.get('temperature', '').lower())


def missing(pairs, facts):
    return {pair for pair in pairs if not any(supports(item, pair) for item in facts)}


def safe_requirements(pairs):
    return [{'test_reference': hashlib.sha256(code.encode()).hexdigest()[:16],
             'kind': kind, 'qualifier': qualifier} for code, kind, qualifier in sorted(pairs)]


def supplement_selection(query, selection, sources):
    """Never repair an invalid selection, infer a fact, retrieve, or invoke an LLM.

    Return a copy. Original LLM selections and their diagnostics remain unchanged.
    Entire application-owned spans retain headings/table context for typed parsing.
    """
    diagnostic = {'version': VERSION, 'status': 'not_applied', 'missing_before': [],
                  'missing_after': [], 'supplemental_excerpts': [], 'conflict': False}
    if not validate_selection(selection, sources) or selection.conflicting_evidence:
        diagnostic['status'] = 'invalid_or_conflicting_original_selection'
        return selection, diagnostic, False
    pool = {sid: doc for sid, doc in sources.items() if trusted(doc) and re.fullmatch(r'S\d+(?:_\d+)?', sid)}
    scoped, _ = scope_sources(query, pool)
    # Do not introduce newly scoped identifiers or change source boundaries here.
    scoped = {sid: doc for sid, doc in scoped.items() if sid in pool and doc.page_content == pool[sid].page_content}
    owners = {doc.metadata['test_code'] for doc in scoped.values() if doc.metadata.get('test_code')}
    pairs = requirements(query, owners)
    selected_facts = extract_facts(selection, sources,extended_layouts=True)
    before = missing(pairs, selected_facts)
    diagnostic['missing_before'] = safe_requirements(before)
    if not owners:
        diagnostic['status'] = 'unresolved_trusted_identity'
        return selection, diagnostic, False
    if not pairs:
        diagnostic['status'] = 'no_supported_required_fields_for_supplement'
        return selection, diagnostic, True
    candidates, all_facts = [], []
    for sid, doc in scoped.items():
        if not doc.page_content.strip():continue
        spans = [{'start': 0, 'end': len(doc.page_content), 'text': doc.page_content}] if len(doc.page_content) <= 3000 else excerpt_spans(sid, doc.page_content)
        for span in spans:
            excerpt = SupportedExcerpt(source_id=sid, quote=span['text'])
            facts = extract_facts(AnswerSelection(excerpts=[excerpt]), sources,extended_layouts=True)
            all_facts.extend(facts)
            candidates.append((excerpt, span, facts))
    if re.search(r'specimen types|specimens.*accepted',query,re.I):
        alternative_owners={f.test_code for f in all_facts if f.kind=='specimen_type' and f.data.get('specimen_role')}
        for owner in alternative_owners:
            pairs.discard((owner,'specimen_type',''))
            pairs.update((owner,'specimen_type',role) for role in ('preferred','acceptable'))
        before=missing(pairs,selected_facts)
        diagnostic['missing_before']=safe_requirements(before)
    # Check all explicit retrieved values for requested fields, not just the first
    # candidate. Competing values must not be resolved by source order.
    fields = {}
    for item in [*selected_facts, *all_facts]:
        if not any(supports(item, pair) for pair in pairs) or item.kind == 'collection_instruction':continue
        key = (item.test_code, item.kind)
        if item.kind=='specimen_type' and item.data.get('specimen_role'):key+=(item.data['specimen_role'],)
        data = dict(item.data)
        if item.kind == 'storage':
            temperature = re.sub(r'\s*\(preferred\)', '', data['temperature'], flags=re.I).lower()
            key += (data['specimen'].lower(), temperature)
            data['temperature'] = temperature
        fields.setdefault(key, set()).add(json.dumps(data, sort_keys=True))
    if any(len(values) > 1 for values in fields.values()):
        diagnostic.update(status='conflicting_retrieved_evidence', conflict=True,
                          missing_after=safe_requirements(before))
        return selection, diagnostic, False
    effective = selection.model_copy(deep=True)
    remaining = before
    for excerpt, span, facts in candidates:
        if not any(supports(item, pair) for pair in remaining for item in facts):continue
        if len(effective.excerpts) >= 12:break
        if any(e.source_id == excerpt.source_id and e.quote == excerpt.quote for e in effective.excerpts):continue
        effective.excerpts.append(excerpt)
        diagnostic['supplemental_excerpts'].append({'source_id': excerpt.source_id,
            'start': span['start'], 'end': span['end'],
            'quote_sha256': hashlib.sha256(excerpt.quote.encode()).hexdigest()})
        remaining = missing(pairs, extract_facts(effective, sources,extended_layouts=True))
    diagnostic['missing_after'] = safe_requirements(remaining)
    valid = validate_selection(effective, sources)
    diagnostic['status'] = 'coverage_verified' if valid and not remaining else 'required_evidence_missing'
    return effective, diagnostic, bool(valid and not remaining)
