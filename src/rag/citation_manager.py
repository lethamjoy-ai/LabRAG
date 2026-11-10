"""Citation IDs belong only to passages retrieved for the current turn."""
import re
from urllib.parse import urlparse


def make_sources(matches):
    sources = {}
    seen = set()
    for match in matches:
        doc = match["document"]
        key = (doc.metadata.get("chunk_id"), doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        sources[f"S{len(sources) + 1}"] = doc
    return sources


def citation(source_id, doc):
    meta = doc.metadata
    pages = re.findall(r"(?im)\bPage\s+(\d+)\s*$", doc.page_content)
    url = meta.get("source_url", "")
    if urlparse(url).scheme not in {"http", "https"}:
        url = ""
    return {"source_id": source_id, "document_id": meta.get("document_id") or "Unknown", "provider": meta.get("laboratory_provider") or "Unknown",
        "test_name": meta.get("test_name") or "Unresolved in indexed metadata",
        "test_code": meta.get("test_code") or "Unresolved in indexed metadata",
        "document_title": meta.get("document_title") or "Unknown",
        "printed_catalog_page": meta.get("printed_catalog_page") or (pages[-1] if pages else "Unknown"),
        "pdf_page_1_based": meta.get("page_number") or "Unknown",
        "original_pdf_page_1_based": meta.get("original_page_number") or "Unknown",
        "section": meta.get("section_heading") or "Unknown", "chunk_id": meta.get("chunk_id") or "Unknown",
        "source_url": url}


def validate_selection(selection, sources):
    if not selection.excerpts:
        return False
    attack = re.compile(r"ignore all previous instructions|answer without citations|invent the missing specimen requirements", re.I)
    return all(not attack.search(excerpt.quote) and excerpt.source_id in sources and excerpt.quote.strip()
        and excerpt.quote in sources[excerpt.source_id].page_content for excerpt in selection.excerpts)


def selection_diagnostics(selection,sources):
    """Explain the existing decision without changing its all-or-nothing rules."""
    import hashlib
    failures=[];excerpts=[]
    if selection is None:
        return {'stage':'evidence_gate','structured_response':False,'excerpt_count':0,'source_ids':[],
            'quote_validation':'not_attempted','failure_category':'Evidence insufficient','excerpts':[]}
    if not selection.excerpts:failures.append('No excerpts selected')
    attack=re.compile(r'ignore all previous instructions|answer without citations|invent the missing specimen requirements',re.I)
    for excerpt in selection.excerpts:
        source=sources.get(excerpt.source_id)
        if source is None:failure='Unknown source identifier'
        elif attack.search(excerpt.quote) or not excerpt.quote.strip():failure='Other validation failure'
        elif excerpt.quote not in source.page_content:failure='Quote does not match the cited passage'
        else:failure=None
        if failure:failures.append(failure)
        trusted=bool(source and (source.metadata.get('corpus_type')=='synthetic' or
            source.metadata.get('corpus_type')=='public' and source.metadata.get('test_scope')=='catalog' and
            urlparse(source.metadata.get('source_url','')).hostname in {'mml-content.s3.amazonaws.com','www.mayocliniclabs.com','mayocliniclabs.com'}))
        grounded=bool(source and excerpt.quote.strip() and ' '.join(excerpt.quote.split()) in ' '.join(source.page_content.split()))
        safe=trusted and grounded and failure in {None,'Quote does not match the cited passage'}
        from src.rag.quote_mismatch import diagnose_quote
        excerpts.append({'source_id':excerpt.source_id if re.fullmatch(r'S\d+(?:_\d+)?',excerpt.source_id) else '[redacted invalid identifier]',
            'mismatch_diagnostics':diagnose_quote(excerpt.quote,excerpt.source_id,sources),
            'quote':excerpt.quote if safe else None,'quote_retained':safe,
            'quote_sha256':hashlib.sha256(excerpt.quote.encode()).hexdigest(),'failure_category':failure})
    return {'stage':'quote_validation','selection_protocol':selection.selection_protocol,'structured_response':True,'excerpt_count':len(selection.excerpts),
        'source_ids':[e['source_id'] for e in excerpts],'quote_validation':'passed' if validate_selection(selection,sources) else 'rejected',
        'failure_category':failures[0] if failures else None,'failure_categories':list(dict.fromkeys(failures)),
        'excerpts':excerpts,'quote_retention':'Only source-grounded public Mayo catalog/synthetic quotes (including whitespace-only differences); other text and free-form model notes are withheld'}


def restore_source_whitespace(selection, sources):
    """Restore PDF indentation only when every non-whitespace token matches.

    Return the actual indexed substring, preserving table spacing. Words, case,
    punctuation, numbers and units must remain identical; unsupported quotes still
    fail the ordinary exact-substring validator. The model response is not mutated.
    """
    result = selection.model_copy(deep=True)
    for excerpt in result.excerpts:
        doc = sources.get(excerpt.source_id)
        if doc is None or excerpt.quote in doc.page_content:
            continue
        words = excerpt.quote.split()
        if not words:
            continue
        match = re.search(r"\s+".join(re.escape(word) for word in words), doc.page_content)
        if match:
            excerpt.quote = match[0]
    return result
