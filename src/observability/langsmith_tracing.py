"""Optional authorized course-demo I/O; privacy summaries remain the default.

Automatic LangChain/LangGraph child tracing is suppressed, including provider
errors. Custom spans receive no settings, documents, prompts or callables as
arguments. Exceptions are re-raised locally after recording a fixed failure flag.
"""
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from langsmith import Client, traceable, tracing_context

_active = ContextVar('labrag_trace_config', default=None)
_operation = ContextVar('labrag_trace_operation', default=None)
DEMO_QUESTION = 'What specimen is required for BUN?'


@lru_cache(maxsize=1)
def demo_questions():
    """Repository-owned public course questions; no uploads or user files."""
    root = Path(__file__).resolve().parents[2] / 'evaluation'
    questions = {DEMO_QUESTION}
    for filename in ('mayo_day4_cases.json', 'mayo_day5_v3_overlap15_ground_truth.json'):
        try:
            data = json.loads((root / filename).read_text(encoding='utf-8'))
            for item in data.get('cases', data.get('questions', [])):
                for key in ('question', 'resolved_question'):
                    if isinstance(item.get(key), str):questions.add(item[key])
                questions.update(q for q in item.get('conversation', []) if isinstance(q, str))
        except (OSError, ValueError, TypeError):
            continue  # Unknown inputs stay filtered if the bank is unavailable.
    return frozenset(questions)


def authorized_demo(input):
    if not isinstance(input, dict) or input.get('question') not in demo_questions():return False
    return all(isinstance(turn, dict) and turn.get('question') in demo_questions()
               for turn in input.get('history', []))


SENSITIVE = re.compile(r'\b(?:MRN|SSN|DOB|date of birth|patient[_ -]?name)\s*[:=]|'
                       r'\b\d{3}-\d{2}-\d{4}\b|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
SECRET = re.compile(r'\b(?:sk-[A-Za-z0-9_-]+|gsk_[A-Za-z0-9_-]+|lsv2_[A-Za-z0-9_-]+)|'
                    r'\bBearer\s+\S+|\b(?:api[_ -]?key|authorization|access[_ -]?token|password)\s*[:=]\s*\S+', re.I)


def safe_text(value):
    if not isinstance(value, str):return value
    config = _active.get()
    for secret in config[4] if config and len(config) > 4 else ():
        if secret:value = value.replace(secret, '[redacted credential]')
    value = SECRET.sub('[redacted credential]', value)
    return '[withheld sensitive content]' if SENSITIVE.search(value) else value


def safe_reference(source):
    result = {}
    for key in ('source_id', 'document_id', 'document_title', 'provider', 'test_code', 'test_name',
                'printed_catalog_page', 'pdf_page_1_based', 'original_pdf_page_1_based',
                'section', 'chunk_id', 'citation_role'):
        value = source.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):result[key] = safe_text(str(value))
    url = source.get('source_url')
    if isinstance(url, str):
        try:
            parsed = urlsplit(url)
            if parsed.scheme in {'https', 'http'} and parsed.hostname:
                # Withhold URL credentials, query tokens and fragments.
                result['source_url'] = safe_text(urlunsplit((parsed.scheme, parsed.hostname, parsed.path, '', '')))
        except ValueError:
            pass
    return result


def sensitive_sources(value):
    for doc in (value.get('sources') or {}).values():
        meta = getattr(doc, 'metadata', {})
        public = meta.get('corpus_type') == 'public' and meta.get('test_scope') == 'catalog'
        try:
            public = public and urlsplit(meta.get('source_url', '')).hostname in {
                'mml-content.s3.amazonaws.com', 'www.mayocliniclabs.com', 'mayocliniclabs.com'}
        except (TypeError, ValueError, AttributeError):
            public = False
        if not public and meta.get('corpus_type') != 'synthetic':return True
        if any(re.search(r'patient|mrn|ssn|dob|birth|email|phone|address', str(key), re.I) and item
               for key, item in meta.items()):return True
        if SENSITIVE.search(getattr(doc, 'page_content', '')):return True
    return False


def full_summary(value, result):
    """Explicit demo I/O, never raw state/settings, histories or provider bodies."""
    config = _active.get()
    if not config or len(config) < 4 or not config[3]:return result
    if sensitive_sources(value):
        result['capture_status'] = 'sensitive_source_withheld'
        return result
    result['capture_status'] = 'authorized_course_demo'
    for key in ('question', 'answer', 'display_answer', 'extractive_answer'):
        if isinstance(value.get(key), str):result[key] = safe_text(value[key])
    if isinstance(value.get('citations'), list):
        result['citations'] = [safe_reference(c) for c in value['citations'] if isinstance(c, dict)]
    if isinstance(value.get('sources'), dict):
        result['source_references'] = []
        for sid, doc in value['sources'].items():
            meta = doc.metadata
            result['source_references'].append(safe_reference({'source_id':sid,
                **{k:meta.get(k) for k in ('document_id','document_title','test_code','test_name','chunk_id','source_url')},
                'provider':meta.get('laboratory_provider'), 'section':meta.get('section_heading'),
                'pdf_page_1_based':meta.get('page_number'),
                'original_pdf_page_1_based':meta.get('original_page_number'),
                'printed_catalog_page':meta.get('printed_catalog_page')}))
    if isinstance(value.get('metrics'), dict):
        result['timings'] = {k:v for k,v in value['metrics'].items()
                            if re.fullmatch('[a-z_]+_seconds', k) and type(v) in (int,float) and v >= 0}
    mode = value.get('answer_generation_mode')
    if mode in {'extractive_v1','grounded_conversational_v1','grounded_conversational_v2'}:
        result['answer_generation_mode'] = mode
    diag = value.get('synthesis_diagnostics') or {}
    coverage = diag.get('evidence_coverage')
    if isinstance(coverage, dict):
        safe = {k:safe_text(coverage[k]) for k in ('version','status') if isinstance(coverage.get(k),str)}
        if type(coverage.get('conflict')) is bool:safe['conflict']=coverage['conflict']
        for key, fields in [('missing_before',('test_reference','kind','qualifier')),
                            ('missing_after',('test_reference','kind','qualifier')),
                            ('supplemental_excerpts',('source_id','start','end','quote_sha256'))]:
            safe[key]=[{k:safe_text(item[k]) for k in fields if type(item.get(k)) in (str,int)}
                       for item in coverage.get(key,[]) if isinstance(item,dict)]
        result['evidence_coverage'] = safe
    if type(diag.get('validated')) is bool:result['synthesis_validated'] = diag['validated']
    validation=value.get('selection_diagnostics') or {}
    result['validation']={k:safe_text(validation[k]) for k in
        ('stage','structured_response','excerpt_count','quote_validation','failure_category','final_answer_status')
        if k in validation and type(validation[k]) in (str,int,bool)}
    return result


@lru_cache(maxsize=4)
def _client(key, endpoint):
    return Client(api_key=key, api_url=endpoint,
                  omit_traced_runtime_info=True)


def _digest(value):
    return sha256(str(value).encode()).hexdigest() if value is not None else None


def summary(value, permit_full=True):
    """No free-form text, filenames, URLs, user IDs or arbitrary metadata."""
    if not isinstance(value, dict):
        return {'payload': 'withheld'}
    result = {}
    if isinstance(value.get('question'), str):
        result['question_characters'] = len(value['question'])
        # One exact, verified catalog demo question; never echo arbitrary input.
        result['question'] = DEMO_QUESTION if value['question'] == DEMO_QUESTION else '[withheld]'
    if value.get('status') in {'answered', 'abstained', 'needs_clarification'}:
        result['status'] = value['status']
    if type(value.get('sufficient')) is bool:
        result['evidence_sufficient'] = value['sufficient']
    if isinstance(value.get('answer'), str):
        result['answer_characters'] = len(value['answer'])
    if isinstance(value.get('citations'), list):
        result['citation_count'] = len(value['citations'])
    selection = value.get('selection')
    if selection is not None:
        result['selected_excerpt_count'] = len(getattr(selection, 'excerpts', []))
    if isinstance(value.get('sources'), dict):
        documents = []
        for doc in list(value['sources'].values())[:100]:
            meta = getattr(doc, 'metadata', {})
            item = {'document_sha256': _digest(meta.get('document_id')),
                    'chunk_sha256': _digest(meta.get('chunk_id'))}
            for field in ('page_number', 'original_page_number', 'printed_catalog_page'):
                page = meta.get(field)
                if type(page) is int and page > 0:
                    item[field] = page
                elif isinstance(page, str) and page.isascii() and page.isdigit() and len(page) < 8:
                    item[field] = int(page)
            documents.append(item)
        result['documents'] = documents
        result['source_count'] = len(value['sources'])
    diag = value.get('selection_diagnostics')
    if isinstance(diag, dict) and diag.get('quote_validation') in {'passed', 'rejected', 'not_attempted'}:
        result['quote_validation'] = diag['quote_validation']
    return full_summary(value, result) if permit_full else result


@traceable(name='LabRAG stage', run_type='chain', process_inputs=lambda data: data,
           process_outputs=lambda data: data)
def _record(safe_input):
    return _operation.get()()


@traceable(name='LabRAG provider selection', run_type='llm',
           process_inputs=lambda data: data, process_outputs=lambda data: data)
def _record_llm(safe_input):
    return _operation.get()()


def observe(name, operation, state):
    config = _active.get()
    if config is None:
        with tracing_context(enabled=False):
            return operation()
    box = {}
    permit_full = not isinstance(state, dict) or not sensitive_sources(state)

    def execute():
        box['executed'] = True
        # Never let SDK automatic instrumentation see raw graph/provider data.
        with tracing_context(enabled=False):
            try:
                box['result'] = operation()
            except BaseException as exc:
                box['error'] = exc
        return {'execution': 'failed'} if 'error' in box else summary(box['result'], permit_full)

    token = _operation.set(execute)
    try:
        try:
            with tracing_context(enabled=True, client=config[0], project_name=config[1],
                                 metadata={'labrag_provider': config[2]}):
                recorder = _record_llm if name == 'generate_answer' else _record
                recorder(summary(state, permit_full), langsmith_extra={'name': name})
        except Exception:
            # Telemetry failure must not repeat a chargeable operation.
            if not box.get('executed'):
                execute()
    finally:
        _operation.reset(token)
    if 'error' in box:
        raise box['error']
    return box['result']


class TracedGraph:
    """Proxy preserves the compiled graph API; supported entry point is invoke."""
    def __init__(self, graph, settings):
        self.graph, self.settings = graph, settings

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def invoke(self, input, *args, **kwargs):
        config = None
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key.strip():
            try:
                config = (_client(self.settings.langsmith_api_key, self.settings.langsmith_endpoint), self.settings.langsmith_project or 'LabRAG',
                          self.settings.llm_provider,
                          self.settings.langsmith_capture_full_io and authorized_demo(input),
                          tuple(key for key in (self.settings.langsmith_api_key,self.settings.groq_api_key,
                              self.settings.openai_api_key,self.settings.pinecone_api_key) if key))
            except Exception:
                pass  # Optional telemetry must not break the RAG workflow.
        token = _active.set(config)
        try:
            return observe('LabRAG workflow', lambda: self.graph.invoke(input, *args, **kwargs), input)
        finally:
            _active.reset(token)
