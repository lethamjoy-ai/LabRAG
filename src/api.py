"""Local API adapter. Import and liveness checks make no external calls."""
from typing import Literal
from uuid import uuid4
import json
import logging
import time
import re
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.rag.service import LocalRAGService

logger=logging.getLogger('uvicorn.error')


def safe_execution(value):
    """Whitelist telemetry fields; never serialize result/state/exception bodies."""
    result={}
    if value.get('provider') in {'openai','groq'}:result['provider']=value['provider']
    model=value.get('model')
    if isinstance(model,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:-]{0,120}',model) and not re.match(r'sk-|gsk_|lsv2_|pcsk_',model,re.I):
        result['model']=model
    for key in ('retrieval_seconds','generation_seconds'):
        item=value.get(key)
        if type(item) in (int,float) and math.isfinite(item) and item>=0:result[key]=item
    if type(value.get('generation_executed')) is bool:result['generation_executed']=value['generation_executed']
    return result


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=3000)

    @field_validator('question')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Enter a nonempty question.')
        return value.strip()


class Citation(BaseModel):
    source_id: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    provider: str | None = None
    test_code: str | None = None
    test_name: str | None = None
    printed_catalog_page: str | None = None
    pdf_page_1_based: str | None = None
    original_pdf_page_1_based: str | None = None
    section: str | None = None
    chunk_id: str | None = None
    source_url: str | None = None
    citation_role: str | None = None


class MatchedTest(BaseModel):
    test_code: str | None = None
    test_name: str | None = None


class QueryResponse(BaseModel):
    status: Literal['answered', 'needs_clarification', 'insufficient_evidence']
    answer: str
    citations: list[Citation]
    matched_tests: list[MatchedTest]
    request_id: str


def known(value):
    if value is None or str(value).strip() in {'', 'Unknown', 'Unresolved in indexed metadata'}:
        return None
    return str(value)


def adapt_result(result, request_id):
    # Only an explicit pipeline clarification can be labeled as such. Identity
    # uncertainty otherwise retains the graph's conservative abstention.
    status = result.get('status')
    status = status if status in {'answered', 'needs_clarification'} else 'insufficient_evidence'
    citations = [Citation(**{key: known(source.get(key)) for key in Citation.model_fields})
                 for source in result.get('citations', [])]
    tests = []
    seen = set()
    identities = [(c.test_code, c.test_name) for c in citations]
    identities += [(known(d.metadata.get('test_code')), known(d.metadata.get('test_name')))
                   for d in result.get('sources', {}).values()]
    for code, name in identities:
        if (code or name) and (code, name) not in seen:
            tests.append(MatchedTest(test_code=code, test_name=name))
            seen.add((code, name))
    return QueryResponse(status=status, answer=result.get('display_answer') or result['answer'],
                         citations=citations, matched_tests=tests, request_id=request_id)


def create_app(service=None):
    app = FastAPI(title='LabRAG Local API', version='1.0.0')
    app.state.rag_service = service if service is not None else LocalRAGService()

    @app.middleware('http')
    async def request_identifier(request: Request, call_next):
        started=time.perf_counter()
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        if request.url.path=='/api/v1/query':
            elapsed=time.perf_counter()-started
            diagnostics=safe_execution(getattr(request.state,'execution',{}))
            response.headers['X-LabRAG-API-Seconds']=f'{elapsed:.6f}'
            for field,header in [('provider','X-LabRAG-Provider'),('model','X-LabRAG-Model'),
                ('retrieval_seconds','X-LabRAG-Retrieval-Seconds'),('generation_seconds','X-LabRAG-Generation-Seconds')]:
                value=diagnostics.get(field)
                if isinstance(value,(str,int,float)):
                    response.headers[header]=str(value)
            if 'generation_executed' in diagnostics:
                response.headers['X-LabRAG-Generation-Executed']=str(diagnostics['generation_executed']).lower()
            logger.info('labrag_request %s',json.dumps({'request_id':request.state.request_id,
                'http_status':response.status_code,'api_seconds':elapsed,**diagnostics},ensure_ascii=True))
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # FastAPI's default validation body echoes user input; withhold it.
        return JSONResponse(status_code=422, content={'error': 'invalid_request',
            'message': 'Supply only question: a nonempty string of at most 3000 characters.',
            'request_id': request.state.request_id})

    @app.get('/health')
    def health():
        return {'status': 'ok', 'external_connectivity': 'not_checked'}

    @app.post('/api/v1/query', response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request):
        # Sync route runs in Starlette's worker pool, never on the event loop.
        try:
            result=app.state.rag_service.query(payload.question)
            metrics=result.get('metrics',{})
            request.state.execution={k:v for k,v in {'provider':result.get('llm_provider'),
                'model':result.get('llm_model'),'retrieval_seconds':metrics.get('retrieve_evidence_seconds'),
                'generation_seconds':metrics.get('generate_answer_seconds',0),
                'generation_executed':'generate_answer_seconds' in metrics}.items() if v is not None}
            return adapt_result(result, request.state.request_id)
        except Exception as exc:
            from src.llm.groq_service import LLMError
            diagnostic = exc.error_diagnostics if isinstance(exc, LLMError) else {}
            status = diagnostic.get('http_status')
            category = diagnostic.get('category', diagnostic.get('provider_error_category'))
            runtime=getattr(app.state.rag_service,'runtime',None)
            request.state.execution={k:runtime[k] for k in ('provider','model') if isinstance(runtime,dict) and isinstance(runtime.get(k),str)}
            if isinstance(exc, TimeoutError) or status in {408, 504} or category in {'timeout', 'request_timeout'}:
                code, error = 504, 'upstream_timeout'
            elif status == 429:
                code, error = 503, 'provider_rate_limited'
            else:
                code, error = 503, 'service_unavailable'
            message='Laboratory query could not complete. Check server configuration or retry later.'
            if category=='missing_credentials' and diagnostic.get('provider') in {'openai','groq'}:
                error='configuration_error'
                message='Set '+('OPENAI_API_KEY' if diagnostic['provider']=='openai' else 'GROQ_API_KEY')+' in the API server .env and restart the API.'
            return JSONResponse(status_code=code, content={'error': error,
                'message': message,
                'request_id': request.state.request_id})

    return app


app = create_app()
