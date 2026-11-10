from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import create_app
from src.llm.groq_service import LLMError
from src.vectorstore.pinecone_client import PineconeError


def client(result=None, error=None):
    service = Mock()
    service.query.return_value = result
    service.query.side_effect = error
    return TestClient(create_app(service)), service


def test_health_does_not_initialize_providers():
    api, service = client()
    assert api.get('/health').json() == {'status': 'ok', 'external_connectivity': 'not_checked'}
    service.query.assert_not_called()


def test_answer_and_citations():
    result = {'status': 'answered', 'answer': 'Serum. [S1]', 'display_answer': 'Serum.',
        'citations': [{'source_id': 'S1', 'document_title': 'Mayo', 'test_code': 'CA',
            'test_name': 'Calcium, Total, Serum', 'printed_catalog_page': 590,
            'pdf_page_1_based': 12, 'source_url': 'https://example.org/catalog'}]}
    api, service = client(result)
    response = api.post('/api/v1/query', json={'question': ' What specimen for CA? '})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'Serum.'
    assert body['citations'][0]['printed_catalog_page'] == '590'
    assert body['citations'][0]['pdf_page_1_based'] == '12'
    assert body['citations'][0]['chunk_id'] is None
    assert body['matched_tests'] == [{'test_code': 'CA', 'test_name': 'Calcium, Total, Serum'}]
    assert body['request_id'] == response.headers['X-Request-ID']
    service.query.assert_called_once_with('What specimen for CA?')


@pytest.mark.parametrize('payload', [{}, {'question': ''}, {'question': '  '},
    {'question': 'x'*3001}, {'question': 'CA?', 'api_key': 'secret'}])
def test_invalid_questions_do_not_call_pipeline(payload):
    api, service = client()
    response = api.post('/api/v1/query', json=payload)
    assert response.status_code == 422
    assert 'secret' not in response.text
    service.query.assert_not_called()


@pytest.mark.parametrize('status,expected', [('abstained', 'insufficient_evidence'),
    ('needs_clarification', 'needs_clarification')])
def test_uncertainty_preserved(status, expected):
    api, _ = client({'status': status, 'answer': 'Please specify the test.', 'sources': {
        'S1': Document(page_content='', metadata={'test_code': 'Unresolved in indexed metadata'})}})
    body = api.post('/api/v1/query', json={'question': 'Specimen?'}).json()
    assert body['status'] == expected
    assert body['matched_tests'] == []
    assert body['citations'] == []


@pytest.mark.parametrize('error,code', [(PineconeError('secret'), 503),
    (LLMError('secret'), 503), (TimeoutError('secret'), 504),
    (LLMError('secret', {'http_status': 429}), 503),
    (LLMError('secret', {'http_status': 504}), 504)])
def test_errors_are_sanitized(error, code):
    api, _ = client(error=error)
    response = api.post('/api/v1/query', json={'question': 'Specimen for CA?'})
    assert response.status_code == code
    assert 'secret' not in response.text
    assert response.json()['request_id']


def test_shared_service_uses_existing_graph(monkeypatch):
    from src.rag.service import answer_question
    graph = Mock()
    graph.invoke.return_value = {'answer': 'verified'}
    factory = Mock(return_value=graph)
    monkeypatch.setattr('src.rag.rag_graph.build_rag_graph', factory)
    assert answer_question('CA?', [], 'retriever', 'llm', 'settings')['answer'] == 'verified'
    graph.invoke.assert_called_once_with({'question': 'CA?', 'history': []})
    assert factory.call_args.kwargs['answer_generation_mode'] == 'grounded_conversational_v2'
