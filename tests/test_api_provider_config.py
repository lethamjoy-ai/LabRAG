import logging
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from src.api import create_app
from src.rag.service import LocalRAGService
from src.llm.provider_factory import OpenAIService, create_llm


def test_server_defaults_openai_and_keeps_embeddings_separate(monkeypatch):
    for name in ('LLM_PROVIDER','LLM_MODEL','OPENAI_LLM_MODEL'):
        monkeypatch.delenv(name,raising=False)
    settings=Settings.from_env()
    assert settings.llm_provider=='openai' and settings.llm_model=='gpt-4.1-mini'
    assert settings.embedding_provider=='local'
    assert settings.embedding_model=='sentence-transformers/all-MiniLM-L6-v2'


@pytest.mark.parametrize('provider,model', [('openai','gpt-4.1'),('groq','openai/gpt-oss-120b')])
def test_explicit_server_override(monkeypatch,provider,model):
    monkeypatch.setenv('LLM_PROVIDER',provider);monkeypatch.setenv('LLM_MODEL',model)
    settings=Settings.from_env()
    assert settings.llm_provider==provider and settings.llm_model==model


def test_provider_specific_model_fallback(monkeypatch):
    monkeypatch.delenv('LLM_MODEL',raising=False)
    monkeypatch.setenv('LLM_PROVIDER','groq')
    assert Settings.from_env().llm_model=='openai/gpt-oss-120b'
    monkeypatch.setenv('LLM_PROVIDER','openai');monkeypatch.setenv('OPENAI_LLM_MODEL','gpt-4.1')
    assert Settings.from_env().llm_model=='gpt-4.1'


def test_api_missing_openai_key_fails_before_pinecone(monkeypatch):
    monkeypatch.setattr(Settings,'from_env',classmethod(lambda cls:Settings(openai_api_key='')))
    forbidden=Mock(side_effect=AssertionError('No network initialization'))
    monkeypatch.setattr('src.vectorstore.pinecone_client.PineconeConnection.connect',forbidden)
    response=TestClient(create_app(LocalRAGService())).post('/api/v1/query',json={'question':'What specimen is required for BUN?'})
    assert response.status_code==503 and response.json()['error']=='configuration_error'
    assert 'OPENAI_API_KEY' in response.json()['message']
    forbidden.assert_not_called()


@pytest.mark.parametrize('provider', ['openai','groq'])
def test_lab_agent_request_uses_server_settings_and_preserves_citations(monkeypatch,caplog,provider):
    settings=Settings(llm_provider=provider,llm_model='gpt-4.1-mini' if provider=='openai' else 'openai/gpt-oss-120b',
                      openai_api_key='secret-canary',groq_api_key='groq-canary')
    monkeypatch.setattr(Settings,'from_env',classmethod(lambda cls:settings))
    monkeypatch.setattr('src.vectorstore.pinecone_client.PineconeConnection.connect',Mock(return_value=Mock()))
    monkeypatch.setattr('src.embeddings.embedding_service.get_configured_embedding_service',Mock(return_value=Mock()))
    factory=Mock(wraps=create_llm)
    monkeypatch.setattr('src.llm.provider_factory.create_llm',factory)
    def answer(question,history,retriever,llm,actual):
        assert actual is settings and history==[]
        assert isinstance(llm,OpenAIService)==(provider=='openai')
        return {'status':'answered','answer':'Serum. [S1]',
                'citations':[{'source_id':'S1','test_code':'BUN','document_title':'Mayo catalog'}],
                'metrics':{'retrieve_evidence_seconds':.25,'generate_answer_seconds':.10}}
    generate=Mock(side_effect=answer);monkeypatch.setattr('src.rag.service.answer_question',generate)
    api=TestClient(create_app(LocalRAGService()))
    caplog.set_level(logging.INFO,logger='uvicorn.error')
    response=api.post('/api/v1/query',json={'question':'What specimen is required for BUN?'})
    assert response.status_code==200 and response.json()['status']=='answered'
    assert response.json()['citations'][0]['source_id']=='S1'
    assert response.headers['X-LabRAG-Provider']==provider
    assert response.headers['X-LabRAG-Model']==settings.llm_model
    assert float(response.headers['X-LabRAG-Retrieval-Seconds'])==.25
    assert float(response.headers['X-LabRAG-Generation-Seconds'])==.10
    assert float(response.headers['X-LabRAG-API-Seconds'])>=0
    for secret in ('What specimen','secret-canary','groq-canary'):
        assert secret not in caplog.text
    assert provider in caplog.text and 'retrieval_seconds' in caplog.text
    assert api.post('/api/v1/query',json={'question':'BUN?','provider':'groq'}).status_code==422
    factory.assert_called_once();generate.assert_called_once()
