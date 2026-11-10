from unittest.mock import Mock

from config.settings import Settings
from src.llm.groq_service import ExcerptIdSelection
from src.llm.provider_factory import build_backend
from src.observability import langsmith_tracing as tracing


def test_urls_loaded_from_dotenv(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('OPENAI_BASE_URL=https://openai.example/v1\n'
                    'LANGSMITH_ENDPOINT=https://tracing.example\n')
    monkeypatch.setattr('config.settings.ENV_FILE', path)
    monkeypatch.delenv('PYTHON_DOTENV_DISABLED', raising=False)
    settings = Settings.from_env()
    assert settings.openai_base_url == 'https://openai.example/v1'
    assert settings.langsmith_endpoint == 'https://tracing.example'


def test_empty_urls_keep_defaults(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('OPENAI_BASE_URL=\nLANGSMITH_ENDPOINT=\n')
    monkeypatch.setattr('config.settings.ENV_FILE', path)
    monkeypatch.delenv('PYTHON_DOTENV_DISABLED', raising=False)
    settings = Settings.from_env()
    assert settings.openai_base_url == Settings().openai_base_url
    assert settings.langsmith_endpoint == Settings().langsmith_endpoint


def test_openai_uses_configured_url(monkeypatch):
    backend = Mock()
    monkeypatch.setattr('langchain_openai.ChatOpenAI', backend)
    build_backend(Settings(openai_api_key='test-only',
                           openai_base_url='https://openai.example/v1'), ExcerptIdSelection)
    assert backend.call_args.kwargs['base_url'] == 'https://openai.example/v1'


def test_tracing_client_cache_includes_endpoint(monkeypatch):
    client = Mock()
    monkeypatch.setattr(tracing, 'Client', client)
    tracing._client.cache_clear()
    try:
        tracing._client('test-only', 'https://one.example')
        tracing._client('test-only', 'https://two.example')
        assert client.call_count == 2
        assert client.call_args.kwargs['api_url'] == 'https://two.example'
    finally:
        tracing._client.cache_clear()
