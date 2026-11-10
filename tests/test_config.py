import pytest
from labrag.config import Settings, ConfigurationError


def test_missing_keys_actionable(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        Settings.from_env()


def test_secrets_redacted():
    assert "secret" not in repr(Settings(pinecone_api_key="secret", llm_api_key="secret"))
