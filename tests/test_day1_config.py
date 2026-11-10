import pytest
from config.settings import Settings


def test_defaults_no_keys():
    settings = Settings()
    assert settings.embedding_dimension == 384
    assert settings.pinecone_index_name == "lab-rag"
    assert len(settings.warnings) == 2


@pytest.mark.parametrize("dimension", [0, -1, 768])
def test_invalid_dimensions(dimension):
    with pytest.raises(ValueError):
        Settings(embedding_dimension=dimension)


def test_secrets_and_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSION", "invalid")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()
    assert "sensitive" not in repr(Settings(groq_api_key="sensitive"))
