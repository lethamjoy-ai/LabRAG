from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from config.settings import Settings
from src.embeddings.openai_service import OpenAIEmbeddingService
from src.embeddings.embedding_service import EmbeddingError
from src.vectorstore.pinecone_client import namespace_for


def fake_client():
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[0.1 + i / 100] * kwargs["dimensions"])
        for i in reversed(range(len(kwargs["input"])))])
    return client


def test_explicit_model_dimensions_batch_order_cache_and_query():
    client = fake_client()
    service = OpenAIEmbeddingService(batch_size=2, client=client)
    result = service.embed_documents(["Alpha", "Beta", "Gamma"])
    assert len(result) == 3 and all(len(v) == 384 for v in result)
    assert result[0][0] == 0.1 and result[1][0] == 0.11
    assert client.embeddings.create.call_count == 2
    call = client.embeddings.create.call_args_list[0].kwargs
    assert call["dimensions"] == 384 and call["model"] == "text-embedding-3-small"
    assert service.embed_query("Alpha") == result[0]
    assert client.embeddings.create.call_count == 2


def test_larger_than_minilm_text_allowed_and_long_input_rejected():
    service = OpenAIEmbeddingService(client=fake_client())
    assert len(service.embed_query("clinical reference " * 300)) == 384
    with pytest.raises(EmbeddingError, match="8192"):
        service.validate_text("reference " * 9000)
    assert service.embed_documents([]) == []
    with pytest.raises(EmbeddingError):
        service.embed_query("")


def test_missing_key_and_api_errors_redacted():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingService().embed_query("Question")
    client = fake_client()
    client.embeddings.create.side_effect = RuntimeError("sensitive-test-secret")
    with pytest.raises(EmbeddingError) as error:
        OpenAIEmbeddingService(client=client).embed_query("Question")
    assert "sensitive-test-secret" not in str(error.value)


def test_invalid_vector_or_response_indices_rejected():
    client = fake_client()
    for data in ([SimpleNamespace(index=0, embedding=[float("nan")] * 384)],
                 [SimpleNamespace(index=0, embedding=[0.1] * 383)],
                 [SimpleNamespace(index=1, embedding=[0.1] * 384)]):
        client.embeddings.create.side_effect = None
        client.embeddings.create.return_value = SimpleNamespace(data=data)
        with pytest.raises(EmbeddingError):
            OpenAIEmbeddingService(client=client).embed_query("Question")


def test_configuration_and_namespace_separation(monkeypatch, tmp_path):
    configured = replace(Settings(), embedding_provider="openai", embedding_model="text-embedding-3-small", openai_api_key="synthetic-secret")
    assert "synthetic-secret" not in repr(configured)
    assert namespace_for(configured, "fixed") == "fixed-chunks-openai-text-embedding-3-small-384"
    assert namespace_for(configured, "section", "synthetic").endswith("384-synthetic")
    assert namespace_for(configured, "fixed") != namespace_for(Settings(), "fixed")
    for kwargs in ({"embedding_model": "text-embedding-ada-002"}, {"embedding_dimension": 2000}):
        with pytest.raises(ValueError):
            replace(configured, **kwargs)
