import pytest
from tests.day3_fakes import embedding_service
from src.embeddings.embedding_service import EmbeddingError


def test_batch_embeddings_dimension_cache_and_query():
    service = embedding_service()
    vectors = service.embed_documents(["Alpha", "Beta", "Gamma", "Alpha"])
    assert len(vectors) == 4 and all(len(v) == 384 for v in vectors)
    assert service.backend.embed_documents.call_count == 2
    assert service.backend.embed_documents.call_args_list[1].args[0] == ["Gamma"]
    service.embed_documents(["Alpha"])
    assert service.backend.embed_documents.call_count == 2
    assert len(service.embed_query("Question")) == 384


def test_empty_invalid_and_truncation_blocked():
    service = embedding_service()
    assert service.embed_documents([]) == []
    for value in (None, "", " "):
        with pytest.raises(EmbeddingError):
            service.embed_query(value)
    with pytest.raises(EmbeddingError, match="not truncated"):
        service.embed_documents(["word " * 300])


@pytest.mark.parametrize("vector", [[0.0] * 383, [float("nan")] * 384, [float("inf")] * 384, [True] * 384])
def test_invalid_vectors(vector):
    with pytest.raises(EmbeddingError):
        embedding_service().validate_vector(vector)
