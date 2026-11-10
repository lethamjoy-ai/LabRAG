from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from config.settings import Settings
from src.vectorstore.pinecone_client import PineconeConnection, PineconeError, namespace_for, retry


def client():
    fake = Mock()
    fake.list_indexes.return_value.names.return_value = ["lab-rag"]
    fake.describe_index.return_value = SimpleNamespace(dimension=384, metric="cosine", status={"ready": True}, host="example")
    return fake


def test_connection_validation_and_no_automatic_creation():
    fake = client()
    connection = PineconeConnection(Settings(pinecone_api_key="testing"), fake)
    fake.list_indexes.assert_not_called()
    assert connection.connect().index is fake.Index.return_value
    fake.create_index.assert_not_called()
    fake.delete_index.assert_not_called()


def test_missing_key_and_index():
    with pytest.raises(PineconeError, match="PINECONE_API_KEY"):
        PineconeConnection(Settings(), client()).connect()
    fake = client()
    fake.list_indexes.return_value.names.return_value = []
    with pytest.raises(PineconeError, match="does not exist"):
        PineconeConnection(Settings(pinecone_api_key="testing"), fake).connect()


@pytest.mark.parametrize("change", [{"dimension": 768}, {"metric": "euclidean"}, {"status": {"ready": False}}])
def test_incompatible_index(change):
    fake = client()
    for key, value in change.items():
        setattr(fake.describe_index.return_value, key, value)
    with pytest.raises(PineconeError):
        PineconeConnection(Settings(pinecone_api_key="testing"), fake).connect()


def test_retry_namespace_and_error_redaction():
    operation = Mock(side_effect=[TimeoutError("sensitive"), "ok"])
    sleep = Mock()
    assert retry(operation, sleep=sleep) == "ok"
    sleep.assert_called_once_with(1)
    with pytest.raises(PineconeError) as error:
        retry(Mock(side_effect=ValueError("sensitive")), sleep=sleep)
    assert "sensitive" not in str(error.value)
    assert namespace_for(Settings(), "fixed") == "fixed-chunks"
    assert namespace_for(Settings(), "section", "synthetic") == "section-chunks-synthetic"


def test_day3_settings_validation():
    for changes in ({"namespace_fixed": "section-chunks"}, {"retrieval_score_threshold": float("nan")},
                    {"embedding_batch_size": 0}, {"pinecone_upsert_batch_size": 1001}, {"retrieval_top_k": 101}):
        with pytest.raises(ValueError):
            replace(Settings(), **changes)


def test_actual_langchain_adapter_reconstructs_metadata_without_network():
    from tests.day3_fakes import embedding_service
    conn = PineconeConnection(Settings(pinecone_api_key="testing"), client())
    conn.index = Mock()
    conn.index.config = SimpleNamespace(host="https://example.invalid", api_key="testing")
    conn.index.query.return_value = {"matches": [{"id": "vector-1", "score": 0.9,
        "metadata": {"text": "Returned source passage", "chunk_id": "chunk-1", "source_url": "https://example.org"}}]}
    store = conn.vector_store(embedding_service(), "section-chunks")
    result = store.similarity_search_by_vector_with_score([0.1] * 384, k=1)
    assert result[0][0].page_content == "Returned source passage"
    assert result[0][0].metadata["chunk_id"] == "chunk-1"
    assert result[0][1] == 0.9
    assert conn.index.query.call_args.kwargs["namespace"] == "section-chunks"
