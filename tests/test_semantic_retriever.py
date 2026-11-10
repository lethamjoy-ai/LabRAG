from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from config.settings import Settings
from tests.day3_fakes import embedding_service, connection
from src.retrieval.semantic_retriever import SemanticRetriever


def retriever(results=None):
    conn = connection()
    conn.vector_store.return_value.similarity_search_by_vector_with_score.return_value = results or []
    return SemanticRetriever(Settings(), embedding_service(), conn), conn


def test_search_namespace_topk_filters_score_sources():
    doc = Document(page_content="Source passage", metadata={"source_url": "https://example.org/reference", "page_number": 2})
    service, conn = retriever([(doc, 0.85)])
    result = service.search("Specimen?", "fixed", 3, {"laboratory_provider": "Mayo"})
    assert result.matches[0]["document"] == doc and result.matches[0]["score"] == 0.85
    assert result.namespace == "fixed-chunks" and result.latency_seconds >= 0
    call = conn.vector_store.return_value.similarity_search_by_vector_with_score.call_args
    assert call.kwargs["k"] == 3
    assert {"laboratory_provider": {"$eq": "Mayo"}} in call.kwargs["filter"]["$and"]
    assert service.search("Unknown").matches[0]["document"].page_content == "Source passage"


def test_zero_results_and_comparison_shared_query_embedding():
    service, conn = retriever()
    assert service.search("Unknown").matches == []
    service.embeddings.backend.embed_query.reset_mock()
    results = service.compare_retrieval("Question", corpus="synthetic")
    assert set(results) == {"fixed", "section"}
    service.embeddings.backend.embed_query.assert_called_once()
    assert results["fixed"].namespace == "fixed-chunks-synthetic"


@pytest.mark.parametrize("kwargs", [{"query": ""}, {"query": "X", "top_k": 0},
    {"query": "X", "filters": {"text": {"$ne": "bad"}}}, {"query": "X", "strategy": "invalid"}])
def test_invalid_requests(kwargs):
    service, _ = retriever()
    with pytest.raises(ValueError):
        service.search(**kwargs)
