from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from labrag.graph import build_graph, ABSTENTION, valid_citations
from labrag.evaluation import evaluate


class Store:
    def search(self, question, **kwargs):
        return [Document(page_content="Demo container is purple.", metadata={"source": "demo", "page": 1})]


class Ranker:
    def rank(self, question, documents):
        return documents


def test_graph_generates_and_checks_labels():
    graph = build_graph(Store(), Ranker(), FakeListChatModel(responses=["Purple [1]."]))
    result = graph.invoke({"question": "Container?"})
    assert result["citation_valid"]
    assert len(result["sources"]) == 1
    frame = evaluate(graph, [{"question": "Container?", "expected_sources": ["demo"], "expected_terms": ["purple"]}])
    assert frame.iloc[0]["recall_at_4"] == 1
    assert frame.iloc[0]["term_coverage"] == 1


def test_invalid_citation_fails_closed():
    graph = build_graph(Store(), Ranker(), FakeListChatModel(responses=["Purple [9]."]))
    assert graph.invoke({"question": "Container?"})["answer"] == ABSTENTION
    assert not valid_citations("Unsupported assertion", 1)


def test_empty_retrieval_skips_llm():
    class EmptyStore:
        def search(self, *args, **kwargs):
            return []
    graph = build_graph(EmptyStore(), Ranker(), FakeListChatModel(responses=[]))
    assert graph.invoke({"question": "Unknown?"})["answer"] == ABSTENTION
