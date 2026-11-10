from unittest.mock import Mock
from langchain_core.documents import Document
from config.settings import Settings
from src.retrieval.semantic_retriever import SearchResult
from src.llm.groq_service import AnswerSelection
from src.rag.rag_graph import build_rag_graph


def graph_with(text):
    retriever, backend = Mock(), Mock()
    matches = [] if text is None else [{"document": Document(page_content=text), "score": .9}]
    retriever.search.return_value = SearchResult("q", "section", "section-chunks", matches, .01)
    # Legacy generators still exercise strict downstream citation validation.
    generator = Mock()
    generator.select.side_effect = lambda messages: AnswerSelection.model_validate(backend.invoke(messages))
    return build_rag_graph(retriever, generator, Settings()), retriever, backend


def test_no_evidence_skips_llm_and_traces_actual_path():
    graph, retriever, backend = graph_with(None)
    result = graph.invoke({"question": "What specimen for BUN?"})
    assert result["status"] == "abstained" and result["citations"] == []
    backend.invoke.assert_not_called()
    assert "abstain" in [n["node"] for n in result["trace"]]


def test_answer_and_fresh_followup_retrieval():
    graph, retriever, backend = graph_with("BUN Serum specimen. Frozen 365 days.")
    backend.invoke.return_value = {"excerpts": [{"source_id": "S1", "quote": "Frozen 365 days."}]}
    result = graph.invoke({"question": "How long can BUN be frozen?"})
    assert result["status"] == "answered" and "365" in result["answer"]
    graph.invoke({"question": "How long can it be frozen?", "history": [{"question": "What specimen for BUN?"}]})
    assert retriever.search.call_count == 2
    assert "BUN" in retriever.search.call_args.args[0]


def test_unsupported_generated_quote_abstains():
    graph, _, backend = graph_with("BUN Serum specimen. Frozen 365 days.")
    backend.invoke.return_value = {"excerpts": [{"source_id": "S1", "quote": "Frozen 100 days."}]}
    assert graph.invoke({"question": "How long can BUN be frozen?"})["status"] == "abstained"
