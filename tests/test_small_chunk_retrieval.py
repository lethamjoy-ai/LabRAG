"""Regressions for a heading and its instructions indexed as separate chunks."""
import json
import pytest
from unittest.mock import Mock
from langchain_core.documents import Document
from config.settings import Settings
from src.chunking.section_chunker import chunk_document_by_section
from src.llm.groq_service import AnswerSelection
from src.rag.rag_graph import build_rag_graph
from src.rag.evidence_validator import identity_query
from src.retrieval.semantic_retriever import SemanticRetriever
from tests.day3_fakes import embedding_service, connection


def panel_chunks(laboratory_document):
    pages = [
        "CMAMA Comprehensive Metabolic Panel, Serum\n113631\n"
        "Specimen Requirements:\nCollection Container/Tube:\nPreferred: Serum gel\n"
        "Acceptable: Red top\nSubmission Container/Tube: Amber vial\nSpecimen Volume: 0.6 mL\n"
        "Collection Instructions:\nSerum gel tubes should be centrifuged and protected from light within 2 hours of collection.\nPage 926",
        "Specimen Minimum Volume: 0.5 mL\nTransport Temperature:\n"
        "Specimen Type        Temperature        Time\nSerum        Refrigerated        24 hours\n"
        "NEXTT Next Test, Plasma\n12345\nSpecimen Requirements:\nPlasma required.\nPage 927",
    ]
    return chunk_document_by_section([
        laboratory_document(text, test_name="", test_code="", page_number=427 + i)
        for i, text in enumerate(pages)], max_tokens=100, overlap=10)


def test_exact_panel_question_recovers_requirements_and_continuation(laboratory_document):
    chunks = panel_chunks(laboratory_document)
    heading = next(doc for doc in chunks if "CMAMA" in doc.page_content)
    distractors = [Document(page_content="Unrelated specimen instructions", metadata={
        "document_id": "other", "page_number": p, "chunk_id": f"other-{p}"}) for p in [10, 20, 30]]
    conn = connection()
    calls = []

    def search(vector, k, filter):
        calls.append(filter)
        clauses = filter["$and"]
        page = next((clause["page_number"]["$eq"] for clause in clauses if "page_number" in clause), None)
        if page is not None:
            return [(doc, .6) for doc in chunks if doc.metadata["page_number"] == page]
        # Heading is below the original three-page expansion limit.
        if len([c for c in calls if not any("page_number" in x for x in c["$and"])]) == 1:
            return [(doc, .9) for doc in [*distractors, heading]]
        return [(heading, .85)]

    conn.vector_store.return_value.similarity_search_by_vector_with_score.side_effect = search
    retriever = SemanticRetriever(Settings(), embedding_service(), conn)
    backend = Mock()

    def select(messages):
        evidence = json.loads(messages[-1].content)["retrieved_evidence"]
        excerpts = []
        for quote in ["Preferred: Serum gel", "Acceptable: Red top", "Specimen Minimum Volume: 0.5 mL"]:
            source = next(item for item in evidence if quote in item["text"])
            excerpts.append({"source_id": source["source_id"], "quote": quote})
        assert all("Plasma required" not in item["text"] for item in evidence)
        return {"excerpts": excerpts}

    backend.invoke.side_effect = select
    generator = Mock()
    generator.select.side_effect = lambda messages: AnswerSelection.model_validate(backend.invoke(messages))
    graph = build_rag_graph(retriever, generator, Settings())
    result = graph.invoke({"question": "What Specimen type is required for Comprehensive metabolic panel ?"})
    assert result["status"] == "answered"
    assert result["retrieval_details"]["identity_query"] == "comprehensive metabolic panel"
    assert "0.5 mL" in result["answer"]
    assert {c["test_code"] for c in result["citations"]} == {"CMAMA"}
    assert any(c["printed_catalog_page"] == "927" for c in result["citations"])
    assert all(call.args[1] == "section-chunks" for call in conn.vector_store.call_args_list)
    conn.index.upsert.assert_not_called()
    conn.index.delete.assert_not_called()


def test_identity_query_removes_instruction_words_without_removing_test_name():
    assert identity_query("What specimen type is required for Comprehensive metabolic panel?") == "comprehensive metabolic panel"
    assert identity_query("What are the storage conditions for CMAMA?") == "CMAMA"
    assert identity_query("What is the minimum sample volume for Calcium, Ionized, Serum?") == "calcium ionized serum"
    assert identity_query("What is the Submission container for Comprehensive Metabolic Panel, Serum ?") == "comprehensive metabolic panel serum"


@pytest.mark.parametrize("question", [
    "What is the Submission container for Comprehensive Metabolic Panel, Serum ?",
    "What is the submission container for Comprehensive Metabolic Panel?",
    "Explain the submission receptacle for Comprehensive Metabolic Panel, Serum?",
])
def test_submission_container_identity_and_answer(question):
    from src.rag.evidence_validator import scope_sources
    from src.retrieval.semantic_retriever import SearchResult
    doc = Document(page_content="Collection Container/Tube: Serum gel\nSubmission Container/Tube: Amber vial", metadata={
        "test_name": "Comprehensive Metabolic Panel, Serum", "test_code": "CMAMA", "chunk_id": "submission"})
    scoped, _ = scope_sources(question, {"S1": doc})
    assert scoped
    retriever = Mock()
    retriever.search.return_value = SearchResult(question, "section", "section-chunks", [{"document": doc, "score": .7}], .01)
    backend = Mock()
    backend.invoke.return_value = {"excerpts": [{"source_id": "S1", "quote": "Submission Container/Tube: Amber vial"}]}
    generator = Mock()
    generator.select.side_effect = lambda messages: AnswerSelection.model_validate(backend.invoke(messages))
    result = build_rag_graph(retriever, generator, Settings()).invoke({"question": question})
    assert result["status"] == "answered"
    assert "Submission Container/Tube: Amber vial" in result["answer"]
    assert "Serum gel" not in result["answer"]


def test_explorer_filters_heading_and_other_test_preserving_corpus_and_filters(laboratory_document):
    chunks = panel_chunks(laboratory_document)
    heading = next(doc for doc in chunks if "CMAMA" in doc.page_content)
    conn = connection()
    requested_filters = []

    def search(vector, k, filter):
        requested_filters.append(filter["$and"])
        page = next((clause["page_number"]["$eq"] for clause in filter["$and"] if "page_number" in clause), None)
        return [(d, .6 if "CMAMA" not in d.page_content else .99) for d in chunks
                if d.metadata["page_number"] == page] if page else [(heading, .99)]

    conn.vector_store.return_value.similarity_search_by_vector_with_score.side_effect = search
    retriever = SemanticRetriever(Settings(), embedding_service(), conn)
    result = retriever.search_with_context(
        "What Specimen type is required for Comprehensive metabolic panel ?", top_k=5,
        filters={"laboratory_provider": "Fictional laboratory"}, corpus="synthetic")
    assert result.matches and len(result.matches) <= 5
    assert all("CMAMA" not in m["document"].page_content for m in result.matches)
    assert all(m["document"].metadata["test_code"] == "CMAMA" for m in result.matches)
    assert any("Preferred: Serum gel" in m["document"].page_content for m in result.matches)
    assert all({"laboratory_provider": {"$eq": "Fictional laboratory"}} in clauses for clauses in requested_filters)
    assert all({"corpus_type": {"$eq": "synthetic"}} in clauses for clauses in requested_filters)
    assert all(call.args[1] == "section-chunks-synthetic" for call in conn.vector_store.call_args_list)
