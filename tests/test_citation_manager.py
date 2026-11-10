from langchain_core.documents import Document
from src.rag.citation_manager import citation, validate_selection, restore_source_whitespace
from src.llm.groq_service import AnswerSelection


def test_pages_do_not_conflate_and_unknown_is_explicit():
    doc = Document(page_content="Frozen 365 days.\nPage 501", metadata={"page_number": 2, "original_page_number": 502})
    result = citation("S1", doc)
    assert result["printed_catalog_page"] == "501"
    assert result["pdf_page_1_based"] == 2
    assert result["original_pdf_page_1_based"] == 502
    assert citation("S2", Document(page_content="No footer"))["printed_catalog_page"] == "Unknown"


def test_fabricated_ids_and_quotes_fail():
    sources = {"S1": Document(page_content="Frozen 365 days.")}
    for source, quote in [("S2", "Frozen 365 days."), ("S1", "Frozen 7 days.")]:
        assert not validate_selection(AnswerSelection(excerpts=[{"source_id": source, "quote": quote}]), sources)


def test_pdf_indentation_is_restored_without_changing_clinical_tokens():
    source = "Preferred: Serum gel\n\n        Acceptable: Red top\nVolume: 0.6 mL"
    sources = {"S1": Document(page_content=source)}
    selection = AnswerSelection(excerpts=[{"source_id": "S1", "quote": "Preferred: Serum gel\nAcceptable: Red top"}])
    restored = restore_source_whitespace(selection, sources)
    assert validate_selection(restored, sources)
    assert restored.excerpts[0].quote in source
    assert restored.excerpts[0].quote != selection.excerpts[0].quote
    for unsupported in ["Volume: 0.5 mL", "Volume: 0.6 mg", "Preferred: Red top", "Acceptable: red top"]:
        response = AnswerSelection(excerpts=[{"source_id": "S1", "quote": unsupported}])
        assert not validate_selection(restore_source_whitespace(response, sources), sources)
