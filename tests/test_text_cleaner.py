import pytest
from src.ingestion.text_cleaner import clean_laboratory_document, clean_documents
from src.ingestion.document_loader import load_document


def test_units_values_ranges_and_rows_survive():
    text = "Specimen Stability\r\n\r\nAnalyte | Range | Unit\nAlpha | -2.5–4.7 | mmol/L\nBeta | ≥0.25 | µmol/L\nStore | 2–8 °C | 24 hours\n"
    cleaned = clean_laboratory_document(text)
    for phrase in ["-2.5–4.7", "mmol/L", "≥0.25", "µmol/L", "2–8 °C", "24 hours"]:
        assert phrase in cleaned
    assert "mmol/L\nBeta" in cleaned


def test_html_rows_and_boilerplate():
    text = "<nav>Menu</nav><h1>Reference Values</h1><table><tr><th>Unit</th><th>Value</th></tr><tr><td>mg/dL</td><td>1.25</td></tr></table><script>bad()</script>"
    cleaned = clean_laboratory_document(text)
    assert "Menu" not in cleaned and "bad()" not in cleaned
    assert "Unit | Value\nmg/dL | 1.25" in cleaned


def test_safe_empty_invalid_and_idempotent():
    assert clean_laboratory_document(None) == ""
    assert clean_laboratory_document("  \n") == ""
    with pytest.raises(ValueError):
        clean_laboratory_document(123)
    value = "Heading\nAlpha  Beta\n0.25 mg/dL"
    assert clean_laboratory_document(clean_laboratory_document(value)) == value


def test_confident_repeated_footer_only(laboratory_document):
    pages = [laboratory_document(f"Rejection Criteria\nDo not discard this repeated clinical heading.\nPage {i} of 3", page_number=i) for i in range(1, 4)]
    result = clean_documents(pages)
    assert all("Page " not in d.page_content for d in result)
    assert all("Rejection Criteria" in d.page_content for d in result)
    assert "Page 1" in result[0].metadata["original_text"]
    assert "Page 1 of 1" in clean_laboratory_document("Test\nPage 1 of 1")


def test_loader_keeps_original_extraction_and_table_spaces():
    doc = load_document(b"Title\n\n\nValue  Unit\n1.25  mg/L", "demo.txt")[0]
    assert "\n\n\n" in doc.metadata["original_text"]
    assert "Value  Unit" in doc.page_content
