import pytest
from src.ingestion.document_loader import load_document


def test_text_and_html():
    doc = load_document(b"Heading\n\nValue -1.2 mg/L", "ref.txt")[0]
    assert "-1.2 mg/L" in doc.page_content
    assert len(doc.metadata["content_hash"]) == 64
    html = load_document(b"<nav>noise</nav><h1>Title</h1><script>bad()</script><p>Content</p>", "ref.html")[0]
    assert "noise" not in html.page_content and "bad" not in html.page_content
    assert html.metadata["section_heading"] == "Title"


def test_pdf_pages():
    import pymupdf
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Synthetic page 1")
        pdf.new_page().insert_text((72, 72), "Synthetic page 2")
        docs = load_document(pdf.tobytes(), "demo.pdf")
    assert [d.metadata["page_number"] for d in docs] == [1, 2]


@pytest.mark.parametrize("data,name", [(b"", "a.txt"), (b"hello", "a.csv"), (b"broken", "a.pdf"),
                                         (b"%PDF-broken", "a.pdf"), (b"\xff", "a.txt"), (b"   ", "a.txt")])
def test_invalid_files(data, name):
    with pytest.raises(ValueError):
        load_document(data, name)


def test_size_limit():
    with pytest.raises(ValueError, match="size"):
        load_document(b"abcd", "a.txt", max_file_bytes=3)
