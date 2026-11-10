import pytest
from labrag.ingestion import clean_text, parse_document, chunk_documents, load_manifest


def test_clinical_symbols_and_page_metadata_survive():
    docs = parse_document(b"Heading\n\nValue: -2.5 mg/dL\n\x00", "ref.txt", {"source": "manual"})
    assert "-2.5 mg/dL" in docs[0].page_content
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "manual"
    assert "\x00" not in docs[0].page_content


def test_html_removes_active_content_preserves_table():
    docs = parse_document(b"<h1>Test</h1><script>bad()</script><table><tr><td>Unit</td><td>mg/L</td></tr></table>", "ref.html")
    assert "bad" not in docs[0].page_content
    assert "Unit | mg/L" in docs[0].page_content


def test_chunk_ids_stable_and_provenance_sensitive():
    docs = parse_document(("Heading\n" + "text " * 200).encode(), "ref.txt")
    first = chunk_documents(docs, size=200, overlap=30)
    assert len(first) > 1
    assert [d.metadata["chunk_id"] for d in first] == [d.metadata["chunk_id"] for d in chunk_documents(docs, 200, 30)]
    assert len({d.metadata["chunk_id"] for d in first}) == len(first)


@pytest.mark.parametrize("data,name", [(b"", "empty.txt"), (b"x", "x.csv"), (b"\xff", "x.txt")])
def test_invalid_uploads(data, name):
    with pytest.raises((ValueError, UnicodeError)):
        parse_document(data, name)


def test_manifest_invalid_and_valid():
    with pytest.raises(ValueError):
        load_manifest(b'[{}]')
    assert len(load_manifest(open("data/source_manifest.json", "rb").read())) == 1


def test_pdf_page_provenance():
    import pymupdf
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Page one")
        pdf.new_page().insert_text((72, 72), "Page two")
        docs = parse_document(pdf.tobytes(), "ref.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 2]
