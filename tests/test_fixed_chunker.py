import pytest
from src.chunking.fixed_chunker import chunk_document_fixed
from src.chunking.chunk_validator import validate_chunks
from src.utils.token_counter import count_tokens


def test_limits_metadata_ids_coverage_and_overlap(laboratory_document):
    text = " ".join(f"synthetic{i}" for i in range(180))
    doc = laboratory_document(text)
    chunks = chunk_document_fixed([doc], 50, 10)
    assert len(chunks) > 1
    assert all(count_tokens(d.page_content) <= 50 for d in chunks)
    assert any(a.metadata["end_char"] > b.metadata["start_char"] for a, b in zip(chunks, chunks[1:]))
    assert all(d.metadata["laboratory_provider"] == "Fictional laboratory" for d in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert not validate_chunks(chunks, [doc], 50).errors
    assert [c.metadata["chunk_id"] for c in chunks] == [c.metadata["chunk_id"] for c in chunk_document_fixed([doc], 50, 10)]
    changed = laboratory_document(text + " updated")
    assert chunks[0].metadata["chunk_id"] != chunk_document_fixed([changed], 50, 10)[0].metadata["chunk_id"]


def test_pages_and_providers_never_mixed(laboratory_document):
    docs = [laboratory_document("Source A", "a"), laboratory_document("Source B", "b", laboratory_provider="Other")]
    chunks = chunk_document_fixed(docs, 50, 10)
    assert [c.page_content for c in chunks] == ["Source A", "Source B"]
    assert not validate_chunks(chunks, docs, 50).errors


def test_repeated_text_unicode_no_loss(laboratory_document):
    doc = laboratory_document(("µmol/L 2–8 °C 合成测试 " * 90))
    chunks = chunk_document_fixed([doc], 45, 8)
    assert not validate_chunks(chunks, [doc], 45).errors
    assert all("�" not in c.page_content for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (20, 20), (20, -1), (True, 0)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document_fixed([], size, overlap)


def test_empty(laboratory_document):
    assert chunk_document_fixed([]) == []
    assert chunk_document_fixed([laboratory_document("   ")]) == []
