from copy import deepcopy
from src.chunking.fixed_chunker import chunk_document_fixed
from src.chunking.chunk_validator import validate_chunks, comparison_metrics


def test_tampering_empty_missing_and_oversized(laboratory_document):
    doc = laboratory_document("Synthetic -2.5 mg/dL with 24 hours of fictional handling.")
    chunks = chunk_document_fixed([doc], 40, 5)
    broken = deepcopy(chunks)
    broken[0].page_content = broken[0].page_content.replace("-2.5", "99.5")
    assert any("source span" in error for error in validate_chunks(broken, [doc], 40).errors)
    broken[0].page_content = ""
    broken[0].metadata.pop("laboratory_provider")
    assert validate_chunks(broken, [doc], 40).empty_chunks == 1
    assert any("missing metadata" in e for e in validate_chunks(broken, [doc], 40).errors)
    assert any("exceeds token limit" in e for e in validate_chunks(chunks, [doc], 1).errors)


def test_missing_text_and_duplicate_detection(laboratory_document):
    doc = laboratory_document(" ".join(f"token{i}" for i in range(100)))
    chunks = chunk_document_fixed([doc], 40, 5)
    assert validate_chunks(chunks[:-1], [doc], 40).errors
    assert any("duplicate" in e for e in validate_chunks(chunks + chunks, [doc], 40).errors)


def test_actual_comparison_metrics(laboratory_document):
    chunks = chunk_document_fixed([laboratory_document("Synthetic short text")], 40, 5)
    result = validate_chunks(chunks, max_tokens=40)
    metrics = comparison_metrics({"fixed": result})["fixed"]
    assert metrics["Total chunks"] == len(chunks)
    assert metrics["Average tokens"] == sum(c.metadata["token_count"] for c in chunks) / len(chunks)
    assert metrics["Validation warnings"] == len(result.warnings)
    assert metrics["Empty chunks"] == 0
