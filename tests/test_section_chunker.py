from src.chunking.section_chunker import chunk_document_by_section
from src.chunking.chunk_validator import validate_chunks
from src.utils.token_counter import count_tokens


def test_headings_isolate_instructions(laboratory_document):
    doc = laboratory_document("Title\nSpecimen Required\nFictional specimen placeholder.\nInterpretation\nFictional meaning placeholder.")
    chunks = chunk_document_by_section([doc], 50, 5)
    specimen = next(c for c in chunks if c.metadata["section_heading"] == "Specimen Required")
    assert "Interpretation" not in specimen.page_content
    assert specimen.page_content.startswith("Specimen Required")
    assert not validate_chunks(chunks, [doc], 50).errors


def test_long_section_repeats_heading_and_limits(laboratory_document):
    doc = laboratory_document("Patient Preparation\n" + " ".join(f"placeholder{i}" for i in range(150)))
    chunks = chunk_document_by_section([doc], 50, 8)
    assert len(chunks) > 1
    assert all("Patient Preparation" in c.page_content for c in chunks)
    assert all(count_tokens(c.page_content) <= 50 for c in chunks)
    assert not validate_chunks(chunks, [doc], 50).errors


def test_fallback_and_custom_heading(laboratory_document):
    doc = laboratory_document("Unrecognized title\n" + "word " * 100)
    chunks = chunk_document_by_section([doc], 40, 5)
    assert all(c.metadata["fallback_to_fixed"] for c in chunks)
    custom = chunk_document_by_section([laboratory_document("CUSTOM HEADER\nBody")], 40, 5,
                                       heading_patterns=[r"CUSTOM HEADER"])
    assert custom[0].metadata["section_heading"] == "CUSTOM HEADER"
    assert not custom[0].metadata["fallback_to_fixed"]


def test_oversized_table_is_retained_and_flagged(laboratory_document):
    text = "Reference Values\nAnalyte | Range | Units\n" + "\n".join(f"Synthetic{i} | -2.5–4.7 | mmol/L" for i in range(30))
    doc = laboratory_document(text)
    chunks = chunk_document_by_section([doc], 50, 5)
    table_chunk = next(c for c in chunks if c.metadata["unsplittable"])
    assert table_chunk.page_content == text
    result = validate_chunks(chunks, [doc], 50)
    assert not result.errors
    assert any("oversized" in warning for warning in result.warnings)


def test_numbered_markdown_and_missing_headings(laboratory_document):
    doc = laboratory_document("## 1. SPECIMEN REQUIRED:\nA\n## 2. Interpretation:\nB")
    chunks = chunk_document_by_section([doc], 40, 5)
    assert len(chunks) == 2
    assert not validate_chunks(chunks, [doc], 40).errors
    assert chunk_document_by_section([]) == []


def test_small_table_does_not_make_surrounding_prose_unsplittable(laboratory_document):
    text = "Specimen Required\n" + "Instruction placeholder. " * 80 + "\n\nSpecimen Type    Temperature    Time\nSerum    Refrigerated    14 days\n\n" + "CPT placeholder. " * 60
    doc = laboratory_document(text)
    chunks = chunk_document_by_section([doc], 60, 8)
    assert not any(c.metadata["unsplittable"] for c in chunks)
    assert all(count_tokens(c.page_content) <= 60 for c in chunks)
    table = next(c for c in chunks if "Specimen Type" in c.page_content)
    assert "Serum    Refrigerated    14 days" in table.page_content
    assert table.metadata["table_review_required"]
    assert not validate_chunks(chunks, [doc], 60).errors


def test_indented_prose_is_not_a_table(laboratory_document):
    doc = laboratory_document("Specimen Required\n" + "      Ordinary instruction text.\n" * 25)
    chunks = chunk_document_by_section([doc], 50, 5)
    assert not any(c.metadata["table_review_required"] for c in chunks)
    assert not validate_chunks(chunks, [doc], 50).errors
