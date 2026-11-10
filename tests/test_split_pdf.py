from decimal import Decimal
import pytest
import pymupdf
from scripts.split_pdf import split_pdf


def make_pdf(path, pages=8):
    with pymupdf.open() as pdf:
        for index in range(pages):
            pdf.new_page().insert_text((72, 72), f"Synthetic page {index + 1}")
        pdf.save(path)


def test_split_actual_sizes_page_order_and_original_preserved(tmp_path):
    source = tmp_path / "catalog.pdf"
    make_pdf(source)
    original = source.read_bytes()
    parts = split_pdf(source, tmp_path / "parts", max_mb="0.0015")
    assert len(parts) > 1
    assert all(path.stat().st_size < 1500 for path in parts)
    texts = []
    for path in parts:
        with pymupdf.open(path) as pdf:
            texts.extend(page.get_text().strip() for page in pdf)
    assert texts == [f"Synthetic page {i}" for i in range(1, 9)]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        split_pdf(source, tmp_path / "parts", max_mb="0.0015")


def test_small_pdf_and_strict_boundary(tmp_path):
    source = tmp_path / "small.pdf"
    make_pdf(source, 1)
    first = split_pdf(source, tmp_path / "first")[0]
    exact_limit = Decimal(first.stat().st_size) / 1_000_000
    with pytest.raises(ValueError, match="Page 1 alone"):
        split_pdf(source, tmp_path / "second", exact_limit)
    assert not list((tmp_path / "second").glob("*.pdf"))


@pytest.mark.parametrize("limit", ["0", "-1", "NaN", "Infinity", "invalid"])
def test_invalid_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="max-mb"):
        split_pdf(tmp_path / "unused.pdf", max_mb=limit)


def test_invalid_and_missing_files(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        split_pdf(tmp_path / "missing.pdf")
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="Cannot open"):
        split_pdf(source)


def test_page_limit_and_remainder(tmp_path):
    source = tmp_path / "catalog.pdf"
    make_pdf(source, 11)
    parts = split_pdf(source, max_pages=5)
    counts = []
    texts = []
    for path in parts:
        with pymupdf.open(path) as pdf:
            counts.append(pdf.page_count)
            texts.extend(page.get_text().strip() for page in pdf)
    assert counts == [5, 5, 1]
    assert texts == [f"Synthetic page {i}" for i in range(1, 12)]


@pytest.mark.parametrize("pages", [0, -1, True, 2.5])
def test_invalid_page_limit(tmp_path, pages):
    with pytest.raises(ValueError, match="max-pages"):
        split_pdf(tmp_path / "unused.pdf", max_pages=pages)
