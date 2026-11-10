"""Split PDFs at page boundaries, verifying every part is strictly below a byte limit.

Usage: python scripts/split_pdf.py catalog.pdf --max-pages 500 --output-dir catalog_parts
MB means 1,000,000 bytes. Uses the project's existing PyMuPDF dependency.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import shutil
import tempfile

import pymupdf


def split_pdf(input_pdf, output_dir=None, max_mb=200, max_pages=500) -> list[Path]:
    """Preserve page order and content; refuse overwrites and oversized single pages.

    Compression is lossless; images are not downsampled. Files are staged before
    publishing so an oversized page cannot leave an apparently complete split.
    Outputs are independent PDFs; document-level signatures/bookmarks are not retained.
    """
    if type(max_pages) is not int or max_pages <= 0:
        raise ValueError("--max-pages must be a positive integer.")
    try:
        size = Decimal(str(max_mb))
        if not size.is_finite() or size <= 0:
            raise ValueError
        limit = int(size * 1_000_000)
        if limit < 2:
            raise ValueError
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("--max-mb must be a finite positive size of at least 2 bytes.") from None
    source_path = Path(input_pdf).resolve()
    if not source_path.is_file():
        raise ValueError(f"Input PDF does not exist: {source_path}")
    destination = Path(output_dir).resolve() if output_dir else source_path.parent / f"{source_path.stem}_parts"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        source = pymupdf.open(source_path)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Cannot open the input PDF. Check its format and integrity.") from exc
    with source, tempfile.TemporaryDirectory(prefix=".pdf-split-", dir=destination) as temporary:
        if not source.is_pdf:
            raise ValueError("Input must be a PDF document.")
        if source.needs_pass:
            raise ValueError("Input is password-protected. Provide an unlocked PDF first.")
        if not source.page_count:
            raise ValueError("Input PDF has no pages.")
        pending = [(first, min(first + max_pages, source.page_count) - 1)
                   for first in reversed(range(0, source.page_count, max_pages))]
        staged = []
        while pending:
            first, last = pending.pop()
            candidate = Path(temporary) / f"pages_{first + 1}_{last + 1}.pdf"
            with pymupdf.open() as part:
                part.insert_pdf(source, from_page=first, to_page=last)
                part.save(candidate, garbage=4, deflate=True)
            actual_size = candidate.stat().st_size
            if actual_size < limit:
                staged.append((first, last, candidate))
                continue
            candidate.unlink()
            if first == last:
                raise ValueError(f"Page {first + 1} alone is {actual_size:,} bytes and cannot fit below "
                                 f"{limit:,} bytes without changing its content. Increase --max-mb or optimize that page separately.")
            middle = (first + last) // 2
            pending.extend([(middle + 1, last), (first, middle)])
        staged.sort(key=lambda item: item[0])
        outputs = [(candidate, destination / f"{source_path.stem}_part_{index:03d}_pages_{first + 1}-{last + 1}.pdf")
                   for index, (first, last, candidate) in enumerate(staged, 1)]
        for _, output in outputs:
            if output.exists():
                raise ValueError(f"Refusing to overwrite {output}. Choose a different output directory.")
        for candidate, output in outputs:
            # Exclusive creation protects both the original PDF and existing outputs.
            with candidate.open("rb") as reader, output.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if output.stat().st_size >= limit:
                raise RuntimeError(f"Output size validation failed: {output}")
        return [output for _, output in outputs]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split a PDF into up to 500 pages per part, also enforcing a file-size limit.")
    parser.add_argument("input_pdf", type=Path, help="PDF to split (original remains unchanged)")
    parser.add_argument("--output-dir", type=Path, help="Default: <input filename>_parts beside the original")
    parser.add_argument("--max-mb", default="200", help="Maximum decimal MB per part (default: 200; strict upper bound)")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages per part (default: 500)")
    args = parser.parse_args(argv)
    try:
        outputs = split_pdf(args.input_pdf, args.output_dir, args.max_mb, args.max_pages)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for output in outputs:
        print(f"{output} ({output.stat().st_size:,} bytes)")
    print(f"Created {len(outputs)} part(s). Original PDF unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
