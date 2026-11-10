"""Conservative heading detection; never combine pages, documents, or providers."""
import re
from src.chunking.common import make_chunks, text_spans
from src.utils.token_counter import count_tokens
from src.ingestion.test_metadata import test_regions

DEFAULT_HEADING_PATTERNS = (
    r"Test Overview", r"Specimen(?: Required| Requirements| Information)?", r"Patient Preparation",
    r"Preparation(?: Instructions)?", r"Collection Instructions", r"Specimen Stability",
    r"Stability", r"Rejection Criteria", r"Clinical Information", r"Interpretation",
    r"Reference Values", r"Cautions", r"Handling(?: Requirements)?", r"Limitations",
)


def table_blocks(text):
    """Separate contiguous table-like lines from prose without changing any text.

    Keep each detected table intact: inferring missing column labels or splitting
    wrapped rows is unsafe. Indentation by itself is not evidence of a table.
    """
    blocks = []
    start = offset = 0
    current_table = None
    for line in text.splitlines(keepends=True):
        row = line.strip()
        is_table = bool("|" in row or "\t" in row or re.search(r"\S {2,}\S", row))
        if current_table is not None and is_table != current_table:
            blocks.append((start, offset, current_table))
            start = offset
        current_table = is_table
        offset += len(line)
    if current_table is not None:
        blocks.append((start, offset, current_table))
    return blocks


def detect_sections(text, heading_patterns=None):
    patterns = DEFAULT_HEADING_PATTERNS if heading_patterns is None else heading_patterns
    matcher = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:" + "|".join(patterns) + r")\s*:?\s*$", re.I)
    headings = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if matcher.fullmatch(line.strip()):
            headings.append((offset, line.strip()))
        offset += len(line)
    if not headings:
        return []
    sections = []
    if text[:headings[0][0]].strip():
        sections.append((0, headings[0][0], ""))
    for i, (start, heading) in enumerate(headings):
        sections.append((start, headings[i + 1][0] if i + 1 < len(headings) else len(text), heading))
    return sections


def chunk_document_by_section(documents, max_tokens=500, overlap=75, heading_patterns=None):
    documents = list(documents)
    def parts(doc):
        sections = []
        for region_start, region_end in test_regions(doc.page_content):
            region = doc.page_content[region_start:region_end]
            found = detect_sections(region, heading_patterns)
            sections.extend((region_start + start, region_start + end, heading, not found)
                            for start, end, heading in (found or [(0, len(region), "")]))
        for start, end, heading, fallback in sections:
            section = doc.page_content[start:end]
            for block_start, block_end, table in table_blocks(section):
                block_start, block_end = start + block_start, start + block_end
                text = doc.page_content[block_start:block_end]
                if not text.strip():
                    continue
                prefix = heading + "\n" if heading else ""
                needs_prefix = bool(prefix and block_start > start)
                oversized_table = table and count_tokens((prefix if needs_prefix else "") + text) > max_tokens
                extra = {"fallback_to_fixed": fallback, "table_review_required": table,
                         "unsplittable": oversized_table, "section_start": start, "section_end": end}
                if table:
                    yield block_start, block_end, heading, {**extra, "heading_prefix": prefix if needs_prefix else ""}
                    continue
                budget = max_tokens - count_tokens(prefix) - 2 if prefix and (needs_prefix or count_tokens(text) > max_tokens) else max_tokens
                if budget <= overlap:
                    raise ValueError("Section heading leaves insufficient token space. Increase size or reduce overlap.")
                for local_start, local_end, _ in text_spans(text, budget, overlap):
                    yield block_start + local_start, block_start + local_end, heading, {
                        **extra, "heading_prefix": prefix if block_start + local_start > start else ""}
    return make_chunks(documents, "section", max_tokens, overlap, parts)
