"""Recover page/test context from indexed spans only; never read local PDF content.

Incomplete pages, conflicting overlaps and uncertain boundaries are rejected.
Metadata enrichment is in-memory and never written back to Pinecone.
"""
import re
from collections import defaultdict
from langchain_core.documents import Document


def headers(text):
    events = {}
    # Sorted extraction: CODE    title. Unsorted extraction: CODE / ID / title.
    for m in re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9]{1,14})[ \t]+([^\n]{4,180})", text):
        if m[1] in {"CPT", "OR", "TECHONLY", "SERUM", "DNA", "RNA"} or ":" in m[2]:
            continue
        name = m[2].strip()
        continuation = re.match(r"\n[ \t]*\d{3,10}[ \t]+([^\n]+)", text[m.end():])
        if continuation:
            name += " " + continuation[1].strip()
        events[m.start()] = (m.start(), name, m[1])
    for m in re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9]{1,14})[ \t]*\n\s*\d{3,10}[ \t]*\n\s*([^\n]{4,180})", text):
        events[m.start()] = (m.start(), m[2].strip(), m[1])
    # An incomplete new entry still ends the preceding test's region.
    for m in re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9]{1,14})[ \t]*\n", text):
        if m.start() not in events:
            events[m.start()] = (m.start(), "", m[1])
    # Some sorted PDF footers interleave the next test code with 'as of'.
    for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,14})as of\b", text):
        events[m.start()] = (m.start(), "", m[1])
    return sorted(events.values())


def enrich_indexed_matches(matches):
    groups = defaultdict(list)
    for match in matches:
        m = match["document"].metadata
        if m.get("document_id") and m.get("source_content_hash") and m.get("page_number"):
            groups[(m["document_id"], m["source_content_hash"], int(m["page_number"]))].append(match)
    complete = {}
    for group, rows in groups.items():
        buffer = []
        valid = True
        for row in sorted(rows, key=lambda r: r["document"].metadata.get("start_char", -1)):
            doc = row["document"]
            meta = doc.metadata
            if "start_char" not in meta or "end_char" not in meta:
                valid = False
                break
            start, end = int(meta["start_char"]), int(meta["end_char"])
            prefix = meta.get("heading_prefix", "")
            raw = doc.page_content[len(prefix):] if prefix and doc.page_content.startswith(prefix) else doc.page_content
            if start < 0 or end <= start or len(raw) != end - start or start > len(buffer) + 3 or end > 100000:
                valid = False
                break
            buffer.extend(" " * max(0, end - len(buffer)))
            for pos, char in enumerate(raw, start):
                if buffer[pos] != " " and buffer[pos] != char:
                    valid = False
                    break
                buffer[pos] = char
        text = "".join(buffer)
        footer = re.search(r"\bPage\s+(\d+)\b", text)
        if valid and footer:
            complete[group] = (text, headers(text), footer[1])
    output = []
    for match in matches:
        doc = match["document"]
        meta = doc.metadata
        try:
            key = (meta["document_id"], meta["source_content_hash"], int(meta["page_number"]))
        except (KeyError, ValueError, TypeError):
            output.append(match)
            continue
        if key not in complete:
            output.append(match)
            continue
        text, entries, printed = complete[key]
        previous = complete.get((*key[:2], key[2] - 1))
        if previous and previous[1] and previous[1][-1][1]:
            _, name, code = previous[1][-1]
            entries = [(0, name, code), *entries] if not entries or entries[0][0] > 0 else entries
        start, end = int(meta["start_char"]), int(meta["end_char"])
        emitted = False
        prefix = meta.get("heading_prefix", "")
        raw_offset = len(prefix) if prefix and doc.page_content.startswith(prefix) else 0
        for i, (boundary, name, code) in enumerate(entries):
            limit = entries[i + 1][0] if i + 1 < len(entries) else len(text)
            left, right = max(start, boundary), min(end, limit)
            if not name or right <= left:
                continue
            fragment = doc.page_content[raw_offset + left - start:raw_offset + right - start]
            output.append({**match, "document": Document(page_content=fragment,
                metadata={**meta, "test_name": name, "test_code": code, "printed_catalog_page": printed,
                          "identity_context": "verified contiguous indexed page spans"})})
            emitted = True
        if not emitted:
            output.append(match)
    return output
