"""Conservative, source-backed test labels. No clinical inference or external calls."""
import re
from dataclasses import dataclass, asdict
from langchain_core.documents import Document

NAME = re.compile(r"^(?:test\s+name|test\s+title)\s*:\s*(.*)$", re.I)
CODE = re.compile(r"^(?:test\s+(?:id|code|definition|number))\s*:\s*(.*)$", re.I)
CODE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")
STOP = re.compile(r"^(?:test\b|specimen\b|patient preparation|collection\b|interpretation\b|"
                  r"reference\b|clinical\b|reporting name|useful for|method\b|cautions\b|"
                  r"rejection\b|stability\b|handling\b|page\b|copyright\b|©|source\b|provider\b)", re.I)
TABLE_SEPARATOR = re.compile(r"\s*\|\s*|\t+| {2,}")


@dataclass(frozen=True)
class TestEntry:
    start: int
    end: int
    test_name: str = ""
    test_code: str = ""
    name_evidence: str = ""
    code_evidence: str = ""


def detect_test_entries(text: str) -> list[TestEntry]:
    """Recognize labeled fields, Mayo-style Test Definition/ID headers, and labeled tables.

    Offsets refer to the supplied extracted text. Unknown layouts remain unresolved.
    No inference from filenames, clinical paragraphs, CPT codes, or bare numbers.
    """
    lines = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        lines.append((offset, raw.strip()))
        offset += len(raw)
    line_ends = {start: lines[i + 1][0] if i + 1 < len(lines) else len(text)
                 for i, (start, _) in enumerate(lines)}
    events = []
    consumed = set()
    for i, (start, line) in enumerate(lines):
        if i in consumed:
            continue
        # Explicitly labeled pipe/tab tables only; do not guess column meanings.
        cells = [c.strip() for c in TABLE_SEPARATOR.split(line.strip("|"))]
        lowered = [c.casefold() for c in cells]
        name_cols = [j for j, c in enumerate(lowered) if c in {"test name", "test title"}]
        code_cols = [j for j, c in enumerate(lowered) if c in {"test id", "test code", "test number"}]
        if len(name_cols) == len(code_cols) == 1:
            for j in range(i + 1, len(lines)):
                row = [c.strip() for c in TABLE_SEPARATOR.split(lines[j][1].strip("|"))]
                if all(re.fullmatch(r"[: -]+", cell or " ") for cell in row):
                    consumed.add(j)
                    continue
                if len(row) != len(cells):
                    break
                name, code = row[name_cols[0]], row[code_cols[0]]
                if not name or not CODE_VALUE.fullmatch(code):
                    break
                events.append((lines[j][0], "pair", name, code, lines[j][1]))
                consumed.add(j)
            continue
        match = NAME.fullmatch(line)
        kind = "name"
        if not match:
            match = CODE.fullmatch(line)
            kind = "code"
        if not match:
            continue
        value = match[1].strip()
        evidence = line
        next_index = i + 1
        # A label may be on a line of its own. Require an immediately adjacent value.
        if not value and next_index < len(lines):
            candidate = lines[next_index][1]
            if candidate and not STOP.match(candidate) and not ":" in candidate:
                value = candidate
                evidence += "\n" + candidate
                consumed.add(next_index)
                next_index += 1
        if not value or len(value) > 250:
            continue
        if kind == "code" and not CODE_VALUE.fullmatch(value):
            continue
        events.append((start, kind, value, "", evidence))
        # Mayo-style: Test Definition: CODE followed immediately by a title line.
        name_already_labeled = len(events) >= 2 and events[-2][1] == "name" and text[events[-2][0]:start].count("\n") <= 5
        if kind == "code" and not name_already_labeled and next_index < len(lines):
            candidate = lines[next_index][1]
            following = next((value for _, value in lines[next_index + 1:next_index + 4] if value), "")
            title_context = bool(STOP.match(following))
            instruction = re.match(r"^(?:store|keep|collect|submit|refrigerate|freeze|transport|avoid|do not|note|the|this|requires?)\b", candidate, re.I)
            if candidate and re.search(r"[A-Za-z]", candidate) and not STOP.match(candidate) and not instruction and not candidate.endswith(".") and ":" not in candidate and len(candidate) <= 250 and title_context and not re.search(r"\d\s*(?:°C|mg/|mmol/|hours?)\b", candidate):
                events.append((start, "name", candidate, "", candidate))
                consumed.add(next_index)
    records = []
    current = None
    for start, kind, value, code, evidence in sorted(events, key=lambda event: event[0]):
        # Two occurrences of the same label start different tests. Pair nearby name/code labels.
        field = "test_name" if kind == "name" else "test_code"
        distant = current and text[current["last"]:start].count("\n") > 5
        if current and (kind == "pair" or current.get(field) or distant):
            records.append(current)
            current = None
        if current is None:
            current = {"start": start, "last": start, "test_name": "", "test_code": "", "name_evidence": "", "code_evidence": ""}
        current["last"] = start
        if kind == "pair":
            current.update(test_name=value, test_code=code, name_evidence=evidence, code_evidence=evidence)
            # Directory rows describe the row only, not subsequent unrelated paragraphs.
            current["explicit_end"] = line_ends[start]
        else:
            current[field] = value
            current["name_evidence" if kind == "name" else "code_evidence"] = evidence
    if current:
        records.append(current)
    return [TestEntry(start=r["start"], end=min(r.get("explicit_end", len(text)), records[i + 1]["start"] if i + 1 < len(records) else len(text)),
                      **{k: r[k] for k in ("test_name", "test_code", "name_evidence", "code_evidence")})
            for i, r in enumerate(records)]


def detected_test_rows(documents):
    return [{"page_number": doc.metadata.get("page_number"), **asdict(entry)}
            for doc in documents for entry in detect_test_entries(doc.page_content)]


def summarize_tests(documents):
    identities = {(row["test_name"], row["test_code"]) for row in detected_test_rows(documents)}
    if len(identities) == 1:
        name, code = next(iter(identities))
        return {"test_name": name, "test_code": code, "test_scope": "single"}
    return {"test_name": "", "test_code": "", "test_scope": "catalog" if identities else "unknown"}


def annotate_documents(documents):
    """Infer document summaries without assigning one catalog test to every page."""
    groups = {}
    for index, doc in enumerate(documents):
        groups.setdefault(doc.metadata.get("document_id", f"part-{index}"), []).append(doc)
    summaries = {key: summarize_tests(group) for key, group in groups.items()}
    result = []
    for index, doc in enumerate(documents):
        summary = summaries[doc.metadata.get("document_id", f"part-{index}")]
        scope = "catalog" if doc.metadata.get("test_scope") == "catalog" or summary["test_scope"] == "catalog" else doc.metadata.get("test_scope", summary["test_scope"])
        meta = {**doc.metadata, "test_scope": scope}
        for field in ("test_name", "test_code"):
            meta[field] = "" if scope == "catalog" else doc.metadata.get(field) or summary[field]
        result.append(Document(page_content=doc.page_content, metadata=meta))
    return result


def test_regions(text):
    entries = detect_test_entries(text)
    if not entries:
        return [(0, len(text))]
    regions = []
    cursor = 0
    for entry in entries:
        if cursor < entry.start:
            regions.append((cursor, entry.start))
        regions.append((entry.start, entry.end))
        cursor = entry.end
    if cursor < len(text):
        regions.append((cursor, len(text)))
    return regions


def test_metadata_for_span(doc, start, end):
    entries = [entry for entry in detect_test_entries(doc.page_content) if entry.start < end and entry.end > start]
    contained = [entry for entry in entries if entry.start <= start and end <= entry.end]
    if len(contained) == 1:
        entry = contained[0]
        return {"test_name": entry.test_name, "test_code": entry.test_code, "test_metadata_source": "detected",
                "test_name_evidence": entry.name_evidence, "test_code_evidence": entry.code_evidence}
    if entries:
        return {"test_name": "", "test_code": "", "test_metadata_source": "ambiguous"}
    if doc.metadata.get("test_scope") != "catalog":
        return {"test_name": doc.metadata.get("test_name", ""), "test_code": doc.metadata.get("test_code", ""),
                "test_metadata_source": "document" if doc.metadata.get("test_name") or doc.metadata.get("test_code") else "unresolved"}
    return {"test_name": "", "test_code": "", "test_metadata_source": "unresolved"}
