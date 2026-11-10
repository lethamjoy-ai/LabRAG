"""Structural and source-span checks, not a clinical semantic validator."""
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from src.chunking.common import REQUIRED, chunk_identity
from src.chunking.section_chunker import detect_sections
from src.utils.token_counter import count_tokens
from src.ingestion.test_metadata import annotate_documents, test_metadata_for_span


@dataclass
class ValidationResult:
    status: str
    warnings: list[str]
    errors: list[str]
    chunk_count: int
    minimum_token_count: int
    maximum_token_count: int
    average_token_count: float
    empty_chunks: int

    def to_dict(self):
        return asdict(self)


def validate_chunks(chunks, documents=None, max_tokens=500):
    chunks = list(chunks)
    sources = annotate_documents(list(documents)) if documents is not None else None
    source_hashes = {}
    if sources is not None:
        grouped = defaultdict(list)
        for source in sources:
            grouped[source.metadata.get("document_id", "")].append(source.page_content)
        source_hashes = {identity: hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
                         for identity, texts in grouped.items()}
    errors, warnings, counts = [], [], []
    ids, positions, text_keys = set(), set(), set()
    spans = defaultdict(list)
    for i, chunk in enumerate(chunks):
        meta, text = chunk.metadata, chunk.page_content
        count = count_tokens(text)
        counts.append(count)
        label = f"Chunk {i}"
        malformed = False
        for key in ("chunk_id", "document_id", "chunking_strategy", "source_content_hash"):
            if key in meta and not isinstance(meta[key], str):
                errors.append(f"{label}: {key} must be a string.")
                malformed = True
        for key in ("chunk_index", "source_part", "start_char", "end_char", "token_count", "chunk_size", "chunk_overlap"):
            if key in meta and type(meta[key]) is not int:
                errors.append(f"{label}: {key} must be an integer.")
                malformed = True
        if malformed:
            continue
        if not text.strip():
            errors.append(f"{label}: empty text.")
        for key in (*REQUIRED, "chunk_id", "chunk_index", "chunking_strategy", "token_count",
                    "source_content_hash", "source_part", "start_char", "end_char"):
            if key not in meta or (key not in {"source_url", "test_name"} and meta[key] in (None, "")):
                errors.append(f"{label}: missing metadata {key}.")
        for key in ("document_id", "document_title", "laboratory_provider", "accessed_date", "content_hash"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                errors.append(f"{label}: {key} must be a nonempty string.")
        if not isinstance(meta.get("test_name"), str):
            errors.append(f"{label}: test_name must be text or blank.")
        if not meta.get("test_name") or meta.get("test_metadata_source") in {"unresolved", "ambiguous"}:
            warnings.append(f"{label}: test identity is incomplete or unresolved; review the source labels.")
        if not meta.get("source_url"):
            warnings.append(f"{label}: local/synthetic source without a source URL; verify attribution before indexing.")
        if meta.get("chunking_strategy") not in {"fixed", "section"}:
            errors.append(f"{label}: invalid strategy.")
        if type(meta.get("chunk_index")) is not int or meta.get("chunk_index", -1) < 0:
            errors.append(f"{label}: invalid chunk sequence.")
        if meta.get("token_count") != count:
            errors.append(f"{label}: token count does not match content.")
        if count > max_tokens:
            if meta.get("unsplittable") and meta.get("table_review_required"):
                warnings.append(f"{label}: oversized table-containing section requires manual review; not ready for indexing.")
            else:
                errors.append(f"{label}: exceeds token limit {max_tokens}.")
        if meta.get("table_review_required"):
            warnings.append(f"{label}: table-like layout requires row/column and unit review.")
        if meta.get("fallback_to_fixed"):
            warnings.append(f"{label}: no recognized headings; used fixed-size fallback.")
        if meta.get("chunk_id") in ids:
            errors.append(f"{label}: duplicate chunk ID.")
        ids.add(meta.get("chunk_id"))
        if meta.get("chunk_id") != chunk_identity(meta, text):
            errors.append(f"{label}: deterministic identity does not match content/provenance.")
        if meta.get("chunk_content_hash") != hashlib.sha256(text.encode()).hexdigest():
            errors.append(f"{label}: chunk content hash mismatch.")
        position = (meta.get("document_id"), meta.get("chunking_strategy"), meta.get("source_part"),
                    meta.get("start_char"), meta.get("end_char"))
        if position in positions:
            errors.append(f"{label}: duplicate source span.")
        positions.add(position)
        key = (meta.get("document_id"), meta.get("chunking_strategy"), text)
        if key in text_keys:
            warnings.append(f"{label}: repeated text at a distinct source position; retained for provenance review.")
        text_keys.add(key)
        if sources is not None:
            part, start, end = (meta.get(k) for k in ("source_part", "start_char", "end_char"))
            if not all(type(v) is int for v in (part, start, end)) or not (0 <= part < len(sources)):
                errors.append(f"{label}: invalid source span metadata.")
                continue
            source = sources[part]
            if meta.get("source_content_hash") != source_hashes.get(source.metadata.get("document_id", "")):
                errors.append(f"{label}: source snapshot hash differs from the current document.")
            prefix = meta.get("heading_prefix", "")
            if prefix and (meta.get("chunking_strategy") != "section" or prefix != meta.get("section_heading", "") + "\n"):
                errors.append(f"{label}: invalid repeated heading prefix.")
            if not 0 <= start < end <= len(source.page_content) or prefix + source.page_content[start:end] != text:
                errors.append(f"{label}: text differs from its exact source span (including values/units).")
            for key in REQUIRED + ("page_number",):
                if key == "test_name" and "test_metadata_source" in meta:
                    continue
                if meta.get(key) != source.metadata.get(key):
                    errors.append(f"{label}: source attribution changed: {key}.")
            if "test_metadata_source" in meta:
                expected_test = test_metadata_for_span(source, start, end)
                for key in ("test_name", "test_code", "test_metadata_source", "test_name_evidence", "test_code_evidence"):
                    if meta.get(key, "") != expected_test.get(key, ""):
                        errors.append(f"{label}: test identity/evidence differs from detected source context: {key}.")
            spans[part].append((start, end, text))
            if meta.get("chunking_strategy") == "section" and not meta.get("fallback_to_fixed"):
                boundaries = [s[0] for s in detect_sections(source.page_content)]
                if any(start < boundary < end for boundary in boundaries):
                    errors.append(f"{label}: crosses a detected section boundary.")
                if not meta.get("section_start", -1) <= start < end <= meta.get("section_end", -1):
                    errors.append(f"{label}: outside recorded section bounds.")
                if meta.get("section_heading"):
                    actual_heading = source.page_content[meta.get("section_start", 0):].splitlines()[0].strip()
                    if actual_heading != meta["section_heading"]:
                        errors.append(f"{label}: section heading does not match source.")
    if sources is not None:
        for part, source in enumerate(sources):
            cursor = 0
            valid_spans = sorted(spans[part])
            for start, end, _ in valid_spans:
                if source.page_content[cursor:start].strip():
                    errors.append(f"Source part {part}: text is missing before character {start}.")
                cursor = max(cursor, end)
            if source.page_content[cursor:].strip():
                errors.append(f"Source part {part}: trailing text is missing.")
            # Check contextual numeric phrases, not just isolated digit presence.
            for match in re.finditer(r"[<>≤≥+-]?\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(?:mg/dL|mmol/L|µmol/L|ng/mL|°C|hours?|minutes?)", source.page_content):
                if not any(start <= match.start() and end >= match.end() for start, end, _ in valid_spans):
                    warnings.append(f"Source part {part}: a numeric/unit phrase spans a chunk boundary; inspect context.")
    else:
        warnings.append("Source documents not supplied; coverage, attribution equality, and value preservation were not verified.")
    warnings.append("Structural checks do not establish clinical meaning or complete semantic preservation.")
    return ValidationResult("error" if errors else "warning" if warnings else "ok", warnings, errors,
        len(chunks), min(counts, default=0), max(counts, default=0), sum(counts) / len(counts) if counts else 0,
        sum(not c.page_content.strip() for c in chunks))


def comparison_metrics(results):
    return {name: {"Total chunks": r.chunk_count, "Average tokens": r.average_token_count,
                   "Minimum tokens": r.minimum_token_count, "Maximum tokens": r.maximum_token_count,
                   "Empty chunks": r.empty_chunks, "Validation warnings": len(r.warnings),
                   "Validation errors": len(r.errors)} for name, r in results.items()}
