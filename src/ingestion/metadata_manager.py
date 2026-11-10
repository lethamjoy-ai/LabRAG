"""Validated local CSV metadata and atomic writes for a single-user prototype."""
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, date, timezone
from urllib.parse import urlparse
import pandas as pd
from src.ingestion.test_metadata import annotate_documents, summarize_tests

COLUMNS = ["document_id", "test_name", "test_code", "laboratory_provider", "source_url", "document_title",
           "document_type", "source_revision_date", "accessed_date", "section_heading", "page_number",
           "content_hash", "ingestion_timestamp", "test_scope"]


def validate_metadata(record):
    record = dict(record)
    record.setdefault("test_scope", "single" if record.get("test_name") else "unknown")
    if record["test_scope"] not in {"single", "catalog", "unknown"}:
        raise ValueError("test_scope must be single, catalog, or unknown.")
    for name in ["document_id", "laboratory_provider", "document_title", "document_type", "content_hash", "ingestion_timestamp", "accessed_date"]:
        if not str(record.get(name, "")).strip():
            raise ValueError(f"{name} is required.")
    for name in ("test_name", "test_code"):
        record.setdefault(name, "")
        if not isinstance(record[name], str):
            raise ValueError(f"{name} must be text or blank.")
    if record["test_scope"] == "catalog":
        record["test_name"] = record["test_code"] = ""
    if record["document_type"] not in {"pdf", "html", "txt"}:
        raise ValueError("Invalid document_type.")
    for key in ["accessed_date", "source_revision_date"]:
        if record.get(key):
            date.fromisoformat(str(record[key]))
    datetime.fromisoformat(str(record["ingestion_timestamp"]))
    if len(str(record["content_hash"])) != 64 or any(c not in "0123456789abcdef" for c in str(record["content_hash"])):
        raise ValueError("content_hash must be a SHA-256 hex digest.")
    if record.get("source_url"):
        url = urlparse(record["source_url"])
        if url.scheme != "https" or not url.netloc:
            raise ValueError("source_url must be a valid HTTPS URL or blank for synthetic/local sources.")
    return {column: record.get(column, "") for column in COLUMNS}


class MetadataManager:
    def __init__(self, path="data/source_manifest.csv"):
        self.path = Path(path)

    def list_records(self):
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if "test_scope" not in frame:
            frame["test_scope"] = frame["test_name"].map(lambda value: "single" if value else "unknown")
        return frame.reindex(columns=COLUMNS, fill_value="")

    def _write(self, frame):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=self.path.parent,
                                         suffix=".csv", delete=False) as handle:
            temporary = handle.name
            frame.to_csv(handle, index=False)
        os.replace(temporary, self.path)

    def _check_duplicate(self, frame, record):
        if frame.empty:
            return
        duplicate = (frame.document_id == record["document_id"]) | (frame.content_hash == record["content_hash"])
        if record["source_url"]:
            duplicate |= ((frame.source_url == record["source_url"]) &
                          (frame.test_name == record["test_name"]) &
                          (frame.laboratory_provider == record["laboratory_provider"]))
        if duplicate.any():
            raise ValueError("Duplicate document ID, content hash, or source identifier. Update the existing record instead.")

    def add(self, record):
        record = validate_metadata(record)
        frame = self.list_records()
        self._check_duplicate(frame, record)
        self._write(pd.concat([frame, pd.DataFrame([record])], ignore_index=True))

    def update(self, document_id, changes):
        frame = self.list_records()
        matches = frame.document_id == document_id
        if matches.sum() != 1:
            raise ValueError("Document ID not found or ambiguous.")
        if any(k in changes for k in ["document_id", "content_hash", "document_type"]):
            raise ValueError("Content identity fields cannot be edited.")
        record = validate_metadata({**frame.loc[matches].iloc[0].to_dict(), **changes})
        self._check_duplicate(frame.loc[~matches], record)
        frame.loc[matches, COLUMNS] = [record[k] for k in COLUMNS]
        self._write(frame)


def save_document(data, documents, fields, root="data"):
    root = Path(root)
    documents = annotate_documents(list(documents))
    detected = summarize_tests(documents)
    fields = {**detected, **fields}
    if detected["test_scope"] == "catalog":
        fields.update(test_scope="catalog", test_name="", test_code="")
    elif fields.get("test_scope") != "catalog":
        for key in ("test_name", "test_code"):
            fields[key] = fields.get(key) or detected[key]
    first = documents[0].metadata
    record = validate_metadata({**fields, **{k: first[k] for k in ["document_id", "document_type", "content_hash"]},
        "section_heading": first.get("section_heading", ""), "page_number": first.get("page_number") or "",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat()})
    manager = MetadataManager(root / "source_manifest.csv")
    manager._check_duplicate(manager.list_records(), record)
    for folder in ["raw", "processed"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    suffix = record["document_type"]
    raw = root / "raw" / f"{record['document_id']}.{suffix}"
    processed = root / "processed" / f"{record['document_id']}.json"
    raw.write_bytes(data)
    processed.write_text(json.dumps([{"text": d.page_content, "original_text": d.metadata.get("original_text", d.page_content),
                                     "metadata": {**{k: v for k, v in d.metadata.items() if k != "original_text"}, **record,
                                                  "page_number": d.metadata.get("page_number"),
                                                  "section_heading": d.metadata.get("section_heading", "")}}
                                     for d in documents], indent=2), encoding="utf-8")
    manager.add(record)
    return record
