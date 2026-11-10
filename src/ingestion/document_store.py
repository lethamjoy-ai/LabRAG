"""Remove one local document and its derived files without touching source originals."""
import os
import tempfile
from pathlib import Path
from src.ingestion.metadata_manager import MetadataManager
from src.chunking.chunk_manager import ChunkManager


def remove_document(document_id: str, root="data"):
    if not isinstance(document_id, str) or len(document_id) != 64 or any(c not in "0123456789abcdef" for c in document_id):
        raise ValueError("Invalid document identifier.")
    root = Path(root).resolve()
    manager = MetadataManager(root / "source_manifest.csv")
    records = manager.list_records()
    matches = records.document_id == document_id
    if matches.sum() != 1:
        raise ValueError("Document not found or its metadata is ambiguous. Refresh and try again.")
    record = records.loc[matches].iloc[0].to_dict()
    suffix = record["document_type"]
    if suffix not in {"pdf", "html", "txt"}:
        raise ValueError("Unsupported stored document type.")
    chunks = ChunkManager(root / "processed")
    targets = [root / "raw" / f"{document_id}.{suffix}",
               root / "processed" / f"{document_id}.json",
               chunks.path_for(document_id, "fixed"), chunks.path_for(document_id, "section")]
    # Resolve and check every target before moving/deleting any data. No glob or recursive deletion.
    for path in [manager.path, *targets]:
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Stored file path is outside the local data directory or is a symbolic link.")
        if path.exists() and not path.is_file():
            raise ValueError("Expected a stored document file, but found a directory.")
    # Stage files on the same filesystem; restore them if moving or updating the manifest fails.
    with tempfile.TemporaryDirectory(prefix=".remove-document-", dir=root) as staging:
        moved = []
        try:
            for index, path in enumerate(targets):
                if path.exists():
                    staged = Path(staging) / str(index)
                    os.replace(path, staged)
                    moved.append((path, staged))
            manager._write(records.loc[~matches].reset_index(drop=True))
        except Exception:
            for path, staged in reversed(moved):
                os.replace(staged, path)
            raise
    return record
