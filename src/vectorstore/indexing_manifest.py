"""Atomic checkpoints for a trusted single-writer indexing process."""
import json
import os
import tempfile
from pathlib import Path


class IndexingManifest:
    def __init__(self, path="data/indexing_manifest.json"):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            return {"version": 1, "entries": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("version") != 1 or not isinstance(value.get("entries"), dict):
                raise ValueError
            return value
        except (OSError, ValueError, AttributeError):
            raise ValueError("Indexing manifest is corrupt or unreadable. Restore it before indexing to retain vector ownership records.") from None

    def write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
