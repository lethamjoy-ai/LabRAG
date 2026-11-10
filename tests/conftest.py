import hashlib
import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def isolate_project_dotenv(request, tmp_path, monkeypatch):
    # Unit tests must never read developer credentials. Explicit integration tests opt in.
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr("config.settings.ENV_FILE", tmp_path / ".env")


@pytest.fixture
def laboratory_document():
    def build(text, document_id="synthetic-1", **metadata):
        return Document(page_content=text, metadata={"document_id": document_id,
            "document_title": "Synthetic laboratory reference", "test_name": "Fictional test",
            "test_code": "DEMO", "laboratory_provider": "Fictional laboratory", "source_url": "https://example.org/synthetic",
            "accessed_date": "2026-10-17", "source_revision_date": "", "page_number": 1,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(), **metadata})
    return build
