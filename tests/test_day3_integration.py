"""Opt-in only: uploads labeled synthetic text to unique test namespaces, then purges owned IDs."""
import os
import time
import uuid
from dataclasses import replace
import pytest
from config.settings import Settings
from src.embeddings.embedding_service import get_configured_embedding_service
from src.vectorstore.pinecone_client import PineconeConnection
from src.vectorstore.index_manager import IndexManager
from src.retrieval.semantic_retriever import SemanticRetriever
from src.chunking.pipeline import process_document
from src.chunking.chunk_manager import export_jsonl

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv("LABRAG_RUN_INTEGRATION") != "1",
              reason="Live Pinecone integration requires explicit LABRAG_RUN_INTEGRATION=1")]


def test_live_index_search_and_comparison(tmp_path, laboratory_document):
    prefix = "labrag-it-" + uuid.uuid4().hex[:12]
    settings = replace(Settings.from_env(), namespace_fixed=prefix + "-fixed", namespace_section=prefix + "-section")
    embeddings = get_configured_embedding_service(settings)
    connection = PineconeConnection(settings).connect()
    manager = IndexManager(settings, embeddings, connection, tmp_path / "manifest.json")
    doc = laboratory_document("Test Overview\nSynthetic integration fixture: fictional Alpha laboratory reference. No clinical guidance.")
    try:
        for strategy in ("fixed", "section"):
            _, chunks, _ = process_document([doc], strategy)
            path = tmp_path / f"{strategy}.jsonl"
            path.write_text(export_jsonl(chunks), encoding="utf-8")
            assert manager.index_file(path, strategy, "synthetic")["status"] == "complete"
        retriever = SemanticRetriever(settings, embeddings, connection)
        for _ in range(15):
            result = retriever.compare_retrieval("Find fictional Alpha reference", corpus="synthetic")
            if all(value.matches for value in result.values()):
                break
            time.sleep(2)
        assert all(value.matches for value in result.values())
        assert all(value.matches[0]["document"].metadata["document_id"] == doc.metadata["document_id"] for value in result.values())
    finally:
        for strategy in ("fixed", "section"):
            if manager._key(doc.metadata["document_id"], strategy, "synthetic") in manager.manifest.read()["entries"]:
                manager.remove_indexed_document(doc.metadata["document_id"], strategy, "synthetic")
