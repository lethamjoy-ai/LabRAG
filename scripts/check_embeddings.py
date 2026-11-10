"""Explicit local model smoke check; does not connect to Pinecone or load API keys."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embeddings.embedding_service import EmbeddingService


def main():
    service = EmbeddingService()
    vectors = service.embed_documents(["Synthetic laboratory reference for software testing.", "Fictional specimen placeholder."])
    query = service.embed_query("Find the synthetic reference.")
    assert len(vectors) == 2 and all(len(vector) == 384 for vector in vectors) and len(query) == 384
    print("Local embedding smoke check passed: 2 document vectors and 1 query vector, each 384 finite dimensions.")


if __name__ == "__main__":
    main()
