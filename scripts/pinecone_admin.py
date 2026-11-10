"""Explicit administration only. No index creation/deletion occurs at app startup."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import Settings
from src.vectorstore.pinecone_client import PineconeConnection, PineconeError, retry
from src.vectorstore.index_manager import IndexManager


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit LabRAG Pinecone administration")
    parser.add_argument("action", choices=["validate", "create", "remove-document"])
    parser.add_argument("--confirm-create", action="store_true")
    parser.add_argument("--confirm-remove", action="store_true")
    parser.add_argument("--document-id")
    parser.add_argument("--strategy", choices=["fixed", "section"], default="section")
    parser.add_argument("--corpus", choices=["public", "synthetic"], default="public")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if not settings.pinecone_api_key.strip():
            raise ValueError("Set PINECONE_API_KEY in .env.")
        if args.action == "create":
            if not args.confirm_create:
                raise ValueError("Index creation requires --confirm-create. Cloud resources may incur charges.")
            from pinecone import Pinecone, ServerlessSpec
            client = Pinecone(api_key=settings.pinecone_api_key)
            if settings.pinecone_index_name in retry(lambda: client.list_indexes().names()):
                raise ValueError("Index already exists. It will not be changed or recreated.")
            retry(lambda: client.create_index(name=settings.pinecone_index_name,
                dimension=settings.embedding_dimension, metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region)))
            print("Index creation requested. Wait for readiness, then validate. Region availability depends on your account.")
        elif args.action == "validate":
            connection = PineconeConnection(settings).connect()
            print(connection.stats())
        else:
            if not args.confirm_remove or not args.document_id:
                raise ValueError("Scoped removal requires --document-id and --confirm-remove.")
            IndexManager(settings, None, PineconeConnection(settings)).remove_indexed_document(
                args.document_id, args.strategy, args.corpus)
            print("Removed only vector IDs recorded for this document/model/index/namespace.")
    except (ValueError, PineconeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    main()
