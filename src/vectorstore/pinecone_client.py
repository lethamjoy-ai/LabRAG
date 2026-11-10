import time


class PineconeError(RuntimeError):
    pass


def retry(operation, attempts=3, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            status = getattr(exc, "status", getattr(exc, "status_code", None))
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or status in {408, 429, 500, 502, 503, 504}
            if not transient or attempt + 1 == attempts:
                raise PineconeError("Pinecone request failed. Check credentials, permissions, index availability, quota, and network connectivity.") from None
            sleep(2 ** attempt)


def namespace_for(settings, strategy, corpus="public"):
    if strategy not in {"fixed", "section"} or corpus not in {"public", "synthetic"}:
        raise ValueError("Choose fixed/section strategy and public/synthetic corpus.")
    namespace = settings.namespace_fixed if strategy == "fixed" else settings.namespace_section
    if settings.embedding_provider == "openai":
        namespace += f"-openai-{settings.embedding_model}-{settings.embedding_dimension}"
    return namespace if corpus == "public" else namespace + "-synthetic"


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class PineconeConnection:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client, self.index = client, None

    def connect(self):
        if not self.settings.pinecone_api_key.strip():
            raise PineconeError("Set PINECONE_API_KEY in .env before connecting. Local ingestion still works without it.")
        if self.client is None:
            from pinecone import Pinecone
            self.client = Pinecone(api_key=self.settings.pinecone_api_key)
        names = retry(lambda: self.client.list_indexes().names())
        if self.settings.pinecone_index_name not in names:
            raise PineconeError("Configured index does not exist. Create it manually with the configured dimension and cosine metric; see docs/pinecone_setup.md.")
        description = retry(lambda: self.client.describe_index(self.settings.pinecone_index_name))
        if field(description, "dimension") != self.settings.embedding_dimension:
            raise PineconeError(f"Index dimension mismatch. Expected {self.settings.embedding_dimension}; use a matching index.")
        if field(description, "metric") != "cosine":
            raise PineconeError("Index must use cosine similarity. Existing indexes are never recreated automatically.")
        if not field(field(description, "status", {}), "ready", False):
            raise PineconeError("Index is not ready. Wait for provisioning and retry.")
        self.index = self.client.Index(host=field(description, "host"))
        return self

    def stats(self):
        if self.index is None:
            self.connect()
        return retry(lambda: self.index.describe_index_stats())

    def vector_store(self, embeddings, namespace):
        if self.index is None:
            self.connect()
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(index=self.index, embedding=embeddings, text_key="text", namespace=namespace)
