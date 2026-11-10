"""Sentence Transformer embeddings, Pinecone persistence, and cross-encoder reranking."""
from langchain_core.documents import Document


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class PineconeStore:
    def __init__(self, settings, embedder):
        from pinecone import Pinecone
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.embedder = embedder
        self.namespace = settings.namespace
        self.model_name = settings.embedding_model
        description = self.client.describe_index(settings.index_name)
        if description.dimension != embedder.dimension:
            raise ValueError(f"Index dimension must be {embedder.dimension} for EMBEDDING_MODEL. Create a matching index.")
        if description.metric != "cosine":
            raise ValueError("Use a Pinecone dense index with cosine similarity.")
        if not description.status["ready"]:
            raise ValueError("Pinecone index is not ready. Retry after provisioning completes.")
        self.index = self.client.Index(host=description.host)

    def upsert(self, chunks):
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            vectors = self.embedder.encode([d.page_content for d in batch])
            records = []
            for doc, vector in zip(batch, vectors, strict=True):
                meta = {k: v for k, v in doc.metadata.items()
                        if isinstance(v, (str, int, float, bool))}
                meta.update(text=doc.page_content, embedding_model=self.model_name)
                records.append({"id": doc.metadata["chunk_id"], "values": vector, "metadata": meta})
            self.index.upsert(vectors=records, namespace=self.namespace)
        return len(chunks)

    def search(self, question, k=12, test_name=None):
        filters = {"embedding_model": {"$eq": self.model_name}}
        if test_name:
            filters["test_name"] = {"$eq": test_name}
        result = self.index.query(vector=self.embedder.encode([question])[0], top_k=k,
                                  namespace=self.namespace, include_metadata=True, filter=filters)
        documents = []
        for match in result.matches:
            meta = dict(match.metadata or {})
            content = meta.pop("text", "")
            if content:
                documents.append(Document(page_content=content,
                    metadata={**meta, "retrieval_score": float(match.score)}))
        return documents


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rank(self, question, documents, k=4):
        if not documents:
            return []
        scores = self.model.predict([(question, d.page_content) for d in documents])
        ranked = []
        for doc, score in zip(documents, scores, strict=True):
            ranked.append(Document(page_content=doc.page_content,
                                   metadata={**doc.metadata, "rerank_score": float(score)}))
        return sorted(ranked, key=lambda d: d.metadata["rerank_score"], reverse=True)[:k]
