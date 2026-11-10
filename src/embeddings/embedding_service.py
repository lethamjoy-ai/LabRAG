"""Lazy local embeddings with exact model-token limits and bounded memory caching."""
import hashlib
import math
from collections import OrderedDict
from functools import lru_cache
from langchain_core.embeddings import Embeddings


class EmbeddingError(ValueError):
    pass


class EmbeddingService(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384,
                 batch_size=32, backend=None):
        if type(dimension) is not int or dimension <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise EmbeddingError("Embedding dimension and batch size must be positive integers.")
        self.model_name, self.dimension, self.batch_size = model_name, dimension, batch_size
        self._backend = backend
        self._cache = OrderedDict()

    @property
    def backend(self):
        if self._backend is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._backend = HuggingFaceEmbeddings(model_name=self.model_name,
                    cache_folder=".model-cache/hub",
                    model_kwargs={"device": "cpu", "trust_remote_code": False},
                    encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True})
            except Exception:
                raise EmbeddingError("Embedding model could not load. Check installed Day 3 dependencies, disk space, and Hugging Face connectivity.") from None
        return self._backend

    def validate_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Embedding input must be nonempty text.")
        model = getattr(self.backend, "_client", None)
        if model is None:
            raise EmbeddingError("Cannot inspect the embedding model tokenizer; indexing is blocked to avoid truncation.")
        tokens = model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        if len(tokens) > model.max_seq_length:
            raise EmbeddingError(f"Input has {len(tokens)} embedding-model tokens; limit is {model.max_seq_length}. "
                                 "Reprocess with smaller chunks/overlap or review the table. Text was not truncated.")
        return len(tokens)

    def validate_vector(self, vector):
        if len(vector) != self.dimension:
            raise EmbeddingError(f"Expected {self.dimension}-dimensional embeddings.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding contains invalid or nonfinite numbers.")
        return [float(value) for value in vector]

    def embed_documents(self, texts):
        if not isinstance(texts, list):
            raise EmbeddingError("embed_documents expects a list of strings.")
        if not texts:
            return []
        for text in texts:
            self.validate_text(text)
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            missing = list(dict.fromkeys(text for text in batch if hashlib.sha256(text.encode()).hexdigest() not in self._cache))
            if missing:
                try:
                    vectors = self.backend.embed_documents(missing)
                except Exception:
                    raise EmbeddingError("Local document embedding failed. Check model availability and memory.") from None
                if len(vectors) != len(missing):
                    raise EmbeddingError("Embedding result count does not match the input batch.")
                for text, vector in zip(missing, vectors, strict=True):
                    self._cache[hashlib.sha256(text.encode()).hexdigest()] = self.validate_vector(vector)
            result.extend(list(self._cache[hashlib.sha256(text.encode()).hexdigest()]) for text in batch)
            while len(self._cache) > 1024:
                self._cache.popitem(last=False)
        return result

    def embed_query(self, question):
        self.validate_text(question)
        try:
            return self.validate_vector(self.backend.embed_query(question))
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError("Local query embedding failed. Check the model and available memory.") from None


@lru_cache(maxsize=2)
def get_embedding_service(model_name, dimension=384, batch_size=32):
    return EmbeddingService(model_name, dimension, batch_size)


def get_configured_embedding_service(settings):
    if settings.embedding_provider == "openai":
        from src.embeddings.openai_service import get_openai_service
        return get_openai_service(settings.embedding_model, settings.embedding_dimension,
                                  settings.embedding_batch_size, settings.openai_api_key)
    return get_embedding_service(settings.embedding_model, settings.embedding_dimension, settings.embedding_batch_size)
