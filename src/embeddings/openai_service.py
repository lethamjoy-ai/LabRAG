"""Optional paid OpenAI embeddings, explicit dimensions and no silent truncation."""
from functools import lru_cache
from src.embeddings.embedding_service import EmbeddingService, EmbeddingError
from src.utils.token_counter import count_tokens


class OpenAIBackend:
    def __init__(self, model_name, dimension, api_key, client=None):
        self.model_name, self.dimension = model_name, dimension
        self._client = client
        self._api_key = api_key

    def embed_documents(self, texts):
        if self._client is None:
            if not self._api_key.strip():
                raise EmbeddingError("Set OPENAI_API_KEY in .env before using OpenAI embeddings.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, timeout=60, max_retries=2)
        response = self._client.embeddings.create(model=self.model_name, dimensions=self.dimension,
            input=texts, encoding_format="float")
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise EmbeddingError("OpenAI embedding result indices do not match the input batch.")
        return [item.embedding for item in data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, model_name="text-embedding-3-small", dimension=384, batch_size=32, api_key="", client=None):
        super().__init__(model_name, dimension, batch_size, backend=OpenAIBackend(model_name, dimension, api_key, client))

    def validate_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Embedding input must be nonempty text.")
        count = count_tokens(text)
        if count > 8192:
            raise EmbeddingError("OpenAI embedding input exceeds 8192 tokens. Reprocess smaller chunks; text was not truncated.")
        return count

    def embed_documents(self, texts):
        if not isinstance(texts, list):
            raise EmbeddingError("embed_documents expects a list of strings.")
        result, batch, tokens = [], [], 0
        for text in texts:
            count = self.validate_text(text)
            if batch and (len(batch) >= self.batch_size or tokens + count > 290_000):
                result.extend(self._embed_batch(batch))
                batch, tokens = [], 0
            batch.append(text)
            tokens += count
        if batch:
            result.extend(self._embed_batch(batch))
        return result

    def _embed_batch(self, texts):
        try:
            return super().embed_documents(texts)
        except Exception:
            raise EmbeddingError("OpenAI embedding request failed. Check OPENAI_API_KEY, API billing/quota, model access, and connectivity.") from None

    def embed_query(self, question):
        return self.embed_documents([question])[0]


@lru_cache(maxsize=2)
def get_openai_service(model_name, dimension, batch_size, api_key):
    return OpenAIEmbeddingService(model_name, dimension, batch_size, api_key)
