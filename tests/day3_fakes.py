from types import SimpleNamespace
from unittest.mock import Mock
from src.embeddings.embedding_service import EmbeddingService


def embedding_service(dimension=384, batch_size=2):
    tokenizer = lambda text, **kwargs: {"input_ids": list(range(len(text.split()) + 2))}
    backend = SimpleNamespace(_client=SimpleNamespace(tokenizer=tokenizer, max_seq_length=256))
    backend.embed_documents = Mock(side_effect=lambda texts: [[0.1] * dimension for _ in texts])
    backend.embed_query = Mock(return_value=[0.1] * dimension)
    return EmbeddingService(dimension=dimension, batch_size=batch_size, backend=backend)


def connection():
    index = Mock()
    return SimpleNamespace(index=index, connect=Mock(), vector_store=Mock())
