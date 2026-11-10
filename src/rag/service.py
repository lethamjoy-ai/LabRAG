"""UI-free entry point for the existing chat graph. No evaluation changes."""
from threading import Lock


def answer_question(question, history, retriever, llm, settings, strategy=None,
                    top_k=None, reranker=None, candidate_count=15,
                    answer_generation_mode='grounded_conversational_v2'):
    from src.rag.rag_graph import build_rag_graph
    graph = build_rag_graph(retriever, llm, settings, strategy, top_k,
        reranker=reranker, candidate_count=candidate_count,
        answer_generation_mode=answer_generation_mode)
    return graph.invoke({'question': question, 'history': history})


class LocalRAGService:
    """Lazy, reusable clients; serialize requests because provider usage is mutable.

    Each API request is an independent conversation. Initialization never writes
    to Pinecone. Failed initialization is retried on the next explicit request.
    """
    def __init__(self):
        self._lock = Lock()
        self._clients = None
        self.runtime = None

    def query(self, question):
        with self._lock:
            if self._clients is None:
                from config.settings import Settings
                from src.vectorstore.pinecone_client import PineconeConnection
                from src.embeddings.embedding_service import get_configured_embedding_service
                from src.retrieval.semantic_retriever import SemanticRetriever
                from src.llm.provider_factory import create_llm,validate_credentials
                settings = Settings.from_env()
                safe_model=settings.llm_model
                if any(secret and secret in safe_model for secret in
                    (settings.openai_api_key,settings.groq_api_key,settings.pinecone_api_key,settings.langsmith_api_key)):
                    safe_model='[redacted]'
                self.runtime={'provider':settings.llm_provider,'model':safe_model}
                validate_credentials(settings)
                connection = PineconeConnection(settings).connect()
                embeddings = get_configured_embedding_service(settings)
                self._clients = (settings, SemanticRetriever(settings, embeddings, connection), create_llm(settings))
            settings, retriever, llm = self._clients
            result=answer_question(question, [], retriever, llm, settings)
            return {**result,'llm_provider':self.runtime['provider'],'llm_model':self.runtime['model']}
