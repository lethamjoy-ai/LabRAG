import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class Settings:
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    openai_llm_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1200
    rag_default_strategy: str = "section"
    rag_max_history_turns: int = 5
    rag_enable_tracing: bool = False
    langsmith_tracing: bool = False
    langsmith_capture_full_io: bool = False
    langsmith_project: str = 'LabRAG'
    langsmith_api_key: str = field(default='', repr=False)
    pinecone_index_name: str = "lab-rag"
    namespace_fixed: str = "fixed-chunks"
    namespace_section: str = "section-chunks"
    max_file_bytes: int = 20 * 1024 * 1024
    chunk_size: int = 500
    chunk_overlap: int = 75
    section_max_tokens: int = 500
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_batch_size: int = 32
    pinecone_upsert_batch_size: int = 50
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0
    groq_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)

    def __post_init__(self):
        if self.llm_provider not in {"groq","openai"}:
            raise ValueError("LLM_PROVIDER must be groq or openai.")
        if not self.openai_llm_model.strip():raise ValueError('OPENAI_LLM_MODEL must be nonempty.')
        if not self.llm_model.strip():
            raise ValueError("LLM_MODEL must be nonempty.")
        if not math.isfinite(self.llm_temperature) or not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
        if not 128 <= self.llm_max_tokens <= 8192 or not 5 <= self.rag_max_history_turns <= 50:
            raise ValueError("LLM_MAX_TOKENS must be 128–8192; RAG_MAX_HISTORY_TURNS must be 5–50.")
        if self.rag_default_strategy not in {"fixed", "section"}:
            raise ValueError("RAG_DEFAULT_STRATEGY must be fixed or section.")
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be local or openai.")
        if self.embedding_provider == "openai":
            limits = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
            if self.embedding_model not in limits:
                raise ValueError("OpenAI embeddings require EMBEDDING_MODEL=text-embedding-3-small or text-embedding-3-large.")
            if not 1 <= self.embedding_dimension <= limits[self.embedding_model]:
                raise ValueError("Embedding dimensions exceed the selected OpenAI model's supported output dimensions.")
        elif self.embedding_model.startswith("text-embedding-"):
            raise ValueError("Set EMBEDDING_PROVIDER=openai for an OpenAI embedding model.")
        for value in (self.embedding_batch_size, self.pinecone_upsert_batch_size, self.retrieval_top_k):
            if type(value) is not int or value <= 0:
                raise ValueError("Embedding/upsert batch sizes and RETRIEVAL_TOP_K must be positive integers.")
        if self.embedding_batch_size > 512 or self.pinecone_upsert_batch_size > 1000 or self.retrieval_top_k > 100:
            raise ValueError("Embedding batch size must be <=512, upsert batch size <=1000, and retrieval top-k <=100.")
        if not math.isfinite(self.retrieval_score_threshold) or not -1 <= self.retrieval_score_threshold <= 1:
            raise ValueError("RETRIEVAL_SCORE_THRESHOLD must be between -1 and 1.")
        if not self.namespace_fixed or not self.namespace_section or self.namespace_fixed == self.namespace_section:
            raise ValueError("Fixed and section namespaces must be nonempty and distinct.")
        if any(type(value) is not int for value in (self.embedding_dimension, self.max_file_bytes,
                self.chunk_size, self.chunk_overlap, self.section_max_tokens)):
            raise ValueError("Dimension, file size, and chunk settings must be integers.")
        if self.chunk_size <= 0 or self.section_max_tokens <= 0 or not 0 <= self.chunk_overlap < min(self.chunk_size, self.section_max_tokens):
            raise ValueError("CHUNK_SIZE and SECTION_MAX_TOKENS must be positive; CHUNK_OVERLAP must be nonnegative and smaller than both.")
        if self.embedding_dimension <= 0 or self.max_file_bytes <= 0:
            raise ValueError("EMBEDDING_DIMENSION and MAX_FILE_BYTES must be positive integers.")
        if self.embedding_model == "sentence-transformers/all-MiniLM-L6-v2" and self.embedding_dimension != 384:
            raise ValueError("all-MiniLM-L6-v2 requires EMBEDDING_DIMENSION=384.")

    @property
    def warnings(self):
        messages = [f"{name} is not set; optional for local processing." for name, value in
                    [("GROQ_API_KEY", self.groq_api_key), ("PINECONE_API_KEY", self.pinecone_api_key)] if not value]
        if self.embedding_provider == "openai" and not self.openai_api_key.strip():
            messages.append("OPENAI_API_KEY is not set; required for OpenAI embeddings.")
        if self.embedding_provider == "local" and self.embedding_model != "sentence-transformers/all-MiniLM-L6-v2":
            messages.append("Custom embedding model dimension must be verified before future indexing.")
        return messages

    @classmethod
    def from_env(cls):
        # Read the project file on every rerun. load_dotenv() would leave a prior
        # empty key in os.environ and ignore a newly saved value by default.
        environment = dict(os.environ)
        if environment.get("PYTHON_DOTENV_DISABLED", "").lower() not in {"1", "true", "t", "yes", "y"}:
            environment.update({key: value for key, value in dotenv_values(ENV_FILE, encoding="utf-8-sig").items()
                                if value is not None})
        mapping = {"embedding_model": "EMBEDDING_MODEL", "embedding_dimension": "EMBEDDING_DIMENSION",
                   "llm_provider": "LLM_PROVIDER", "llm_model": "LLM_MODEL", "pinecone_index_name": "PINECONE_INDEX_NAME",
                   "namespace_fixed": "PINECONE_NAMESPACE_FIXED", "namespace_section": "PINECONE_NAMESPACE_SECTION",
                   "max_file_bytes": "MAX_FILE_BYTES", "groq_api_key": "GROQ_API_KEY", "pinecone_api_key": "PINECONE_API_KEY"}
        mapping.update(chunk_size="CHUNK_SIZE", chunk_overlap="CHUNK_OVERLAP", section_max_tokens="SECTION_MAX_TOKENS")
        mapping.update(embedding_provider="EMBEDDING_PROVIDER", openai_api_key="OPENAI_API_KEY")
        mapping.update(openai_llm_model='OPENAI_LLM_MODEL', openai_base_url='OPENAI_BASE_URL',
                       langsmith_endpoint='LANGSMITH_ENDPOINT')
        mapping.update(langsmith_api_key='LANGSMITH_API_KEY', langsmith_project='LANGSMITH_PROJECT',
                       langsmith_tracing='LANGSMITH_TRACING')
        mapping.update(langsmith_capture_full_io='LANGSMITH_CAPTURE_FULL_IO')
        mapping.update(llm_temperature="LLM_TEMPERATURE", llm_max_tokens="LLM_MAX_TOKENS",
            rag_default_strategy="RAG_DEFAULT_STRATEGY", rag_max_history_turns="RAG_MAX_HISTORY_TURNS",
            rag_enable_tracing="RAG_ENABLE_TRACING")
        mapping.update(pinecone_cloud="PINECONE_CLOUD", pinecone_region="PINECONE_REGION",
            embedding_batch_size="EMBEDDING_BATCH_SIZE", pinecone_upsert_batch_size="PINECONE_UPSERT_BATCH_SIZE",
            retrieval_top_k="RETRIEVAL_TOP_K", retrieval_score_threshold="RETRIEVAL_SCORE_THRESHOLD")
        values = {}
        for attr, env in mapping.items():
            value = environment.get(env)
            if value is not None:
                value = value.strip()
                if attr in {"embedding_dimension", "max_file_bytes", "chunk_size", "chunk_overlap", "section_max_tokens",
                            "embedding_batch_size", "pinecone_upsert_batch_size", "retrieval_top_k", "llm_max_tokens", "rag_max_history_turns"}:
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(f"{env} must be a positive integer (overlap may be zero).") from None
                if attr in {"retrieval_score_threshold", "llm_temperature"}:
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"{env} must be a number.") from None
                if attr in {"rag_enable_tracing", "langsmith_tracing", "langsmith_capture_full_io"}:
                    if value.lower() not in {"true", "false", "1", "0"}:
                        raise ValueError(f"{env} must be true or false.")
                    value = value.lower() in {"true", "1"}
                values[attr] = value
        if not environment.get('LLM_PROVIDER','').strip():values.pop('llm_provider',None)
        if not environment.get('LLM_MODEL','').strip():
            provider=values.get('llm_provider','openai')
            values['llm_model']=values.get('openai_llm_model','gpt-4.1-mini') if provider=='openai' else 'openai/gpt-oss-120b'
        for name in ("openai_base_url", "langsmith_endpoint"):
            if not values.get(name, "").strip(): values.pop(name, None)
        return cls(**values)
