"""Configuration with redacted secrets and actionable errors."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = field(repr=False)
    llm_api_key: str = field(repr=False)
    index_name: str = "labrag"
    namespace: str = "lab-reference-v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    provider: str = "groq"
    base_url: str | None = None
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def from_env(cls):
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        if provider not in {"groq", "openai"}:
            raise ConfigurationError("LLM_PROVIDER must be groq or openai.")
        key = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        required = ["PINECONE_API_KEY", key]
        missing = [name for name in required if not os.getenv(name, "").strip()]
        if missing:
            raise ConfigurationError("Set " + ", ".join(missing) + " in .env; copy .env.example first.")
        defaults = cls.__dataclass_fields__
        values = {}
        for attr, env in [("index_name", "PINECONE_INDEX_NAME"), ("namespace", "PINECONE_NAMESPACE"),
                          ("embedding_model", "EMBEDDING_MODEL"), ("llm_model", "LLM_MODEL"),
                          ("reranker_model", "RERANKER_MODEL")]:
            values[attr] = os.getenv(env, str(defaults[attr].default)).strip()
            if not values[attr]:
                raise ConfigurationError(f"{env} cannot be empty.")
        return cls(pinecone_api_key=os.environ["PINECONE_API_KEY"],
                   llm_api_key=os.environ[key], provider=provider,
                   base_url=os.getenv("OPENAI_BASE_URL") or None, **values)
