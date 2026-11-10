"""cl100k_base counts approximate embedding tokens; no embedding model is loaded."""
from functools import lru_cache
import tiktoken


@lru_cache(maxsize=1)
def tokenizer():
    # tiktoken downloads/caches its public encoding asset on first use.
    return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, disallowed_special=()))
