from unittest.mock import Mock
import pytest
from config.settings import Settings
from src.llm.groq_service import GroqService, LLMError


@pytest.mark.parametrize("status,code,expected", [
    (404, "model_not_found", "model is unavailable"),
    (400, "tool_use_failed", "structured tool response"),
    (400, "context_length_exceeded", "input limit"),
    (400, "invalid_request_error", "request format"),
    (401, "invalid_api_key", "authentication"),
    (429, "rate_limit_exceeded", "rate limit"),
])
def test_safe_actionable_provider_errors(status, code, expected):
    error = RuntimeError("SECRET KEY AND REQUEST CONTENT")
    error.status_code = status
    error.body = {"error": {"code": code, "message": "SECRET BODY"}}
    backend = Mock()
    backend.invoke.side_effect = error
    with pytest.raises(LLMError) as caught:
        GroqService(Settings(), backend).select([])
    assert expected in str(caught.value)
    assert "SECRET" not in str(caught.value)
