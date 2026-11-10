from unittest.mock import Mock
import pytest
from config.settings import Settings
from src.llm.groq_service import GroqService, LLMError


def test_lazy_and_injectable():
    service = GroqService(Settings())
    assert service.backend is None
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        service.select([])
    backend = Mock()
    backend.invoke.return_value = {"excerpts": [{"source_id": "S1", "quote": "Evidence"}]}
    assert GroqService(Settings(), backend).select([]).excerpts[0].quote == "Evidence"


def test_provider_errors_are_sanitized():
    backend = Mock()
    backend.invoke.side_effect = RuntimeError("SECRET-KEY")
    with pytest.raises(LLMError) as error:
        GroqService(Settings(), backend).select([])
    assert "SECRET" not in str(error.value)
