import pytest
from config.settings import Settings


def test_defaults_and_environment(monkeypatch):
    settings = Settings()
    assert (settings.chunk_size, settings.chunk_overlap, settings.section_max_tokens) == (500, 75, 500)
    monkeypatch.setenv("CHUNK_SIZE", "600")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("SECTION_MAX_TOKENS", "700")
    configured = Settings.from_env()
    assert (configured.chunk_size, configured.chunk_overlap, configured.section_max_tokens) == (600, 50, 700)


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"chunk_overlap": -1}, {"chunk_overlap": 500},
                                    {"section_max_tokens": 50}, {"chunk_size": True}])
def test_invalid_chunk_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
