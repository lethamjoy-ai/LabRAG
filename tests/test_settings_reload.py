from config.settings import Settings


def test_dotenv_saved_key_replaces_stale_empty_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("PINECONE_API_KEY", "")
    path = tmp_path / ".env"
    monkeypatch.setattr("config.settings.ENV_FILE", path)
    path.write_text("PINECONE_API_KEY=\n", encoding="utf-8")
    assert Settings.from_env().pinecone_api_key == ""
    path.write_text("PINECONE_API_KEY=synthetic-test-key\n", encoding="utf-8")
    assert Settings.from_env().pinecone_api_key == "synthetic-test-key"
    path.write_text("PINECONE_API_KEY=updated-synthetic-key\n", encoding="utf-8")
    assert Settings.from_env().pinecone_api_key == "updated-synthetic-key"


def test_dotenv_path_independent_of_working_directory_and_bom(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    path = tmp_path / ".env"
    path.write_text("PINECONE_API_KEY=synthetic-test-key\n", encoding="utf-8-sig")
    monkeypatch.setattr("config.settings.ENV_FILE", path)
    child = tmp_path / "elsewhere"
    child.mkdir()
    monkeypatch.chdir(child)
    assert Settings.from_env().pinecone_api_key == "synthetic-test-key"
