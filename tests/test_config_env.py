from utils import load_config


def test_embedding_api_key_can_be_configured_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_API_KEY", "deepseek-key")
    monkeypatch.setenv("OMBRE_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OMBRE_EMBEDDING_API_KEY", "gemini-key")
    monkeypatch.setenv(
        "OMBRE_EMBEDDING_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    monkeypatch.setenv("OMBRE_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))

    config = load_config(config_path=str(tmp_path / "missing.yaml"))

    assert config["dehydration"]["api_key"] == "deepseek-key"
    assert config["dehydration"]["base_url"] == "https://api.deepseek.com/v1"
    assert config["embedding"]["api_key"] == "gemini-key"
    assert (
        config["embedding"]["base_url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert config["embedding"]["model"] == "gemini-embedding-001"


def test_embedding_api_key_falls_back_to_dehydration_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_API_KEY", "shared-key")
    monkeypatch.delenv("OMBRE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))

    config = load_config(config_path=str(tmp_path / "missing.yaml"))

    assert "api_key" not in config.get("embedding", {})
    assert config["dehydration"]["api_key"] == "shared-key"
