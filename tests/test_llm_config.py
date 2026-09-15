"""Tests for core/llm_config.py -- validation, .env loading, no-secret rules."""

import pytest

from core.llm_config import LLMConfig, load_env_file, load_llm_config


def test_defaults_when_section_missing():
    config = load_llm_config(None)

    assert config.provider == "ollama"  # local-first default
    assert config.model == ""  # empty = auto-discover the installed model
    assert config.temperature == 0.0
    assert config.timeout_sec == 12.0
    assert config.keep_alive_sec == 3600
    assert config.think is False


def test_invalid_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        load_llm_config({"provider": "skynet"})


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        load_llm_config({"temperature": 9.5})


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError, match="timeout_sec"):
        load_llm_config({"timeout_sec": 0.1})


def test_invalid_max_tokens_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        load_llm_config({"max_tokens": "lots"})


def test_base_url_requires_scheme():
    with pytest.raises(ValueError, match="base_url"):
        load_llm_config(
            {"provider": "openai-compatible", "base_url": "localhost:1234"}
        )


def test_api_key_never_comes_from_settings(monkeypatch):
    """A key pasted into settings.yaml must be ignored entirely."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_llm_config(
        {"provider": "anthropic", "api_key": "sk-ant-SHOULD-BE-IGNORED"}
    )
    assert config.api_key == ""


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    config = load_llm_config({"provider": "anthropic"})
    assert config.api_key == "test-key-123"


def test_custom_api_key_env_var(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_KEY", "custom-key")
    config = load_llm_config(
        {"provider": "anthropic", "api_key_env": "MY_CUSTOM_KEY"}
    )
    assert config.api_key == "custom-key"
    assert config.api_key_env_var == "MY_CUSTOM_KEY"


def test_local_provider_needs_no_key(monkeypatch):
    """A stock local Ollama install requires no API key."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config = load_llm_config({"provider": "ollama"})
    assert config.api_key == ""  # that's fine -- local has no auth
    assert config.base_url == ""  # empty = Ollama's default endpoint


def test_keep_alive_and_think_are_read():
    config = load_llm_config(
        {"provider": "ollama", "keep_alive_sec": -1, "think": True}
    )
    assert config.keep_alive_sec == -1
    assert config.think is True


def test_invalid_keep_alive_rejected():
    with pytest.raises(ValueError, match="keep_alive_sec"):
        load_llm_config({"provider": "ollama", "keep_alive_sec": "forever"})


def test_invalid_base_url_scheme_rejected_for_ollama():
    with pytest.raises(ValueError, match="base_url"):
        load_llm_config({"provider": "ollama", "base_url": "127.0.0.1:11434"})


def test_load_env_file_applies_missing_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nTEST_LLM_KEY_ABC="abc-value"\nBROKEN LINE\n\n'
        "EMPTY_KEY=\nTEST_LLM_KEY_XYZ=xyz-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_LLM_KEY_ABC", raising=False)
    monkeypatch.delenv("TEST_LLM_KEY_XYZ", raising=False)
    monkeypatch.setenv("EMPTY_KEY", "already-set")

    applied = load_env_file(env_file)

    assert applied["TEST_LLM_KEY_ABC"] == "abc-value"
    assert applied["TEST_LLM_KEY_XYZ"] == "xyz-value"
    assert set(applied) == {"TEST_LLM_KEY_ABC", "TEST_LLM_KEY_XYZ"}
    import os

    assert os.environ["TEST_LLM_KEY_ABC"] == "abc-value"
    # Existing environment wins over .env (setdefault semantics).
    assert os.environ["EMPTY_KEY"] == "already-set"


def test_load_env_file_missing_file_is_noop(tmp_path):
    assert load_env_file(tmp_path / "does-not-exist.env") == {}
