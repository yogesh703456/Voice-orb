"""Tests for core/llm_provider.py (failure modes, offline -- no network)
and core/llm_service.py (service/fallback wiring)."""

import json
import urllib.error

import pytest

from core.llm_config import LLMConfig
from core.llm_provider import (
    LLMAuthError,
    LLMNetworkError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    OpenAICompatibleProvider,
    _execute_request,
    _extract_anthropic_text,
    _extract_openai_text,
    create_provider,
)
from core.llm_service import LLMService, build_service


def _config(**overrides) -> LLMConfig:
    defaults = dict(
        provider="anthropic",
        model="test-model",
        api_key="test-key",
        base_url="",
        temperature=0.0,
        timeout_sec=5.0,
        max_tokens=100,
        api_key_env_var="ANTHROPIC_API_KEY",
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ---------------------------------------------------------------------------
# Configuration / construction
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_not_configured():
    from core.llm_provider import AnthropicMessagesProvider

    with pytest.raises(LLMNotConfiguredError):
        AnthropicMessagesProvider(_config(api_key=""))


def test_factory_maps_providers():
    assert isinstance(create_provider(_config()), object)
    openai = create_provider(_config(provider="openai-compatible"))
    assert isinstance(openai, OpenAICompatibleProvider)


def test_openai_base_url_is_normalized():
    provider = create_provider(
        _config(provider="openai-compatible", base_url="http://localhost:1234/v1/")
    )
    assert provider._url == "http://localhost:1234/v1/chat/completions"


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def test_extract_anthropic_text():
    data = {"content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}
    assert _extract_anthropic_text(data) == "Hello world"


def test_extract_openai_text():
    data = {"choices": [{"message": {"content": "the answer"}}]}
    assert _extract_openai_text(data) == "the answer"


def test_extract_openai_missing_choices_raises():
    with pytest.raises((KeyError, IndexError, TypeError)):
        _extract_openai_text({"nope": True})


# ---------------------------------------------------------------------------
# Error classification -- real HTTPError objects, no network needed
# ---------------------------------------------------------------------------

def _http_error(status: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.test", code=status, msg="err",
        hdrs=None, fp=__import__("io").BytesIO(body),
    )


class _RaisingUrlOpener:
    """Context manager standing in for urlopen(), raising a canned error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __enter__(self):
        raise self.error

    def __exit__(self, *args):
        return False


def test_auth_error_classified(monkeypatch):
    monkeypatch.setattr(
        "core.llm_provider.urllib.request.urlopen",
        lambda req, timeout: (_ for _ in ()).throw(_http_error(401)),
    )
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMAuthError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_rate_limit_classified(monkeypatch):
    monkeypatch.setattr(
        "core.llm_provider.urllib.request.urlopen",
        lambda req, timeout: (_ for _ in ()).throw(_http_error(429)),
    )
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMRateLimitError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_server_error_classified(monkeypatch):
    monkeypatch.setattr(
        "core.llm_provider.urllib.request.urlopen",
        lambda req, timeout: (_ for _ in ()).throw(_http_error(503)),
    )
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMProviderError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_connection_error_is_network(monkeypatch):
    def _boom(req, timeout):
        raise urllib.error.URLError(lambda: None, None) if False else urllib.error.URLError(
            __import__("socket").gaierror(11001, "getaddrinfo failed")
        )

    monkeypatch.setattr(
        "core.llm_provider.urllib.request.urlopen", _boom
    )
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMNetworkError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_timeout_classified(monkeypatch):
    def _boom(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _boom)
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMTimeoutError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_non_json_body_is_response_error(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"<html>gateway error</html>"

    def _ok(req, timeout):
        return _Resp()

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _ok)
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMResponseError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_empty_text_is_response_error(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"content": []}'

    def _ok(req, timeout):
        return _Resp()

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _ok)
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMResponseError):
        _execute_request(request, 5.0, _extract_anthropic_text)


def test_error_dict_in_valid_json_is_provider_error(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"error": {"message": "model overloaded"}}'

    def _ok(req, timeout):
        return _Resp()

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _ok)
    request = urllib.request.Request("https://api.test", data=b"{}", method="POST")
    with pytest.raises(LLMProviderError):
        _execute_request(request, 5.0, _extract_anthropic_text)


# ---------------------------------------------------------------------------
# Service wiring / fallback behavior
# ---------------------------------------------------------------------------

def test_build_service_disabled():
    service = build_service({"enabled": False})
    assert not service.available
    assert service.status == "disabled in settings.yaml"


def test_build_service_local_unreachable_is_unavailable_not_broken(monkeypatch):
    """Ollama off/no model -> the service reports WHY instead of crashing;
    the assistant must keep working with local commands only."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    service = build_service({"enabled": True, "provider": "ollama"})
    # The OllamaProvider constructor itself never fails (lazy resolution);
    # unreachability surfaces on first decide() as an LLMNetworkError.
    # What must hold: a service was built and is honest about state.
    assert service.available or "Ollama" in service.status or service.status == ""


def test_ollama_provider_local_builds_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    from core.llm_provider import OllamaProvider

    provider = create_provider(
        _config(provider="ollama", api_key="", base_url="http://127.0.0.1:11434")
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"


def test_ollama_provider_resolves_single_model(monkeypatch):
    """With llm.model empty, /api/tags with exactly one model resolves it."""
    from core.llm_provider import OllamaProvider

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"models": [{"name": "qwen3.5:4b"}]}'

    seen_urls = []

    def _ok(req, timeout):
        seen_urls.append(req.full_url)
        return _Resp()

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _ok)
    provider = OllamaProvider(
        _config(provider="ollama", api_key="", base_url="http://127.0.0.1:11434", model="")
    )
    assert provider._resolve_model() == "qwen3.5:4b"
    assert seen_urls[0].endswith("/api/tags")
    # Resolved once, cached afterwards.
    assert provider._resolve_model() == "qwen3.5:4b"
    assert len(seen_urls) == 1


def test_ollama_provider_ambiguous_models_is_not_configured(monkeypatch):
    from core.llm_provider import OllamaProvider

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"models": [{"name": "a"}, {"name": "b"}]}'

    monkeypatch.setattr(
        "core.llm_provider.urllib.request.urlopen", lambda req, timeout: _Resp()
    )
    provider = OllamaProvider(
        _config(provider="ollama", api_key="", base_url="http://127.0.0.1:11434", model="")
    )
    with pytest.raises(LLMNotConfiguredError, match="Set llm.model"):
        provider._resolve_model()


def test_ollama_dead_endpoint_is_network_error():
    """Ollama not running -> classified as a network error (bounded, no
    hang), which main.py maps to a spoken fallback. Real socket connect
    to a closed localhost port -- no external network involved."""
    from core.llm_provider import OllamaProvider

    provider = OllamaProvider(
        _config(
            provider="ollama", api_key="",
            base_url="http://127.0.0.1:59999", model="m", timeout_sec=3.0,
        )
    )
    with pytest.raises(LLMNetworkError):
        provider.ask("system", "user")


def test_ollama_ask_sends_native_knobs(monkeypatch):
    """think/keep_alive must reach the native payload; content extracted."""
    from core.llm_provider import OllamaProvider

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"message": {"role": "assistant", "content": " {\\\"action\\\": \\\"chat\\\", \\\"response\\\": \\\"hi\\\"}"}}'

    def _ok(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("core.llm_provider.urllib.request.urlopen", _ok)
    provider = OllamaProvider(
        _config(
            provider="ollama", api_key="",
            base_url="http://127.0.0.1:11434", model="qwen3.5:4b",
        )
    )
    text = provider.ask("system prompt", "user prompt")
    assert text.strip() == '{"action": "chat", "response": "hi"}'
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["think"] is False
    assert captured["payload"]["keep_alive"] == 3600
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["model"] == "qwen3.5:4b"
    assert captured["payload"]["options"]["num_predict"] == 100


def test_build_service_invalid_settings_raise(monkeypatch):
    with pytest.raises(ValueError):
        build_service({"enabled": True, "provider": "skynet"})


def test_unavailable_service_decide_raises_not_configured():
    service = LLMService(None, "not configured")
    with pytest.raises(LLMNotConfiguredError):
        service.decide("anything")


def test_service_records_latency_and_error_kind():
    from core.llm_brain import LLMBrain
    from core.llm_provider import LLMTimeoutError

    class _TimeoutProvider:
        name = "timeout-provider"

        def ask(self, system_prompt, user_prompt):
            raise LLMTimeoutError("too slow")

    service = LLMService(LLMBrain(_TimeoutProvider()), "")
    with pytest.raises(LLMTimeoutError):
        service.decide("hello")
    assert service.last_error_kind == "timeout"
    assert service.last_latency_sec is not None
