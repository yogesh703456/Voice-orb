"""
core/llm_provider.py

Provider abstraction for the LLM layer.

Architecture:
    LLMError (base, with .kind)
      ├── LLMNotConfiguredError   no API key / bad settings
      ├── LLMNetworkError         no internet / DNS / connection refused
      ├── LLMTimeoutError         request exceeded the configured timeout
      ├── LLMAuthError            401/403 -- key rejected
      ├── LLMRateLimitError       429
      ├── LLMProviderError        any other 5xx/4xx from the provider
      └── LLMResponseError        response arrived but unusable (bad JSON/shape)

Two providers ship out of the box, both over stdlib urllib (no new
dependencies, no SDK pinned to one vendor):
  * OllamaProvider            -- the LOCAL runtime (default). Talks to Ollama's
                                  native /api/chat endpoint because it exposes
                                  two knobs the OpenAI-compatible layer lacks:
                                  `think` (the model's reasoning mode -- off by
                                  default, it is far too slow for voice on CPU)
                                  and `keep_alive` (keep the model resident).
  * AnthropicMessagesProvider -- api.anthropic.com /v1/messages
  * OpenAICompatibleProvider  -- any OpenAI-schema /chat/completions endpoint
                                  (OpenAI, Groq, OpenRouter, LM Studio, vLLM, ...)

All three implement the same tiny interface (ask()), so adding a provider
later (e.g. a direct llama.cpp binding) means implementing one method.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from core.llm_config import LLMConfig, OLLAMA_DEFAULT_BASE_URL


class LLMError(Exception):
    """Base class for every failure the LLM layer can produce."""

    kind = "error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.kind = type(self).kind


class LLMNotConfiguredError(LLMError):
    kind = "not_configured"


class LLMNetworkError(LLMError):
    kind = "network"


class LLMTimeoutError(LLMError):
    kind = "timeout"


class LLMAuthError(LLMError):
    kind = "auth"


class LLMRateLimitError(LLMError):
    kind = "rate_limit"


class LLMProviderError(LLMError):
    kind = "provider"


class LLMResponseError(LLMError):
    kind = "response"


class LLMProvider(Protocol):
    """The one method every provider must implement."""

    def ask(self, system_prompt: str, user_prompt: str) -> str: ...


_TIMEOUT_MAP = {
    # (urllib exception, mapped error kind)
    urllib.error.URLError: LLMNetworkError,
}


def _classify_http_error(status: int, body: str) -> LLMError:
    if status in (401, 403):
        return LLMAuthError(
            f"LLM API rejected the credentials (HTTP {status}). "
            f"Check that the API key environment variable is set to a valid key. {body[:200]}"
        )
    if status == 429:
        return LLMRateLimitError(f"LLM API rate limit hit (HTTP 429). {body[:200]}")
    if 500 <= status <= 599:
        return LLMProviderError(f"LLM provider server error (HTTP {status}). {body[:200]}")
    return LLMProviderError(f"LLM API returned HTTP {status}. {body[:200]}")


class OllamaProvider:
    """The user's LOCAL model, served by Ollama's native HTTP API.

    Deliberately NOT the OpenAI-compatibility endpoint: the native API is
    the only way to set `think` (qwen-class models otherwise burn hundreds
    of reasoning tokens -- tens of seconds on CPU -- before answering) and
    `keep_alive` (without it Ollama unloads the model after ~5 idle
    minutes and every request pays the cold-load again).

    No API key is required for a stock local install. If OLLAMA_API_KEY is
    present in the environment it is sent as a Bearer header, which covers
    key-secured Ollama proxies without special-casing anything.

    The model name resolves lazily on first ask(): an empty llm.model in
    settings.yaml means "use whatever is installed" when Ollama hosts
    exactly one model, and an explicit error listing the choices when it
    hosts several (never guess). Resolution happens once per process; the
    resolved name is cached for the process lifetime.
    """

    name = "ollama"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self._api_key = config.api_key
        self._resolved_model: str | None = config.model or None

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model(self) -> str:
        """Return the configured model, or auto-discover the single
        installed one. Raises LLMNotConfiguredError with actionable text
        when discovery is ambiguous or Ollama has nothing installed."""
        if self._resolved_model:
            return self._resolved_model

        request = urllib.request.Request(self._base_url + "/api/tags", method="GET")
        raw = _execute_request(
            request,
            min(self._config.timeout_sec, 10.0),
            _extract_ollama_models_json,
        )
        try:
            models = json.loads(raw)
        except json.JSONDecodeError:
            models = []
        names = [
            str(m.get("name") or "").strip()
            for m in models
            if isinstance(m, dict) and str(m.get("name") or "").strip()
        ]
        if not names:
            raise LLMNotConfiguredError(
                "Ollama is reachable but has no models installed. "
                "Install one (e.g. `ollama pull <model>`) or set llm.model "
                "in config/settings.yaml."
            )
        if len(names) > 1:
            raise LLMNotConfiguredError(
                "Several Ollama models are installed ("
                + ", ".join(names)
                + "). Set llm.model in config/settings.yaml to exactly one."
            )
        self._resolved_model = names[0]
        return self._resolved_model

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._resolve_model(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Reasoning mode off by default: measured ~68s of thinking on
            # CPU before a trivial answer. Opt in via llm.think.
            "think": self._config.think,
            # Keep the model resident so warm requests stay fast.
            "keep_alive": self._config.keep_alive_sec,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            url=self._base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        return _execute_request(request, self._config.timeout_sec, _extract_ollama_text)


class AnthropicMessagesProvider:
    """Claude via api.anthropic.com's /v1/messages endpoint."""

    name = "anthropic"

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise LLMNotConfiguredError(
                f"No API key found. Set the {config.api_key_env_var} "
                "environment variable (or put it in a git-ignored .env file)."
            )
        self._config = config

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._config.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        return _execute_request(request, self._config.timeout_sec, _extract_anthropic_text)


class OpenAICompatibleProvider:
    """Any endpoint speaking the OpenAI /chat/completions schema."""

    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise LLMNotConfiguredError(
                f"No API key found. Set the {config.api_key_env_var} "
                "environment variable (or put it in a git-ignored .env file)."
            )
        self._config = config
        base = config.base_url or "https://api.openai.com/v1"
        self._url = base.rstrip("/") + "/chat/completions"

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
            method="POST",
        )
        return _execute_request(request, self._config.timeout_sec, _extract_openai_text)


def _execute_request(request: urllib.request.Request, timeout_sec: float,
                     text_extractor) -> str:
    """Send the request, classify every failure mode, extract the text."""
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        # Never propagate the request headers (which contain the key).
        raise _classify_http_error(exc.code, error_body) from None
    except TimeoutError as exc:
        raise LLMTimeoutError(
            f"LLM request timed out after {timeout_sec:.0f}s."
        ) from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise LLMTimeoutError(
                f"LLM request timed out after {timeout_sec:.0f}s."
            ) from None
        raise LLMNetworkError(
            f"Cannot reach the LLM provider: {reason}"
        ) from None
    except OSError as exc:
        raise LLMNetworkError(f"Cannot reach the LLM provider: {exc}") from None

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMResponseError(f"LLM returned non-JSON data: {exc}") from None

    if not isinstance(data, dict):
        raise LLMResponseError("LLM JSON response was not an object.")

    if data.get("error"):
        message = str(data["error"].get("message", "")) if isinstance(data["error"], dict) else str(data["error"])
        raise LLMProviderError(f"LLM provider reported an error: {message[:300]}")

    try:
        text = text_extractor(data)
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(
            f"LLM response did not have the expected shape ({exc})."
        ) from None

    if not text or not text.strip():
        raise LLMResponseError("LLM returned an empty response.")

    return text.strip()


def _extract_ollama_models_json(data: dict) -> str:
    """Extractor for GET /api/tags: re-serialize the models list so the
    caller can decode it. (Keeps _execute_request's single text contract.)"""
    return json.dumps(data.get("models") or [])


def _extract_ollama_text(data: dict) -> str:
    message = data["message"]
    if not isinstance(message, dict):
        raise TypeError("message is not an object")
    return str(message.get("content") or "")


def _extract_anthropic_text(data: dict) -> str:
    content = data["content"]
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _extract_openai_text(data: dict) -> str:
    return data["choices"][0]["message"]["content"]


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory: map a validated config onto the right provider class."""
    if config.provider == "ollama":
        return OllamaProvider(config)
    if config.provider == "anthropic":
        return AnthropicMessagesProvider(config)
    if config.provider == "openai-compatible":
        return OpenAICompatibleProvider(config)
    raise LLMNotConfiguredError(f"Unknown provider: {config.provider!r}")
