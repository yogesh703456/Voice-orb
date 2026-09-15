"""
core/llm_config.py

Centralized configuration for the optional LLM layer.

Rules this module enforces for the whole project:
  * Every LLM knob lives in one place: config/settings.yaml under `llm:`.
  * API keys NEVER live in source or settings files -- they are read from
    the environment (optionally seeded from a git-ignored .env file).
  * Bad configuration fails loudly and early, with a message that says how
    to fix it -- the assistant must never half-configure itself.

The LLM layer is entirely optional: when `llm.enabled` is false, or no API
key is present, the assistant behaves exactly as it did before (regex
intents + executor only). Nothing else in the app is allowed to read LLM
settings from anywhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_PROVIDERS = ("ollama", "anthropic", "openai-compatible")

# Default Ollama endpoint. Ollama's own installer binds here; the value is
# overridable per machine via llm.base_url in settings.yaml.
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# Env vars consulted for API keys, per provider. Keys always come from the
# environment (optionally seeded by a git-ignored .env file) -- never from
# settings.yaml, source code, or command-line arguments.
# Ollama: a vanilla localhost install needs NO key; OLLAMA_API_KEY is read
# only so key-secured Ollama deployments (reverse proxies etc.) work too.
_API_KEY_ENV_VARS = {
    "ollama": ("OLLAMA_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY"),
}


@dataclass(frozen=True)
class LLMConfig:
    """Fully validated LLM settings. Construct via load_llm_config()."""

    provider: str = "ollama"    # local-first default: the local runtime wins
    model: str = ""
    api_key: str = ""
    base_url: str = ""          # ollama/openai-compatible; empty = provider default
    temperature: float = 0.0
    timeout_sec: float = 12.0
    max_tokens: int = 300
    keep_alive_sec: int = 3600  # ollama only: how long the model stays resident (-1 = always)
    think: bool = False         # ollama only: enable the model's reasoning mode (slow on CPU)
    api_key_env_var: str = ""   # which env var the key came from (for logs/diagnostics)


def load_env_file(path: Path | str | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ (setdefault
    semantics: real environment variables always win over .env values).

    Deliberately tiny and stdlib-only -- no python-dotenv dependency.
    Only complete KEY=VALUE lines are recognized; comments (#) and blank
    lines are skipped. Values are never logged or returned into settings
    files; they land solely in the process environment.

    Returns the dict of pairs that were actually applied (for tests).
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    applied: dict[str, str] = {}

    if not env_path.is_file():
        return applied

    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[LLM] Could not read .env file ({exc}); ignoring it.")
        return applied

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied


def _require_number(section: dict, key: str, default: float,
                    minimum: float, maximum: float) -> float:
    value = section.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"llm.{key} must be a number, got {value!r}"
        ) from None
    if not minimum <= value <= maximum:
        raise ValueError(
            f"llm.{key} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def load_llm_config(section: object) -> LLMConfig:
    """Validate the `llm:` mapping from settings.yaml into an LLMConfig.

    Raises ValueError with an actionable message on any invalid value.
    A missing/None section is valid and yields defaults (the brain builder
    decides separately whether the layer is usable).
    """
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(
            "config/settings.yaml: the 'llm:' section must be a mapping "
            f"(key: value lines), got {type(section).__name__}."
        )

    provider = str(section.get("provider") or "ollama").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"llm.provider must be one of {', '.join(SUPPORTED_PROVIDERS)}, "
            f"got {provider!r}."
        )

    model = str(section.get("model") or "").strip()
    if not model:
        # Sensible per-provider defaults so a bare `llm: {enabled: true}`
        # section still works; explicit model in settings always wins.
        # For ollama, an empty model means "auto-discover the installed
        # model(s) at connection time" -- the provider resolves it, so the
        # settings file never hardcodes a machine-specific model name.
        if provider == "anthropic":
            model = "claude-sonnet-4-5"
        elif provider == "openai-compatible":
            model = "gpt-4o-mini"

    base_url = str(section.get("base_url") or "").strip()
    if provider in ("ollama", "openai-compatible") and base_url \
            and not base_url.startswith(("http://", "https://")):
        raise ValueError(
            f"llm.base_url must start with http:// or https://, got {base_url!r}."
        )

    # Ollama-specific knobs. keep_alive_sec: how long the model stays
    # loaded in the Ollama server between requests (-1 = forever). think:
    # the model's reasoning mode -- powerful but SLOW on CPU, and a voice
    # assistant needs sub-10s answers, so it defaults to off.
    keep_alive_raw = section.get("keep_alive_sec", 3600)
    try:
        keep_alive_sec = int(keep_alive_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"llm.keep_alive_sec must be an integer (-1 = keep forever), got {keep_alive_raw!r}"
        ) from None
    if not -1 <= keep_alive_sec <= 604800:
        raise ValueError(
            "llm.keep_alive_sec must be between -1 and 604800 (one week), "
            f"got {keep_alive_sec}"
        )

    think_raw = section.get("think", False)
    if isinstance(think_raw, str):
        think = think_raw.strip().lower() in ("true", "yes", "on", "1")
    else:
        think = bool(think_raw)

    temperature = _require_number(section, "temperature", 0.0, 0.0, 2.0)
    timeout_sec = _require_number(section, "timeout_sec", 12.0, 1.0, 120.0)

    max_tokens_raw = section.get("max_tokens", 300)
    try:
        max_tokens = int(max_tokens_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"llm.max_tokens must be an integer, got {max_tokens_raw!r}"
        ) from None
    if not 16 <= max_tokens <= 4096:
        raise ValueError(
            f"llm.max_tokens must be between 16 and 4096, got {max_tokens}"
        )

    # The key: environment only. Which env var to consult can itself be
    # configured (api_key_env), defaulting to the provider's standard name.
    api_key_env_var = str(section.get("api_key_env") or "").strip() or (
        _API_KEY_ENV_VARS[provider][0]
    )
    api_key = os.environ.get(api_key_env_var, "")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
        keep_alive_sec=keep_alive_sec,
        think=think,
        api_key_env_var=api_key_env_var,
    )
