"""
core/llm_service.py

Process-wide LLM service wiring: config -> provider -> brain, built at
most once (never per command -- rule 15/16 of the architecture), with an
honest availability probe the pipeline can consult.

The service is *optional by design*: if it is not configured (no API key,
llm.enabled false) every caller must degrade to the pre-LLM behavior, and
main.py does exactly that.
"""

from __future__ import annotations

import threading
import time

from core import llm_config
from core.llm_brain import LLMBrain, LLMDecision
from core.llm_provider import (
    LLMError,
    LLMNotConfiguredError,
    create_provider,
)

_lock = threading.Lock()
_service: "LLMService | None" = None


class LLMService:
    """Owns the single LLMBrain instance for the process."""

    def __init__(self, brain: LLMBrain | None, status: str) -> None:
        self.brain = brain
        self.status = status  # human-readable why-not when brain is None
        self.last_latency_sec: float | None = None
        self.last_error_kind: str | None = None

    @property
    def available(self) -> bool:
        return self.brain is not None

    def decide(self, transcript: str) -> LLMDecision:
        """Ask the brain; record latency and the failure kind for the
        pipeline's fallback decision and diagnostics."""
        if self.brain is None:
            raise LLMNotConfiguredError(self.status or "LLM not configured.")

        started = time.perf_counter()
        try:
            decision = self.brain.decide(transcript)
        except LLMError as exc:
            self.last_error_kind = exc.kind
            raise
        finally:
            self.last_latency_sec = time.perf_counter() - started
        self.last_error_kind = None
        return decision

    def describe(self) -> str:
        if self.brain is None:
            return f"LLM disabled ({self.status})"
        return f"LLM ready (provider={self.brain._provider.name})"


def build_service(llm_settings: object) -> LLMService:
    """Build an LLMService from the settings.yaml `llm:` section.
    Never raises for "not configured" -- it returns an unavailable service
    with a reason, so the app keeps working without the LLM. Raises only
    for *invalid* configuration the user should fix."""
    try:
        config = llm_config.load_llm_config(llm_settings)
    except ValueError as exc:
        raise ValueError(f"Invalid LLM configuration: {exc}") from None

    enabled_raw = (llm_settings or {}).get("enabled", True) if isinstance(llm_settings, dict) else True
    if not enabled_raw:
        return LLMService(None, "disabled in settings.yaml")

    try:
        provider = create_provider(config)
    except LLMNotConfiguredError as exc:
        return LLMService(None, str(exc))

    brain = LLMBrain(provider)
    service = LLMService(brain, "")
    return service


def get_service(settings: dict | None = None) -> LLMService:
    """Process-wide accessor. Builds once; later calls reuse the instance
    (an explicit settings dict on the first call wins)."""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                section = (settings or {}).get("llm") if settings else None
                _service = build_service(section)
    return _service


def reset_service() -> None:
    """Test/teardown hook: drop the process-wide instance."""
    global _service
    with _lock:
        _service = None
