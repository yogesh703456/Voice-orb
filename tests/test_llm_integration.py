"""End-to-end integration tests (offline): LLM decision -> validator ->
ParsedIntent -> the real executor, plus the pipeline's fallback mapping.

No network access happens anywhere in this file: provider responses are
faked at the ask() boundary, exactly where the real HTTP call would be.
"""

import pytest

from core.executor import execute
from core.intent import IntentType, parse_commands
from core.llm_brain import LLMBrain, LLMDecision, LLMParseError
from core.llm_provider import LLMError, LLMNotConfiguredError, LLMTimeoutError
from core.llm_service import LLMService
from core.llm_validator import LLMValidationError, validate_decision


class _ScriptedProvider:
    """Returns canned replies keyed by the user prompt it was given."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[str] = []

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        return self.replies[user_prompt]


def _service_with(replies: dict[str, str]) -> LLMService:
    return LLMService(LLMBrain(_ScriptedProvider(replies)), "")


# ---------------------------------------------------------------------------
# Known commands still take the fast path (LLM never consulted)
# ---------------------------------------------------------------------------

def test_known_command_does_not_need_llm():
    intents = parse_commands("open chrome")
    assert len(intents) == 1
    assert intents[0].type is IntentType.LAUNCH_APP


def test_unknown_command_yields_single_unknown_intent():
    # Something the regex parser genuinely cannot interpret -- a complex,
    # conversational request with no command verb it knows. ("find the
    # python project I was working on yesterday" DOES parse as
    # SEARCH_FILES -- verified -- so it is not a valid LLM-trigger test.)
    intents = parse_commands("please help me figure out what to make for dinner tonight")
    assert len(intents) == 1
    assert intents[0].type is IntentType.UNKNOWN


# ---------------------------------------------------------------------------
# Full decision -> intent -> executor path
# ---------------------------------------------------------------------------

def test_llm_command_intent_executes_search(capsys):
    decision = validate_decision(
        LLMDecision(action="command", intent="search_files", query="dbms notes")
    )
    result = execute(decision)
    # Executor ran with its own real file search (empty index -> not found
    # response is fine; what matters is the intent was accepted).
    assert isinstance(result.spoken_response, str)
    assert result.spoken_response


def test_llm_volume_command_executes():
    decision = validate_decision(LLMDecision(action="command", intent="volume", key="up"))
    result = execute(decision)
    assert result.success
    assert "olume" in result.spoken_response


def test_llm_time_command_executes():
    decision = validate_decision(LLMDecision(action="command", intent="time"))
    result = execute(decision)
    assert result.success


def test_llm_battery_graceful_without_psutil_change():
    decision = validate_decision(LLMDecision(action="command", intent="battery"))
    result = execute(decision)
    # Runs either way; must simply produce a spoken response.
    assert result.spoken_response


# ---------------------------------------------------------------------------
# _consult_llm pipeline behavior (imported from main.py)
# ---------------------------------------------------------------------------

def _consult(transcript: str, service: LLMService):
    from main import _consult_llm

    return _consult_llm(transcript, service)


def test_consult_chat_returns_spoken_response():
    service = _service_with({"User said: why is the sky blue": '{"action": "chat", "response": "Rayleigh scattering."}'})
    outcome = _consult("why is the sky blue", service)
    assert outcome is not None
    assert outcome.chat_response == "Rayleigh scattering."
    assert outcome.validated_intent is None


def test_consult_command_returns_validated_intent():
    service = _service_with({
        "User said: find my python project": '{"action": "command", "intent": "search_files", "query": "python project"}'
    })
    outcome = _consult("find my python project", service)
    assert outcome is not None
    assert outcome.validated_intent is not None
    assert outcome.validated_intent.type is IntentType.SEARCH_FILES


def test_consult_unknown_returns_none():
    service = _service_with({"User said: gibberish": '{"action": "unknown"}'})
    assert _consult("gibberish", service) is None


def test_consult_timeout_yields_fallback_response():
    class _Timeout:
        name = "timeout"

        def ask(self, system, user):
            raise LLMTimeoutError("too slow")

    outcome = _consult("anything", LLMService(LLMBrain(_Timeout()), ""))
    assert outcome is not None
    assert "too slow" in outcome.chat_response.lower() or "slow" in outcome.chat_response.lower()


def test_consult_malformed_json_yields_fallback():
    service = _service_with({"User said: hello there": "I am sorry, I cannot help with that at all."})
    outcome = _consult("hello there", service)
    assert outcome is not None
    assert outcome.chat_response  # graceful spoken fallback, no crash
    assert outcome.validated_intent is None


def test_consult_validator_rejection_yields_safe_response():
    service = _service_with({
        "User said: do something dangerous": '{"action": "command", "intent": "shell", "query": "rm -rf /"}'
    })
    outcome = _consult("do something dangerous", service)
    assert outcome is not None
    assert "safely" in outcome.chat_response.lower()
    assert outcome.validated_intent is None


def test_consult_never_raises():
    """Whatever the provider does, _consult_llm must return, not raise."""

    class _Chaos:
        name = "chaos"

        def ask(self, system, user):
            raise RuntimeError("total chaos")

    outcome = _consult("anything", LLMService(LLMBrain(_Chaos()), ""))
    assert outcome is not None
    assert outcome.chat_response


# ---------------------------------------------------------------------------
# Timing/latency instrumentation sanity (perf requirement 22)
# ---------------------------------------------------------------------------

def test_service_latency_recorded_on_success():
    service = _service_with({"User said: hi": '{"action": "chat", "response": "hello"}'})
    _consult("hi", service)
    assert service.last_latency_sec is not None
    assert service.last_latency_sec >= 0


def test_service_latency_recorded_on_timeout():
    class _Slow:
        name = "slow"

        def ask(self, system, user):
            raise LLMTimeoutError("timeout")

    service = LLMService(LLMBrain(_Slow()), "")
    _consult("anything", service)
    assert service.last_latency_sec is not None
