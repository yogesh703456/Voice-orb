"""Tests for core/llm_brain.py -- strict parsing of LLM decisions."""

import pytest

from core.llm_brain import (
    LLMBrain,
    LLMParseError,
    LLMDecision,
    build_system_prompt,
    parse_decision,
)


def test_parse_valid_command_decision():
    decision = parse_decision(
        '{"action": "command", "intent": "open_app", "app": "chrome"}'
    )
    assert decision.action == "command"
    assert decision.intent == "open_app"
    assert decision.app == "chrome"


def test_parse_chat_decision():
    decision = parse_decision(
        '{"action": "chat", "response": "Polymorphism lets objects take many forms."}'
    )
    assert decision.action == "chat"
    assert "Polymorphism" in decision.response


def test_parse_unknown_decision():
    decision = parse_decision('{"action": "unknown"}')
    assert decision.action == "unknown"


def test_number_fields_coerced_to_strings():
    decision = parse_decision('{"action": "command", "intent": "volume", "key": 50}')
    assert decision.key == "50"


def test_malformed_json_rejected():
    with pytest.raises(LLMParseError):
        parse_decision("this is not json at all")


def test_unbalanced_json_rejected():
    with pytest.raises(LLMParseError):
        parse_decision('{"action": "chat", "response": "oops')


def test_json_array_rejected():
    with pytest.raises(LLMParseError):
        parse_decision('[{"action": "chat"}]')


def test_unknown_action_rejected():
    with pytest.raises(LLMParseError):
        parse_decision('{"action": "destroy_everything"}')


def test_json_wrapped_in_prose_is_recovered():
    decision = parse_decision(
        'Sure! Here is my decision: {"action": "command", "intent": "open_app", "app": "notepad"}'
        " -- hope that helps."
    )
    assert decision.intent == "open_app"


def test_code_fenced_json_is_recovered():
    decision = parse_decision(
        '```json\n{"action": "chat", "response": "hello"}\n```'
    )
    assert decision.action == "chat"


def test_non_string_field_types_ignored_safely():
    decision = parse_decision(
        '{"action": "command", "intent": "open_app", "app": {"nested": "object"}}'
    )
    # A dict where a string was expected becomes an empty field, which the
    # validator then rejects as a missing parameter.
    assert decision.app == ""
    assert decision.action == "command"


def test_boolean_fields_ignored_safely():
    decision = parse_decision('{"action": "command", "intent": "open_app", "app": true}')
    assert decision.app == ""


def test_missing_fields_default_to_empty():
    decision = parse_decision('{"action": "command", "intent": "open_app"}')
    assert decision.app == ""


class _FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


def test_brain_hands_transcript_to_provider_and_parses():
    provider = _FakeProvider('{"action": "chat", "response": "Hi there."}')
    brain = LLMBrain(provider)
    decision = brain.decide("hello jarvis")

    assert decision.action == "chat"
    assert len(provider.calls) == 1
    _system, user = provider.calls[0]
    assert "hello jarvis" in user


def test_system_prompt_lists_real_capabilities():
    prompt = build_system_prompt()
    # The prompt teaches only intents the executor actually has.
    for required in ("open_app", "search_files", "volume", "battery", "delete_file"):
        assert required in prompt
    # ...and never advertises intents the executor must not expose to the LLM
    # (checked as intent-list entries, so prose mentions don't false-positive).
    for forbidden in ("- shutdown", "- restart", "- run_code", "- shell", "- reboot"):
        assert forbidden not in prompt
