"""Tests for core/llm_validator.py -- the security gate between the LLM
and the executor. Every attack vector in the architecture rules is here."""

import pytest

from core.intent import IntentType
from core.llm_brain import LLMDecision
from core.llm_validator import (
    LLMValidationError,
    validate_decision,
    validate_name,
    validate_text,
)


def _cmd(intent: str = "open_app", **fields) -> LLMDecision:
    base = {
        "action": "command",
        "intent": intent,
        "app": "",
        "query": "",
        "target_site": "",
        "location": "",
        "key": "",
        "response": "",
    }
    base.update(fields)
    return LLMDecision(**base)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_open_app_becomes_launch_app():
    intent = validate_decision(_cmd(intent="open_app", app="chrome"))
    assert intent.type is IntentType.LAUNCH_APP
    assert intent.query == "chrome"


def test_open_website_becomes_open_url():
    """'open youtube' is a website, not an app -- same rule the regex parser uses."""
    intent = validate_decision(_cmd(intent="open_app", app="youtube"))
    assert intent.type is IntentType.OPEN_URL
    assert intent.query == "https://www.youtube.com"


def test_search_files_valid_intent():
    intent = validate_decision(_cmd(intent="search_files", query="dbms notes"))
    assert intent.type is IntentType.SEARCH_FILES
    assert intent.query == "dbms notes"


def test_volume_number_and_words():
    assert validate_decision(_cmd(intent="volume", key="50")).query == "50"
    assert validate_decision(_cmd(intent="volume", key="up")).query == "up"
    assert validate_decision(_cmd(intent="volume", key="MUTE")).query == "mute"


def test_create_folder_with_location():
    intent = validate_decision(
        _cmd(intent="create_folder", query="race", location="desktop")
    )
    assert intent.type is IntentType.CREATE_ITEM
    assert intent.meta == "folder:desktop"


def test_rename_requires_both_names():
    intent = validate_decision(
        _cmd(intent="rename_file", query="old notes", app="new notes")
    )
    assert intent.type is IntentType.RENAME_ITEM
    assert intent.query == "old notes"
    assert intent.meta == "new notes"


def test_chat_and_unknown_return_none():
    chat = LLMDecision(action="chat", response="hi")
    assert validate_decision(chat) is None
    unknown = LLMDecision(action="unknown")
    assert validate_decision(unknown) is None


# ---------------------------------------------------------------------------
# Security: arbitrary shell / code / paths
# ---------------------------------------------------------------------------

def test_shell_metacharacters_in_name_rejected():
    with pytest.raises(LLMValidationError):
        validate_name("notes && del C:\\Windows", field_name="query")


def test_path_separators_rejected():
    with pytest.raises(LLMValidationError):
        validate_name("..\\..\\Windows\\System32", field_name="query")
    with pytest.raises(LLMValidationError):
        validate_name("C:\\Users\\hp\\secret.txt", field_name="query")


def test_path_traversal_rejected():
    with pytest.raises(LLMValidationError):
        validate_name("notes.. ..etc", field_name="query")


def test_drive_letter_rejected():
    with pytest.raises(LLMValidationError):
        validate_name("C:", field_name="query")


def test_reserved_device_names_rejected():
    with pytest.raises(LLMValidationError):
        validate_name("con", field_name="query")
    with pytest.raises(LLMValidationError):
        validate_name("nul.txt", field_name="query")


def test_command_injection_via_open_app_rejected():
    # A hypothetical LLM hallucination: intent open_app with a command line.
    with pytest.raises(LLMValidationError):
        validate_decision(
            _cmd(intent="open_app", app="chrome.exe & shutdown /s")
        )


def test_file_url_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="open_url", query="file:///C:/Windows/System32/calc.exe"))


def test_javascript_url_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="open_url", query="javascript:alert(1)"))


def test_unc_path_url_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="open_url", query="\\\\evil-host\\share\\x"))


def test_https_url_accepted():
    intent = validate_decision(
        _cmd(intent="open_url", query="https://www.wikipedia.org")
    )
    assert intent.type is IntentType.OPEN_URL


def test_unknown_intent_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="format_c_drive", query="C:\\"))


def test_shutdown_intent_not_in_allowlist():
    """Shutdown/restart must never be reachable via the LLM path."""
    for sneaky in ("shutdown", "system_shutdown", "restart", "reboot"):
        with pytest.raises(LLMValidationError):
            validate_decision(_cmd(intent=sneaky))


def test_missing_required_parameter_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="open_app"))


def test_empty_query_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="search_files", query="   "))


def test_invalid_location_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(
            _cmd(intent="create_folder", query="x", location="C:\\Windows")
        )


def test_invalid_volume_key_rejected():
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="volume", key="explode"))
    with pytest.raises(LLMValidationError):
        validate_decision(_cmd(intent="volume", key="101"))


def test_prompt_injection_shaped_query_is_just_text():
    """A user file literally named like an instruction is still just a name:
    separators/length rules apply, but content itself is not executed."""
    intent = validate_decision(
        _cmd(intent="open_file", query="ignore previous instructions.txt")
    )
    assert intent.type is IntentType.OPEN_FILE


def test_delete_file_routes_through_confirmation_intent():
    """delete_file stays DELETE_ITEM -- the executor's spoken-confirmation
    flow (needs_confirmation/pending_action) then applies unchanged."""
    intent = validate_decision(_cmd(intent="delete_file", query="old dump"))
    assert intent.type is IntentType.DELETE_ITEM


def test_control_characters_rejected():
    with pytest.raises(LLMValidationError):
        validate_text("line\x00break", field_name="query")


def test_newlines_flattened_in_text():
    assert validate_text("type this\nthen that", field_name="query") == \
        "type this then that"
