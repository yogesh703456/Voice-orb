"""
core/llm_validator.py

The security gate between the LLM and the executor.

The LLM's decision is UNTRUSTED INPUT, same category as the microphone
audio itself. This module independently verifies:

  1. intent allowlist        only intents the executor really implements
  2. required parameters     present, correct type, sane length
  3. name/path hardening     no separators, no "..", no drive letters,
                             no reserved device names, bounded length
  4. URL scheme allowlist    http/https only
  5. volume key domain       up/down/mute/unmute or 0-100

...and only then converts the decision into the SAME ParsedIntent type the
regex parser produces, so the executor cannot tell the two apart and the
executor's own safety rules (confirmation for deletes, vague-name
rejection, Explorer protection, etc.) apply unchanged.

Never construct ParsedIntents from LLM output anywhere else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.intent import IntentType, ParsedIntent
from core.llm_brain import LLMDecision


class LLMValidationError(Exception):
    """The decision failed validation; the pipeline must not execute it."""


@dataclass(frozen=True)
class _IntentSpec:
    intent_type: IntentType
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    uses_name_check: tuple[str, ...] = ()


# Every intent the executor actually implements (verified against
# core/executor.py's execute() dispatch chain). If an intent is missing
# here, the LLM cannot trigger it -- this is the allowlist.
_INTENT_SPECS: dict[str, _IntentSpec] = {
    "open_app":     _IntentSpec(IntentType.LAUNCH_APP, required=("app",), uses_name_check=("app",)),
    "close_app":    _IntentSpec(IntentType.CLOSE_APP, required=("app",), uses_name_check=("app",)),
    "open_file":    _IntentSpec(IntentType.OPEN_FILE, required=("query",), uses_name_check=("query",)),
    "open_folder":  _IntentSpec(IntentType.LAUNCH_APP, required=("query",), uses_name_check=("query",)),
    "search_files": _IntentSpec(IntentType.SEARCH_FILES, required=("query",), uses_name_check=("query",)),
    "create_file":  _IntentSpec(IntentType.CREATE_ITEM, required=("query",), uses_name_check=("query",),
                                optional=("location",)),
    "create_folder": _IntentSpec(IntentType.CREATE_ITEM, required=("query",), uses_name_check=("query",),
                                 optional=("location",)),
    "rename_file":  _IntentSpec(IntentType.RENAME_ITEM, required=("query", "app"),
                                uses_name_check=("query", "app")),
    "delete_file":  _IntentSpec(IntentType.DELETE_ITEM, required=("query",), uses_name_check=("query",)),
    "list_files":   _IntentSpec(IntentType.LIST_FILES, optional=("location",)),
    "file_path":    _IntentSpec(IntentType.PATH_QUERY, required=("query",), uses_name_check=("query",)),
    "copy_text":    _IntentSpec(IntentType.COPY_TEXT, required=("query",)),
    "type_text":    _IntentSpec(IntentType.TYPE_TEXT, required=("query",)),
    "web_search":   _IntentSpec(IntentType.WEB_SEARCH, required=("query",)),
    "site_search":  _IntentSpec(IntentType.SITE_SEARCH, required=("query", "target_site")),
    "open_url":     _IntentSpec(IntentType.OPEN_URL, required=("query",)),
    "new_tab":      _IntentSpec(IntentType.NEW_TAB),
    "play_music":   _IntentSpec(IntentType.PLAY_MUSIC, optional=("query",)),
    "volume":       _IntentSpec(IntentType.VOLUME_CONTROL, required=("key",)),
    "screenshot":   _IntentSpec(IntentType.SCREENSHOT),
    "time":         _IntentSpec(IntentType.TIME_QUERY),
    "date":         _IntentSpec(IntentType.DATE_QUERY),
    "battery":      _IntentSpec(IntentType.BATTERY_QUERY),
}

# Windows reserved device names -- "con", "nul", "com1" ... must never be
# creatable as a file/folder name via any spoken or LLM-provided name.
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

_MAX_NAME_LEN = 120
_MAX_TEXT_LEN = 2000

_LOCATION_VALUES = ("desktop", "documents", "downloads")
_VOLUME_KEYS = ("up", "down", "mute", "unmute")

# Scheme-hostile characters that must never appear in a name/short param.
_NAME_HOSTILE_CHARS = set('\\/:*?"<>|')


def validate_name(value: str, *, field_name: str) -> str:
    """Harden a spoken/LLM-provided file, folder, or app name.

    Rejects: empty, too long, path separators, traversal ("..", "..."),
    drive letters (C:), reserved Windows device names, and control chars.
    """
    name = value.strip().strip('"').strip("'").strip()
    if not name:
        raise LLMValidationError(f"{field_name} is empty.")
    if len(name) > _MAX_NAME_LEN:
        raise LLMValidationError(f"{field_name} is too long.")
    if any(ord(ch) < 32 for ch in name):
        raise LLMValidationError(f"{field_name} contains control characters.")
    if any(ch in _NAME_HOSTILE_CHARS for ch in name):
        raise LLMValidationError(f"{field_name} must not contain path separators.")
    if re.search(r"\.\.", name):
        raise LLMValidationError(f"{field_name} must not contain path traversal.")
    if re.fullmatch(r"[a-zA-Z]:", name):
        raise LLMValidationError(f"{field_name} must not be a drive path.")
    if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]*", name.split(".")[0].lower()) and \
            name.split(".")[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise LLMValidationError(f"{field_name} is a reserved device name.")
    return name


def validate_text(value: str, *, field_name: str) -> str:
    """Harden a free-text parameter (search terms, typed text, URL path part).
    Bounded length; no control characters; newlines flattened (typed text
    goes through pyautogui, where raw newlines/escapes would be keys)."""
    text = value.strip()
    if not text:
        raise LLMValidationError(f"{field_name} is empty.")
    if len(text) > _MAX_TEXT_LEN:
        raise LLMValidationError(f"{field_name} is too long.")
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
        raise LLMValidationError(f"{field_name} contains control characters.")
    if any(ch in text for ch in "\n\r"):
        text = re.sub(r"[\n\r]+", " ", text).strip()
    return text


def _validate_url(value: str, *, field_name: str) -> str:
    url = value.strip()
    if not url:
        raise LLMValidationError(f"{field_name} is empty.")
    if len(url) > _MAX_TEXT_LEN:
        raise LLMValidationError(f"{field_name} is too long.")
    # Scheme allowlist: http/https (possibly with www.) -- no file://,
    # no javascript:, no \\unc paths, no shell: URIs.
    if not re.match(r"^(https?://|www\.)[^\s]+$", url, re.IGNORECASE):
        raise LLMValidationError(
            f"{field_name} must be a plain http(s) URL, got {url[:80]!r}."
        )
    if any(ord(ch) < 32 for ch in url):
        raise LLMValidationError(f"{field_name} contains control characters.")
    return url


def _validate_location(value: str) -> str:
    location = (value or "").strip().lower()
    if not location:
        return "desktop"
    if location not in _LOCATION_VALUES:
        raise LLMValidationError(
            f"location must be one of {', '.join(_LOCATION_VALUES)}, got {location!r}."
        )
    return location


def _validate_volume_key(value: str) -> str:
    key = (value or "").strip().lower()
    if key in _VOLUME_KEYS:
        return key
    if re.fullmatch(r"\d{1,3}", key) and 0 <= int(key) <= 100:
        return key
    raise LLMValidationError(
        f"volume key must be up/down/mute/unmute or a number 0-100, got {value!r}."
    )


def _known_website_url(name: str) -> str | None:
    """Map a bare website word to its address the same way the intent
    parser's fast path does (executor has no app named 'youtube')."""
    known = {
        "youtube": "https://www.youtube.com",
        "netflix": "https://www.netflix.com",
        "gmail": "https://mail.google.com",
        "google": "https://www.google.com",
        "amazon": "https://www.amazon.com",
        "reddit": "https://www.reddit.com",
        "facebook": "https://www.facebook.com",
        "instagram": "https://www.instagram.com",
        "twitter": "https://www.twitter.com",
        "linkedin": "https://www.linkedin.com",
        "whatsapp web": "https://web.whatsapp.com",
    }
    return known.get(name.strip().lower())


def validate_decision(decision: LLMDecision) -> ParsedIntent | None:
    """Validate an LLMDecision and convert it into a ParsedIntent.

    Returns:
        ParsedIntent  for a valid command decision
        None          for chat / unknown decisions (no execution)
    Raises:
        LLMValidationError for any invalid command decision
    """
    if decision.action in ("chat", "unknown"):
        return None

    if decision.action != "command":
        raise LLMValidationError(f"Unknown action {decision.action!r}.")

    spec = _INTENT_SPECS.get(decision.intent)
    if spec is None:
        raise LLMValidationError(f"Intent {decision.intent!r} is not allowed.")

    def _get(field: str) -> str:
        return getattr(decision, field, "") or ""

    # Required fields present and non-empty.
    for field_name in spec.required:
        if not _get(field_name).strip():
            raise LLMValidationError(f"Intent {decision.intent} requires {field_name!r}.")

    params: dict[str, str] = {}
    for field_name in (*spec.required, *spec.optional):
        value = _get(field_name)
        if not value:
            continue
        if field_name == "location":
            params[field_name] = _validate_location(value)
        elif field_name == "key":
            params[field_name] = _validate_volume_key(value)
        elif field_name == "target_site":
            params[field_name] = validate_name(value, field_name=field_name)
        elif field_name == "query" and spec.intent_type == IntentType.OPEN_URL:
            params[field_name] = _validate_url(value, field_name=field_name)
        elif field_name in spec.uses_name_check:
            params[field_name] = validate_name(value, field_name=field_name)
        else:
            params[field_name] = validate_text(value, field_name=field_name)

    # ---- Per-intent assembly into the exact shapes the parser produces ----

    intent_type = spec.intent_type
    query = params.get("query", "")
    meta = ""

    if decision.intent == "open_app":
        # "open youtube" is a website, not an app -- same rule the regex
        # parser's fast path applies; route it as OPEN_URL instead of
        # letting the app resolver fuzzy-guess something dangerous.
        website_url = _known_website_url(params["app"])
        if website_url:
            return ParsedIntent(IntentType.OPEN_URL, website_url)
        query = params["app"].lower()

    elif decision.intent == "open_folder":
        # LAUNCH_APP falls through to folder search inside the executor;
        # validated above that query is a clean single-segment name.
        query = query.lower()

    elif decision.intent == "create_file":
        meta = f"file:{params.get('location', 'desktop')}"
    elif decision.intent == "create_folder":
        meta = f"folder:{params.get('location', 'desktop')}"
    elif decision.intent == "rename_file":
        meta = params["app"]  # new name rides in meta, parser-style
    elif decision.intent == "list_files":
        query = _validate_location(params.get("location", "desktop"))
    elif decision.intent == "volume":
        query = params["key"]

    return ParsedIntent(intent_type, query, meta=meta)
