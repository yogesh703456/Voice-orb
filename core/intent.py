"""
JARVIS Voice ORB - Intent Parser
Converts recognized speech into a simple intent.
"""

from dataclasses import dataclass
from enum import Enum, auto
import re


class IntentType(Enum):
    GREETING = auto()
    THANKS = auto()
    OPEN_FILE = auto()
    SEARCH_FILES = auto()
    LAUNCH_APP = auto()
    CLOSE_APP = auto()
    WEB_SEARCH = auto()
    SITE_SEARCH = auto()
    NEW_TAB = auto()
    OPEN_URL = auto()
    SAVE_FILE = auto()
    COPY_TEXT = auto()
    PASTE_TEXT = auto()
    COPY_FILE = auto()
    CREATE_ITEM = auto()
    RENAME_ITEM = auto()
    DELETE_ITEM = auto()
    LIST_FILES = auto()
    PATH_QUERY = auto()
    TYPE_TEXT = auto()
    KEY_ACTION = auto()
    WINDOW_CONTROL = auto()
    SCREENSHOT = auto()
    TIME_QUERY = auto()
    DATE_QUERY = auto()
    VOLUME_CONTROL = auto()
    BATTERY_QUERY = auto()
    UNKNOWN = auto()


@dataclass
class ParsedIntent:
    type: IntentType
    query: str
    # Extra payload for intents that need more than one string (currently
    # just SITE_SEARCH's target site, e.g. "youtube"). Optional and empty
    # by default so every existing ParsedIntent(type, query) call site
    # keeps working unchanged.
    meta: str = ""


def clean_text(text: str) -> str:
    """Clean Whisper output."""

    text = text.lower().strip()
    text = re.sub(r"^[\s.,!?]+|[\s.,!?]+$", "", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated identical sentence/phrase
    words = text.split()

    if len(words) >= 4:
        half = len(words) // 2

        first = words[:half]
        second = words[half:]

        if first == second:
            words = first

    # Whisper sometimes repeats a command word: "open open chrome".
    compact_words = []
    for word in words:
        if not compact_words or compact_words[-1] != word:
            compact_words.append(word)

    return " ".join(compact_words)


def remove_wake_word(text: str) -> str:
    """Remove JARVIS wake words."""

    wake_patterns = [
        r"^hello (?:jarvis|camp|axiom)\b",
        r"^hey (?:jarvis|axiom)\b",
        r"^hi (?:jarvis|axiom)\b",
        r"^ok(?:ay)? jarvis\b",
        r"^jarvis\b",
    ]

    for pattern in wake_patterns:
        text = re.sub(pattern, "", text, count=1)

    return text.strip()


# Polite/indirect phrasing that Whisper faithfully transcribes but that
# would otherwise stop every pattern below from matching at all, e.g.
# "can you open chrome" or "please open chrome" previously fell straight
# through to UNKNOWN even though the actual command was perfectly clear.
_LEADING_STANDALONE = re.compile(r"^(?:now|then|so|okay|ok|alright|right|well|hey)\s+")
_LEADING_POLITE = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+(?:please\s+|just\s+)?"
    r"|(?:please|just)\s+)+"
)
_TRAILING_FILLER = re.compile(r"\s+(?:please|for me|now|thanks|thank you)$")


def strip_filler(command: str) -> str:
    """Strip polite/indirect wrapping ("can you ...", "... please", "now
    open chrome") from a command so the intent patterns below -- which
    match on a command starting with the actual verb ("open", "search",
    "make"...) -- still fire. Runs after the wake word has already been
    removed."""

    previous = None
    while previous != command:
        previous = command
        command = _LEADING_STANDALONE.sub("", command).strip()
        command = _LEADING_POLITE.sub("", command).strip()
        command = _TRAILING_FILLER.sub("", command).strip()
    return command


_LOCATION_ANYWHERE_RE = re.compile(
    r",?\s*(?:on|in|to)\s+(?:the\s+)?(desktop|documents|downloads)\b\.?"
)
_NAME_MARKER_RE = re.compile(
    r"^,?\s*(?:name\s+it|call\s+it|named|called)\s+(.+)$"
)


def _split_create_target(remainder: str) -> tuple[str, str]:
    """Pull the name and location out of whatever follows "make/create a
    folder/file", in whatever order they were said, e.g. "on desktop,
    name it race" or "named race on desktop" or just "race". Returns
    (name, location) -- name is "" when nothing usable was said at all
    (the executor then falls back to an Explorer-style default name like
    "New Folder"), rather than -- as a purely name-first regex used to --
    mistaking the location phrase itself for the name."""

    remainder = remainder.strip(" .,")

    location = ""
    loc_match = _LOCATION_ANYWHERE_RE.search(remainder)
    if loc_match:
        location = loc_match.group(1)
        remainder = remainder[: loc_match.start()] + " " + remainder[loc_match.end() :]
        remainder = re.sub(r"\s+", " ", remainder).strip(" .,")
        remainder = re.sub(r"^,\s*", "", remainder).strip()

    name_match = _NAME_MARKER_RE.match(remainder)
    if name_match:
        name = name_match.group(1).strip(" .,")
    else:
        # No "named"/"call it" marker -- if anything is left, treat it as
        # the name directly ("make a folder race"); if nothing is left,
        # no name was said at all.
        name = remainder.strip(" .,")

    return name, location


def parse(transcript: str) -> ParsedIntent:

    text = clean_text(transcript)

    if not text:
        return ParsedIntent(IntentType.UNKNOWN, "")

    # ------------------------------------------------
    # GREETINGS / WAKE WORD
    # ------------------------------------------------

    greeting_phrases = [
        "hello",
        "hello jarvis",
        "hello camp",
        "hey jarvis",
        "hey axiom",
        "hi jarvis",
        "hi",
        "hey",
    ]

    thanks_phrases = [
        "thanks",
        "thank you",
        "thanks jarvis",
        "thank you jarvis",
    ]

    if text in thanks_phrases:
        return ParsedIntent(IntentType.THANKS, "")

    if text in greeting_phrases:
        return ParsedIntent(IntentType.GREETING, "")

    # Remove wake word before processing commands
    command = remove_wake_word(text)
    command = command.strip(" .,!?")

    # If user only said "Jarvis"
    if not command:
        return ParsedIntent(IntentType.GREETING, "")

    # Strip polite/indirect wrapping ("can you open chrome", "open chrome
    # please") so the verb-first patterns below still match.
    command = strip_filler(command)
    if not command:
        return ParsedIntent(IntentType.GREETING, "")

    # Common short-command recognition variants from Whisper.
    if re.match(r"^(?:open|launch|start)\b", command):
        command = re.sub(
            r"\b(?:google\s+)?(?:chrome|chrom|crone|crohn|crow|floor)\b",
            "chrome",
            command,
        )

    # ------------------------------------------------
    # OPEN COMMAND
    # ------------------------------------------------

    open_patterns = [r"^show\s+me\s+(.+)$"]

    for pattern in open_patterns:

        match = re.match(pattern, command)

        if match:
            query = match.group(1).strip()

            return ParsedIntent(
                IntentType.OPEN_FILE,
                query
            )

    # ------------------------------------------------
    # BROWSER COMMANDS
    # ------------------------------------------------

    match = re.match(
        r"^(?:web\s+search|search\s+web|search\s+online|let['’]?s\s+search)\s+"
        r"(?:(?:the\s+web\s+)?for\s+|online\s+for\s+)?(.+)$",
        command,
    )
    if match:
        return ParsedIntent(IntentType.WEB_SEARCH, match.group(1).strip())

    # "search youtube for cats" / "search for cats on youtube" / "look up
    # the mona lisa on wikipedia" / "search on netflix toxic" -- search a
    # specific site, not files on disk and not a bare Google query.
    # Checked before the generic SEARCH_FILES patterns further down, which
    # would otherwise swallow the site name as if it were part of a
    # filename query.
    #
    # Two tiers, deliberately different levels of strictness:
    #  - "search ON <site> ..." / "... on <site>" is unambiguous -- the
    #    word "on" can't appear in a plain file search, so ANY site name
    #    works here, not just ones we've heard of. This is what makes
    #    "search on netflix toxic" or "search on hotstar money heist"
    #    work without hardcoding every streaming/shopping site that
    #    exists.
    #  - Bare "search <site> <query>" (no "on") is only recognized for a
    #    short curated list of very common sites, because without the
    #    "on" marker it would otherwise swallow ordinary file searches
    #    ("search resume", "search my project folder").
    _KNOWN_SITES = (
        "youtube", "you tube", "google", "amazon", "wikipedia", "bing",
        "github", "netflix", "reddit", "spotify",
    )
    _KNOWN_SITE_GROUP = "|".join(sorted(_KNOWN_SITES, key=len, reverse=True))

    match = re.match(
        r"^(?:search|look up|find)\s+on\s+(\S+)\s+(?:for\s+)?(.+)$",
        command,
    )
    if match:
        site = match.group(1).strip()
        return ParsedIntent(IntentType.SITE_SEARCH, match.group(2).strip(), meta=site)

    match = re.match(
        r"^(?:search|look up|find)\s+(?:for\s+)?(.+?)\s+on\s+(\S+)$",
        command,
    )
    if match:
        site = match.group(2).strip()
        return ParsedIntent(IntentType.SITE_SEARCH, match.group(1).strip(), meta=site)

    match = re.match(
        rf"^(?:search|look up|find)\s+({_KNOWN_SITE_GROUP})\s+(?:for\s+)?(.+)$",
        command,
    )
    if match:
        site = match.group(1).replace("you tube", "youtube")
        return ParsedIntent(IntentType.SITE_SEARCH, match.group(2).strip(), meta=site)

    if command in ("new tab", "open new tab", "create new tab"):
        return ParsedIntent(IntentType.NEW_TAB, "")

    match = re.match(r"^copy file\s+(.+)$", command)
    if match:
        return ParsedIntent(IntentType.COPY_FILE, match.group(1).strip())

    match = re.match(r"^(?:copy|copy text)\s+(.+)$", command)
    if match:
        return ParsedIntent(IntentType.COPY_TEXT, match.group(1).strip())

    if command in ("paste", "paste text", "paste it"):
        return ParsedIntent(IntentType.PASTE_TEXT, "")

    # Bare domains ("open google.com", "go to bbc.co.uk") are common and
    # should open as a URL, not be treated as an app/file name -- only
    # recognized suffixes are matched so ordinary app names that happen to
    # contain a dot ("node.js") don't get misrouted.
    _COMMON_TLDS = (
        r"com|org|net|io|co|gov|edu|dev|app|ai|info|biz|tv|me|us|uk|in"
    )
    match = re.match(
        r"^(?:open|go to|navigate to)\s+"
        r"(https?://\S+|www\.\S+|"
        rf"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:{_COMMON_TLDS})(?:\.[a-z]{{2,3}})?(?:/\S*)?)$",
        command,
    )
    if match:
        return ParsedIntent(IntentType.OPEN_URL, match.group(1).strip())

    # ------------------------------------------------
    # CLOSE / SAVE COMMANDS
    # ------------------------------------------------

    match = re.match(r"^(?:close|exit|quit|stop|shut ?down)\s+(?:the\s+)?(.+)$", command)
    if match:
        return ParsedIntent(IntentType.CLOSE_APP, match.group(1).strip())

    # Bare "shut down" / "shutdown" with no app named and no "jarvis" --
    # not the shutdown phrase (that's caught earlier in main.py) and not
    # a close command either since nothing to close was named. Keep it as
    # UNKNOWN but with a query is_shutdown_like() below can recognize, so
    # the executor can give a helpful nudge instead of a flat "I didn't
    # understand that."
    if command in ("shut down", "shutdown"):
        return ParsedIntent(IntentType.UNKNOWN, command)

    match = re.match(r"^save\s+(?:the\s+)?(?:file|document)?\s*(.*)$", command)
    if match and match.group(1).strip():
        return ParsedIntent(IntentType.SAVE_FILE, match.group(1).strip())

    # ------------------------------------------------
    # FILE / FOLDER MANAGEMENT
    # ------------------------------------------------
    # Scoped to Desktop/Documents/Downloads, same as everything else that
    # touches the filesystem in this project -- "on desktop"/"in
    # documents"/"in downloads" picks where, defaulting to Desktop when
    # unstated (create) or searching all three (delete/rename).
    #
    # People don't always say the name and the location in the same
    # order, or use "called"/"named" at all -- "make a folder on desktop,
    # name it race" is just as natural as "make a folder named race on
    # desktop". _split_create_target below finds the location and name
    # wherever they land instead of assuming one fixed word order, and
    # falls back to a Explorer-style default ("New Folder") if no name
    # was said at all, rather than (as it used to) mistaking the location
    # phrase itself for the name.

    match = re.match(
        r"^(?:make|create)\s+(?:a\s+|an\s+)?(?:new\s+)?folder\b\s*(.*)$",
        command,
    )
    if match:
        name, location = _split_create_target(match.group(1))
        return ParsedIntent(IntentType.CREATE_ITEM, name, meta=f"folder:{location}")

    match = re.match(
        r"^(?:make|create)\s+(?:a\s+|an\s+)?(?:new\s+)?file\b\s*(.*)$",
        command,
    )
    if match:
        name, location = _split_create_target(match.group(1))
        return ParsedIntent(IntentType.CREATE_ITEM, name, meta=f"file:{location}")

    match = re.match(
        r"^delete\s+(?:the\s+)?(?:(file|folder)\s+)?(?:(?:called|named)\s+)?(.+)$",
        command,
    )
    if match:
        kind_hint = match.group(1) or ""
        return ParsedIntent(IntentType.DELETE_ITEM, match.group(2).strip(), meta=kind_hint)

    match = re.match(
        r"^rename\s+(?:the\s+)?(?:file|folder)?\s*(?:called|named)?\s*(.+?)\s+(?:to|as)\s+(.+)$",
        command,
    )
    if match:
        return ParsedIntent(IntentType.RENAME_ITEM, match.group(1).strip(), meta=match.group(2).strip())

    match = re.match(
        r"^(?:list|show)\s+(?:my\s+)?files(?:\s+(?:on|in)\s+(?:the\s+)?(desktop|documents|downloads))?$",
        command,
    )
    if match:
        return ParsedIntent(IntentType.LIST_FILES, match.group(1) or "desktop")

    match = re.match(
        r"^what(?:'s| is)\s+(?:in|on)\s+(?:my\s+)?(?:the\s+)?(desktop|documents|downloads)$",
        command,
    )
    if match:
        return ParsedIntent(IntentType.LIST_FILES, match.group(1))

    # "give me the path of that folder you created" / "where is rohan" /
    # "give me the address of the folder" -- a very natural follow-up
    # after creating or finding something, previously totally unhandled.
    match = re.match(
        r"^(?:what(?:'s| is)\s+(?:the\s+)?|give me\s+(?:the\s+)?|tell me\s+(?:the\s+)?)?"
        r"(?:path|address|location)\s+(?:of\s+)?(?:the\s+|that\s+)?(.+)$",
        command,
    )
    if not match:
        match = re.match(r"^where\s+is\s+(?:the\s+|that\s+)?(.+)$", command)
    if match:
        reference = match.group(1).strip()
        # "...that folder you created" / "...the file I just made" --
        # strip the trailing clause, it's not part of a name to search for.
        reference = re.sub(
            r"\s+(?:you\s+(?:just\s+)?(?:created|made)|i\s+(?:just\s+)?(?:created|made))$",
            "",
            reference,
        ).strip()
        return ParsedIntent(IntentType.PATH_QUERY, reference)

    # ------------------------------------------------
    # SYSTEM / UTILITY
    # ------------------------------------------------

    if re.match(r"^(?:take\s+(?:a\s+)?)?screenshot$", command) or command == "capture screen":
        return ParsedIntent(IntentType.SCREENSHOT, "")

    if re.match(r"^what(?:'s| is)?\s+(?:the\s+)?time(?:\s+is\s+it)?$", command) or command == "what time is it":
        return ParsedIntent(IntentType.TIME_QUERY, "")

    if re.match(r"^what(?:'s| is)\s+(?:the\s+)?date$", command) or command in (
        "what day is it", "what is today's date", "today's date",
    ):
        return ParsedIntent(IntentType.DATE_QUERY, "")

    if re.match(r"^(?:what(?:'s| is)\s+(?:my\s+)?)?battery(?:\s+level|\s+percentage)?$", command) or command in (
        "how much battery do i have", "check battery",
    ):
        return ParsedIntent(IntentType.BATTERY_QUERY, "")

    match = re.match(r"^(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3})(?:\s*%|\s*percent)?$", command)
    if match:
        return ParsedIntent(IntentType.VOLUME_CONTROL, match.group(1))
    if command in ("volume up", "increase volume", "turn up the volume", "turn the volume up"):
        return ParsedIntent(IntentType.VOLUME_CONTROL, "up")
    if command in ("volume down", "decrease volume", "turn down the volume", "turn the volume down", "lower the volume"):
        return ParsedIntent(IntentType.VOLUME_CONTROL, "down")
    if command in ("mute", "mute volume", "mute the volume", "silence"):
        return ParsedIntent(IntentType.VOLUME_CONTROL, "mute")
    if command in ("unmute", "unmute volume", "unmute the volume"):
        return ParsedIntent(IntentType.VOLUME_CONTROL, "unmute")

    # ------------------------------------------------
    # WORKING INSIDE APPS -- typing, keys, window control
    # ------------------------------------------------
    # These act on whatever window currently has focus (the app you just
    # opened, or whatever you clicked into) -- e.g. "open notepad" then
    # "type meeting notes for today" actually dictates that text into it.

    match = re.match(r"^(?:type|write|dictate)\s+(.+)$", command)
    if match:
        return ParsedIntent(IntentType.TYPE_TEXT, match.group(1).strip())

    _KEY_ACTIONS = {
        "press enter": "enter", "hit enter": "enter", "press return": "enter",
        "new line": "enter",
        "press tab": "tab", "hit tab": "tab",
        "press escape": "escape", "hit escape": "escape", "press esc": "escape",
        "press backspace": "backspace", "delete that": "backspace",
        "press delete": "delete",
        "select all": "select_all",
        "undo": "undo", "undo that": "undo",
        "redo": "redo", "redo that": "redo",
        "cut": "cut", "cut that": "cut",
        # Deliberately NOT "delete everything" -- that phrase is
        # ambiguous with the file/folder DELETE_ITEM pattern checked
        # earlier ("delete <name>"), which must keep meaning file
        # deletion. "clear" avoids the collision entirely.
        "clear everything": "select_all_delete",
        "clear the document": "select_all_delete",
    }
    if command in _KEY_ACTIONS:
        return ParsedIntent(IntentType.KEY_ACTION, _KEY_ACTIONS[command])

    _WINDOW_ACTIONS = {
        "minimize this": "minimize", "minimize the window": "minimize",
        "minimize this window": "minimize", "minimize it": "minimize",
        "maximize this": "maximize", "maximize the window": "maximize",
        "maximize this window": "maximize", "maximize it": "maximize",
        "show desktop": "show_desktop", "minimize everything": "show_desktop",
        "minimize all windows": "show_desktop",
    }
    if command in _WINDOW_ACTIONS:
        return ParsedIntent(IntentType.WINDOW_CONTROL, _WINDOW_ACTIONS[command])

    match = re.match(r"^switch to\s+(?:the\s+)?(.+)$", command)
    if match:
        return ParsedIntent(IntentType.WINDOW_CONTROL, "switch", meta=match.group(1).strip())

    # ------------------------------------------------
    # SEARCH COMMAND
    # ------------------------------------------------

    search_patterns = [
        r"^search\s+for\s+(.+)$",
        r"^search\s+(.+)$",
        r"^find\s+(.+)$",
    ]

    for pattern in search_patterns:

        match = re.match(pattern, command)

        if match:
            query = match.group(1).strip()

            return ParsedIntent(
                IntentType.SEARCH_FILES,
                query
            )

    # ------------------------------------------------
    # LAUNCH APPLICATION
    # ------------------------------------------------

    launch_patterns = [
        r"^launch\s+(?:the\s+)?(.+)$",
        r"^start\s+(?:the\s+)?(.+)$",
        r"^open\s+app\s+(.+)$",
        # Generic catch-all: "open X" now always tries to launch X as an
        # app first. Anything installed with a Start Menu entry resolves
        # here (see core/apps.py) instead of needing to be hardcoded by
        # name; the executor falls back to a file search if nothing
        # matches as an app, so "open <filename>" still works too.
        r"^open\s+(?:the\s+)?(.+)$",
    ]

    for pattern in launch_patterns:

        match = re.match(pattern, command)

        if match:
            query = match.group(1).strip()
            # Strip generic trailing filler words: "open brave software",
            # "launch spotify app" -> just the app name itself.
            query = re.sub(
                r"\s+(?:app|application|software|program|browser)$", "", query
            ).strip()

            return ParsedIntent(
                IntentType.LAUNCH_APP,
                query
            )

    # ------------------------------------------------
    # UNKNOWN
    # ------------------------------------------------

    return ParsedIntent(
        IntentType.UNKNOWN,
        command
    )


# ------------------------------------------------------------------
# MULTIPLE COMMANDS IN ONE UTTERANCE
# ------------------------------------------------------------------
# "open chrome and search youtube for cats", "open notepad then close
# calculator" -- only split at a connector when it's immediately followed
# by another command verb. Without that guard, an ordinary query that
# happens to contain "and" ("search for cats and dogs") would get sliced
# in half; requiring a verb right after the connector keeps that as one
# command while still splitting genuinely separate instructions.
_COMMAND_VERBS = (
    r"open|close|exit|quit|stop|shut ?down|launch|start|search|find|look up|"
    r"go to|navigate to|show me|new tab|copy|paste|save|web search|"
    r"make|create|delete|rename|list|show|screenshot|volume|mute|unmute|"
    r"type|write|dictate|switch to|press|select all|undo|redo|"
    r"minimize|maximize"
)
_SPLIT_RE = re.compile(
    rf"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\balso\b)\s*(?=(?:{_COMMAND_VERBS})\b)"
)


def split_commands(text: str) -> list[str]:
    """Split one utterance into multiple command strings, if it contains
    more than one. Returns [text] unchanged when there's nothing to split."""
    text = text.strip()
    if not text:
        return []
    parts = [part.strip(" .,!?") for part in _SPLIT_RE.split(text)]
    parts = [part for part in parts if part]
    return parts or [text]


def parse_commands(transcript: str) -> list[ParsedIntent]:
    """Like parse(), but returns one ParsedIntent per command found in the
    utterance so a single sentence like "open notepad and search youtube
    for lofi" performs both actions instead of only the first."""
    cleaned = clean_text(transcript)
    if not cleaned:
        return [ParsedIntent(IntentType.UNKNOWN, "")]
    return [parse(segment) for segment in split_commands(cleaned)]
