"""
JARVIS Voice ORB - Command Executor
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse
import webbrowser
from typing import Callable, Optional
from core.intent import ParsedIntent, IntentType
from core import apps
from indexer.file_index import FileIndex

try:
    from rapidfuzz import fuzz, process as fuzzy_process
except ImportError:
    fuzzy_process = None

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class ExecutionResult:
    success: bool
    spoken_response: str
    # Set for actions that shouldn't run unattended (currently just
    # deletion). When True, the caller (main.py) speaks spoken_response as
    # a yes/no question, listens for one more short reply, and only calls
    # pending_action() if the answer was affirmative.
    needs_confirmation: bool = False
    pending_action: Optional[Callable[[], "ExecutionResult"]] = None


def _find_chrome_exe() -> str | None:
    """Locate a real chrome.exe. Chrome's installer does NOT add itself to
    PATH on Windows, so `subprocess.Popen(["chrome.exe", ...])` reliably
    fails with WinError 2 even when Chrome is installed -- shutil.which()
    almost never finds it either, for the same reason. Check PATH first
    (in case it was added manually), then the two standard install
    locations (system-wide vs. per-user installs land in different
    Program Files roots)."""
    for name in ("chrome.exe", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(env_var)
        if not base:
            continue
        candidate = Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if candidate.is_file():
            return str(candidate)

    return None


def _open_chrome(url: str | None = None, new_window: bool = False) -> bool:
    """Launch Chrome specifically. Used only when the user explicitly asks
    for Chrome (e.g. "open chrome") -- everything else (searches, plain
    URLs, new tab) goes through _open_default_browser() instead, since
    forcing Chrome there would ignore whatever browser is actually set as
    default on the machine."""
    target = url or "about:blank"
    flags = "--new-window" if new_window else "--new-tab"

    chrome_path = _find_chrome_exe()
    if chrome_path:
        try:
            subprocess.Popen([chrome_path, flags, target])
            return True
        except OSError:
            pass

    # Chrome isn't installed at a location we know about (or launching it
    # failed anyway) -- fall back to whatever the default browser is.
    return webbrowser.open(target, new=1 if new_window else 0)


def _open_default_browser(url: str, new_window: bool = False) -> bool:
    """Open a URL in the user's actual default browser (Edge, Firefox,
    Chrome, whatever it's set to) -- used for every search and plain link,
    since a search command has no reason to override that choice."""
    try:
        return webbrowser.open(url, new=1 if new_window else 2)
    except Exception:
        return False


_SITE_SEARCH_URLS = {
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "google": "https://www.google.com/search?q={q}",
    "amazon": "https://www.amazon.com/s?k={q}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "github": "https://github.com/search?q={q}",
    "netflix": "https://www.netflix.com/search?q={q}",
    "reddit": "https://www.reddit.com/search/?q={q}",
    "spotify": "https://open.spotify.com/search/{q}",
    "imdb": "https://www.imdb.com/find/?q={q}",
    "stackoverflow": "https://stackoverflow.com/search?q={q}",
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "linkedin": "https://www.linkedin.com/search/results/all/?keywords={q}",
    "pinterest": "https://www.pinterest.com/search/pins/?q={q}",
    "quora": "https://www.quora.com/search?q={q}",
    "twitter": "https://twitter.com/search?q={q}",
    "x": "https://x.com/search?q={q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "primevideo": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
    "hotstar": "https://www.hotstar.com/in/search?q={q}",
}

_SITE_LABELS = {
    "youtube": "YouTube", "google": "Google", "amazon": "Amazon",
    "wikipedia": "Wikipedia", "bing": "Bing", "github": "GitHub",
    "netflix": "Netflix", "reddit": "Reddit", "spotify": "Spotify",
    "imdb": "IMDb", "stackoverflow": "Stack Overflow", "ebay": "eBay",
    "flipkart": "Flipkart", "linkedin": "LinkedIn", "pinterest": "Pinterest",
    "quora": "Quora", "twitter": "Twitter", "x": "X", "maps": "Google Maps",
    "primevideo": "Prime Video", "hotstar": "Hotstar",
}

# Aliases people actually say -> the canonical key above.
_SITE_ALIASES = {
    "you tube": "youtube", "you-tube": "youtube",
    "prime video": "primevideo", "amazon prime video": "primevideo",
    "stack overflow": "stackoverflow", "google maps": "maps",
}


def _resolve_site_search(site: str, query: str) -> tuple[str, str]:
    """Turn a spoken site name + query into (url, display_label).

    Sites we know get an accurate search URL. Anything else still gets a
    best-effort guess -- most sites expose a `/search?q=` endpoint on their
    own domain -- rather than refusing outright, since the whole point of
    "search on X" is that X doesn't have to be pre-registered. It won't be
    right 100% of the time for an unlisted site, but it's far more useful
    than only supporting a fixed handful of names.
    """
    key = _SITE_ALIASES.get(site.strip().lower(), site.strip().lower())
    q = urllib.parse.quote_plus(query)
    if key in _SITE_SEARCH_URLS:
        return _SITE_SEARCH_URLS[key].format(q=q), _SITE_LABELS.get(key, key.title())

    domain = re.sub(r"[^a-z0-9]", "", key)
    if not domain:
        domain = "google"
    return f"https://www.{domain}.com/search?q={q}", site.strip().title()


_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    "AppData",
}


def _fuzzy_pick(needle: str, candidates: list[tuple[Path, str]], limit: int) -> list[Path]:
    """Rescue pass for when no exact substring matched anything: fuzzy-
    match the needle against every candidate name collected during the
    walk. This is what lets a mis-transcribed multi-word name ("raw
    paper scissor mini project" said for a folder actually named "Rock
    Paper Scissor Mini Project") still resolve, instead of silently
    failing just because the STT output wasn't a literal substring of
    the real name."""
    if fuzzy_process is None or not candidates:
        return []
    # Lowercase both sides explicitly -- rapidfuzz's raw scorers are
    # case-sensitive, and `needle` is already lowercased by the caller,
    # so leaving real filenames in their original case would silently
    # tank every score (e.g. 74 vs. 62 for the same pair, enough to fall
    # below the cutoff) for no reason related to the actual match quality.
    names_lower = [name.lower() for _, name in candidates]
    results = fuzzy_process.extract(
    needle,
    names_lower,
    scorer=fuzz.WRatio,
    score_cutoff=90,
    limit=limit,
    )
    return [candidates[index][0] for _name, _score, index in results]


def _find_matches(query: str, *, mode: str, limit: int, max_items: int) -> list[Path]:
    """Shared walk-and-match implementation for _search_user_files and
    _search_user_folders (mode="files" or "dirs"): exact substring
    matches first (fast, precise), then a fuzzy fallback over everything
    seen if that finds nothing at all."""
    needle = query.strip().lower()
    # Below ~4 characters a raw substring match starts hitting unrelated
    # files purely by coincidence -- "you" is a substring of "layout.tsx",
    # "it" of "editor.py", etc. A garbled/short transcript should fail
    # cleanly rather than confidently open the wrong thing.
    if not needle or len(needle) < 4:
        return []

    roots = [Path.home() / folder for folder in ("Desktop", "Documents", "Downloads")]
    substring_matches: list[Path] = []
    candidates: list[tuple[Path, str]] = []
    scanned = 0

    for root in roots:
        if not root.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name not in _SKIP_DIR_NAMES]
                names = list(dirnames) if mode == "dirs" else filenames
                for name in names:
                    scanned += 1
                    
                    path = Path(dirpath) / name
                    candidates.append((path, name))
                    if needle in name.lower():
                        substring_matches.append(path)
                        if len(substring_matches) >= limit:
                            return substring_matches
        except OSError:
            continue

    if substring_matches:
        return substring_matches
    return _fuzzy_pick(needle, candidates, limit)


def _search_user_files(
    query: str,
    limit: int = 5,
    max_files: int = 2500,
) -> list[Path]:
    if _file_index is None:
        return []

    return [
        Path(path)
        for _, path in _file_index.search(
            query,
            limit=limit,
            folders_only=False,
        )
    ]


def _search_user_folders(
    query: str,
    limit: int = 5,
    max_dirs: int = 2500,
) -> list[Path]:
    if _file_index is None:
        return []

    return [
        Path(path)
        for _, path in _file_index.search(
            query,
            limit=limit,
            folders_only=True,
        )
    ]


_LOCATIONS = {
    "desktop": lambda: Path.home() / "Desktop",
    "documents": lambda: Path.home() / "Documents",
    "downloads": lambda: Path.home() / "Downloads",
}


def _resolve_location(name: str) -> Path:
    key = (name or "desktop").strip().lower()
    return _LOCATIONS.get(key, _LOCATIONS["desktop"])()


def _sanitize_name(name: str) -> Optional[str]:
    """Reject anything that looks like it's trying to escape the intended
    folder (path separators, "..") or is empty -- a spoken name should
    never turn into a multi-level path."""
    name = name.strip().strip('"').strip()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    return name


def _next_available_name(directory: Path, base: str, suffix: str = "") -> str:
    """Explorer-style auto naming ("New Folder", "New Folder (2)", ...)
    for when no name was actually said."""
    candidate = f"{base}{suffix}"
    if not (directory / candidate).exists():
        return candidate
    n = 2
    while True:
        candidate = f"{base} ({n}){suffix}"
        if not (directory / candidate).exists():
            return candidate
        n += 1


# Remembers the most recently created/found file or folder for this
# session, so "give me the path of that folder" (referring to something
# just talked about, with no name repeated) has something concrete to
# resolve to instead of failing outright.
_LAST_CREATED_PATH: Optional[Path] = None
_file_index: FileIndex | None = None


def configure_file_index(index: FileIndex) -> None:
    global _file_index
    _file_index = index


def _delete_path(path: Path, is_dir: bool) -> ExecutionResult:
    """Move a confirmed item to the Recycle Bin; never delete permanently."""
    global _LAST_CREATED_PATH

    if send2trash is None:
        return ExecutionResult(
            False,
            "Deletion is disabled because Recycle Bin support is unavailable."
        )

    try:
        send2trash(str(path))

        if _LAST_CREATED_PATH == path:
            _LAST_CREATED_PATH = None

        return ExecutionResult(
            True,
            f"Moved {path.name} to the Recycle Bin."
        )

    except OSError as exc:
        return ExecutionResult(
            False,
            f"I couldn't move {path.name} to the Recycle Bin: {exc}"
        )


def _type_text(text: str) -> ExecutionResult:
    """Type into whatever window currently has focus -- this is what makes
    "open notepad" then "type meeting notes" actually dictate into it,
    rather than JARVIS being limited to launching/closing apps only."""
    try:
        import pyautogui
    except ImportError:
        return ExecutionResult(False, "Typing needs the pyautogui package installed.")
    try:
        pyautogui.typewrite(text, interval=0.01)
        return ExecutionResult(True, f"Typed: {text}")
    except Exception as exc:
        return ExecutionResult(False, f"I couldn't type that: {exc}")


_KEY_ACTIONS = {
    "enter": (("enter",), "Pressed enter."),
    "tab": (("tab",), "Pressed tab."),
    "escape": (("esc",), "Pressed escape."),
    "backspace": (("backspace",), "Pressed backspace."),
    "delete": (("delete",), "Pressed delete."),
}
_HOTKEY_ACTIONS = {
    "select_all": (("ctrl", "a"), "Selected all."),
    "undo": (("ctrl", "z"), "Undone."),
    "redo": (("ctrl", "y"), "Redone."),
    "cut": (("ctrl", "x"), "Cut."),
}


def _do_key_action(action: str) -> ExecutionResult:
    """Send a keystroke/shortcut to the active window -- lets JARVIS
    actually operate inside whatever app is open (select all, undo,
    clear a document...) instead of only launching/closing it."""
    try:
        import pyautogui
    except ImportError:
        return ExecutionResult(False, "Key actions need the pyautogui package installed.")
    try:
        if action in _KEY_ACTIONS:
            keys, label = _KEY_ACTIONS[action]
            pyautogui.press(keys[0])
            return ExecutionResult(True, label)
        if action in _HOTKEY_ACTIONS:
            keys, label = _HOTKEY_ACTIONS[action]
            pyautogui.hotkey(*keys)
            return ExecutionResult(True, label)
        if action == "select_all_delete":
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            return ExecutionResult(True, "Cleared.")
        return ExecutionResult(False, "I don't know that key action.")
    except Exception as exc:
        return ExecutionResult(False, f"I couldn't do that: {exc}")


def _control_window(action: str, target: str = "") -> ExecutionResult:
    """Minimize/maximize the active window, show the desktop, or switch
    focus to an already-running app's window."""
    if action == "switch":
        try:
            import pygetwindow as gw
        except ImportError:
            return ExecutionResult(False, "Switching windows needs the pygetwindow package installed.")
        try:
            needle = target.lower()
            titles = [t for t in gw.getAllTitles() if t.strip() and needle in t.lower()]
            if not titles:
                return ExecutionResult(False, f"I couldn't find an open window for {target}.")
            window = gw.getWindowsWithTitle(titles[0])[0]
            window.activate()
            return ExecutionResult(True, f"Switched to {titles[0]}.")
        except Exception as exc:
            return ExecutionResult(False, f"I couldn't switch to that: {exc}")

    try:
        if action == "show_desktop":
            import pyautogui
            pyautogui.hotkey("win", "d")
            return ExecutionResult(True, "Showing the desktop.")

        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        SW_MINIMIZE, SW_MAXIMIZE = 6, 3
        if action == "minimize":
            user32.ShowWindow(hwnd, SW_MINIMIZE)
            return ExecutionResult(True, "Minimized.")
        if action == "maximize":
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
            return ExecutionResult(True, "Maximized.")
        return ExecutionResult(False, "I don't know that window action.")
    except Exception as exc:
        return ExecutionResult(False, f"I couldn't do that: {exc}")


def _play_music(query: str) -> ExecutionResult:
    """"Play music" with no target -- open a local music app if one's
    installed, else fall back to YouTube Music. "Play <song/artist>" --
    can't drive Spotify's actual playback without its API, so the honest,
    useful thing is to open a YouTube search for it, which does play."""
    query = query.strip()
    if not query:
        if apps.is_resolvable_name("spotify"):
            match = apps.resolve_app("spotify")
            if match and apps.launch_by_app_id(match["app_id"]):
                return ExecutionResult(True, "Opening Spotify.")
        _open_default_browser("https://music.youtube.com")
        return ExecutionResult(True, "Opening YouTube Music.")

    url, _ = _resolve_site_search("youtube", query)
    _open_default_browser(url)
    return ExecutionResult(True, f"Playing {query} on YouTube.")


def _system_shutdown(command: str) -> ExecutionResult:
    """Shut down the whole PC -- unlike closing an app, this is
    irreversible for whatever else was open, so it always requires a
    spoken "yes" first (same pattern as file deletion), never runs
    unattended in a multi-command batch, and gives a real cancel window
    (Windows delays the actual shutdown, and "cancel shutdown" runs
    `shutdown /a` to abort it during that window)."""
    if command == "cancel":
        try:
            subprocess.run(["shutdown", "/a"], check=False)
            return ExecutionResult(True, "Shutdown cancelled.")
        except OSError as exc:
            return ExecutionResult(False, f"I couldn't cancel the shutdown: {exc}")

    def _confirmed_shutdown() -> ExecutionResult:
        try:
            subprocess.Popen(["shutdown", "/s", "/t", "15"])
            return ExecutionResult(
                True,
                "Shutting down in 15 seconds. Say \"cancel shutdown\" to stop it.",
            )
        except OSError as exc:
            return ExecutionResult(False, f"I couldn't shut down the PC: {exc}")

    return ExecutionResult(
        True,
        "Are you sure you want to shut down the PC? Say yes to confirm.",
        needs_confirmation=True,
        pending_action=_confirmed_shutdown,
    )


def _control_volume(command: str) -> ExecutionResult:
    """Volume up/down/mute/unmute via simulated media keys -- works on any
    Windows machine with no extra dependency. Setting an exact percentage
    needs the audio endpoint API (pycaw); if it's not installed, nudge the
    volume in the right direction instead of just failing outright."""
    try:
        import ctypes

        VK_VOLUME_MUTE = 0xAD
        VK_VOLUME_DOWN = 0xAE
        VK_VOLUME_UP = 0xAF
        KEYEVENTF_EXTENDEDKEY = 0x1
        KEYEVENTF_KEYUP = 0x2

        def _tap(vk: int, times: int = 1) -> None:
            for _ in range(times):
                ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
                ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

        if command == "mute":
            _tap(VK_VOLUME_MUTE)
            return ExecutionResult(True, "Muted.")
        if command == "unmute":
            _tap(VK_VOLUME_MUTE)  # mute is a toggle on Windows
            return ExecutionResult(True, "Unmuted.")
        if command == "up":
            _tap(VK_VOLUME_UP, 3)
            return ExecutionResult(True, "Volume up.")
        if command == "down":
            _tap(VK_VOLUME_DOWN, 3)
            return ExecutionResult(True, "Volume down.")

        # A specific percentage was requested ("set volume to 50").
        try:
            target_percent = max(0, min(100, int(command)))
        except ValueError:
            return ExecutionResult(False, "I didn't catch what volume you wanted.")

        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(target_percent / 100.0, None)
            return ExecutionResult(True, f"Volume set to {target_percent} percent.")
        except ImportError:
            # No pycaw -- approximate with taps instead of failing outright.
            _tap(VK_VOLUME_DOWN, 16)  # reset toward 0 first for a rough baseline
            _tap(VK_VOLUME_UP, round(target_percent / 100 * 16))
            return ExecutionResult(
                True,
                f"I nudged the volume toward {target_percent} percent "
                "(install pycaw for an exact level).",
            )
    except Exception as exc:
        return ExecutionResult(False, f"I couldn't change the volume: {exc}")


def ai_fallback(command: str) -> Optional[str]:
    """Extension point for the "else: send_to_ai(command)" branch --
    called only when nothing else (fast_dispatch's cheap checks, then the
    full regex pattern cascade) could make sense of the command. Returns
    the text JARVIS should speak, or None to fall back to the default "I
    didn't understand" response.

    Disabled by default: wiring this up needs an actual provider and API
    key, which isn't something to hardcode here. To enable it, replace
    this function's body with a call to whichever LLM you want --
    Anthropic, OpenAI, a local model, anything that takes a string and
    returns one back. Because this only ever runs on the leftover
    fraction of commands the fast/cheap paths didn't already handle, the
    common commands ("open chrome", "volume up", ...) never pay for it.
    """
    return None


def warm_up() -> None:
    """Eagerly import every optional dependency this module otherwise
    only imports lazily inside its handler functions (pyautogui,
    pygetwindow, PIL's screenshot support, pycaw) -- so the *first* time
    someone actually asks to type something, switch windows, take a
    screenshot, or set an exact volume level, that command isn't the one
    paying a one-off import cost on top of doing the thing. Call once at
    startup, before the wait loop, alongside the Whisper/TTS/wake-word/
    app-list preloading in main.py. Each import is best-effort: if an
    optional package isn't installed, that specific feature still reports
    it plainly the first time it's actually used, exactly as before --
    this only removes the *latency*, not the dependency."""
    for module_name in ("pyautogui", "pygetwindow", "PIL.ImageGrab", "pycaw.pycaw", "comtypes"):
        try:
            __import__(module_name)
        except ImportError:
            pass


def execute(intent: ParsedIntent) -> ExecutionResult:

    # ---------------------------------------------
    # GREETING
    # ---------------------------------------------

    if intent.type == IntentType.GREETING:

        return ExecutionResult(
            True,
            "Hello! How can I help you?"
        )

    if intent.type == IntentType.THANKS:
        return ExecutionResult(
            True,
            "You're welcome!"
        )

    if intent.type == IntentType.NEW_TAB:
        _open_default_browser("about:blank")
        return ExecutionResult(True, "Opening a new browser tab.")

    if intent.type == IntentType.OPEN_URL:
        url = intent.query
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        _open_default_browser(url, new_window=True)
        return ExecutionResult(True, f"Opening {url}.")

    if intent.type == IntentType.WEB_SEARCH:
        query = urllib.parse.quote_plus(intent.query)
        _open_default_browser(f"https://www.google.com/search?q={query}")
        return ExecutionResult(True, f"Searching the web for {intent.query}.")

    if intent.type == IntentType.SITE_SEARCH:
        url, label = _resolve_site_search(intent.meta or "google", intent.query)
        _open_default_browser(url)
        return ExecutionResult(True, f"Searching {label} for {intent.query}.")

    if intent.type == IntentType.CREATE_ITEM:
        global _LAST_CREATED_PATH
        kind, _, location_name = (intent.meta or "").partition(":")
        location = _resolve_location(location_name)
        location_label = (location_name or "desktop").strip().lower() or "desktop"

        raw_name = intent.query.strip()
        if raw_name:
            name = _sanitize_name(raw_name)
            if name is None:
                return ExecutionResult(False, "That's not a name I can use for a file or folder.")
        else:
            # No name was said at all ("make a folder on desktop") --
            # rather than fail or (as this used to) mistake the location
            # phrase itself for the name, default to an Explorer-style
            # auto-numbered name the same way right-click > New Folder does.
            name = None

        if kind == "folder":
            if name is None:
                name = _next_available_name(location, "New Folder")
            target = location / name
            try:
                target.mkdir(parents=True, exist_ok=False)
                _LAST_CREATED_PATH = target
                return ExecutionResult(True, f"Created the folder {name} on your {location_label}.")
            except FileExistsError:
                return ExecutionResult(False, f"A folder named {name} already exists there.")
            except OSError as exc:
                return ExecutionResult(False, f"I couldn't create that folder: {exc}")
        else:
            if name is None:
                file_name = _next_available_name(location, "New Text Document", suffix=".txt")
            else:
                # No extension spoken ("make a file named rohan") -- default
                # to .txt so the file is at least immediately openable/editable
                # instead of an extensionless file Windows won't know how to
                # handle by default.
                file_name = name if "." in name else f"{name}.txt"
            target = location / file_name
            if target.exists():
                return ExecutionResult(False, f"A file named {file_name} already exists there.")
            try:
                target.touch(exist_ok=False)
                _LAST_CREATED_PATH = target
                return ExecutionResult(True, f"Created {file_name} on your {location_label}.")
            except OSError as exc:
                return ExecutionResult(False, f"I couldn't create that file: {exc}")

    if intent.type == IntentType.DELETE_ITEM:
        name = _sanitize_name(intent.query)
        if name is None:
            return ExecutionResult(False, "That's not a name I can use for a file or folder.")
        kind_hint = (intent.meta or "").strip().lower()

        candidate: Optional[Path] = None
        is_dir = False
        literal = Path(name).expanduser()
        if literal.is_file() or literal.is_dir():
            candidate = literal
            is_dir = literal.is_dir()
        else:
            if kind_hint != "folder":
                matches = _search_user_files(name, limit=1)
                if matches:
                    candidate = matches[0]
            if candidate is None and kind_hint != "file":
                folder_matches = _search_user_folders(name, limit=1)
                if folder_matches:
                    candidate = folder_matches[0]
                    is_dir = True

        if candidate is None:
            return ExecutionResult(False, f"I couldn't find a file or folder named {name}.")

        kind_word = "folder" if is_dir else "file"
        found_candidate, found_is_dir = candidate, is_dir

        def _confirmed_delete() -> ExecutionResult:
            return _delete_path(found_candidate, found_is_dir)

        return ExecutionResult(
            True,
            f'Delete the {kind_word} "{candidate.name}"? Say yes to confirm.',
            needs_confirmation=True,
            pending_action=_confirmed_delete,
        )

    if intent.type == IntentType.RENAME_ITEM:
        old_name = _sanitize_name(intent.query)
        new_name = _sanitize_name(intent.meta)
        if old_name is None or new_name is None:
            return ExecutionResult(False, "I need both a valid current name and a valid new name.")

        literal = Path(old_name).expanduser()
        if literal.is_file() or literal.is_dir():
            candidate = literal
        else:
            file_matches = _search_user_files(old_name, limit=1)
            candidate = file_matches[0] if file_matches else None
            if candidate is None:
                folder_matches = _search_user_folders(old_name, limit=1)
                candidate = folder_matches[0] if folder_matches else None

        if candidate is None:
            return ExecutionResult(False, f"I couldn't find a file or folder named {old_name}.")

        # Keep the original extension if a bare new name was given for a file.
        if candidate.is_file() and "." not in new_name and candidate.suffix:
            new_name = new_name + candidate.suffix

        destination = candidate.with_name(new_name)
        if destination.exists():
            return ExecutionResult(False, f"Something named {new_name} already exists there.")
        try:
            candidate.rename(destination)
            if _LAST_CREATED_PATH == candidate:
                _LAST_CREATED_PATH = destination
            return ExecutionResult(True, f"Renamed {candidate.name} to {new_name}.")
        except OSError as exc:
            return ExecutionResult(False, f"I couldn't rename that: {exc}")

    if intent.type == IntentType.LIST_FILES:
        location = _resolve_location(intent.query)
        label = (intent.query or "desktop").strip().lower() or "desktop"
        if not location.exists():
            return ExecutionResult(False, f"I couldn't find your {label} folder.")
        try:
            names = sorted(p.name for p in location.iterdir() if not p.name.startswith("."))
        except OSError as exc:
            return ExecutionResult(False, f"I couldn't read that folder: {exc}")

        if not names:
            return ExecutionResult(True, f"Your {label} folder is empty.")
        shown = names[:8]
        remainder = len(names) - len(shown)
        spoken = ", ".join(shown)
        if remainder > 0:
            spoken += f", and {remainder} more"
        return ExecutionResult(True, f"Your {label} folder has: {spoken}.")

    if intent.type == IntentType.PATH_QUERY:
        reference = intent.query.strip().lower()

        # A generic reference ("the folder", "it", "that") with no actual
        # name -- resolve to whatever was most recently created, since
        # that's overwhelmingly what "that folder you created" means.
        if reference in ("folder", "file", "it", "that", "this", "item", "one", "", "the folder", "the file"):
            if _LAST_CREATED_PATH is not None and _LAST_CREATED_PATH.exists():
                return ExecutionResult(True, f"It's at {_LAST_CREATED_PATH}.")
            return ExecutionResult(False, "I don't have anything recent to give you the path for.")

        literal = Path(reference).expanduser()
        if literal.exists():
            return ExecutionResult(True, f"It's at {literal}.")

        matches = _search_user_files(reference, limit=1) or _search_user_folders(reference, limit=1)
        if matches:
            return ExecutionResult(True, f"It's at {matches[0]}.")
        return ExecutionResult(False, f"I couldn't find anything named {intent.query}.")

    if intent.type == IntentType.SCREENSHOT:
        try:
            from PIL import ImageGrab
        except ImportError:
            return ExecutionResult(False, "Screenshot support needs the Pillow package installed.")
        try:
            image = ImageGrab.grab()
            filename = f"Screenshot_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
            target = Path.home() / "Desktop" / filename
            image.save(target)
            return ExecutionResult(True, f"Saved a screenshot as {filename} on your Desktop.")
        except Exception as exc:
            return ExecutionResult(False, f"I couldn't take a screenshot: {exc}")

    if intent.type == IntentType.TIME_QUERY:
        now = datetime.now()
        hour = now.strftime("%I").lstrip("0") or "12"
        return ExecutionResult(True, f"It's {hour}:{now:%M %p}.")

    if intent.type == IntentType.DATE_QUERY:
        return ExecutionResult(True, f"Today is {datetime.now():%A, %B %d, %Y}.")

    if intent.type == IntentType.BATTERY_QUERY:
        if psutil is None:
            return ExecutionResult(False, "I need psutil installed to check the battery.")
        battery = psutil.sensors_battery()
        if battery is None:
            return ExecutionResult(True, "This device doesn't report a battery -- probably a desktop.")
        state = "charging" if battery.power_plugged else "on battery"
        return ExecutionResult(True, f"Battery is at {round(battery.percent)} percent, {state}.")

    if intent.type == IntentType.VOLUME_CONTROL:
        return _control_volume(intent.query)

    if intent.type == IntentType.PLAY_MUSIC:
        return _play_music(intent.query)

    if intent.type == IntentType.SYSTEM_SHUTDOWN:
        return _system_shutdown(intent.query)

    if intent.type == IntentType.TYPE_TEXT:
        return _type_text(intent.query)

    if intent.type == IntentType.KEY_ACTION:
        return _do_key_action(intent.query)

    if intent.type == IntentType.WINDOW_CONTROL:
        return _control_window(intent.query, intent.meta)

    if intent.type == IntentType.COPY_TEXT:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(intent.query)
            root.update()
            root.destroy()
            return ExecutionResult(True, "Text copied to the clipboard.")
        except Exception as exc:
            return ExecutionResult(False, f"I couldn't copy that text: {exc}")

    if intent.type == IntentType.PASTE_TEXT:
        return ExecutionResult(
            False,
            "Paste is available through the active application. I won't type into an unknown window.",
        )

    if intent.type == IntentType.COPY_FILE:
        source = Path(intent.query.strip().strip('"')).expanduser()
        destination = Path.home() / "Desktop" / source.name
        if not source.is_file():
            return ExecutionResult(False, f"I couldn't find {intent.query}.")
        try:
            shutil.copy2(source, destination)
            return ExecutionResult(True, f"Copied {source.name} to your Desktop.")
        except OSError as exc:
            return ExecutionResult(False, f"I couldn't copy that file: {exc}")

    # ---------------------------------------------
    # UNKNOWN
    # ---------------------------------------------

    if intent.type == IntentType.UNKNOWN:

        ai_response = ai_fallback(intent.query)
        if ai_response:
            return ExecutionResult(True, ai_response)

        return ExecutionResult(
            False,
            "I didn't understand that command."
        )

    # ---------------------------------------------
    # TEMPORARY LAUNCH TEST
    # ---------------------------------------------

    if intent.type == IntentType.LAUNCH_APP:

        app = intent.query.lower().strip()

        if "chrome" in app:
            # Special-cased instead of going through the generic table
            # below: _open_chrome() knows how to actually find chrome.exe
            # (see _find_chrome_exe) and falls back to the default browser
            # if Chrome genuinely isn't installed.
            _open_chrome(new_window=True)
            return ExecutionResult(True, "Opening Chrome.")

        # Fast path: a short list of very common apps, launched directly
        # with no subprocess/PowerShell round-trip. Anything not in this
        # list falls through to the generic resolver below instead of
        # failing outright.
        fast_paths = {
            "whatsapp": ("WhatsApp", [
                "explorer.exe",
                r"shell:AppsFolder\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
            ]),
            "notepad": ("Notepad", ["notepad.exe"]),
            "calculator": ("Calculator", ["calc.exe"]),
            "paint": ("Paint", ["mspaint.exe"]),
            "explorer": ("File Explorer", ["explorer.exe"]),
        }

        for app_name, (display_name, command) in fast_paths.items():
            if app_name in app:
                try:
                    subprocess.Popen(command)
                    return ExecutionResult(True, f"Opening {display_name}.")
                except OSError:
                    break  # fall through to the generic resolver below

        # Generic path: ask Windows what's actually installed (covers
        # anything with a Start Menu entry -- Brave, Spotify, VS Code,
        # Calendar, Mail, etc.) instead of only the apps hardcoded above.
        # Skip both the app match *and* the file-search fallback below for
        # anything too short/vague/compound to safely match against
        # (see apps.is_resolvable_name) -- otherwise a mis-transcribed or
        # nonsense word ("open you") reliably matches *something*, just
        # not the right thing (previously this let "open you" open an
        # unrelated file called layout.tsx, since "you" happens to be a
        # substring of "layout").
        if apps.is_resolvable_name(app):
            match = apps.resolve_app(app)
            if match:
                if apps.launch_by_app_id(match["app_id"]):
                    return ExecutionResult(True, f"Opening {match['name']}.")

            # Nothing matched as an app -- it might be a file or folder
            # instead. Folders matter just as much as files here: "open
            # my project folder" is exactly as common a request as "open
            # my resume", but folder search used to not be attempted at
            # all in this path (only file search was), so any project
            # directory could never be opened this way no matter what was
            # said.
            matches = _search_user_files(app, limit=7)
            if matches:
                os.startfile(str(matches[0]))
                return ExecutionResult(True, f"Opening {matches[0].name}.")

            folder_matches = _search_user_folders(app, limit=5)

            exact_folder = next(
                (
                    folder for folder in folder_matches
                    if app == folder.name.lower()
                    or app in folder.name.lower()
                ),
                None,
            )

            if exact_folder:
                os.startfile(str(exact_folder))
                return ExecutionResult(
                    True,
                    f"Opening the {exact_folder.name} folder."
                )

            if folder_matches:
                choices = ", ".join(folder.name for folder in folder_matches)
                return ExecutionResult(
                    False,
                    f"I found possible folders: {choices}. Please say the full folder name."
                )

        return ExecutionResult(
            False,
            f"I couldn't find an app or file called {intent.query}."
        )

    if intent.type == IntentType.CLOSE_APP:
        app = intent.query.lower().strip()

        if "explorer" in app:
            return ExecutionResult(
                False,
                "I won't force-close File Explorer because it is part of Windows.",
            )

        # Fast path for the handful of apps whose process name doesn't
        # obviously match how people say it out loud.
        fast_names = {
            "google chrome": "chrome",
            "chrome": "chrome",
            "whatsapp": "whatsapp",
            "calculator": "calculatorapp",
        }
        process_hint = next(
            (name for key, name in fast_names.items() if key in app), None
        )

        target = process_hint or app
        if not apps.is_resolvable_name(target):
            return ExecutionResult(
                False,
                f"I'm not sure what you mean by \"{intent.query}\" -- "
                "could you name the app?"
            )

        pids = apps.resolve_running_pids(target)
        if not pids:
            return ExecutionResult(False, f"I couldn't find {intent.query} running.")

        closed = apps.close_pids(pids)
        if closed:
            return ExecutionResult(True, f"Closed {intent.query}.")
        return ExecutionResult(
            False, f"I found {intent.query} running but couldn't close it."
        )

    if intent.type == IntentType.SAVE_FILE:
        return ExecutionResult(
            False,
            "I can open files, but saving requires knowing which app and file to edit.",
        )

    # ---------------------------------------------
    # OPEN FILE
    # ---------------------------------------------

    if intent.type == IntentType.OPEN_FILE:
        query = intent.query.strip().strip('"')
        candidate = Path(query).expanduser()

        if not candidate.is_file():
            matches = _search_user_files(query, limit=1)
            candidate = matches[0] if matches else None

        if candidate and candidate.is_file():
            os.startfile(str(candidate))
            return ExecutionResult(True, f"Opening {candidate.name}.")

        return ExecutionResult(False, f"I couldn't find a file named {query}.")

    # ---------------------------------------------
    # SEARCH
    # ---------------------------------------------

    if intent.type == IntentType.SEARCH_FILES:
        query = intent.query.strip().lower()
        matches = _search_user_files(query, limit=5)

        if matches:
            names = ", ".join(path.name for path in matches)
            return ExecutionResult(True, f"I found {len(matches)} match(es): {names}.")

        return ExecutionResult(False, f"I couldn't find anything matching {intent.query}.")

    return ExecutionResult(
        False,
        "I didn't understand that command."
    )


def execute_all(intents: list[ParsedIntent]) -> ExecutionResult:
    """Run a batch of intents from one utterance ("open notepad and search
    youtube for lofi") in sequence and combine their spoken responses into
    a single reply -- one TTS turnaround for the whole batch instead of
    one per step, which is both faster and less disjointed to listen to."""
    if not intents:
        return ExecutionResult(False, "I didn't understand that command.")
    if len(intents) == 1:
        return execute(intents[0])

    responses: list[str] = []
    overall_success = False
    for intent in intents:
        if intent.type in (IntentType.DELETE_ITEM, IntentType.SYSTEM_SHUTDOWN):
            # Both need a spoken yes/no confirmation, which only main.py's
            # single-command path knows how to do -- silently running (or
            # silently skipping) a delete or a full shutdown buried inside
            # a multi-command batch would be exactly the kind of
            # surprising destructive action this project should never
            # produce. Shutdown especially: it would also cut off
            # whatever other actions were queued after it in the batch.
            label = "Deletions" if intent.type == IntentType.DELETE_ITEM else "Shutting down"
            example = f"delete {intent.query}" if intent.type == IntentType.DELETE_ITEM else "shutdown pc"
            responses.append(
                f'For safety, {label.lower()} need to be their own command -- '
                f'say "{example}" by itself'
            )
            continue
        result = execute(intent)
        overall_success = overall_success or result.success
        if result.spoken_response:
            responses.append(result.spoken_response.rstrip("."))

    if not responses:
        return ExecutionResult(False, "I didn't understand that command.")
    return ExecutionResult(overall_success, ". ".join(responses) + ".")
