"""
core/apps.py

Generic "resolve a spoken app name" layer, replacing the old approach of
hand-picking a handful of apps (chrome, notepad, calculator...) inside
executor.py. Two separate problems, two separate data sources:

LAUNCHING something by name -> asks Windows what's actually installed.
`Get-StartApps` (PowerShell) lists every Start Menu entry: both
traditional .exe-backed shortcuts (Brave, Spotify, VS Code, ...) and
UWP/Store apps (Calendar, Mail, ...). Every entry it returns can be
launched the same way -- `explorer.exe shell:AppsFolder\\<AppID>` -- so
one launch mechanism covers both kinds of app.

CLOSING something by name -> asks Windows what's currently *running*
(via psutil), since the thing you want to close is a live process, not
a Start Menu shortcut, and its process name often doesn't match its
Start Menu display name (e.g. Calculator's Start Menu entry vs. its
process name differ across Windows versions).

Both use rapidfuzz to fuzzy-match the spoken name against whichever
list is relevant, so "brave", "open calendar", "close chrome" etc. all
work without a name needing to be hardcoded anywhere.

The Start Apps list is the slow part (a few hundred ms of PowerShell
startup), so it's fetched once per process and cached -- see preload().
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from typing import Optional

try:
    from rapidfuzz import fuzz
    from rapidfuzz import process as fuzzy_process
except ImportError:
    fuzz = None
    fuzzy_process = None

try:
    import psutil
except ImportError:
    psutil = None


_cache_lock = threading.Lock()
_start_apps: Optional[list[dict]] = None

_VAGUE_TARGETS = {
    "it", "that", "this", "them", "one", "thing", "app", "program", "software",
    "you", "your", "yourself", "me", "myself", "him", "her", "someone",
    "something", "anything", "anyone",
}


def is_resolvable_name(spoken_name: str) -> bool:
    """Reject names too short, too vague, or too compound to safely
    fuzzy-match. Without this, a 2-letter pronoun ("it") or a whole
    sentence ("brave and search youtube") gets matched against every
    installed app or running process, and fuzzy scoring on garbage input
    reliably finds *something* that scores just high enough -- usually
    the wrong thing (an uninstaller, a maintenance tool, an unrelated
    running process). Better to say "I don't know what you mean" than
    to confidently do the wrong thing."""
    name = spoken_name.strip().lower()
    if not name or len(name) < 3:
        return False
    if name in _VAGUE_TARGETS:
        return False
    if re.search(r"\b(?:and|then)\b", name):
        return False
    return True


def _load_start_apps() -> list[dict]:
    global _start_apps
    with _cache_lock:
        if _start_apps is not None:
            return _start_apps
        apps: list[dict] = []
        try:
            result = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-Command",
                    "Get-StartApps | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            data = json.loads(result.stdout or "[]")
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                name = entry.get("Name")
                app_id = entry.get("AppID")
                if name and app_id:
                    apps.append({"name": name, "app_id": app_id})
        except Exception as exc:
            print(f"[apps] Could not list installed apps: {exc}")
        _start_apps = apps
        return _start_apps


def preload() -> None:
    """Kick off the Start Menu scan early (call this once, right after
    the pipeline starts) so the first 'open <app>' of the session isn't
    the one paying for the PowerShell round-trip."""
    threading.Thread(target=_load_start_apps, daemon=True).start()


def resolve_app(spoken_name: str, score_cutoff: float = 85.0) -> Optional[dict]:
    """Fuzzy-match a spoken app name against every installed Start Menu
    entry (Win32 + UWP alike). Returns {'name', 'app_id'} for the best
    match, or None if nothing crosses the confidence cutoff -- callers
    should fall back to file search or an error at that point rather
    than guess."""
    apps = _load_start_apps()
    if not apps:
        return None

    needle = spoken_name.strip().lower()

    # Exact/substring hit first. Short common names ("photos", "mail",
    # "calculator") often score *below* the fuzzy cutoff purely because
    # they're much shorter than the full Start Menu entry ("Microsoft
    # Photos"), even though they're an unambiguous match -- so check for
    # a literal match before falling back to fuzzy scoring at all.
    exact = [a for a in apps if needle == a["name"].strip().lower()]
    if exact:
        return exact[0]
    substring = [a for a in apps if needle and needle in a["name"].strip().lower()]
    if substring:
        # Prefer the shortest matching name: closest to what was actually said.
        return min(substring, key=lambda a: len(a["name"]))

    if fuzzy_process is None:
        return None

    names = [a["name"] for a in apps]
    # Try a strict cutoff first, then relax it once. The lower pass exists
    # specifically to rescue names mangled by imperfect STT ("fotos" for
    # "Photos", "kalkulator" for "Calculator") that the strict cutoff
    # would otherwise reject outright.
    for cutoff in (score_cutoff, max(60.0, score_cutoff - 20)):
        match = fuzzy_process.extractOne(
            needle, names, scorer=fuzz.WRatio, score_cutoff=cutoff
        )
        if match:
            _matched_name, _score, index = match
            return apps[index]
    return None


def launch_by_app_id(app_id: str) -> bool:
    """Launch any Get-StartApps entry -- Win32 or UWP -- via the same
    mechanism the actual Start Menu uses to launch it."""
    try:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        return True
    except OSError:
        return False


def _running_process_names() -> dict[str, list[int]]:
    """Currently running processes, keyed by lowercased name with the
    .exe suffix stripped, e.g. {'chrome': [pid, pid, ...]}."""
    grouped: dict[str, list[int]] = {}
    if psutil is None:
        return grouped
    for proc in psutil.process_iter(["pid", "name"]):
        raw_name = (proc.info.get("name") or "").strip()
        if not raw_name:
            continue
        base_name = raw_name.rsplit(".", 1)[0].lower()
        grouped.setdefault(base_name, []).append(proc.info["pid"])
    return grouped


def resolve_running_pids(spoken_name: str, score_cutoff: float = 82.0) -> list[int]:
    """Fuzzy-match a spoken app name against currently running process
    names and return the pids of the best-matching process group."""
    grouped = _running_process_names()
    if not grouped:
        return []

    # Try an exact/substring hit first -- cheap and avoids rapidfuzz
    # picking an unrelated process when the spoken name is already a
    # clean match (e.g. "chrome" should never need fuzzy help).
    needle = spoken_name.strip().lower()
    for name, pids in grouped.items():
        if needle == name or needle in name or name in needle:
            return pids

    if fuzzy_process is None:
        return []

    match = fuzzy_process.extractOne(
        needle, list(grouped.keys()), scorer=fuzz.WRatio, score_cutoff=score_cutoff
    )
    if not match:
        return []
    matched_name, _score, _index = match
    return grouped[matched_name]


def close_pids(pids: list[int]) -> int:
    """Ask each pid to terminate. Returns how many were actually
    signaled (not necessarily how many have exited yet)."""
    if psutil is None:
        return 0
    closed = 0
    for pid in pids:
        try:
            psutil.Process(pid).terminate()
            closed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return closed
