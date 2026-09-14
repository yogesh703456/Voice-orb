# Voice Orb — always-on voice assistant with a floating corner widget

A lightweight, always-on-boot voice assistant for Windows. Sits as a small
draggable glowing orb in the corner of your screen, listens for a wake word,
transcribes your command locally, executes it (open/search files, launch
apps), and replies out loud.

> Placeholder name — rename the package/repo once you're happy with a name
> for it (fits the same naming pattern as your other named projects).

## Design goals (from planning discussion)

- **Always-on wake word**, not push-to-talk
- **Low resource use** — must run comfortably on a mid-spec office laptop
- **Low latency** — end-to-end well under 5 seconds per command, not minutes
- **Visible presence** — a small draggable glowing orb in a screen corner,
  not a hidden tray icon, with state-based color/pulse (idle / listening /
  processing / speaking)
- **Spoken confirmation** (TTS), not just visual
- **Full background auto-start** on Windows boot (tray + Task Scheduler)

## Pipeline

```
mic (always on)
  -> wake word detector (openWakeWord, local, ~idle CPU)
  -> command capture (VAD-based recording of the utterance)
  -> STT on that short clip only (faster-whisper, tiny/base, int8)
  -> intent parser (rule-based, no model call)
  -> action executor (file search/open, app launch)
  -> TTS confirmation (pyttsx3, offline)
```

The orb widget is a thin observer on top of this pipeline's state machine —
it never drives logic, it just reflects `idle / listening / processing /
speaking / error`.

## Folder structure

```
voice-orb/
├── main.py                  # entrypoint: wires pipeline + orb + tray together
├── requirements.txt
├── config/
│   └── settings.yaml         # wake word phrase, model sizes, orb position, etc.
├── orb/
│   ├── widget.py              # PySide6 floating draggable orb window
│   └── state.py                # shared state enum + simple pub/sub for the pipeline -> widget link
├── core/
│   ├── wake_word.py           # openWakeWord listener loop
│   ├── capture.py              # VAD-based command recording
│   ├── stt.py                    # faster-whisper transcription
│   ├── intent.py                 # rule-based command parser
│   ├── executor.py             # runs the parsed intent (open/search/launch)
│   └── tts.py                    # offline speech feedback
├── indexer/
│   ├── file_index.py           # SQLite-backed filename index
│   └── watcher.py               # filesystem watcher to keep the index fresh
└── assets/
    └── (icons, sounds if any)
```

## Setup (once implementation lands)

```bash
<<<<<<< HEAD
python -m venv .venv
.venv\Scripts\activate
=======
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
>>>>>>> 45763e9 (Initial commit)
pip install -r requirements.txt
python main.py
```

## Auto-start on Windows boot

Handled by a setup script (added in the tray/autostart milestone) that
registers a Task Scheduler entry running `main.py` at logon, hidden.

## Status

Framework/skeleton stage — folder structure and module stubs only. Build
plan and milestones tracked in the project discussion; implementation
follows the day-by-day plan (wake word -> STT -> file search -> intent
parsing -> TTS -> orb widget -> autostart -> polish).
