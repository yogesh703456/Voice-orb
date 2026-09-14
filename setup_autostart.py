from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_FILE = PROJECT_DIR / "main.py"

PYTHON_EXE = Path(sys.executable)
PYTHONW_EXE = PYTHON_EXE.with_name("pythonw.exe")

if not PYTHONW_EXE.exists():
    raise RuntimeError(
        f"pythonw.exe was not found beside {PYTHON_EXE}. "
        "Run this script using the project's virtual environment."
    )

startup_folder = (
    Path(os.environ["APPDATA"])
    / "Microsoft"
    / "Windows"
    / "Start Menu"
    / "Programs"
    / "Startup"
)

startup_file = startup_folder / "VoiceOrb.cmd"

startup_file.write_text(
    f'@echo off\nstart "" /b "{PYTHONW_EXE}" "{MAIN_FILE}"\n',
    encoding="utf-8",
)

print(f"Autostart created successfully:\n{startup_file}")
print("Voice Orb will start automatically next time you sign in.")