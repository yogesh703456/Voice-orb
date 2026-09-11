"""
orb/state.py

The single source of truth for "what is the assistant doing right now".
The backend pipeline (running on its own thread) writes to this; the Qt
orb widget (running on the main/GUI thread) polls it on every animation
tick. This indirection matters for a concrete reason: Qt widgets are not
thread-safe to call into directly from a non-GUI thread, and the pipeline
thread does blocking microphone/network work, so it cannot live on the
Qt event-loop thread either. A plain attribute on this shared object is
safe to write from the backend thread and read from the GUI thread
because CPython attribute get/set are atomic under the GIL — no lock
needed for these simple scalar values.

core/ modules should only ever import `state_bus` from here, never
anything from orb.widget (that keeps core/ free of any UI framework
import, per the project's original design goal).
"""

from __future__ import annotations

from enum import Enum


class OrbState(str, Enum):
    IDLE = "idle"
    WAKE = "wake"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    RESPONDING = "responding"
    ERROR = "error"


class StateBus:
    """Thread-safe-by-construction shared state.

    Backend pipeline stages call set_state()/set_audio_level() as they
    move through wake -> listen -> think -> execute -> respond -> idle.
    The orb widget's animation timer reads .state / .audio_level once
    per frame (~60fps) and never subscribes/blocks, so there's no
    cross-thread callback machinery to get wrong.
    """

    def __init__(self) -> None:
        self._state: OrbState = OrbState.IDLE
        self._audio_level: float = 0.0

    def set_state(self, state: "OrbState | str") -> None:
        try:
            self._state = state if isinstance(state, OrbState) else OrbState(str(state).lower())
        except ValueError:
            pass

    def set_audio_level(self, level: float) -> None:
        try:
            self._audio_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self._audio_level = 0.0

    @property
    def state(self) -> OrbState:
        return self._state

    @property
    def audio_level(self) -> float:
        return self._audio_level


# Process-wide singleton — every module (backend and UI) shares this instance.
state_bus = StateBus()
