"""
orb/widget.py

The floating, draggable, always-on-top orb. Frameless transparent window,
positioned per config, animated with a live waveform/particle renderer.

This module owns ONLY presentation. It never imports from core/ directly —
every frame it reads the current OrbState + audio level from
orb.state.state_bus, which the backend pipeline writes to from its own
thread. That's the whole integration surface between frontend and backend.

A manual API (set_state / set_audio_level / emit / on / off) is also kept
so the widget can still be driven directly and previewed standalone with
`python -m orb.widget` (1-7 keys to force a state, 8/9/space/0 to test the
audio-reactive rendering, Esc to quit) without a live microphone or model
downloads.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, QPointF ,QSettings
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QApplication, QWidget

from orb.state import OrbState, state_bus


@dataclass(frozen=True)
class Profile:
    pulse_speed: float
    pulse_depth: float
    wave_speed: float
    wave_amp: float
    rotation: float
    brightness: float
    turbulence: float
    particle_alpha: float


PROFILES = {
    OrbState.IDLE:       Profile(.70, .055, .62, .82, .10, .82, .10, .34),
    OrbState.WAKE:       Profile(2.0, .12, 2.5, 1.45, .60, 1.18, .50, .48),
    OrbState.LISTENING:  Profile(1.1, .09, 3.1, 1.75, .30, 1.12, .72, .40),
    OrbState.THINKING:   Profile(1.5, .08, 1.8, 1.10, .95, 1.02, 1.00, .42),
    OrbState.EXECUTING:  Profile(1.85, .10, 3.35, 1.38, 1.35, 1.15, 1.20, .46),
    OrbState.RESPONDING: Profile(1.25, .095, 2.2, 1.48, .38, 1.10, .55, .38),
    OrbState.ERROR:      Profile(.9, .025, .40, .32, .0, .85, .08, .22),
}


class OrbEventBus:
    """Tiny dependency-free event bus, kept for manual/standalone driving."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable):
        self._listeners.setdefault(str(event), []).append(callback)
        return callback

    def off(self, event: str, callback: Callable) -> None:
        listeners = self._listeners.get(str(event), [])
        if callback in listeners:
            listeners.remove(callback)

    def emit(self, event: str, payload=None) -> None:
        for callback in tuple(self._listeners.get(str(event), [])):
            callback(payload)


class OrbWidget(QWidget):
    """The always-on-top glowing orb.

    Normal usage: the backend pipeline never touches this class directly.
    It calls state_bus.set_state(...) / state_bus.set_audio_level(...) from
    orb.state, and this widget's animation timer picks the change up on the
    next frame (see _sync_from_bus). set_state()/set_audio_level() below
    can also be called directly for standalone preview/testing.
    """

    def __init__(self, position: str = "bottom-right", size_px: int = 56, draggable: bool = True) -> None:
        super().__init__()
        self.position = position
        self.draggable = draggable

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.bus = OrbEventBus()
        self.state = OrbState.IDLE
        self.target_state = OrbState.IDLE
        self.transition = 1.0
        self.t = 0.0
        self.audio_level = 0.0
        self.audio_target = 0.0
        self.drag_offset = None
        self.settings = QSettings("VoiceOrb", "VoiceOrb")

        screen = QApplication.primaryScreen().availableGeometry()
        # Honor any configured size_px (clamped to a sane minimum so the
        # wave/particle renderer still has room to draw). The auto-computed
        # fallback only kicks in when no size was configured at all --
        # previously anything at or below the shipped config default of 56
        # was silently overridden by a much larger auto-computed size, so
        # the config file's own documented default could never actually
        # take effect.
        if size_px:
            side = max(40, size_px)
        else:
            side = max(190, min(int(min(screen.width(), screen.height()) * .205), 360))
        self.resize(side, side)
        saved_position = self.settings.value("orb_position")

        if saved_position is not None:
            self.move(saved_position)
        else:
            self._place(screen, side)

        rng = random.Random(91)
        self.particles = []
        for _ in range(480):
            a = rng.random() * math.tau
            z = rng.uniform(-1, 1)
            r = math.sqrt(max(0, 1 - z * z))
            self.particles.append(
                (math.cos(a) * r, math.sin(a) * r, z, rng.uniform(.35, 1.15))
            )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
    def closeEvent(self, event) -> None:
        self.settings.setValue("orb_position", self.pos())
        self.settings.sync()
        event.accept()    

    def _place(self, screen, side: int) -> None:
        margin = max(24, int(side * .12))
        positions = {
            "bottom-right": (screen.right() - side - margin, screen.bottom() - side - margin),
            "bottom-left": (screen.left() + margin, screen.bottom() - side - margin),
            "top-right": (screen.right() - side - margin, screen.top() + margin),
            "top-left": (screen.left() + margin, screen.top() + margin),
        }
        x, y = positions.get(self.position, positions["bottom-right"])
        self.move(x, y)

    # ---------- Public integration API ----------

    def on(self, event: str, callback: Callable):
        return self.bus.on(event, callback)

    def off(self, event: str, callback: Callable) -> None:
        self.bus.off(event, callback)

    def emit(self, event: str, payload=None) -> None:
        self.bus.emit(event, payload)
        try:
            self.set_state(event)
        except ValueError:
            pass

    def set_state(self, state) -> None:
        try:
            s = state if isinstance(state, OrbState) else OrbState(str(state).lower())
        except ValueError:
            return
        if s != self.target_state:
            self.target_state = s
            self.transition = 0.0

    def set_audio_level(self, level: float) -> None:
        """Set normalized microphone/TTS energy: 0.0 to 1.0."""
        try:
            self.audio_target = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self.audio_target = 0.0

    def reset_audio(self) -> None:
        self.audio_target = 0.0

    # ---------- Backend hookup ----------

    def _sync_from_bus(self) -> None:
        """Pull the latest state/level the pipeline thread published."""
        self.set_state(state_bus.state)
        self.set_audio_level(state_bus.audio_level)

    # ---------- Animation ----------

    def profile(self) -> Profile:
        a, b = PROFILES[self.state], PROFILES[self.target_state]
        q = min(1, self.transition)
        q = q * q * (3 - 2 * q)
        return Profile(
            *(getattr(a, k) * (1 - q) + getattr(b, k) * q
              for k in Profile.__dataclass_fields__)
        )

    def tick(self) -> None:
        self._sync_from_bus()

        self.t += .016
        if self.transition < 1:
            self.transition = min(1, self.transition + .045)
            if self.transition >= 1:
                self.state = self.target_state

        # Smooth audio envelope, preventing jitter from raw microphone values.
        self.audio_level += (self.audio_target - self.audio_level) * .16
        self.update()

    # ---------- Desktop interaction ----------

    def mousePressEvent(self, e) -> None:
        if self.draggable and e.button() == Qt.MouseButton.LeftButton:
            self.drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self.draggable and self.drag_offset is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self.drag_offset)

    def mouseReleaseEvent(self, _) -> None:
        self.drag_offset = None

    def keyPressEvent(self, e) -> None:
        """Manual overrides for standalone preview. Ignored once the real
        pipeline is driving state_bus every frame, since _sync_from_bus()
        will just overwrite these on the next tick."""
        mapping = {
            Qt.Key.Key_1: OrbState.IDLE,
            Qt.Key.Key_2: OrbState.WAKE,
            Qt.Key.Key_3: OrbState.LISTENING,
            Qt.Key.Key_4: OrbState.THINKING,
            Qt.Key.Key_5: OrbState.EXECUTING,
            Qt.Key.Key_6: OrbState.RESPONDING,
            Qt.Key.Key_7: OrbState.ERROR,
        }
        if e.key() in mapping:
            self.set_state(mapping[e.key()])
        elif e.key() == Qt.Key.Key_0:
            self.set_audio_level(0.0)
        elif e.key() == Qt.Key.Key_8:
            self.set_audio_level(.25)
        elif e.key() == Qt.Key.Key_9:
            self.set_audio_level(.65)
        elif e.key() == Qt.Key.Key_Space:
            self.set_audio_level(1.0 if self.audio_target < .5 else 0.0)
        elif e.key() == Qt.Key.Key_Escape:
            self.close()

    # ---------- Renderer ----------

    def draw_wave_ring(self, painter, cx, cy, radius, profile, layer, inner=False) -> None:
        phase = self.t * profile.wave_speed * math.tau + layer * 1.37
        n = 280
        pts = []

        # Audio energy scales the wave deformation. Listening/responding get
        # the strongest response; other states still accept the signal.
        audio_boost = 1.0 + self.audio_level * (
            1.55 if self.state == OrbState.LISTENING else 1.10
        )

        for i in range(n):
            a = i / n * math.tau
            f1 = math.sin(a * (5 + layer) + phase)
            f2 = math.sin(a * (10 + layer * 2) - phase * 1.18 + 1.4)
            f3 = math.sin(a * 17 + phase * .55 + layer * .9)
            deformation = (
                .025 * f1 + .012 * f2 + .005 * f3
            ) * profile.wave_amp * audio_boost
            deformation += .004 * math.sin(self.t * 2.1 + layer) * profile.turbulence
            rr = radius * (1 + deformation)

            x = cx + math.cos(a) * rr
            y = cy + math.sin(a) * rr * (.985 + .012 * math.sin(a * 2 - phase))
            pts.append(QPointF(x, y))

        if inner:
            alpha = int((110 - 18 * layer) * profile.brightness)
            width = 1.0
        else:
            alpha = int((210 - 22 * layer) * profile.brightness)
            width = 1.35 if layer == 0 else 1.05

        if self.state == OrbState.ERROR:
            color = QColor(235, 238, 245, max(1, alpha))
        elif layer % 2 == 0:
            color = QColor(255, 255, 255, max(1, alpha))
        else:
            color = QColor(0, 0, 0, max(1, int(alpha * .72)))

        pen = QPen(color)
        pen.setWidthF(width)
        painter.setPen(pen)
        for i in range(n - 1):
            painter.drawLine(pts[i], pts[i + 1])
        painter.drawLine(pts[-1], pts[0])

    def paintEvent(self, _) -> None:
        p = self.profile()
        pulse = 1 + math.sin(self.t * math.tau * p.pulse_speed) * p.pulse_depth
        # Audio also adds a very small radial response, while pulse remains primary.
        radius = min(self.width(), self.height()) * .335 * (
            pulse + self.audio_level * .028
        )
        cx, cy = self.width() / 2, self.height() / 2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Primary outer flowing wave boundary.
        for layer in range(7):
            self.draw_wave_ring(
                painter, cx, cy, radius * (1 + layer * .008), p, layer
            )
        for layer in range(3):
            self.draw_wave_ring(
                painter, cx, cy, radius * (1.055 + layer * .018), p, layer
            )

        # Secondary sparse particles.
        rot = self.t * p.rotation
        for x, y, z, size in self.particles:
            a = math.atan2(y, x) + rot
            rr = math.sqrt(x * x + y * y)
            px = cx + math.cos(a) * rr * radius * (.91 + .07 * max(0, z))
            py = cy + math.sin(a) * rr * radius * (.91 + .07 * max(0, z))
            alpha = int(
                65 * (.4 + .6 * (z + 1) / 2) * p.brightness * p.particle_alpha *
                (1 + .45 * self.audio_level)
            )
            painter.setPen(QPen(
                QColor(240, 244, 250, max(1, alpha)), max(.45, size)
            ))
            painter.drawPoint(QPointF(px, py))

        # Translucent core.
        core = radius * .43
        painter.setBrush(QColor(3, 5, 9, 180))
        painter.setPen(QPen(
            QColor(245, 248, 255, int(82 * p.brightness)), 1.2
        ))
        painter.drawEllipse(QPointF(cx, cy), core, core)

        # Inner flowing contours, visibly audio-reactive.
        inner_r = core * .72
        for layer in range(3):
            self.draw_wave_ring(
                painter, cx, cy,
                inner_r * (1 + layer * .07), p, layer, True
            )

        painter.end()


def _standalone_preview() -> None:
    """`python -m orb.widget` — preview the orb with no backend attached.
    1-7 force a state, 8/9/space/0 fake an audio level, Esc quits."""
    app = QApplication(sys.argv)
    orb = OrbWidget()
    orb.show()
    orb.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone_preview()
