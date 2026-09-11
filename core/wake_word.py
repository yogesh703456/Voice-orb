"""
Wake-word listener for Voice Orb.

Listens continuously for a built-in OpenWakeWord model and signals
when the wake word is detected. The microphone stream is released
after a hit so command capture can reuse the same device.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from core.capture import CHUNK_SAMPLES, SAMPLE_RATE, find_microphone

try:
    import openwakeword
    from openwakeword.model import Model
except ImportError:
    openwakeword = None
    Model = None


RETRIGGER_COOLDOWN_SEC = 2.0

# Human phrases -> OpenWakeWord built-in model ids.
# Custom names like "hey axiom" are not shipped models; map them to Jarvis.
BUILTIN_MODELS = {
    "hey jarvis": "hey_jarvis_v0.1",
    "hey_jarvis": "hey_jarvis_v0.1",
    "hey_jarvis_v0.1": "hey_jarvis_v0.1",
    "hello jarvis": "hey_jarvis_v0.1",
    "hey axiom": "hey_jarvis_v0.1",
    "alexa": "alexa_v0.1",
    "hey mycroft": "hey_mycroft_v0.1",
    "hey_mycroft_v0.1": "hey_mycroft_v0.1",
}


def resolve_model_id(phrase: str) -> str:
    key = phrase.strip().lower().replace("-", "_")
    key = " ".join(key.replace("_", " ").split())
    underscored = key.replace(" ", "_")
    return (
        BUILTIN_MODELS.get(key)
        or BUILTIN_MODELS.get(underscored)
        or BUILTIN_MODELS.get(phrase.strip())
        or "hey_jarvis_v0.1"
    )


class WakeWordListener:
    def __init__(
        self,
        phrase: str,
        sensitivity: float = 0.3,
        on_wake_detected: Optional[Callable[[], None]] = None,
        device: Optional[int] = None,
    ) -> None:
        self.phrase = phrase
        self.model_id = resolve_model_id(phrase)
        self.sensitivity = sensitivity
        self.on_wake_detected = on_wake_detected
        self.device = device

        self._running = False
        self._model = None
        self._last_trigger_time = 0.0
        self._wake_event = threading.Event()
        self._stop_requested = False

    def load(self) -> None:
        if Model is None:
            raise ImportError(
                "openwakeword is not installed. Run: pip install openwakeword"
            )

        try:
            openwakeword.utils.download_models()
        except Exception:
            pass

        print(f"[wake_word] Loading model '{self.model_id}' (phrase: '{self.phrase}')")
        self._model = Model(
            wakeword_models=[self.model_id],
            inference_framework="onnx",
        )
        print("[wake_word] Model loaded.")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[wake_word] stream status: {status}")

        if not self._running or self._model is None:
            return

        audio = np.asarray(indata[:, 0], dtype=np.float32)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        try:
            predictions = self._model.predict(pcm)
        except Exception as error:
            print(f"[wake_word] prediction failed: {type(error).__name__}: {error}")
            return

        # Always feed audio (needed for model state). Ignore hits during cooldown.
        now = time.monotonic()
        if now - self._last_trigger_time < RETRIGGER_COOLDOWN_SEC:
            return

        score = _best_score(predictions, self.model_id)
        if score < self.sensitivity:
            return

        self._last_trigger_time = now
        print(f"[wake_word] DETECTED '{self.phrase}' (score={score:.3f})")
        self._wake_event.set()
        self._running = False

        if self.on_wake_detected is not None:
            try:
                self.on_wake_detected()
            except Exception as error:
                print(
                    f"[wake_word] callback failed: {type(error).__name__}: {error}"
                )

    def listen_until_wake(self) -> bool:
        """
        Block until the wake word is heard. Releases the microphone
        afterwards so command recording can start.

        Returns True on detection, False if stopped.
        """
        if self._model is None:
            self.load()

        self._wake_event.clear()
        self._running = True
        self._stop_requested = False
        self._last_trigger_time = 0.0

        device = self.device if self.device is not None else find_microphone()
        self.device = device

        print(
            f"[wake_word] Listening for '{self.phrase}' "
            f"(model={self.model_id}, sensitivity={self.sensitivity})"
        )

        try:
            with sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=self._audio_callback,
            ):
                while self._running and not self._wake_event.is_set():
                    sd.sleep(50)
        except KeyboardInterrupt:
            self._running = False
            raise
        finally:
            self._running = False

        # If stop() was called while we were listening, report "not
        # detected" even though _wake_event may also have been set to
        # unblock the wait loop below -- otherwise the caller (main.py's
        # pipeline loop) treats a shutdown request as an actual wake-word
        # detection and proceeds to record and act on a command.
        return self._wake_event.is_set() and not self._stop_requested

    def stop(self) -> None:
        self._stop_requested = True
        self._running = False
        self._wake_event.set()


def _best_score(predictions, model_id: str) -> float:
    if not isinstance(predictions, dict) or not predictions:
        return 0.0

    values = []
    for key, value in predictions.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        values.append(score)
        if str(key).lower() == str(model_id).lower():
            return score

    return max(values) if values else 0.0
