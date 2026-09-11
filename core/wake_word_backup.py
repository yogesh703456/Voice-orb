"""
Wake-word listener for Voice Orb.

Listens continuously for the configured OpenWakeWord model
and calls on_wake_detected() when the wake word is detected.
"""

import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# Number of samples processed at a time.
# 1280 samples at 16 kHz = 80 ms.
CHUNK_SAMPLES = 1280

# Minimum time between two detections.
RETRIGGER_COOLDOWN_SEC = 2.0


# ============================================================
# WAKE WORD LISTENER
# ============================================================

class WakeWordListener:

    def __init__(
        self,
        phrase: str,
        sensitivity: float = 0.3,
        on_wake_detected: Optional[Callable[[], None]] = None,
    ) -> None:

        self.phrase = phrase
        self.sensitivity = sensitivity
        self.on_wake_detected = on_wake_detected

        # Runtime state
        self._running = False
        self._model: Optional[Model] = None

        # Used to prevent repeated triggers
        self._last_trigger_time = 0.0

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self) -> Model:
        print(f"[DEBUG] Loading wake-word model: {self.phrase}")

        model = Model(
            wakeword_models=[self.phrase],
            inference_framework="onnx",
        )

        return model

    # ========================================================
    # AUDIO CALLBACK
    # ========================================================

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:

        # Show sounddevice errors/warnings
        if status:
            print(f"[wake_word] stream status: {status}")

        # Listener has been stopped
        if not self._running:
            return

        # Current time
        current_time = time.monotonic()

        # Prevent repeated triggers
        if (
            current_time - self._last_trigger_time
            < RETRIGGER_COOLDOWN_SEC
        ):
            return

        # Make sure the model exists
        if self._model is None:
            print("[ERROR] Wake-word model is not loaded.")
            return

        # Convert microphone audio to int16
        audio_chunk = indata[:, 0].astype(np.int16)

        try:
            # Run wake-word prediction
            predictions = self._model.predict(audio_chunk)

        except Exception as error:
            print(
                f"[ERROR] Wake-word prediction failed: "
                f"{type(error).__name__}: {error}"
            )
            return

        # Get score for our wake word
        score = float(
            predictions.get(self.phrase, 0.0)
        )

        # Uncomment this if you want to see scores continuously.
        # print(f"[DEBUG] wake-word score = {score:.3f}")

        # Wake word detected
        if score >= self.sensitivity:

            self._last_trigger_time = current_time

            print(
                f"[wake_word] DETECTED: "
                f"'{self.phrase}' "
                f"(score={score:.3f})"
            )

            # Call the function supplied by the main application
            if self.on_wake_detected is not None:

                try:
                    self.on_wake_detected()

                except Exception as error:
                    print(
                        "[ERROR] Wake detection callback failed: "
                        f"{type(error).__name__}: {error}"
                    )

    # ========================================================
    # START LISTENER
    # ========================================================

    def start(self) -> None:

        print("[DEBUG] start() entered")

        # Load model
        self._model = self._load_model()

        print("[DEBUG] Wake-word model loaded successfully")

        # Start listener
        self._running = True

        # Reset trigger timer
        self._last_trigger_time = 0.0

        print(
            f"[DEBUG] Listener running = {self._running}"
        )

        print("[DEBUG] Opening microphone...")

        try:

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                callback=self._audio_callback,
            ):

                print(
                    f"[wake_word] Listening for "
                    f"'{self.phrase}' "
                    f"(sensitivity={self.sensitivity})"
                )

                print("[DEBUG] Microphone stream opened")

                print(
                    "[DEBUG] Say 'hey jarvis' now..."
                )

                print(
                    "[DEBUG] Press Ctrl+C to stop"
                )

                # Keep listener alive
                while self._running:
                    sd.sleep(100)

        except KeyboardInterrupt:

            print("[DEBUG] Keyboard interrupt received")

        except Exception as error:

            print(
                "[ERROR] Wake-word listener stopped: "
                f"{type(error).__name__}: {error}"
            )

            raise

        finally:

            self._running = False

            print("[DEBUG] Listener stopped")

    # ========================================================
    # STOP LISTENER
    # ========================================================

    def stop(self) -> None:

        print("[DEBUG] stop() called")

        self._running = False
