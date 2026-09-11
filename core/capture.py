"""
Microphone selection and VAD-based command recording.

Records after the wake word, stopping on silence instead of a fixed
window so short commands do not wait out a 3–5 second timer.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz


def find_microphone() -> Optional[int]:
    """Pick a real input device. Prefer the Windows default, skip virtual mics."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"[MIC ERROR] Could not list devices: {exc}")
        return None

    skip = ("camo", "stereo mix", "what u hear", "loopback", "mapper")

    def is_usable(index: int) -> bool:
        try:
            dev = sd.query_devices(index)
        except Exception:
            return False
        name = str(dev.get("name", "")).lower()
        return int(dev.get("max_input_channels", 0)) > 0 and not any(
            token in name for token in skip
        )

    default_in = sd.default.device[0] if sd.default.device else None
    if isinstance(default_in, (int, np.integer)) and is_usable(int(default_in)):
        print(f"[MIC] Using default input device: {int(default_in)}")
        return int(default_in)

    ranked = []
    for index, dev in enumerate(devices):
        if not is_usable(index):
            continue
        name = str(dev.get("name", "")).lower()
        try:
            hostapi = sd.query_hostapis(dev["hostapi"])["name"].lower()
        except Exception:
            hostapi = ""
        score = 0
        if "wasapi" in hostapi:
            score += 2
        if "mme" in hostapi:
            score += 1
        if "realtek" in name or "microphone" in name:
            score += 1
        ranked.append((score, index, dev["name"]))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        _, index, name = ranked[0]
        print(f"[MIC] Selected input device {index}: {name}")
        return index

    print("[MIC ERROR] No usable microphone found.")
    return None


def record_command(
    device: Optional[int] = None,
    silence_timeout_sec: float = 1.2,
    max_duration_sec: float = 8.0,
    sample_rate: int = SAMPLE_RATE,
    on_level: Optional[Callable[[float], None]] = None,
) -> Optional[np.ndarray]:
    """
    Capture a spoken command with energy-based VAD.

    If on_level is given, it's called once per audio chunk with the chunk's
    RMS normalized to roughly 0.0-1.0 (relative to speech_threshold) — this
    is what lets the orb widget's waveform actually react to your voice
    instead of only animating a canned demo level.

    Returns a float32 mono array, or None if recording failed / stayed silent.
    """
    if device is None:
        device = find_microphone()
    if device is None:
        return None

    chunk_sec = CHUNK_SAMPLES / sample_rate
    speech_threshold = 0.012
    started = False
    silent_chunks = 0
    needed_silence = max(1, int(silence_timeout_sec / chunk_sec))
    frames: list[np.ndarray] = []

    print("[MIC] Speak your command...")

    try:
        with sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            # Two independent budgets: how long we'll wait *before* speech
            # starts, and (separately, starting fresh once speech is
            # detected) how long the command itself is allowed to run.
            # Using a single shared chunk-count budget for both meant a
            # normal pause before speaking silently ate into the time
            # available to actually say the command, cutting it off early.
            wait_deadline = time.monotonic() + max_duration_sec + 2.0
            speech_deadline = None
            while True:
                now = time.monotonic()
                if not started and now > wait_deadline:
                    break
                if started and now > speech_deadline:
                    break
                block, overflowed = stream.read(CHUNK_SAMPLES)
                if overflowed:
                    print("[MIC] Input overflow (continuing).")
                audio = np.asarray(block[:, 0], dtype=np.float32)
                rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

                if on_level is not None:
                    try:
                        on_level(min(1.0, rms / (speech_threshold * 4)))
                    except Exception:
                        pass

                if not started:
                    if rms < speech_threshold:
                        continue
                    started = True
                    speech_deadline = time.monotonic() + max_duration_sec

                frames.append(audio)

                if rms < speech_threshold * 0.6:
                    silent_chunks += 1
                    if silent_chunks >= needed_silence:
                        break
                else:
                    silent_chunks = 0
    except Exception as exc:
        print(f"[MIC ERROR] Recording failed: {exc}")
        time.sleep(0.4)
        return None

    if not frames:
        print("[MIC] No speech captured.")
        return None

    audio = np.concatenate(frames)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    print(f"[MIC] Captured {len(audio) / sample_rate:.1f}s (rms={rms:.4f}, peak={peak:.4f})")

    if peak < 0.004:
        print("[MIC] WARNING: Audio is extremely quiet.")
        return None

    return np.clip(audio, -1.0, 1.0)
