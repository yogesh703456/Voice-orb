"""
core/stt.py

Local speech-to-text on the short command clip only — never runs
continuously. Model is loaded once at startup and reused per call, since
model load time (not inference time) is the real cost to avoid paying
repeatedly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - dependency handled by requirements.txt
    WhisperModel = None


class Transcriber:
    def __init__(self, model_size: str = "tiny.en", compute_type: str = "int8") -> None:
        self.model_size = model_size or os.getenv("VOICE_ORB_MODEL", "tiny.en")
        self.compute_type = compute_type
        self._model = None  # loaded once in load()

    def load(self):
        """Create the underlying Whisper model once and cache it."""
        if self._model is not None:
            return self

        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )

        model_name = str(self.model_size)
        model_path = Path(model_name).expanduser()
        model_source = str(model_path) if model_path.is_dir() else model_name

        try:
            self._model = WhisperModel(model_source, compute_type=self.compute_type)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "Whisper model could not be loaded. Connect to the internet for the "
                "first download, or set VOICE_ORB_MODEL to a local model folder. "
                f"Original error: {exc}"
            ) from exc

        return self

    @staticmethod
    def _clean_hallucinated_repetition(text: str) -> str:
        """Drop repeated phrase tails that Whisper sometimes hallucinates."""
        words = text.split()
        if len(words) < 8:
            return text

        for size in range(1, min(20, len(words) // 2) + 1):
            tail = words[-size:]
            index = len(words) - (2 * size)
            repeat_count = 1
            while index >= 0 and words[index:index + size] == tail:
                repeat_count += 1
                index -= size

            if repeat_count >= 3:
                words = words[: len(words) - ((repeat_count - 1) * size)]
                break

        return " ".join(words)

    @staticmethod
    def _normalize_audio(pcm: object) -> np.ndarray:
        """Accept raw PCM bytes, NumPy arrays, or already-normalized floats."""
        if pcm is None:
            return np.empty(0, dtype=np.float32)

        if isinstance(pcm, np.ndarray):
            audio = np.asarray(pcm, dtype=np.float32)
        elif isinstance(pcm, (bytes, bytearray, memoryview)):
            raw = bytes(pcm)
            if not raw:
                return np.empty(0, dtype=np.float32)

            if len(raw) % 4 == 0:
                audio = np.frombuffer(raw, dtype=np.float32)
            elif len(raw) % 2 == 0:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                raise ValueError("PCM data length is neither a whole number of int16 nor float32 samples.")
        else:
            audio = np.asarray(pcm, dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32)
            audio = audio / 32768.0 if peak > 32768 else audio / 2147483648.0
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def transcribe(self, pcm_bytes: bytes | np.ndarray | None) -> str:
        """Convert recorded PCM audio into a short sentence."""
        if pcm_bytes is None:
            return ""

        audio = self._normalize_audio(pcm_bytes)
        if audio.size == 0:
            return ""

        if self._model is None:
            self.load()

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            without_timestamps=True,
            initial_prompt=(
                "Voice commands: open Chrome, open Google Chrome, open Notepad, "
                "open Calculator, launch an application, search for a file, find a document."
            ),
        )

        segments = list(segments)
        if not segments:
            return ""

        text = " ".join(segment.text for segment in segments).strip()
        text = self._clean_hallucinated_repetition(text)

        if len(text.split()) > 40:
            return ""

        return " ".join(text.split())
