import os
import time
from pathlib import Path
import numpy as np
import sounddevice as sd

from core.capture import find_microphone

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class SpeechRecognizer:
    def __init__(self, model_name: str | None = None, compute_type: str = "int8"):
        print("[STT] Loading Whisper English model...")

        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )

        model_name = model_name or os.getenv("VOICE_ORB_MODEL", "small.en")
        model_path = Path(model_name).expanduser()

        try:
            # A local directory avoids any network download. Otherwise,
            # faster-whisper downloads the named model on first startup.
            model_source = str(model_path) if model_path.is_dir() else model_name
            self.model = WhisperModel(model_source, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(
                "Whisper model could not be loaded. Connect to the internet "
                "for the first download, or set VOICE_ORB_MODEL to a local "
                f"model folder. Original error: {exc}"
            ) from exc

        self.sample_rate = 16000
        self.device = find_microphone()

        print("[STT] Whisper model loaded.")
        print(f"[MIC] Selected microphone: {self.device}")

    # ---------------------------------------------------------
    # FIND A REAL MICROPHONE
    # ---------------------------------------------------------
    def find_microphone(self):
        return find_microphone()

    # ---------------------------------------------------------
    # RECORD AUDIO
    # ---------------------------------------------------------
    def record_audio(self, duration=5, sample_rate=16000, device=None):
        """
        Records microphone audio.

        Returns:
            numpy float32 audio array
            None if recording failed
        """

        if device is None:
            device = self.device

        print("\n" + "=" * 60)
        print("LISTENING...")
        print("=" * 60)
        print("Speak now!")

        # If stored device becomes invalid, find another one.
        if device is None:
            device = self.find_microphone()

        if device is None:
            print("[MIC ERROR] No microphone found.")
            time.sleep(1)
            return None

        # Verify device before opening stream.
        try:
            dev = sd.query_devices(device)

            print(f"[MIC] Using input device: {device}")
            print(f"[MIC] Device: {dev['name']}")

            if dev["max_input_channels"] <= 0:
                print("[MIC ERROR] Selected device has no input channels.")
                self.device = self.find_microphone()
                return None

        except Exception as e:
            print(f"[MIC ERROR] Invalid microphone: {e}")

            self.device = self.find_microphone()
            return None

        try:
            # Record using sounddevice.
            # IMPORTANT:
            # We explicitly use the selected valid device.
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                blocking=True
            )

            # Convert (samples, 1) -> (samples,)
            audio = audio.flatten()

            # Check audio level
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))

            print(f"[MIC] RMS level: {rms:.4f}")
            print(f"[MIC] Maximum level: {peak:.4f}")

            if peak < 0.005:
                print("[MIC] WARNING: Audio is extremely quiet.")

            elif peak > 0.98:
                print(
                    "[MIC] WARNING: Microphone may be clipping. "
                    "Lower microphone input volume."
                )

            else:
                print("[MIC] Audio level looks OK.")

            # Prevent extremely loud/clipped values.
            audio = np.clip(audio, -1.0, 1.0)

            if peak > 0.98:
                audio *= 0.5

            return audio

        except Exception as e:
            print(f"[MIC ERROR] Recording failed: {e}")

            # Do NOT repeatedly hammer the same broken device.
            print("[MIC] Searching for another microphone...")

            self.device = self.find_microphone()

            time.sleep(1)

            return None

    # ---------------------------------------------------------
    # TRANSCRIBE
    # ---------------------------------------------------------
    def transcribe(self, audio):
        """
        Convert recorded audio into text using Whisper.
        """

        if audio is None:
            return ""

        if len(audio) == 0:
            return ""

        try:
            print("\n[STT] Transcribing...")

            # Make sure Whisper receives float32.
            audio = np.asarray(audio, dtype=np.float32)

            # Check whether audio is basically silence.
            rms = float(np.sqrt(np.mean(audio ** 2)))

            if rms < 0.003:
                print("[STT] Audio is too quiet.")
                return ""

            # Prevent clipped samples from dominating short commands.
            audio = np.clip(audio, -0.95, 0.95)

            # Remove quiet leading/trailing microphone noise.
            envelope = np.abs(audio)
            active = np.flatnonzero(envelope > max(0.008, rms * 0.18))
            if active.size:
                padding = int(0.15 * self.sample_rate)
                start = max(0, int(active[0]) - padding)
                end = min(len(audio), int(active[-1]) + padding)
                audio = audio[start:end]

            # Give quiet speech a little more signal without amplifying
            # already-loud or clipped recordings.
            trimmed_rms = float(np.sqrt(np.mean(audio ** 2)))
            if 0.008 < trimmed_rms < 0.08:
                audio *= min(2.0, 0.12 / trimmed_rms)
                audio = np.clip(audio, -0.95, 0.95)

            segments, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=5,
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
                    "Voice commands: open Chrome, open Google Chrome, "
                    "open Notepad, open Calculator, launch an application, "
                    "search for a file, find a document."
                ),
            )

            segments = list(segments)
            if not segments:
                print("[STT] No reliable speech segment detected.")
                return ""

            reliable_segments = [
                segment for segment in segments
                if getattr(segment, "no_speech_prob", 0.0) < 0.55
                and getattr(segment, "avg_logprob", -10.0) > -1.2
            ]

            if not reliable_segments:
                print("[STT] Ignoring low-confidence audio.")
                return ""

            text = " ".join(segment.text for segment in reliable_segments).strip()
            text = self._clean_hallucinated_repetition(text)

            # Normal commands are short. Reject runaway Whisper output.
            if len(text.split()) > 40:
                print("[STT] Ignoring repetitive or unclear audio.")
                return ""

            return " ".join(text.split())

        except Exception as e:
            print(f"[STT ERROR] Transcription failed: {e}")
            return ""

    @staticmethod
    def _clean_hallucinated_repetition(text):
        words = text.split()
        if len(words) < 8:
            return text

        for size in range(1, min(20, len(words) // 2) + 1):
            repeated = words[-size:]
            repeat_count = 1
            index = len(words) - (2 * size)
            while index >= 0 and words[index:index + size] == repeated:
                repeat_count += 1
                index -= size

            if repeat_count >= 3:
                words = words[:len(words) - ((repeat_count - 1) * size)]
                break

        return " ".join(words)

    # ---------------------------------------------------------
    # TEST MICROPHONE
    # ---------------------------------------------------------
    def test_microphone(self):
        print("\n[MIC TEST]")
        print("Speak a short phrase for 3 seconds...")

        audio = self.record_audio(
            duration=3,
            sample_rate=self.sample_rate
        )

        if audio is None:
            print("[MIC TEST] Recording failed.")
            return False

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        if rms < 0.003:
            print("[MIC TEST] Microphone is too quiet.")
            return False

        if peak > 0.98:
            print("[MIC TEST] Microphone is clipping; lower input volume.")
            return False

        print(f"[MIC TEST] RMS: {rms:.4f} | Peak: {peak:.4f}")
        print("[MIC TEST] Microphone level is suitable.")
        return True
