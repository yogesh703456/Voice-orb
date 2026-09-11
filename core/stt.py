"""
core/stt.py

Local speech-to-text on the short command clip only — never runs
continuously. Model is loaded once at startup and reused per call, since
model load time (not inference time) is the real cost to avoid paying
repeatedly.
"""


class Transcriber:
    def __init__(self, model_size: str = "tiny.en", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None  # loaded once in load()

    def load(self) -> None:
        """
        TODO (implementation milestone, days 3-4):
        - from faster_whisper import WhisperModel
        - self._model = WhisperModel(self.model_size, compute_type=self.compute_type)
        - Call this once at app startup, not per-command — model load is
          the expensive part, inference on a 2-4s clip is fast once loaded
        """
        raise NotImplementedError("Model load lands in days 3-4 milestone")

    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        TODO:
        - Convert pcm_bytes -> numpy float32 array
        - segments, _ = self._model.transcribe(audio_array, beam_size=1)
          (beam_size=1 trades a little accuracy for speed — right call here)
        - Join segment texts, return stripped string
        """
        raise NotImplementedError("Transcription lands in days 3-4 milestone")
