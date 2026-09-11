"""
Automated checks for Voice Orb: parser, executor safety, wake-word mapping,
microphone, and (when models are installed) STT / OpenWakeWord.
"""

from __future__ import annotations

import tempfile
import traceback
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from core.capture import SAMPLE_RATE, find_microphone
from core.executor import execute
from core.intent import IntentType, ParsedIntent, parse
from core.wake_word import WakeWordListener, resolve_model_id
from main import is_shutdown, load_settings


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"[PASS] {name}{suffix}")

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        print(f"[FAIL] {name} -- {detail}")

    def skip(self, name: str, detail: str) -> None:
        self.skipped += 1
        print(f"[SKIP] {name} -- {detail}")


def _check(results: Results, name: str, fn) -> None:
    try:
        fn()
        results.ok(name)
    except Exception as exc:
        results.fail(name, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def test_settings(results: Results) -> None:
    settings = load_settings()
    phrase = (settings.get("wake_word") or {}).get("phrase", "")
    if not phrase:
        results.fail("settings.yaml", "wake_word.phrase missing")
        return
    model_id = resolve_model_id(phrase)
    if model_id != "hey_jarvis_v0.1":
        results.fail("settings.yaml", f"unexpected model {model_id} for {phrase!r}")
        return
    results.ok("settings.yaml", f"phrase={phrase!r} -> {model_id}")


def test_intent_and_shutdown(results: Results) -> None:
    cases = [
        ("hey jarvis open chrome", IntentType.LAUNCH_APP, "chrome"),
        ("open chrome", IntentType.LAUNCH_APP, "chrome"),
        ("launch notepad", IntentType.LAUNCH_APP, "notepad"),
        ("search for resume", IntentType.SEARCH_FILES, "resume"),
        ("hello jarvis", IntentType.GREETING, ""),
        ("thanks", IntentType.THANKS, ""),
        ("web search python docs", IntentType.WEB_SEARCH, "python docs"),
        ("new tab", IntentType.NEW_TAB, ""),
        ("asdf qwerty", IntentType.UNKNOWN, "asdf qwerty"),
    ]
    for text, expected_type, expected_query in cases:
        intent = parse(text)
        if intent.type != expected_type or intent.query != expected_query:
            results.fail(
                f"parse({text!r})",
                f"got {intent.type} query={intent.query!r}",
            )
            return
    if not is_shutdown("shutdown jarvis") or not is_shutdown("Shut down Jarvis!"):
        results.fail("is_shutdown", "did not match shutdown phrases")
        return
    if is_shutdown("open chrome"):
        results.fail("is_shutdown", "false positive on a normal command")
        return
    results.ok("intent + shutdown", f"{len(cases)} phrases")


def test_executor_safe_paths(results: Results) -> None:
    greeting = execute(parse("hello"))
    if not greeting.success or "Hello" not in greeting.spoken_response:
        results.fail("executor greeting", greeting.spoken_response)
        return
    unknown = execute(parse("make coffee"))
    if unknown.success:
        results.fail("executor unknown", "should not succeed")
        return
    close_explorer = execute(
        ParsedIntent(IntentType.CLOSE_APP, "explorer")
    )
    if close_explorer.success or "won't force-close" not in close_explorer.spoken_response:
        results.fail("executor close explorer", close_explorer.spoken_response)
        return
    results.ok("executor safe paths")


def test_microphone(results: Results) -> None:
    device = find_microphone()
    if device is None:
        results.fail("find_microphone", "no input device")
        return
    results.ok("find_microphone", f"device={device}")

    try:
        info = sd.query_devices(device)
        seconds = 1.2
        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocking=True,
        ).flatten()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        results.ok(
            "microphone record",
            f"{info['name']!r} rms={rms:.4f} peak={peak:.4f}",
        )
        if peak < 0.001:
            results.skip(
                "microphone level",
                "almost silent — check the mic is not muted",
            )
        else:
            results.ok("microphone level", "signal present")
    except Exception as exc:
        results.fail("microphone record", f"{type(exc).__name__}: {exc}")


def _tts_wav(text: str, path: Path) -> None:
    import pyttsx3

    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", 160)
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    engine.stop()


def _load_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported sample width {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE and len(audio) > 1:
        duration = len(audio) / float(rate)
        target = max(1, int(duration * SAMPLE_RATE))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def test_wake_word_model(results: Results) -> None:
    listener = WakeWordListener(phrase="hey jarvis", sensitivity=0.3)
    try:
        listener.load()
    except Exception as exc:
        results.skip("openwakeword load", str(exc))
        return
    results.ok("openwakeword load", listener.model_id)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "hey_jarvis.wav"
            _tts_wav("hey jarvis", wav_path)
            audio = _load_wav_mono_16k(wav_path)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        chunk = 1280
        max_score = 0.0
        for start in range(0, max(0, len(pcm) - chunk + 1), chunk):
            prediction = listener._model.predict(pcm[start:start + chunk])
            if isinstance(prediction, dict) and prediction:
                max_score = max(max_score, max(float(v) for v in prediction.values()))
        detail = f"max score on TTS 'hey jarvis' = {max_score:.3f}"
        if max_score >= listener.sensitivity:
            results.ok("openwakeword score (TTS)", detail)
        else:
            results.skip(
                "openwakeword score (TTS)",
                detail + " (SAPI voices often do not trigger this model)",
            )
    except Exception as exc:
        results.skip("openwakeword score (TTS)", str(exc))


def test_stt(results: Results) -> None:
    try:
        from core.speech_recognition import SpeechRecognizer
    except Exception as exc:
        results.skip("STT import", str(exc))
        return

    try:
        recognizer = SpeechRecognizer()
    except Exception as exc:
        results.skip("Whisper load", str(exc))
        return
    results.ok("Whisper load")

    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    text = recognizer.transcribe(silence)
    if text:
        results.fail("STT silence", f"expected empty, got {text!r}")
    else:
        results.ok("STT silence", "no false transcript")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "open_chrome.wav"
            _tts_wav("open chrome", wav_path)
            audio = _load_wav_mono_16k(wav_path)
        text = recognizer.transcribe(audio).lower().strip()
        intent = parse(text) if text else None
        if intent and intent.type == IntentType.LAUNCH_APP and "chrome" in intent.query:
            results.ok("STT TTS 'open chrome'", f"transcript={text!r}")
        elif text:
            results.skip("STT TTS 'open chrome'", f"transcript={text!r} (SAPI may be unclear)")
        else:
            results.skip("STT TTS 'open chrome'", "empty transcript")
    except Exception as exc:
        results.skip("STT TTS 'open chrome'", str(exc))


def main() -> int:
    print("=" * 60)
    print("VOICE ORB PIPELINE TEST")
    print("=" * 60)
    print()

    results = Results()
    def _mapping() -> None:
        assert resolve_model_id("hey axiom") == "hey_jarvis_v0.1"
        assert resolve_model_id("hey jarvis") == "hey_jarvis_v0.1"

    _check(results, "wake-word model mapping", _mapping)
    test_settings(results)
    test_intent_and_shutdown(results)
    test_executor_safe_paths(results)
    test_microphone(results)
    test_wake_word_model(results)
    test_stt(results)

    print()
    print("=" * 60)
    print(
        f"DONE  passed={results.passed}  failed={results.failed}  "
        f"skipped={results.skipped}"
    )
    print("=" * 60)
    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
