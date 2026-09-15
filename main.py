import re
import sys
import threading
import time
from pathlib import Path

from core.capture import record_command
from core.speech_recognition import SpeechRecognizer
from core.intent import IntentType, ParsedIntent, parse_commands
from core.executor import execute_all, warm_up
from core.tts import Speaker
from core.wake_word import WakeWordListener
from core import apps
from core import llm_service
from core.llm_brain import LLMParseError
from core.llm_provider import LLMError
from core.llm_service import LLMService
from core.llm_validator import LLMValidationError, validate_decision
from orb.state import OrbState, state_bus
from indexer.file_index import FileIndex
from core.executor import configure_file_index
from indexer.watcher import IndexWatcher


def load_settings() -> dict:
    path = Path(__file__).resolve().parent / "config" / "settings.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"Settings file was not found: {path}"
        )

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is not installed. Run: pip install pyyaml"
        ) from exc

    try:
        with path.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"settings.yaml has invalid YAML: {exc}"
        ) from exc

    if not isinstance(settings, dict):
        raise ValueError(
            "settings.yaml must contain a YAML mapping, not a list or plain text."
        )

    return settings

def print_startup_banner() -> None:
    print()
    print("=" * 60)
    print("                 JARVIS - VOICE ORB")
    print("=" * 60)
    print()
    print("[JARVIS] Starting up...")
    print()


def print_ready(wake_phrase: str) -> None:
    print()
    print("[JARVIS] System ready.")
    print(f"Say '{wake_phrase}' to wake me.")
    print("After the wake word, say your command.")
    print("Say 'shutdown jarvis' to stop.")
    print()


def is_shutdown(text: str) -> bool:
    cleaned = re.sub(r"[,.!?]", "", text.lower()).strip()
    return bool(
        re.search(
            r"\b(?:shutdown|shut down|exit|quit|stop)\s+jarvis\b",
            cleaned,
        )
    )


_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed",
    "do it", "go ahead", "affirmative", "correct", "please do",
}


# ---------------------------------------------------------------------------
# LLM BRAIN (optional) -- consultation logic
# ---------------------------------------------------------------------------
# Engaged ONLY when the regex parser produced a single UNKNOWN intent, so
# known commands keep their zero-cost fast path (rule 9). The LLM never
# executes anything: its decision is parsed (llm_brain), validated
# (llm_validator), and only then handed to the same executor the regex
# path uses. Every failure mode degrades to a spoken fallback -- never a
# crash, never an infinite wait.

# Spoken when the brain (local Ollama or cloud) can't be reached, per
# failure kind. Honest and short; never exposes error details or key
# material.
_LLM_FALLBACK_RESPONSES = {
    "timeout": "My brain is too slow to respond right now.",
    "network": "I can't reach my brain right now, so I'm running on local commands only.",
    "auth": "My brain service rejected its credentials, so requests are unavailable.",
    "rate_limit": "My brain is overloaded right now. Try again in a moment.",
    "provider": "My brain service is having problems right now.",
    "response": "My brain gave me a garbled answer, so I'll skip that.",
    "not_configured": "My brain isn't configured.",
    "error": "My brain failed just now.",
}
_LLM_GENERAL_FALLBACK = "My brain failed just now."
_LLM_REJECTED_RESPONSE = "I thought about that, but I can't do it safely."
_LLM_UNPARSEABLE_RESPONSE = "I couldn't quite work out how to do that."


class _LLMOutcome:
    """Result of one LLM consultation: either a direct spoken response
    (chat / graceful fallback) or a validated intent ready for the
    executor. Never both."""

    __slots__ = ("chat_response", "validated_intent")

    def __init__(self, chat_response: str = "", validated_intent: ParsedIntent | None = None) -> None:
        self.chat_response = chat_response
        self.validated_intent = validated_intent


def _clamp_for_speech(text: str, max_chars: int = 600) -> str:
    """Keep TTS output bounded and sentence-clean: never speak a truncated
    mid-sentence blob, never speak an empty response."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(". ", 0, max_chars)
    if cutoff < max_chars // 2:
        cutoff = max_chars
        text = text[:cutoff].rstrip()
        # Trim a trailing partial word.
        if " " in text:
            text = text.rsplit(" ", 1)[0]
        return text + "..."
    return text[: cutoff + 1]


def _consult_llm(transcript: str, llm: LLMService) -> _LLMOutcome | None:
    """Ask the LLM what to do with a transcript the parser couldn't read.

    Returns None when the LLM said "unknown" (the executor's normal
    I-didn't-understand response should run) or when nothing usable came
    back; otherwise an _LLMOutcome. Never raises.
    """
    started = time.perf_counter()
    try:
        decision = llm.decide(transcript)
    except LLMError as exc:
        latency = time.perf_counter() - started
        print(f"[LLM] unavailable after {latency:.1f}s (kind={exc.kind}); using fallback response.")
        return _LLMOutcome(chat_response=_LLM_FALLBACK_RESPONSES.get(exc.kind, _LLM_GENERAL_FALLBACK))
    except LLMParseError as exc:
        latency = time.perf_counter() - started
        print(f"[LLM] malformed decision after {latency:.1f}s: {exc}")
        return _LLMOutcome(chat_response=_LLM_UNPARSEABLE_RESPONSE)
    except Exception as exc:  # never let the optional brain crash the loop
        latency = time.perf_counter() - started
        print(f"[LLM] unexpected error after {latency:.1f}s: {type(exc).__name__}: {exc}")
        return _LLMOutcome(chat_response=_LLM_GENERAL_FALLBACK)

    latency = llm.last_latency_sec or (time.perf_counter() - started)
    print(f"[LLM] decision action={decision.action} intent={decision.intent!r} ({latency:.1f}s)")

    if decision.action == "chat":
        response = _clamp_for_speech(decision.response or "I'm not sure what to say to that.")
        return _LLMOutcome(chat_response=response)

    if decision.action != "command":
        # "unknown": no LLM action -- fall through to the executor's
        # standard response so behavior matches the non-LLM app.
        return None

    try:
        intent = validate_decision(decision)
    except LLMValidationError as exc:
        print(f"[LLM] validator rejected the command: {exc}")
        return _LLMOutcome(chat_response=_LLM_REJECTED_RESPONSE)

    if intent is None:
        return None
    print(f"[LLM] validated -> {intent}")
    return _LLMOutcome(validated_intent=intent)


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def is_affirmative(text: str) -> bool:
    """Used only for the yes/no follow-up after a destructive-action
    confirmation prompt (see core.executor's needs_confirmation). Defaults
    to "no" for anything that isn't clearly a yes -- a misheard or unclear
    answer should never accidentally confirm a deletion."""
    cleaned = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return cleaned in _AFFIRMATIVE or cleaned.startswith("yes")


def run_pipeline(settings: dict, stop_event: threading.Event, on_shutdown) -> None:
    """The voice pipeline loop. Runs on a background thread -- never touches
    Qt directly. Its only connection to the orb widget is state_bus (see
    orb/state.py): every stage of the loop below publishes what it's doing
    (idle / wake / listening / thinking / executing / responding / error)
    so the widget can render it, and record_command()'s on_level callback
    feeds real microphone energy in during LISTENING instead of the fake
    demo animation the standalone frontend used to run.
    """
    wake_cfg = settings.get("wake_word") or {}
    stt_cfg = settings.get("stt") or {}
    tts_cfg = settings.get("tts") or {}
    file_index_cfg = settings.get("file_index") or {}

    index_roots = file_index_cfg.get("roots") or [
    "~/Desktop",
    "~/Documents",
    "~/Downloads",
    ]

    index_path = (
        Path(__file__).resolve().parent
        / "file_index.sqlite3"
    )

    file_index = FileIndex(
        db_path=str(index_path),
        roots=index_roots,
    )

    configure_file_index(file_index)

    if file_index_cfg.get("refresh_on_startup", True):
        print("[INDEX] Building file index...")
        file_index.build()
    index_watcher = None    

    wake_phrase = str(wake_cfg.get("phrase") or "hey jarvis")
    sensitivity = float(wake_cfg.get("sensitivity") or 0.3)
    silence_timeout = float(stt_cfg.get("silence_timeout_sec") or 1.2)
    model_size = str(stt_cfg.get("model_size") or "small.en")
    compute_type = str(stt_cfg.get("compute_type") or "int8")

    # ------------------------------------------------------------
    # STARTUP: everything that costs real time -- the Whisper model, the
    # Start Menu app scan, the TTS engine probe, the wake-word model, and
    # every optional dependency the executor otherwise only imports on
    # first use -- loads here, once, up front. Nothing below this block
    # runs lazily on the first real command; by the time "WAITING FOR
    # ..." prints, the whole pipeline is actually sitting ready in RAM,
    # not still loading pieces of itself in the background.
    # ------------------------------------------------------------
    print_startup_banner()

    try:
        recognizer = SpeechRecognizer(model_name=model_size, compute_type=compute_type)
    except Exception as exc:
        print(f"[JARVIS] Could not start: {exc}")
        state_bus.set_state(OrbState.ERROR)
        return

    apps.preload()

    speaker = Speaker(rate=int(tts_cfg.get("rate") or 175), voice=str(tts_cfg.get("voice") or "default"))
    speaker.load()

    wake_listener = None
    try:
        wake_listener = WakeWordListener(
            phrase=wake_phrase,
            sensitivity=sensitivity,
            device=recognizer.device,
            on_wake_detected=lambda: state_bus.set_state(OrbState.WAKE),
        )
        wake_listener.load()
    except Exception as exc:
        print(f"[JARVIS] Wake-word model unavailable ({exc}).")
        print("[JARVIS] Falling back to listening for the wake phrase in speech.")
        wake_listener = None

    warm_up()

    # LLM brain (optional): built ONCE here -- config load + provider
    # construction -- so no command ever pays initialization cost. When
    # disabled or unconfigured this prints the reason and changes nothing
    # else about the pipeline.
    llm = llm_service.get_service(settings)
    print(f"[LLM] {llm.describe()}")

    if file_index_cfg.get("watch_for_changes", True):
        index_watcher = IndexWatcher(file_index, index_roots)
        index_watcher.start()

    # Only now -- once Whisper, the app list, TTS, the wake-word model,
    # and the optional executor dependencies have all actually finished
    # loading -- is the assistant genuinely ready, so this is where that
    # gets said, not before.
    print_ready(wake_phrase)
    state_bus.set_state(OrbState.IDLE)

    while not stop_event.is_set():
        try:
            if wake_listener is not None:
                print("=" * 60)
                print(f"WAITING FOR '{wake_phrase.upper()}'...")
                print("=" * 60)
                state_bus.set_state(OrbState.IDLE)
                heard = wake_listener.listen_until_wake()
                if not heard:
                    continue
                # WAKE state is set by the on_wake_detected callback above.
            else:
                print("=" * 60)
                print("LISTENING FOR WAKE PHRASE...")
                print("=" * 60)
                state_bus.set_state(OrbState.IDLE)
                audio = record_command(
                    device=recognizer.device,
                    silence_timeout_sec=silence_timeout,
                )
                text = recognizer.transcribe(audio) if audio is not None else ""
                if not text or wake_phrase.lower() not in text.lower():
                    print("[JARVIS] Wake phrase not heard.")
                    continue
                state_bus.set_state(OrbState.WAKE)

            state_bus.set_state(OrbState.LISTENING)
            audio = record_command(
                device=recognizer.device,
                silence_timeout_sec=silence_timeout,
                on_level=state_bus.set_audio_level,
            )
            state_bus.set_audio_level(0.0)

            if audio is None:
                print("[JARVIS] No command captured.")
                state_bus.set_state(OrbState.RESPONDING)
                speaker.say("I didn't catch that.")
                continue

            state_bus.set_state(OrbState.THINKING)
            _turn_started = time.perf_counter()
            _stt_started = time.perf_counter()
            text = recognizer.transcribe(audio)
            _stt_latency = time.perf_counter() - _stt_started
            if not text:
                print("[JARVIS] I didn't hear anything.")
                state_bus.set_state(OrbState.RESPONDING)
                speaker.say("I didn't hear anything.")
                continue

            text = text.strip()
            print()
            print("=" * 60)
            print("RECOGNIZED TEXT")
            print("=" * 60)
            print(text)
            print("=" * 60)

            if is_shutdown(text):
                print()
                print("[JARVIS] Shutting down...")
                state_bus.set_state(OrbState.RESPONDING)
                # Fire-and-forget: nothing happens after this in the
                # session, so there's no reason to let shutdown itself
                # wait on speech. This guarantees the orb closes promptly
                # even if TTS is slow or (despite the safeguards in
                # core/tts.py) gets stuck.
                threading.Thread(
                    target=speaker.say, args=("Goodbye.",), daemon=True
                ).start()
                break

            intents = parse_commands(text)
            print()
            for i, intent in enumerate(intents, 1):
                print(f"[INTENT {i}/{len(intents)}] {intent}")

            # LLM brain (optional): consulted ONLY when the parser produced
            # exactly one UNKNOWN intent, so every recognized command keeps
            # its zero-cost fast path. The LLM's decision is validated
            # before it can reach the executor (see _consult_llm).
            llm_chat_response = ""
            _llm_latency = None
            if (llm.available
                    and len(intents) == 1
                    and intents[0].type is IntentType.UNKNOWN):
                state_bus.set_state(OrbState.THINKING)
                outcome = _consult_llm(text, llm)
                if outcome is not None:
                    if outcome.validated_intent is not None:
                        intents = [outcome.validated_intent]
                    else:
                        llm_chat_response = outcome.chat_response
                    _llm_latency = llm.last_latency_sec

            _executor_started = time.perf_counter()
            if llm_chat_response:
                _executor_latency = time.perf_counter() - _executor_started
                print(
                    f"[TIMING] stt={_fmt_ms(_stt_latency)} "
                    f"llm={_fmt_ms(_llm_latency or 0.0)} "
                    f"executor={_fmt_ms(_executor_latency)} "
                    f"total={_fmt_ms(time.perf_counter() - _turn_started)}"
                )
                state_bus.set_state(OrbState.RESPONDING)
                speaker.say(llm_chat_response)
                time.sleep(0.3)
                continue

            state_bus.set_state(OrbState.EXECUTING)
            result = execute_all(intents)
            _executor_latency = time.perf_counter() - _executor_started
            print(
                f"[TIMING] stt={_fmt_ms(_stt_latency)} "
                f"llm={_fmt_ms(_llm_latency or 0.0)} "
                f"executor={_fmt_ms(_executor_latency)}"
            )
            print()
            print("=" * 60)
            print("JARVIS")
            print("=" * 60)
            print(result.spoken_response)
            print("=" * 60)

            state_bus.set_state(OrbState.RESPONDING)
            speaker.say(result.spoken_response)
            print(f"[TIMING] total={_fmt_ms(time.perf_counter() - _turn_started)}")

            if result.needs_confirmation and result.pending_action is not None:
                # A destructive action (currently just delete) asked a
                # yes/no question in its spoken_response above. Listen for
                # one short follow-up reply right now, without waiting for
                # the wake word again -- then either run the pending
                # action or back out, and speak that outcome too.
                state_bus.set_state(OrbState.LISTENING)
                confirm_audio = record_command(
                    device=recognizer.device,
                    silence_timeout_sec=silence_timeout,
                    max_duration_sec=4.0,
                    on_level=state_bus.set_audio_level,
                )
                state_bus.set_audio_level(0.0)
                confirm_text = recognizer.transcribe(confirm_audio) if confirm_audio is not None else ""
                print(f"[CONFIRM] heard: {confirm_text!r}")

                state_bus.set_state(OrbState.THINKING)
                if is_affirmative(confirm_text):
                    final_result = result.pending_action()
                    final_text = final_result.spoken_response
                else:
                    final_text = "Okay, I won't do that."

                print(final_text)
                state_bus.set_state(OrbState.RESPONDING)
                speaker.say(final_text)

            time.sleep(0.3)

        except KeyboardInterrupt:
            print()
            print("[JARVIS] Stopped by user.")
            break

        except Exception as exc:
            print()
            print(f"[ERROR] {exc}")
            print()
            state_bus.set_state(OrbState.ERROR)
            time.sleep(1)

    if wake_listener is not None:
        wake_listener.stop()

    state_bus.set_state(OrbState.IDLE)
    if index_watcher is not None:
        index_watcher.stop()
    on_shutdown()


def main() -> None:
    settings = load_settings()
    orb_cfg = settings.get("orb") or {}

    # Qt owns the main thread; the voice pipeline runs on a daemon thread
    # and only ever talks to it through orb.state.state_bus.
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    from orb.widget import OrbWidget

    app = QApplication(sys.argv)
    orb = OrbWidget(
        position=str(orb_cfg.get("position") or "bottom-right"),
        size_px=int(orb_cfg.get("size_px") or 56),
        draggable=bool(orb_cfg.get("draggable", True)),
    )
    orb.show()

    # Calling app.quit() straight from the pipeline's background thread
    # is documented as thread-safe, but a Qt signal is the more robust,
    # idiomatic way to cross threads: emitting it from the background
    # thread automatically becomes a queued delivery to this QObject's
    # thread (the GUI thread), where the slot below actually runs --
    # this belt-and-suspenders approach removes any doubt about the
    # shutdown request reliably reaching the Qt event loop.
    class _Shutdown(QObject):
        requested = Signal()

    shutdown = _Shutdown()

    def _do_shutdown() -> None:
        orb.close()
        app.quit()

    shutdown.requested.connect(_do_shutdown)

    stop_event = threading.Event()
    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(settings, stop_event, shutdown.requested.emit),
        daemon=True,
    )
    pipeline_thread.start()

    exit_code = app.exec()

    # Closing the orb window (or "shutdown jarvis" from the pipeline)
    # gets us here; ask the pipeline thread to stop too.
    stop_event.set()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
