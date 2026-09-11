import re
import sys
import threading
import time
from pathlib import Path

from core.capture import record_command
from core.speech_recognition import SpeechRecognizer
from core.intent import parse_commands
from core.executor import execute_all
from core.tts import Speaker
from core.wake_word import WakeWordListener
from core import apps
from orb.state import OrbState, state_bus


def load_settings() -> dict:
    path = Path(__file__).resolve().parent / "config" / "settings.yaml"
    try:
        import yaml

        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def print_header(wake_phrase: str) -> None:
    print()
    print("=" * 60)
    print("                 JARVIS - VOICE ORB")
    print("=" * 60)
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

    wake_phrase = str(wake_cfg.get("phrase") or "hey jarvis")
    sensitivity = float(wake_cfg.get("sensitivity") or 0.3)
    silence_timeout = float(stt_cfg.get("silence_timeout_sec") or 1.2)
    model_size = str(stt_cfg.get("model_size") or "small.en")
    compute_type = str(stt_cfg.get("compute_type") or "int8")

    print_header(wake_phrase)

    try:
        recognizer = SpeechRecognizer(model_name=model_size, compute_type=compute_type)
    except Exception as exc:
        print(f"[JARVIS] Could not start: {exc}")
        state_bus.set_state(OrbState.ERROR)
        return

    # Kick off the Start Menu scan now, in the background, so it's ready
    # by the time you say your first "open <app>" instead of that first
    # command paying for the PowerShell round-trip (see core/apps.py).
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
            text = recognizer.transcribe(audio)
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

            state_bus.set_state(OrbState.EXECUTING)
            result = execute_all(intents)
            print()
            print("=" * 60)
            print("JARVIS")
            print("=" * 60)
            print(result.spoken_response)
            print("=" * 60)

            state_bus.set_state(OrbState.RESPONDING)
            speaker.say(result.spoken_response)

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
