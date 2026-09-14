"""
core/tts.py

Offline speech feedback. pyttsx3 wraps SAPI5 on Windows -- no network
round-trip, which matters since this is on the critical latency path for
every command's confirmation.

Two things this deliberately does NOT do, because both are known
Windows/pyttsx3 failure modes:

1. Reuse one long-lived pyttsx3 engine across the whole session. That
   works for the first call or two, then silently stops producing any
   audio at all (no exception -- it just goes mute).

2. Spawn a brand-new OS thread for every single utterance and call
   pyttsx3.init() fresh on each one. pyttsx3's SAPI5 driver goes through
   COM (via comtypes), and COM apartments are initialized per-thread --
   tearing one down and spinning up a new one on a new thread for every
   utterance is a plausible source of an occasional hang, and matches
   the symptom actually seen (a stuck "Goodbye." blocking shutdown, with
   the orb window never closing).

Instead: ONE dedicated worker thread lives for the life of the process,
created the first time say() is used. COM/SAPI gets initialized once on
that thread and stays there. Each utterance still gets a fresh pyttsx3
engine OBJECT (fixing failure mode #1) but on that same stable thread
(avoiding failure mode #2). say() hands work to the worker over a queue
and waits on an Event with a timeout -- so even if the worker thread
somehow gets stuck on one particular utterance, say() itself can never
block the pipeline for longer than that timeout. A stuck worker just
means every later call falls through to the PowerShell fallback for the
rest of the session (each running in its own fresh process, so it can't
inherit whatever state the worker got stuck in) -- degraded, but never
hung.
"""

import queue
import subprocess
import threading


class Speaker:
    def __init__(self, rate: int = 175, voice: str = "default") -> None:
        self.rate = rate
        self.voice = voice
        self._pyttsx3_available: bool | None = None  # None = not probed yet
        self._queue: "queue.Queue[tuple[str, threading.Event, list]]" = queue.Queue()
        self._worker_started = False
        self._consecutive_failures = 0

    def load(self) -> None:
        """Probe whether pyttsx3 + SAPI5 works at all, and start the
        dedicated worker thread if so. Safe to call more than once."""
        if self._pyttsx3_available is not None:
            return
        try:
            import pyttsx3

            engine = pyttsx3.init("sapi5")
            engine.stop()
            self._pyttsx3_available = True
        except Exception as exc:
            print(f"[TTS] pyttsx3 unavailable ({exc}); using PowerShell fallback.")
            self._pyttsx3_available = False
            return

        if not self._worker_started:
            threading.Thread(target=self._worker_loop, daemon=True).start()
            self._worker_started = True

    def _worker_loop(self) -> None:
        import pyttsx3

        while True:
            text, done, outcome = self._queue.get()
            ok = False
            try:
                # A fresh engine object per utterance (avoids the
                # "goes mute after 1-2 calls" failure mode), but still on
                # this same long-lived, already-COM-initialized thread.
                engine = pyttsx3.init("sapi5")
                engine.setProperty("rate", self.rate)
                if self.voice != "default":
                    for installed_voice in engine.getProperty("voices"):
                        name = getattr(installed_voice, "name", "")
                        if self.voice.lower() in name.lower():
                            engine.setProperty("voice", installed_voice.id)
                            break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                ok = True
            except Exception as exc:
                print(f"[TTS] Engine failed: {exc}")
            finally:
                # Always signal, success or failure, so say() never waits
                # out its full timeout when we already know the answer --
                # but record the outcome too, so a *failed* utterance
                # (exception raised, no audio produced) doesn't get
                # silently treated as spoken just because the worker
                # returned quickly. This is what used to make some
                # commands go completely silent: pyttsx3 can fail without
                # raising all the way out to say(), and the old code only
                # fell back to PowerShell on a timeout, never on a fast
                # failure.
                outcome.append(ok)
                done.set()

    def say(self, text: str) -> None:
        if not text or not str(text).strip():
            return

        if self._pyttsx3_available is None:
            self.load()

        if self._pyttsx3_available:
            done = threading.Event()
            outcome: list[bool] = []
            self._queue.put((text, done, outcome))
            # Rough per-character budget, clamped to a sane range.
            timeout = min(15.0, max(4.0, 0.09 * len(text) + 2.0))
            finished = done.wait(timeout)
            if finished and outcome and outcome[0]:
                self._consecutive_failures = 0
                return

            if finished:
                print("[TTS] pyttsx3 failed to produce audio; using PowerShell fallback.")
            else:
                print("[TTS] pyttsx3 timed out; using PowerShell fallback.")

            self._consecutive_failures += 1
<<<<<<< HEAD
            if self._consecutive_failures >= 2:
=======
            if self._consecutive_failures >= 1:
>>>>>>> 45763e9 (Initial commit)
                # pyttsx3/SAPI5 is a known source of this exact failure
                # mode -- works once or twice, then silently stops
                # producing audio for the rest of the session. Once it's
                # failed twice in a row, stop trusting it: every further
                # call would otherwise pay for a full pyttsx3 attempt
                # (and its timeout) before falling back anyway, which is
                # both slower and still a coin flip on whether you hear
                # anything.
                print("[TTS] pyttsx3 failing repeatedly; switching to PowerShell for the rest of this session.")
                self._pyttsx3_available = False

        self._speak_powershell(text)

    def _speak_powershell(self, text: str) -> None:
        rate = max(-10, min(10, round((self.rate - 175) / 10)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; "
            "$s.Speak($args[0])"
        )
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script, text],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            print("[TTS] PowerShell fallback also timed out.")
