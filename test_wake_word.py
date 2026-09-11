import time
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1280          # 80 ms at 16 kHz
THRESHOLD = 0.30
TEST_SECONDS = 30


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("OPENWAKEWORD MICROPHONE TEST")
print("=" * 60)

print()
print("Loading wake-word model...")

try:
    openwakeword.utils.download_models()
except Exception:
    pass

try:
    model = Model(
        wakeword_models=["hey_jarvis_v0.1"],
        inference_framework="onnx"
    )

    print("Model loaded successfully.")

except Exception as e:
    print()
    print("ERROR loading model:")
    print(e)
    raise


# ============================================================
# CALLBACK
# ============================================================

detected = False
max_score = 0.0


def audio_callback(indata, frames, callback_time, status):
    global detected
    global max_score

    if status:
        print("[AUDIO STATUS]", status)

    # --------------------------------------------------------
    # sounddevice gives us float32 audio in approximately
    # the range -1.0 to +1.0
    # --------------------------------------------------------

    audio = indata[:, 0].copy()

    # Calculate microphone level
    rms = float(np.sqrt(np.mean(audio ** 2)))

    peak = float(np.max(np.abs(audio)))

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert float32 [-1, +1] to signed 16-bit PCM
    # --------------------------------------------------------

    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # --------------------------------------------------------
    # DEBUG AUDIO LEVEL
    # --------------------------------------------------------

    print(
        f"\rMIC RMS={rms:.6f} "
        f"PEAK={peak:.6f} "
        f"PCM={pcm.min()}..{pcm.max()}",
        end=""
    )

    # --------------------------------------------------------
    # Run wake-word model
    # --------------------------------------------------------

    try:
        prediction = model.predict(pcm)

        # Current openWakeWord versions can return a dictionary,
        # so handle both dictionary and list-like results safely.

        if isinstance(prediction, dict):

            for name, value in prediction.items():

                try:
                    score = float(value)
                except Exception:
                    continue

                if score > max_score:
                    max_score = score

                if score >= THRESHOLD and not detected:

                    detected = True

                    print()
                    print()
                    print("=" * 60)
                    print("WAKE WORD DETECTED!")
                    print(f"MODEL : {name}")
                    print(f"SCORE : {score:.4f}")
                    print("=" * 60)

        else:
            # Some versions may return another structure.
            # Print it once for diagnosis.
            pass

    except Exception as e:

        print()
        print()
        print("[MODEL ERROR]")
        print(e)


# ============================================================
# START MICROPHONE
# ============================================================

print()
print()
print("=" * 60)
print("OPENING MICROPHONE")
print("=" * 60)

print()
print("Speak clearly:")
print()
print("        HEY JARVIS")
print()
print("Try saying it several times.")
print()
print(f"Test duration: {TEST_SECONDS} seconds")
print("Press Ctrl+C to stop.")
print()


try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):

        print("MICROPHONE ACTIVE")
        print("-" * 60)

        start_time = time.time()

        while time.time() - start_time < TEST_SECONDS:

            time.sleep(0.1)

except KeyboardInterrupt:

    print()
    print()
    print("Stopped by user.")

except Exception as e:

    print()
    print()
    print("[MICROPHONE ERROR]")
    print(e)


# ============================================================
# RESULT
# ============================================================

print()
print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

print(f"Maximum wake-word score: {max_score:.4f}")

if detected:

    print()
    print("SUCCESS!")
    print("The microphone and wake-word model are working.")

else:

    print()
    print("Wake word was NOT detected.")

print("=" * 60)