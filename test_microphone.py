import sounddevice as sd
import numpy as np

print("================================")
print(" MICROPHONE INPUT TEST")
print("================================")

print("\nAvailable audio devices:\n")

print(sd.query_devices())

print("\nDefault input device:")
print(sd.default.device)

print("\nStarting microphone test...")
print("Speak into your microphone.")
print("You should see RMS values changing.")
print("Press Ctrl+C to stop.\n")


def audio_callback(indata, frames, time, status):

    if status:
        print("[STATUS]", status)

    audio = indata[:, 0].astype(np.float32)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print(f"Microphone RMS: {rms:.4f} | Peak: {peak:.4f}")
    if peak > 0.95:
        print("[WARNING] Clipping detected; lower microphone volume.")
    elif rms < 0.003:
        print("[WARNING] Signal is very quiet.")


try:

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        blocksize=1280,
        callback=audio_callback,
    ):

        print("[OK] Microphone stream started\n")

        while True:
            sd.sleep(1000)

except KeyboardInterrupt:

    print("\nMicrophone test stopped.")

except Exception as e:

    print("\n[ERROR]")
    print(type(e).__name__, ":", e)
