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

    audio = indata[:, 0]

    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))

    print(f"Microphone level: {rms:.2f}")


try:

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
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