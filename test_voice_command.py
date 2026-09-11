from core.speech_recognition import SpeechRecognizer


print("VOICE COMMAND ACCURACY TEST")
print("Say exactly: open Chrome")
print("Speak once, close to the microphone, then wait.")

recognizer = SpeechRecognizer()
audio = recognizer.record_audio(duration=3)
text = recognizer.transcribe(audio)

print(f"\nFINAL TRANSCRIPT: {text or '[nothing reliable detected]'}")
if text.lower() == "open chrome":
    print("[PASS] Command recognized accurately.")
else:
    print("[CHECK] Try lowering microphone volume or speaking closer.")
