import base64
import os
from sarvamai import SarvamAI

# Load .env into os.environ if present
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise ValueError("SARVAM_API_KEY environment variable is not set.")

client = SarvamAI(api_subscription_key=api_key)

text_to_speak = (
    "Preliminary chest X-ray analysis complete. Normal cardiac silhouette "
    "and clear lung fields observed. No acute focal consolidation identified."
)

print(f"Calling Sarvam Text-to-Speech API (model: bulbul:v4-flash)...")
response = client.text_to_speech.convert(
    text=text_to_speak,
    language_code="en-IN",
    model="bulbul:v4-flash",
    speaker="ritu_en_medical",
    output_audio_codec="wav",
)

output_filename = "medical_briefing.wav"

if response.audios:
    # Sarvam returns base64-encoded audio clips
    audio_bytes = base64.b64decode(response.audios[0])
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)
    print(f"Success! Audio briefing saved to {output_filename} ({len(audio_bytes)} bytes)")
else:
    print("Received empty audio response.")
