import base64
import os
import re
from sarvamai import SarvamAI

def _load_env():
    """Load .env file if present without external dependencies."""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

def get_sarvam_client():
    _load_env()
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable is not set.")
    return SarvamAI(api_subscription_key=api_key)

SUPPORTED_LANGUAGES = {
    "hi-IN": {
        "name": "Hindi",
        "aliases": ["hi", "hindi"],
        "model": "bulbul:v4-flash",
        "speaker": "ritu_hi_medical",
    },
    "en-IN": {
        "name": "English (India)",
        "aliases": ["en", "english"],
        "model": "bulbul:v4-flash",
        "speaker": "ritu_en_medical",
    },
    "bn-IN": {
        "name": "Bengali",
        "aliases": ["bn", "bengali", "bangla"],
        "model": "bulbul:v4-flash",
        "speaker": "roopa_bn_conversational",
    },
    "ta-IN": {
        "name": "Tamil",
        "aliases": ["ta", "tamil"],
        "model": "bulbul:v4-flash",
        "speaker": "vijay_ta_narration",
    },
    "te-IN": {
        "name": "Telugu",
        "aliases": ["te", "telugu"],
        "model": "bulbul:v4-flash",
        "speaker": "kavitha_te_narration",
    },
    "kn-IN": {
        "name": "Kannada",
        "aliases": ["kn", "kannada"],
        "model": "bulbul:v4-flash",
        "speaker": "chetan_kn_conversation",
    },
    "mr-IN": {
        "name": "Marathi",
        "aliases": ["mr", "marathi"],
        "model": "bulbul:v4-flash",
        "speaker": "ishita_mr_conversational",
    },
    "gu-IN": {
        "name": "Gujarati",
        "aliases": ["gu", "gujarati"],
        "model": "bulbul:v4-flash",
        "speaker": "pooja_gu_conversational",
    },
    "pa-IN": {
        "name": "Punjabi",
        "aliases": ["pa", "punjabi"],
        "model": "bulbul:v4-flash",
        "speaker": "harpreet_pa_narration",
    },
    "ml-IN": {
        "name": "Malayalam",
        "aliases": ["ml", "malayalam"],
        "model": "bulbul:v3",
        "speaker": "anushka",
    },
    "od-IN": {
        "name": "Odia",
        "aliases": ["od", "odia"],
        "model": "bulbul:v3",
        "speaker": "anushka",
    },
}

def resolve_language_code(lang_input: str) -> str:
    """Normalize language code or common names to standard code."""
    lang_clean = lang_input.strip().lower()
    for code, meta in SUPPORTED_LANGUAGES.items():
        if lang_clean == code.lower() or lang_clean in meta["aliases"]:
            return code
    return "hi-IN" if "hi" in lang_clean else "en-IN"

def _chunk_text_by_sentences(text: str, max_chars: int = 800) -> list:
    """Split long text at sentence boundaries to respect API character limits."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if current_length + len(s) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_length = len(s)
        else:
            current_chunk.append(s)
            current_length += len(s) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def translate_medical_text(text: str, target_lang: str, source_lang: str = "en-IN") -> str:
    """
    Translate medical text using Sarvam AI's Mayura model with automatic chunking.
    """
    client = get_sarvam_client()
    target_code = resolve_language_code(target_lang)
    
    if target_code == source_lang:
        return text

    # Split into chunks under Mayura's 1000-character ceiling
    chunks = _chunk_text_by_sentences(text, max_chars=800)
    translated_parts = []

    for chunk in chunks:
        response = client.text.translate(
            input=chunk,
            source_language_code=source_lang,
            target_language_code=target_code,
            mode="formal",
            model="mayura:v1",
        )
        translated_parts.append(response.translated_text)

    return " ".join(translated_parts)

def generate_voice_briefing(
    text: str,
    output_path: str = None,
    language: str = "hi-IN",
    auto_translate: bool = True
) -> dict:
    """
    Synthesize complete medical dictation into speech in Hindi or regional Indian languages.
    Ensures zero questions are asked and audio contains full clinical analysis.
    """
    client = get_sarvam_client()
    lang_code = resolve_language_code(language)
    lang_meta = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["hi-IN"])

    if output_path is None:
        output_path = f"doctor_briefing_{lang_code}.wav"

    # Translate if necessary
    spoken_text = text
    if auto_translate and lang_code != "en-IN":
        spoken_text = translate_medical_text(text, target_lang=lang_code)

    # Clean text and ensure no questions remain in the dictation
    cleaned_sentences = [
        s.strip() for s in re.split(r'(?<=[.!?।])\s+', spoken_text.strip())
        if s.strip() and not s.strip().endswith("?") and "?" not in s
    ]
    cleaned_text = " ".join(cleaned_sentences).strip()[:2450]

    if not cleaned_text:
        raise ValueError("Text to synthesize cannot be empty.")

    # Call Sarvam Bulbul TTS
    response = client.text_to_speech.convert(
        text=cleaned_text,
        language_code=lang_code,
        model=lang_meta["model"],
        speaker=lang_meta["speaker"],
        output_audio_codec="wav",
    )

    if not response.audios:
        raise RuntimeError("No audio returned from Sarvam Text-to-Speech API.")

    audio_bytes = base64.b64decode(response.audios[0])
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return {
        "output_path": output_path,
        "language_code": lang_code,
        "language_name": lang_meta["name"],
        "speaker": lang_meta["speaker"],
        "spoken_text": cleaned_text,
        "file_size": len(audio_bytes)
    }

if __name__ == "__main__":
    sample_text = (
        "Chest radiograph confirms marked cardiomegaly with concurrent interstitial fibrosis and pleural thickening. "
        "Urgent computed tomography of the chest with intravenous contrast and echocardiography are the definitive next steps."
    )
    res = generate_voice_briefing(sample_text, language="hi-IN")
    print(f"Language: {res['language_name']} ({res['language_code']})")
    print(f"Dictation Text:\n{res['spoken_text']}")
    print(f"Saved: {res['output_path']} ({res['file_size']} bytes)")
