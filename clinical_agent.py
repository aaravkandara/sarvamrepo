import json
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

SYSTEM_PROMPT = """You are an AI Clinical Radiologist Assistant providing a formal, complete radiologic dictation for an attending physician based on quantitative computer-aided detection (CAD) of a chest radiograph.

CRITICAL INSTRUCTIONS:
1. Deliver a FULL, complete clinical and radiologic analysis of the scan.
2. DO NOT ASK ANY QUESTIONS.
3. DO NOT use conversational closing phrases such as "Would you like me to...", "Please let me know...", "Do you want...", or "How can I assist you?".
4. Every sentence must be an objective, definitive clinical statement or recommendation.

Format your response with the following clear markdown sections:

### 1. Radiological Findings & Key Observations
Detailed radiological observations covering heart size, lung parenchyma, pleural spaces, and focal lesions with measured confidence scores.

### 2. Primary Clinical Impressions & Differential Diagnoses
Clear, ranked differential diagnoses explaining each detected pathology.

### 3. Recommended Clinical Follow-up & Next Steps
Definitive recommended follow-up actions (e.g. prior film comparison, high-resolution CT chest, echocardiogram, clinical and laboratory workup).

### 4. Comprehensive Radiologist Dictation
Provide a complete, comprehensive spoken dictation of the entire analysis (Findings, Impressions, and Recommendations) in continuous, formal medical prose. Do NOT ask any questions. Do NOT invite responses. End authoritatively with the final definitive clinical recommendations.
"""

def clean_dictation_script(text: str) -> str:
    """
    Remove any accidental question sentences or conversational closing offers.
    """
    # Remove sentences containing conversational offers or ending in question marks
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned_sentences = []
    
    banned_phrases = [
        "please let me know",
        "would you like",
        "do you want",
        "how can i",
        "let me know if",
        "feel free to",
        "what would you like",
        "if you would like"
    ]
    
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Drop sentences ending with '?'
        if s_clean.endswith("?") or "?" in s_clean:
            continue
        # Drop sentences containing banned conversational phrases
        if any(bp in s_clean.lower() for bp in banned_phrases):
            continue
        cleaned_sentences.append(s_clean)

    result = " ".join(cleaned_sentences).strip(' "\'*\n')
    return result

def generate_clinical_report(findings: dict) -> dict:
    """
    Generate a doctor assistant report from X-ray CAD findings using Sarvam-105b-conversations.
    """
    client = get_sarvam_client()

    top_findings_str = "\n".join(
        [f"- {item['pathology']}: confidence {item['score']:.2%}" for item in findings.get("top_findings", [])]
    )
    
    user_prompt = f"""Patient Chest X-ray CAD Analysis Results:
Image Analyzed: {findings.get('image_path', 'chest_xray')}

Top Radiographic Detections:
{top_findings_str}

Please generate the full clinical evaluation and comprehensive radiologist dictation. Remember: do not ask any questions or offer follow-up prompts.
"""

    response = client.chat.completions(
        model="sarvam-105b-conversations",
        messages=[
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
        ],
        temperature=0.2
    )

    report_text = response.choices[0].message.content or ""

    # Extract dictation script for audio synthesis
    dictation_script = ""
    if "### 4. Comprehensive Radiologist Dictation" in report_text:
        dictation_script = report_text.split("### 4. Comprehensive Radiologist Dictation")[-1].strip()
    elif "Comprehensive Radiologist Dictation" in report_text:
        dictation_script = report_text.split("Comprehensive Radiologist Dictation")[-1].strip()
    elif "### 4." in report_text:
        dictation_script = report_text.split("### 4.")[-1].strip()
    else:
        # Fallback to full report text
        dictation_script = report_text.strip()

    # Filter out questions or conversational filler
    dictation_script = clean_dictation_script(dictation_script)

    return {
        "full_report": report_text,
        "spoken_summary": dictation_script
    }

if __name__ == "__main__":
    from vision_analyzer import analyze_xray
    findings = analyze_xray("sample_xray.png")
    print("Generating complete clinical report and dictation with Sarvam-105b...")
    report = generate_clinical_report(findings)
    print("\n--- FULL REPORT ---")
    print(report["full_report"])
    print("\n--- DICTATION SCRIPT (NO QUESTIONS) ---")
    print(report["spoken_summary"])
