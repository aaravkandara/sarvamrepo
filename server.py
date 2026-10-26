#!/usr/bin/env python3
"""
Flask Web Server for Sarvam AI Chest X-Ray Assistant
---------------------------------------------------
Serves the web UI and provides the /api/analyze endpoint for
radiograph computer-aided detection and multilingual voice dictation.
"""

import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

from vision_analyzer import analyze_xray
from clinical_agent import generate_clinical_report
from voice_briefing import generate_voice_briefing, resolve_language_code, SUPPORTED_LANGUAGES

# Ensure output directory exists (support read-only Vercel serverless filesystem)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.environ.get("VERCEL"):
    GENERATED_DIR = "/tmp/generated"
    os.environ["TORCH_HOME"] = "/tmp/torch"
else:
    GENERATED_DIR = os.path.join(STATIC_DIR, "generated")

os.makedirs(GENERATED_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample_xray")
def sample_xray():
    sample_path = os.path.join(BASE_DIR, "sample_xray.png")
    if os.path.exists(sample_path):
        return send_file(sample_path, mimetype="image/png")
    return "Sample image not found", 404

@app.route("/audio/<path:filename>")
def get_audio(filename):
    return send_from_directory(GENERATED_DIR, filename, mimetype="audio/wav")

@app.route("/api/languages")
def get_languages():
    return jsonify(SUPPORTED_LANGUAGES)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    target_lang = request.form.get("language", "hi-IN")
    use_sample = request.form.get("use_sample", "false").lower() == "true"
    
    unique_id = uuid.uuid4().hex[:8]
    
    # 1. Determine image path
    if use_sample:
        image_path = os.path.join(BASE_DIR, "sample_xray.png")
    else:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        uploaded_file = request.files["image"]
        if uploaded_file.filename == "":
            return jsonify({"error": "No file selected"}), 400
            
        ext = os.path.splitext(uploaded_file.filename)[1] or ".png"
        image_path = os.path.join(GENERATED_DIR, f"upload_{unique_id}{ext}")
        uploaded_file.save(image_path)

    try:
        # 2. Step 1: Vision Feature Extraction (DenseNet-121)
        findings = analyze_xray(image_path)

        # 3. Step 2: Clinical Agent Synthesis (Sarvam-105b)
        report_data = generate_clinical_report(findings)

        # 4. Step 3: Multilingual Voice Dictation (Mayura + Bulbul TTS)
        lang_code = resolve_language_code(target_lang)
        audio_filename = f"dictation_{unique_id}_{lang_code}.wav"
        audio_full_path = os.path.join(GENERATED_DIR, audio_filename)

        briefing_res = generate_voice_briefing(
            text=report_data["spoken_summary"],
            output_path=audio_full_path,
            language=lang_code,
            auto_translate=True
        )

        return jsonify({
            "status": "success",
            "findings": findings["top_findings"],
            "all_scores": findings["all_scores"],
            "report": report_data["full_report"],
            "dictation_script": briefing_res["spoken_text"],
            "audio_url": f"/audio/{audio_filename}",
            "language_name": briefing_res["language_name"],
            "language_code": briefing_res["language_code"],
            "speaker": briefing_res["speaker"],
            "file_size": briefing_res["file_size"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n==================================================================")
    print(f"  SARVAM AI MEDICAL ASSISTANT WEB APPLICATION")
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
