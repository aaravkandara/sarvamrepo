#!/usr/bin/env python3
"""
Sarvam AI Medical Assistant - Multilingual Chest X-Ray Analysis & Voice Agent
-----------------------------------------------------------------------------
Workflow:
1. Vision Analysis: Extracts radiological findings using TorchXRayVision DenseNet-121.
2. Clinical Agent: Synthesizes differential diagnosis & clinical report via Sarvam-105b.
3. Multilingual Voice Briefing: Translates and dictates the analysis in Hindi, Tamil,
   Telugu, Bengali, Kannada, Marathi, Gujarati, Punjabi, Malayalam, or English.
"""

import argparse
import os
import sys
import time

from vision_analyzer import analyze_xray
from clinical_agent import generate_clinical_report
from voice_briefing import generate_voice_briefing, translate_medical_text, resolve_language_code, SUPPORTED_LANGUAGES

def run_pipeline(
    image_path: str,
    language: str = "hi-IN",
    audio_output: str = None,
    skip_voice: bool = False,
    translate_full_report: bool = False
):
    lang_code = resolve_language_code(language)
    lang_info = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["hi-IN"])

    print("=" * 75)
    print("   SARVAM AI MEDICAL ASSISTANT: MULTILINGUAL X-RAY ANALYSIS PIPELINE")
    print("=" * 75)
    print(f"Target Dictation Language: {lang_info['name']} ({lang_code})")
    print(f"Assigned Voice Persona:    {lang_info['speaker']} ({lang_info['model']})")
    
    if not os.path.exists(image_path):
        print(f"\nError: Image file '{image_path}' not found.")
        sys.exit(1)
        
    # --- Step 1: Vision Feature Extraction ---
    print(f"\n[Step 1/3] Scanning Radiograph: {image_path}")
    start_t = time.time()
    findings = analyze_xray(image_path)
    print(f"  ✓ Analysis complete in {time.time() - start_t:.2f}s")
    print("\n  Top Detected Radiographic Features:")
    for item in findings["top_findings"]:
        bar = "█" * int(item["score"] * 20)
        print(f"    • {item['pathology']:<24}: {item['score']:>6.1%}  {bar}")

    # --- Step 2: Clinical Agent Report ---
    print(f"\n[Step 2/3] Consulting Sarvam Clinical Agent (sarvam-105b)...")
    start_t = time.time()
    report = generate_clinical_report(findings)
    print(f"  ✓ Clinical synthesis complete in {time.time() - start_t:.2f}s")

    report_content = report["full_report"]
    if translate_full_report and lang_code != "en-IN":
        print(f"  Translating full clinical report to {lang_info['name']} via Mayura...")
        report_content = translate_medical_text(report_content, target_lang=lang_code)

    print("\n" + "-" * 75)
    print(report_content)
    print("-" * 75)

    # --- Step 3: Multilingual Voice Dictation ---
    if not skip_voice:
        print(f"\n[Step 3/3] Generating Voice Dictation in {lang_info['name']}...")
        start_t = time.time()
        spoken_en = report["spoken_summary"]
        
        briefing_res = generate_voice_briefing(
            text=spoken_en,
            output_path=audio_output,
            language=lang_code,
            auto_translate=True
        )
        file_size_kb = briefing_res["file_size"] / 1024
        print(f"  ✓ Dictation synthesized in {time.time() - start_t:.2f}s")
        print(f"\n  [Dictation Script - {lang_info['name']}]:")
        print(f"  \"{briefing_res['spoken_text']}\"")
        print(f"\n  ✓ Saved Audio File: {briefing_res['output_path']} ({file_size_kb:.1f} KB)")
    else:
        print("\n[Step 3/3] Voice generation skipped by user flag.")

    print("\n" + "=" * 75)
    print("   ASSESSMENT COMPLETE - READY FOR CLINICIAN REVIEW")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(
        description="Sarvam AI Multilingual Medical Assistant for Chest X-Ray Analysis"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="sample_xray.png",
        help="Path to chest X-ray image (default: sample_xray.png)"
    )
    parser.add_argument(
        "--lang", "-l",
        default="hi-IN",
        help="Dictation language: hindi (hi-IN), tamil (ta-IN), telugu (te-IN), bengali (bn-IN), kannada (kn-IN), marathi (mr-IN), gujarati (gu-IN), punjabi (pa-IN), malayalam (ml-IN), odia (od-IN), english (en-IN). Default: hi-IN"
    )
    parser.add_argument(
        "--output-audio", "-o",
        default=None,
        help="Destination path for audio briefing WAV file (default: doctor_briefing_<lang>.wav)"
    )
    parser.add_argument(
        "--translate-report",
        action="store_true",
        help="Also translate the written markdown clinical report into the target language"
    )
    parser.add_argument(
        "--skip-voice",
        action="store_true",
        help="Skip TTS audio synthesis"
    )

    args = parser.parse_args()
    run_pipeline(
        image_path=args.image,
        language=args.lang,
        audio_output=args.output_audio,
        skip_voice=args.skip_voice,
        translate_full_report=args.translate_report
    )

if __name__ == "__main__":
    main()
