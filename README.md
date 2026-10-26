# Sarvam AI Medical Assistant - Multilingual Chest X-Ray Analysis & Voice Agent

An AI web platform and CLI for radiologists and attending physicians that takes chest radiographs, computes quantitative computer-aided detections (CAD), synthesizes an in-depth differential diagnostic report using Sarvam AI's LLM (`sarvam-105b-conversations`), and delivers an audio physician briefing in **Hindi** and **regional Indian languages** using Sarvam AI's Text-to-Speech (`bulbul:v4-flash`).

---

## Web Application

The project includes an interactive web interface:

### 1. Launch the Web Server
```bash
python3 server.py
```
Open your browser at: **[http://localhost:5001](http://localhost:5001)**

### 2. Web Features
- **Drag-and-Drop Radiograph Upload**: Upload your own chest X-rays (PNG, JPG) or click **Load Sample NIH Scan** for instant testing.
- **Multilingual Voice Selector**: Select Hindi, Tamil, Telugu, Bengali, Kannada, Marathi, Gujarati, Punjabi, Malayalam, Odia, or English.
- **Computer-Aided Detection (CAD) Visualizer**: Color-coded progress bars displaying the confidence percentages of 18 detected thoracic pathologies.
- **Integrated Audio Player**: Stream, play, seek, and replay the generated formal radiologist dictation.
- **Full Clinical Report**: Interactive markdown viewer detailing radiological findings, ranked differential diagnoses, and definite clinical next steps.

---

## CLI Usage

### Dictation in Hindi:
```bash
python3 app.py sample_xray.png --lang hindi
```

### Dictation in Regional Languages:
```bash
# Tamil
python3 app.py sample_xray.png --lang tamil

# Telugu
python3 app.py sample_xray.png --lang telugu

# Bengali
python3 app.py sample_xray.png --lang bengali

# Marathi
python3 app.py sample_xray.png --lang marathi

# Kannada
python3 app.py sample_xray.png --lang kannada
```

---

## Supported Voice Personas

| Language | Code | Voice Persona | Model |
| :--- | :--- | :--- | :--- |
| **Hindi** | `hi-IN` | `ritu_hi_medical` | `bulbul:v4-flash` |
| **Tamil** | `ta-IN` | `vijay_ta_narration` | `bulbul:v4-flash` |
| **Telugu** | `te-IN` | `kavitha_te_narration` | `bulbul:v4-flash` |
| **Bengali** | `bn-IN` | `roopa_bn_conversational` | `bulbul:v4-flash` |
| **Kannada** | `kn-IN` | `chetan_kn_conversation` | `bulbul:v4-flash` |
| **Marathi** | `mr-IN` | `ishita_mr_conversational` | `bulbul:v4-flash` |
| **Gujarati** | `gu-IN` | `pooja_gu_conversational` | `bulbul:v4-flash` |
| **Punjabi** | `pa-IN` | `harpreet_pa_narration` | `bulbul:v4-flash` |
| **Malayalam** | `ml-IN` | `anushka` | `bulbul:v3` |
| **Odia** | `od-IN` | `anushka` | `bulbul:v3` |
| **English (India)** | `en-IN` | `ritu_en_medical` | `bulbul:v4-flash` |

---

## Project Structure

- [`server.py`](server.py): Flask web server hosting the web application.
- [`templates/index.html`](templates/index.html): Responsive clinical radiology frontend dashboard.
- [`app.py`](app.py): Master CLI orchestrator that runs the 3-step pipeline.
- [`vision_analyzer.py`](vision_analyzer.py): Pre-trained DenseNet-121 model extracting 18 chest radiograph pathology scores.
- [`clinical_agent.py`](clinical_agent.py): Medical prompt engine connecting to Sarvam AI's chat model (`sarvam-105b-conversations`).
- [`voice_briefing.py`](voice_briefing.py): Multilingual translation & TTS generator using Sarvam Mayura and Bulbul.
- [`sample_xray.png`](sample_xray.png): Test chest radiograph from the NIH Chest X-ray dataset.
