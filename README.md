# 🎤 Hybrid Grammar Scoring Engine

An AI-powered web app that evaluates spoken grammar using both **audio** and **transcribed text**. It combines speech features and textual grammar analysis to generate a grammar score using a **hybrid deep learning model**.

---

## ⚠️ Python Compatibility

This app supports:

* ✅ Python 3.10
* ✅ Python 3.11
* ✅ Python 3.12
  ❌ Python 3.13 is **not yet supported** due to TensorFlow incompatibility.

---

## 📁 Folder Structure

```
APP/
├── data/                       # Training/test datasets and audio
│   ├── audios_train/
│   ├── audios_test/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                     # Saved model & scalers
│   ├── hybrid_model.keras
│   ├── scaler_audio.pkl
│   └── scaler_text.pkl
├── app.py                      # 🔹 Streamlit app (main entry point)
├── GPU_Checker.py              # GPU availability check
├── train-model.py              # Model training script
├── submission.csv              # Submission format (for competitions)
└── requirements.txt            # Required Python packages
```

(Note: `temp.py` is used for debugging and excluded from documentation.)

---

## 🔗 Dataset

This project uses the **SHL Dataset – Grammar Error Audio**.

➞ **Extract into:**

```
APP/data/
```

Include:

* `train.csv`, `test.csv`, `sample_submission.csv`
* Folders: `audios_train/`, `audios_test/`

---

## 🚀 Features

* 🧠 **Transcription** using OpenAI's Whisper ASR model
* 📖 **Grammar checking** via LanguageTool API (local Docker or cloud)
* 🔊 **Audio features**: MFCC, Chroma, ZCR, Spectral Contrast, etc.
* 🧲 **Hybrid scoring** model using Keras and scikit-learn
* 🌐 **Web interface** built with Streamlit
* 🎙️ **Built-in microphone recording** via `st_audiorec`

---

## ✅ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/JOY23072005/Grammer-Scoring-Engine.git
cd hybrid-grammar-engine
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 Train the Model

To train from scratch or re-train with your data:

```bash
python train-model.py
```

It will generate:

* `models/hybrid_model.keras`
* `models/scaler_audio.pkl`
* `models/scaler_text.pkl`

---

## ⚙️ GPU Check (Optional)

To check if GPU is detected by TensorFlow:

```bash
python GPU_Checker.py
```

---

## 📜 .gitignore Suggestions

```gitignore
.venv/
__pycache__/
*.pyc
data/
models/
submission.csv
```

---

## 📜 License

MIT License © 2025 Joydeep Hans
