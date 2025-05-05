import streamlit as st
import librosa
import numpy as np
from transformers import pipeline
import language_tool_python
import tempfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 1. Advanced Caching
@st.cache_resource
def load_resources():
    return {
        'asr': pipeline("automatic-speech-recognition", 
                       model="openai/whisper-small"),
        'grammar': language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org'),
        'model': tf.keras.models.load_model('models/hybrid_model.h5')
    }

# 2. Feature Visualization
def plot_audio_analysis(y, sr):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Waveform')
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")
    ax[1].set_title('Spectrogram')
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[2])
    fig.colorbar(img, ax=ax[2])
    ax[2].set_title('MFCC')
    
    return fig

# 3. Grammar Analysis
def analyze_grammar(text,resources):
    matches = resources['grammar'].check(text)
    error_count = len(matches)
    error_types = set()  # Use a set to store unique error types
    
    for match in matches:
        error_types.add(match.ruleId)  # Add the rule ID to track different types of errors
    
    return {
        'error_count': error_count,
        'error_types': error_types
    }

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Shape: (13,)
    return np.expand_dims(mfcc_mean, axis=0)  # Shape: (1, 13)

def extract_text_features(text):
    # Simple text embedding based on length and word count
    num_chars = len(text)
    num_words = len(text.split())
    avg_word_length = num_chars / num_words if num_words else 0
    return np.array([[num_chars, num_words, avg_word_length]])  # Shape: (1, 3)


# 4. Main Application
def main():
    st.title("🎙️ Advanced Grammar Scoring Engine")
    
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            
            # Audio Analysis
            y, sr = librosa.load(tmp_file.name, sr=16000)
            st.subheader("Audio Analysis")
            st.pyplot(plot_audio_analysis(y, sr))
            
            # Transcription
            resources = load_resources()
            transcription = resources['asr'](tmp_file.name)["text"]
            st.subheader("Transcription")
            st.code(transcription)
            
            # Grammar Analysis
            grammar_report = analyze_grammar(transcription,resources)
            st.subheader("Grammar Report")
            col1, col2 = st.columns(2)
            col1.metric("Total Errors", grammar_report['error_count'])
            col2.metric("Unique Error Types", len(grammar_report['error_types']))
            
            # Prediction
            audio_features = extract_audio_features(tmp_file.name)
            text_features = extract_text_features(transcription)
            prediction = resources['model'].predict([audio_features, text_features])
            
            st.subheader("Grammar Score Prediction")
            st.metric("Predicted Score", f"{prediction[0][0]:.2f}/5.0")
            st.progress(prediction[0][0]/5)

if __name__ == "__main__":
    main()
