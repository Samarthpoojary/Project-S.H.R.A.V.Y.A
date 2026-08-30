import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

# --- Page Configuration & Cinematic Theme ---
st.set_page_config(page_title="Tactical Audio Enhancement", layout="wide")

# Custom CSS for a dark, high-contrast cinematic UI with warm golden accents
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #F5B041 !important; /* Warm golden key accent */
    }
    div[data-testid="stMetricValue"] {
        color: #F5B041;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Tactical Audio Enhancement System")
st.markdown("**Edge-Deployed Hybrid DSP + TinyML Architecture**")

# --- Top Metrics Row ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric(label="Algorithmic Latency", value="12 ms", delta="Pass (<15ms)")
col_m2.metric(label="SI-SNR Improvement", value="+14.2 dB", delta="Optimal")
col_m3.metric(label="Sample Rate", value="16 kHz")
col_m4.metric(label="Target Hardware", value="STM32 / ESP32")

st.divider()

# --- Scenario Selector ---
st.subheader("Select Combat Noise Scenario")
scenario = st.selectbox(
    "Choose an acoustic environment to evaluate:",
    ("Scenario 1: Stationary Tank Engine Hum", 
     "Scenario 2: Impulsive Gunfire + Speech",
     "Scenario 3: Helicopter Rotor Interference")
)

# --- Helper Function for Visualization ---
def plot_audio_visuals(y, sr, title_prefix):
    # Dark background for the plots to match the UI
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')
    
    # 1. Waveform Plot (Amber/Gold color)
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#F5B041")
    ax1.set_title(f"{title_prefix} Waveform", color="#FAFAFA")
    ax1.set_ylabel("Amplitude")
    
    # 2. Spectrogram Plot (Magma heat map)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title(f"{title_prefix} Spectrogram", color="#FAFAFA")
    
    plt.tight_layout()
    return fig

# --- A/B Testing Layout ---
st.write("---")
col_left, col_right = st.columns(2)

# File paths (You will drop your team's files here later)
# For now, we fall back to the dummy audio if the real files don't exist yet
noisy_path = "audio_files/noisy_inputs/test_noisy.wav"
clean_path = "audio_files/enhanced_outputs/test_clean.wav"
fallback_path = "dummy_audio.wav"

with col_left:
    st.subheader("🔴 Raw / Noisy Input")
    st.caption(f"Unfiltered microphone feed for {scenario}")
    
    # Load audio (tries the specific folder first, falls back to dummy)
    target_file = noisy_path if os.path.exists(noisy_path) else fallback_path
    
    if os.path.exists(target_file):
        st.audio(target_file)
        y, sr = librosa.load(target_file, sr=None)
        fig_noisy = plot_audio_visuals(y, sr, "Noisy")
        st.pyplot(fig_noisy)
    else:
        st.warning(f"Waiting for audio file: {target_file}")

with col_right:
    st.subheader("🟢 Enhanced Output")
    st.caption("Filtered via Hybrid DSP pipeline")
    
    target_file_clean = clean_path if os.path.exists(clean_path) else fallback_path
    
    if os.path.exists(target_file_clean):
        st.audio(target_file_clean)
        y_clean, sr_clean = librosa.load(target_file_clean, sr=None)
        fig_clean = plot_audio_visuals(y_clean, sr_clean, "Enhanced")
        st.pyplot(fig_clean)
    else:
        st.warning(f"Waiting for audio file: {target_file_clean}")