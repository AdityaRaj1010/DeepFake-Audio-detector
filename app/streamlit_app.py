import streamlit as st
from pathlib import Path
import tempfile
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to import src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.predict import predict_file
from src.preprocess import load_audio, compute_log_mel

st.set_page_config(page_title="Deepfake Audio Detector", layout="wide")

st.title("🎤 Deepfake Audio Detector — Demo")
st.write("Upload a `.wav` file to analyze whether it's **Real or Fake**")

uploaded = st.file_uploader("Upload WAV File", type=["wav"])
model_path = Path("saved/best_model.pth")
# model_path_input = st.text_input("Path to Model (.pth)", value="saved/best_model.pth")
if st.button("Analyze Audio"):
    if uploaded is None:
        st.warning("Please upload a WAV file.")
    else:
        if not Path(model_path).is_file():
            st.error(f"Model not found at {model_path}")
        else:
            # Save uploaded file temporarily
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()
            tpath = tmp.name

            with st.spinner("Analyzing..."):
                try:
                    # Run prediction
                    result = predict_file(tpath, model_path)
                    prob_real = result["probability"]
                    prob_fake = 1 - prob_real

                    # Decide final label
                    if prob_real >= 0.5:
                        label = "Real"
                        confidence = prob_real * 100
                    else:
                        label = "Fake"
                        confidence = prob_fake * 100

                    # --- UI SECTION STARTS ---
                    st.header(f"Result: **{label} ({confidence:.2f}%)**")

                    # Probability bars
                    st.subheader("Confidence Breakdown")
                    st.write("### Probability Visualization")

                    st.progress(int(prob_real * 100))
                    st.write(f"**Real:** {prob_real*100:.2f}%")

                    st.progress(int(prob_fake * 100))
                    st.write(f"**Fake:** {prob_fake*100:.2f}%")

                    # Load audio and compute features
                    y = load_audio(tpath)
                    mel = compute_log_mel(y)

                    # Visualizations — Mel Spectrogram
                    st.subheader("Mel-Spectrogram")
                    fig1, ax1 = plt.subplots(figsize=(8, 3))
                    img = ax1.imshow(mel, aspect="auto", origin="lower")
                    ax1.set_title("Log-Mel Spectrogram")
                    ax1.set_xlabel("Time Frames")
                    ax1.set_ylabel("Mel Frequency Bins")
                    plt.colorbar(img, ax=ax1)
                    st.pyplot(fig1)

                    # Waveform plot
                    st.subheader("Waveform")
                    fig2, ax2 = plt.subplots(figsize=(8, 2))
                    ax2.plot(y)
                    ax2.set_title("Waveform")
                    ax2.set_xlabel("Sample Index")
                    ax2.set_ylabel("Amplitude")
                    st.pyplot(fig2)

                    # Energy plot
                    st.subheader("Signal Energy Over Time")
                    energy = np.sum(mel, axis=0)
                    fig3, ax3 = plt.subplots(figsize=(8, 2))
                    ax3.plot(energy)
                    ax3.set_title("Energy Curve")
                    ax3.set_xlabel("Time Frames")
                    ax3.set_ylabel("Energy")
                    st.pyplot(fig3)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                finally:
                    try:
                        os.unlink(tpath)
                    except:
                        pass
