import streamlit as st
from pathlib import Path
import tempfile
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.predict import predict_file

st.title("Deepfake Audio Detector — Demo")
st.write("Upload a short voice note (.wav) and click 'Check Audio'.")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])
model_path_input = st.text_input("Path to saved model:", "saved/best_model.pth")

if st.button("Check Audio"):
    if uploaded is None:
        st.warning("Please upload a .wav file")
    else:
        if not Path(model_path_input).is_file():
            st.error(f"Model not found at {model_path_input}")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()   # IMPORTANT!
            tpath = tmp.name

            with st.spinner("Analyzing..."):
                try:
                    result = predict_file(tpath, model_path_input, cfg_path="configs/config.yaml")
                    prob = result["probability"]
                    label = result["label"]
                    st.metric(label, f"{prob*100:.2f}%")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                finally:
                    try:
                        os.unlink(tpath)
                    except:
                        pass
