# 🎤 Deepfake Audio Detector

A lightweight deep learning system for detecting AI-generated (fake) audio from real human speech. Built with a shallow CNN trained on log-mel spectrograms, with a Streamlit web interface for real-time inference.

---

## 📌 Overview

This project trains a binary classifier to distinguish between **real** and **fake (AI-synthesized)** audio clips. It extracts log-mel spectrogram features from `.wav` files and passes them through a compact CNN architecture to produce a probability score.

| Feature | Detail |
|---|---|
| Input | `.wav` audio files |
| Output | Real / Fake + confidence % |
| Model | Shallow CNN (PyTorch) |
| Features | Log-Mel Spectrogram (64 bins) |
| Sample Rate | 16,000 Hz |
| Clip Duration | 3 seconds (padded/trimmed) |
| Dataset | The Fake-or-Real (FoR) Dataset (deepfake audio) |
| Dataset Size | 160,000+ `.wav` files |

---

## 🗂️ Project Structure

```text
deepfake-audio-detector/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI for demo & inference
│
├── configs/
│   └── config.yaml             # All hyperparameters & paths
│
├── src/
│   ├── init.py
│   ├── dataloader.py           # Dataset class & DataLoader factory
│   ├── models.py               # ShallowCNN architecture
│   ├── predict.py              # Single-file inference function
│   ├── preprocess.py           # Audio loading & mel spectrogram computation
│   ├── train.py                # Training entry point
│   ├── trainer.py              # Trainer class (train/eval loop + checkpointing)
│   └── utils.py                # Seed setting utility
│
├── saved/                      # Model checkpoints saved here (gitignored)
│   └── best_model.pth
│
├── data/                       # Audio data directory (gitignored)
│   ├── real/                   # Real speech .wav files
│   └── fake/                   # AI-generated .wav files
│
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces config
│
├── requirements.txt
```
---

## 🧠 Model Architecture — `ShallowCNN`

- Input: (B, 1, 64, T)          ← 1-channel log-mel spectrogram
- Conv2d(1 → 16, 3×3) + BN + ReLU + MaxPool(2×2)
- Conv2d(16 → 32, 3×3) + BN + ReLU + MaxPool(2×2)
- Conv2d(32 → 64, 3×3) + BN + ReLU
- AdaptiveAvgPool2d(1×1)        ← Collapses spatial dims
- Flatten → Linear(64 → 64) → ReLU → Dropout(0.3)
- Linear(64 → 1) → Sigmoid      ← Output probability ∈ [0, 1]
- Output **> 0.5** → **Real**
- Output **≤ 0.5** → **Fake**

---

## ⚙️ Configuration (`configs/config.yaml`)

```yaml
seed: 42
sr: 16000           # Sample rate (Hz)
duration: 3.0       # Clip length in seconds
n_mels: 64          # Mel frequency bins
n_fft: 1024         # FFT window size
hop_length: 256     # Hop length for STFT
batch_size: 32
epochs: 15
lr: 1e-4
save_dir: saved
model_name: shallow_cnn
device: auto        # "auto" uses CUDA if available

real_dir: data/real
fake_dir: data/fake
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AdityaRaj1010/deepfake-audio-detector.git
cd deepfake-audio-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ffmpeg` must be installed system-wide if you plan to use advanced audio augmentations.

### 3. Prepare Data

Organize your `.wav` files as follows:

```text
data/
├── real/    ← genuine human speech clips
└── fake/    ← AI-synthesized / TTS-generated clips
```
Clips will be automatically resampled to 16 kHz and trimmed/padded to 3 seconds during loading.

### 4. Train the Model

```bash
python -m src.train
```

The best model checkpoint (lowest validation loss) will be saved to `saved/best_model.pth`.

### 5. Run the Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Streamlit Demo Features

Upload any `.wav` file and get:

- ✅ **Real / Fake verdict** with confidence percentage
- 📊 **Probability breakdown** bars for both classes
- 🎵 **Log-Mel Spectrogram** visualization
- 📈 **Waveform** plot
- ⚡ **Signal energy** over time curve

---

## 🔍 Inference — Programmatic Usage

```python
from src.predict import predict_file

result = predict_file(
    file_path="path/to/audio.wav",
    model_path="saved/best_model.pth"
)

print(result)
# {'probability': 0.87, 'label': 'Real'}
```

The `probability` field reflects the model's confidence that the audio is **real** (score closer to 1.0 = more likely real).

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training & inference |
| `librosa` | Audio feature extraction |
| `soundfile` | WAV file I/O |
| `numpy` / `scipy` | Numerical processing |
| `matplotlib` | Visualization |
| `streamlit` | Web demo UI |
| `scikit-learn` | Evaluation utilities |
| `pyyaml` | Config loading |
| `tqdm` | Training progress bars |

---

## ☁️ GitHub Codespaces

This repo is fully configured for one-click launch in GitHub Codespaces:

1. Click **Code → Open with Codespaces**
2. Packages install automatically via `requirements.txt`
3. The Streamlit app launches and is forwarded on port `8501`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**AdityaRaj1010**  
Built as a portfolio project exploring audio deepfake detection using classical mel-spectrogram features and lightweight CNN architectures.
```text
├── .gitignore
└── README.md
```
