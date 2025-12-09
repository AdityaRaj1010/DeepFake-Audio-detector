from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
def load_audio(file_path, sr=16000, duration=3.0):
    """Load audio   , convert to mono, resample to sr, and pad/trim to duration seconds."""
    y, orig_sr = sf.read(str(file_path)) if isinstance(file_path, (str, Path)) else (file_path, sr)
    # soundfile returns float64 or int arrays. Convert to float32
    y = np.asarray(y, dtype=np.float32)
    # If stereo, convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
    target_length = int(sr * duration)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    return y
def compute_log_mel(y, sr=16000, n_mels=64, n_fft=1024, hop_length=256):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize per-sample
    S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-9)
    return S_db.astype(np.float32)
if __name__ == "__main__":
    # quick test
    import sys
    p = sys.argv[1]
    y = load_audio(p)
    m = compute_log_mel(y)
    print(m.shape)