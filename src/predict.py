import torch
import yaml
import numpy as np
from pathlib import Path
from .preprocess import load_audio, compute_log_mel
from .models import get_model

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def predict_file(file_path, model_path, cfg_path="configs/config.yaml", device="cpu"):
    cfg = load_config(cfg_path)

    device = torch.device(device)
    model = get_model(cfg["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y = load_audio(file_path, sr=cfg["sr"], duration=cfg["duration"])
    mel = compute_log_mel(y, sr=cfg["sr"], n_mels=cfg["n_mels"],
                          n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])

    mel = mel[np.newaxis, np.newaxis, :, :]   # → 1×1×n_mels×T
    x = torch.tensor(mel, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)                         # shape (1,)
        prob = float(out.cpu().numpy().ravel()[0])

    label = "Real" if prob > 0.5 else "Fake"

    return {"probability": prob, "label": label}
