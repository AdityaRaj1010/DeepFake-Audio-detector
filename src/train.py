import yaml
from src.dataloader import make_dataloaders
from src.models import get_model
from src.trainer import Trainer
import torch
from pathlib import Path
import os




def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)




def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    # device
    device = torch.device("cuda" if (cfg.get("device") == "auto" and torch.cuda.is_available()) or cfg.get("device") == "cuda" else "cpu")
    print("Using device:", device)


    train_loader, test_loader = make_dataloaders(
    cfg.get("real_dir"),
    cfg.get("fake_dir"),
    batch_size=cfg.get("batch_size"),
    sr=cfg.get("sr"),
    duration=cfg.get("duration"),
    n_mels=cfg.get("n_mels"),
    n_fft=cfg.get("n_fft"),
    hop_length=cfg.get("hop_length"),
    test_ratio=cfg.get("test_ratio", 0.2),
    )


    model = get_model(cfg.get("model_name", "shallow_cnn"))
    model = model.to(device)


    tr = Trainer(model, device, save_dir=cfg.get("save_dir"), lr=cfg.get("lr"))
    best_path = tr.fit(train_loader, test_loader, epochs=cfg.get("epochs"))
    print("Training finished. Best model at:", best_path)




if __name__ == "__main__":
    main()