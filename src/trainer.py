import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Trainer:
    def __init__(self, model, device, save_dir="saved", lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCELoss()   # for sigmoid output
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for x, y in tqdm(loader, desc="train", leave=False):
            x = x.to(self.device)
            y = y.to(self.device).float()  # Shape (B,)

            out = self.model(x)            # Shape (B,)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="eval", leave=False):
                x = x.to(self.device)
                y = y.to(self.device).float()

                out = self.model(x)
                loss = self.criterion(out, y)

                total_loss += loss.item() * x.size(0)
                all_preds.append(out.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        return total_loss / len(loader.dataset), preds, labels

    def fit(self, train_loader, test_loader, epochs=10):
        best_loss = float("inf")
        best_path = self.save_dir / "best_model.pth"

        for ep in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.eval_epoch(test_loader)

            print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best model to {best_path}")

        return best_path
