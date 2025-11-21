import torch
import torch.nn as nn

class ShallowCNN(nn.Module):
    def __init__(self, in_channels=1, n_mels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # → Output shape: (B, 64, 1, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),     # IMPORTANT: OUTPUT = 1 NEURON
            nn.Sigmoid(),         # Gives probability 0 to 1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.view(-1)          # ALWAYS returns shape (B,)
        

def get_model(name="shallow_cnn", **kwargs):
    if name == "shallow_cnn":
        return ShallowCNN(**kwargs)
    raise ValueError("Unknown model")
