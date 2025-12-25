# src/dataloader.py
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import random
import numpy as np
from .preprocess import load_audio, compute_log_mel

class AudioDataset(Dataset):
    def __init__(self, file_list, labels, sr=16000, duration=3.0, n_mels=64, n_fft=1024, hop_length=256):
        self.file_list = list(file_list)
        self.labels = list(labels)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        p = self.file_list[idx]
        label = float(self.labels[idx])

        y = load_audio(p, sr=self.sr, duration=self.duration)
        mel = compute_log_mel(y, sr=self.sr, n_mels=self.n_mels,
                              n_fft=self.n_fft, hop_length=self.hop_length)

        # shape -> 1 x n_mels x T
        mel = mel[np.newaxis, :, :]
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def make_dataloaders(real_dir, fake_dir, batch_size=32, test_ratio=0.2,
                     sr=16000, duration=3.0, n_mels=64, n_fft=1024, hop_length=256, shuffle=True):
    """
    Create train and test dataloaders.

    Args:
        real_dir (str or Path): path to real audio folder
        fake_dir (str or Path): path to fake audio folder
        batch_size (int): batch size
        test_ratio (float): fraction to reserve for test
        sr, duration, n_mels, n_fft, hop_length: audio preprocessing params
        shuffle (bool): whether to shuffle dataset before splitting
    Returns:
        train_loader, test_loader
    """
    real_dir = Path(real_dir) # trained with 80405 files
    fake_dir = Path(fake_dir) # trained with 84402 files

    real_files = sorted([p for p in real_dir.glob("**/*.wav")])
    fake_files = sorted([p for p in fake_dir.glob("**/*.wav")])

    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files)

    # shuffle
    combined = list(zip(files, labels))
    if shuffle:
        random.shuffle(combined)
    files, labels = zip(*combined)

    # split
    n_total = len(files)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_files = files[:n_train]
    train_labels = labels[:n_train]
    test_files = files[n_train:]
    test_labels = labels[n_train:]

    train_ds = AudioDataset(train_files, train_labels, sr=sr, duration=duration,
                            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    test_ds = AudioDataset(test_files, test_labels, sr=sr, duration=duration,
                           n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
