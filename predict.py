import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from wettbewerb import get_6montages



# EEG frequency bands
eeg_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 70)
}

# Compute power in a frequency band
def compute_bandpower(sig: np.ndarray, fs: float, band: tuple) -> float:
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=min(2048, len(sig)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])

# Extract features: band powers + basic stats
def extract_features(data: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for ch in data:
        for band in eeg_bands.values():
            feats.append(compute_bandpower(ch, fs, band))
        feats.extend([
            np.mean(ch),
            np.std(ch),
            np.sqrt(np.mean(ch**2)),
            skew(ch),
            kurtosis(ch)
        ])
    return np.array(feats, dtype=np.float32)

# Neural network matching training architecture\ class ClassifierRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super(ClassifierRegressor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU()
        )
        self.cls_head = nn.Linear(64, 1)
        self.reg_head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.shared(x)
        cls_out = torch.sigmoid(self.cls_head(x)).squeeze(1)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

# Load model from shared directory
def load_model(path: str, input_dim: int):
    model = ClassifierRegressor(input_dim)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

# Prediction function (do not change signature)
def predict_labels(channels: List[str], data: np.ndarray, fs: float,
                   reference_system: str, model_name: str = "best_model.pth") -> Dict[str, Any]:
    """
    Predicts seizure presence and onset/offset times.

    Returns a dict with keys:
      seizure_present (bool)
      seizure_confidence (float)
      onset (float)
      onset_confidence (float)
      offset (float)
      offset_confidence (float)
    """
    montage, mont_data, missing = get_6montages(channels, data)
    if missing:
        return {k: 0.0 if isinstance(k, float) else False for k in ['seizure_present','seizure_confidence','onset','onset_confidence','offset','offset_confidence']}

    for i in range(len(mont_data)):
        mont_data[i] = mne.filter.notch_filter(mont_data[i], Fs=fs, freqs=[50,100], verbose=False)
        mont_data[i] = mne.filter.filter_data(mont_data[i], sfreq=fs, l_freq=0.5, h_freq=70, verbose=False)

    feats = extract_features(mont_data, fs)
    model = load_model(model_name, len(feats))
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.no_grad():
        prob, locs = model(x)
    prob = prob.item()
    onset, offset = locs[0].tolist()

    return {
        'seizure_present': prob > 0.5,
        'seizure_confidence': prob,
        'onset': float(onset), 'onset_confidence': 1.0,
        'offset': float(offset), 'offset_confidence': 1.0
    }
