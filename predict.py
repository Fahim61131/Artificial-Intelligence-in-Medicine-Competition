# -*- coding: utf-8 -*-
"""
predict.py

Simple English version for seizure detection and onset/offset regression.
"""

import numpy as np
torch_imported = False
try:
    import torch
    import torch.nn as nn
    torch_imported = True
except ImportError:
    pass

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

# Calculate band power
def compute_bandpower(sig: np.ndarray, fs: float, band: tuple) -> float:
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=min(2048, len(sig)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])

# Extract features from all channels
def extract_features(data: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for ch in data:
        # band powers
        for band in eeg_bands.values():
            feats.append(compute_bandpower(ch, fs, band))
        # time-domain stats
        feats.append(np.mean(ch))
        feats.append(np.std(ch))
        feats.append(np.sqrt(np.mean(ch**2)))
        feats.append(skew(ch))
        feats.append(kurtosis(ch))
    return np.array(feats, dtype=np.float32)

# Simple neural net model
if torch_imported:
    class EEGModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.cls = nn.Linear(64, 1)
            self.reg = nn.Linear(64, 2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            prob = torch.sigmoid(self.cls(x)).squeeze(1)
            locs = self.reg(x)
            return prob, locs

    _model = None
    def load_model(path: str, input_dim: int):
        global _model
        if _model is None or _model.fc1.in_features != input_dim:
            _model = EEGModel(input_dim)
            _model.load_state_dict(torch.load(path, map_location='cpu'))
            _model.eval()
        return _model

# Prediction function (do not change signature)
def predict_labels(channels: List[str], data: np.ndarray, fs: float,
                   reference_system: str, model_name: str = 'best_model.pth') -> Dict[str, Any]:
    """
    Returns:
      seizure_present (bool), seizure_confidence (float),
      onset (float), onset_confidence (float),
      offset (float), offset_confidence (float)
    """
    # montage and check missing channels
    montage, mont_data, missing = get_6montages(channels, data)
    if missing:
        return {
            'seizure_present': False,
            'seizure_confidence': 0.0,
            'onset': 0.0,
            'onset_confidence': 0.0,
            'offset': 0.0,
            'offset_confidence': 0.0
        }
    # filters
    for i in range(len(mont_data)):
        mont_data[i] = mne.filter.notch_filter(mont_data[i], Fs=fs, freqs=[50, 100], verbose=False)
        mont_data[i] = mne.filter.filter_data(mont_data[i], sfreq=fs, l_freq=0.5, h_freq=70, verbose=False)

    # features
    feats = extract_features(mont_data, fs)

    if not torch_imported:
        raise RuntimeError("Torch not available")

    # load and run model
    model = load_model(model_name, len(feats))
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.no_grad():
        prob, locs = model(x)
    prob = prob.item()
    onset, offset = locs[0].tolist()

    return {
        'seizure_present': prob > 0.5,
        'seizure_confidence': prob,
        'onset': float(onset),
        'onset_confidence': 1.0,
        'offset': float(offset),
        'offset_confidence': 1.0
    }
