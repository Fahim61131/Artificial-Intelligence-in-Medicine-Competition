# -*- coding: utf-8 -*-
"""
Seizure Detection Script (LightConvLSTMTransformer version)
Naming and structure adapted for challenge submission
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch
from wettbewerb import get_3montages

TARGET_FS = 250.0  # Not used here, but kept for convention
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Model Definition ========
class LightConvLSTMTransformer(nn.Module):
    def __init__(self, in_channels, n_classes=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)          # (B, T, C)
        x, _ = self.lstm(x)             # (B, T, 128)
        x = self.transformer(x)         # (B, T, 128)
        x = x[:, -1, :]                 # (B, 128)
        return self.fc(self.norm(x)).squeeze(1)

# ======== Helpers ========
def create_filters(fs):
    b_bp, a_bp = butter(4, [0.5/(fs/2), 70.0/(fs/2)], btype='band')
    b_notch, a_notch = iirnotch(60.0/(fs/2), Q=35.0)
    return b_bp, a_bp, b_notch, a_notch

def apply_filters(data, b_bp, a_bp, b_notch, a_notch):
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = filtfilt(b_bp, a_bp, data[:, ch])
        out[:, ch] = filtfilt(b_notch, a_notch, x)
    return out

def extract_windows(data, fs, window_sec=5, stride_sec=2):
    ws = int(window_sec * fs)
    ss = int(stride_sec * fs)
    starts = np.arange(0, data.shape[0] - ws + 1, ss)
    windows = np.stack([data[s:s+ws] for s in starts], axis=0)  # (num_windows, ws, channels)
    # Normalize each window
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-6
    windows = ((windows - mean) / std).transpose(0, 2, 1)  # (num_windows, channels, ws)
    return windows, starts

def smooth_predictions(preds):
    # Remove single outlier predictions (01*0 to 0)
    preds = preds.astype(bool)
    for i in range(1, len(preds) - 1):
        if preds[i] and not preds[i - 1] and not preds[i + 1]:
            preds[i] = False
    return preds.astype(int)

# ======== Main Interface ========
def predict_labels(channels, data, fs, model_name='best_light_convlstm_transformer_model.pth'):
    try:
        # Extract montage
        _, montage_data, _ = get_3montages(channels, data)
        montage_data = montage_data.T  # shape: (samples, channels)

        # Filtering
        b_bp, a_bp, b_notch, a_notch = create_filters(fs)
        filtered = apply_filters(montage_data, b_bp, a_bp, b_notch, a_notch)

        # Windowing
        windows, starts = extract_windows(filtered, fs, window_sec=5, stride_sec=2)

        # Model
        checkpoint = torch.load(model_name, map_location='cpu')
        model = LightConvLSTMTransformer(checkpoint['input_channels']).to(DEVICE).eval()
        model.load_state_dict(checkpoint['model_state_dict'])

        # Prediction
        probs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i:i + batch_size]).float().to(DEVICE)
                logits = model(batch)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        preds = (probs > 0.5).astype(int)
        smoothed = smooth_predictions(preds)

        # Event detection
        events, in_ev = [], False
        for idx, flag in enumerate(smoothed):
            if flag and not in_ev:
                in_ev, ev0 = True, idx
            elif not flag and in_ev:
                events.append((ev0, idx - 1))
                in_ev = False
        if in_ev:
            events.append((ev0, len(smoothed) - 1))

        if not events:
            return {
                "seizure_present": 0,
                "seizure_confidence": float(probs.max()),
                "onset": None,
                "onset_confidence": 0.0,
                "offset": None,
                "offset_confidence": 0.0
            }

        # Pick longest event
        lengths = [j - i for i, j in events]
        si, ei = events[int(np.argmax(lengths))]
        onset = starts[si] / fs
        offset = (starts[ei] + windows.shape[2]) / fs
        conf = float(probs[si:ei + 1].mean())

        return {
            "seizure_present": 1,
            "seizure_confidence": conf,
            "onset": onset,
            "onset_confidence": conf,
            "offset": offset,
            "offset_confidence": conf
        }

    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            "seizure_present": 0,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
