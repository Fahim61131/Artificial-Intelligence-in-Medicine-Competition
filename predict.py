# -*- coding: utf-8 -*-
"""
Final Submission Script for Seizure Detection Task
Compatible with the challenge evaluation system
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from scipy.signal import butter, filtfilt, iirnotch, resample
from wettbewerb import get_3montages

# Constants
TARGET_FS = 250.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Model Definition ========
class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class EEGHybridMultitask(nn.Module):
    def __init__(self, num_channels=3, seq_len=int(TARGET_FS * 2.0)):
        super().__init__()
        self.input_norm = nn.LayerNorm([num_channels, seq_len])
        self.conv_blocks = nn.Sequential(
            ResConvBlock(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            ResConvBlock(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.Dropout(0.1)
        )
        self.proj = nn.Conv1d(32 * num_channels, 128, kernel_size=1)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, 128) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 1)
        self.regressor = nn.Linear(128, 2)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        B, ch, C, T = x.size()
        x = x.view(B, ch * C, T)
        x = self.proj(x)
        x = x.permute(0, 2, 1) + self.pos_enc
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(1), self.regressor(x)

# ======== Helpers ========
def load_model(model_path: str) -> EEGHybridMultitask:
    model = EEGHybridMultitask().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def apply_filters(data: np.ndarray, fs: float) -> np.ndarray:
    b_bp, a_bp = butter(4, [0.5 / (fs / 2), 70.0 / (fs / 2)], btype='band')
    b_notch, a_notch = iirnotch(50.0 / (fs / 2), Q=35)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = filtfilt(b_bp, a_bp, data[:, ch])
        x = filtfilt(b_notch, a_notch, x)
        out[:, ch] = x
    return out

def extract_windows(data: np.ndarray, fs: float, window_sec: float = 2.0, stride_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    W = int(fs * window_sec)
    S = int(fs * stride_sec)
    windows, starts = [], []
    for start in range(0, len(data) - W + 1, S):
        seg = data[start:start + W].T
        std = seg.std(axis=1, keepdims=True)
        std[std < 1e-4] = 1e-4
        seg = (seg - seg.mean(axis=1, keepdims=True)) / std
        windows.append(seg)
        starts.append(start / fs)
    return np.stack(windows), np.array(starts)

def smooth_predictions(preds: List[int], min_consec: int = 2, max_gap: int = 1) -> List[int]:
    result = [0] * len(preds)
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            start = i
            gap = 0
            ones = 1
            i += 1
            while i < len(preds) and (preds[i] == 1 or gap < max_gap):
                if preds[i] == 1:
                    ones += 1
                    gap = 0
                else:
                    gap += 1
                i += 1
            end = i
            if ones >= min_consec:
                for j in range(start, end):
                    result[j] = 1
        else:
            i += 1
    return result

# ======== Main Interface ========
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str = "best_multitask_model.pt") -> Dict[str, Any]:
    try:
        _, montage_data, _ = get_3montages(channels, data)
        if montage_data.shape[1] != 3:
            montage_data = montage_data.T if montage_data.shape[0] == 3 else None
        if montage_data is None:
            raise ValueError("Montage extraction failed")

        # Resample if needed
        if fs != TARGET_FS:
            num_samples = int(montage_data.shape[0] * TARGET_FS / fs)
            montage_data = resample(montage_data, num_samples, axis=0)
            fs = TARGET_FS

        # Filtering
        filtered = apply_filters(montage_data, fs)
        windows, starts = extract_windows(filtered, fs)

        # Model inference
        model = load_model(model_name)
        all_probs, all_onsets, all_offsets = [], [], []
        with torch.no_grad():
            for i in range(0, len(windows), 64):
                batch = torch.tensor(windows[i:i + 64], dtype=torch.float32).to(device)
                logits, regs = model(batch)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                all_onsets.extend(regs[:, 0].cpu().numpy().tolist())
                all_offsets.extend(regs[:, 1].cpu().numpy().tolist())

        probs = np.array(all_probs)
        preds = (probs > 0.5).astype(int)
        smoothed = smooth_predictions(preds)

        seizure_present = 0
        seizure_confidence = 0.0
        onset = None
        onset_confidence = 0.0
        offset = None
        offset_confidence = 0.0

        if 1 in smoothed:
            idxs = np.where(smoothed)[0]
            first, last = idxs[0], idxs[-1]
            seizure_present = 1
            seizure_confidence = float(probs[idxs].mean())
            onset = float(starts[first] + all_onsets[first])
            offset = float(starts[last] + all_offsets[last])
            onset_confidence = float(probs[first])
            offset_confidence = float(probs[last])

            if np.isnan(onset) or np.isnan(offset):
                seizure_present = 0
                seizure_confidence = 0.0
                onset = None
                onset_confidence = 0.0
                offset = None
                offset_confidence = 0.0

        return {
            "seizure_present": seizure_present,
            "seizure_confidence": seizure_confidence,
            "onset": onset,
            "onset_confidence": onset_confidence,
            "offset": offset,
            "offset_confidence": offset_confidence
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
