import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, resample
from typing import List, Dict, Any
from wettbewerb import get_3montages

# ======== DEVICE SETUP ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Target sampling rate (Hz) used during training
TARGET_FS = 250.0

# ======== Model Definition =========
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
        # LayerNorm expects (..., num_channels, seq_len)
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
        # x: (B, channels, seq_len)
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # (B, 1, channels, seq_len)
        x = self.conv_blocks(x)
        B, ch, C, T = x.size()
        x = x.view(B, ch * C, T)
        x = self.proj(x)
        x = x.permute(0, 2, 1) + self.pos_enc  # (B, seq_len, 128)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(1), self.regressor(x)

# ======== Model Loader =========
def load_model(model_path: str) -> EEGHybridMultitask:
    """Load the pretrained model onto the correct device."""
    model = EEGHybridMultitask().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ======== Signal Filtering =========
def apply_filters(data: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    lowcut, highcut = 0.5, 70.0
    notch_freq, Q = 50.0, 35.0
    b_bp, a_bp = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    b_notch, a_notch = iirnotch(notch_freq / (fs / 2), Q)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = filtfilt(b_bp, a_bp, data[:, ch])
        x = filtfilt(b_notch, a_notch, x)
        out[:, ch] = x
    return out

# ======== Window Extraction =========
def extract_windows(data: np.ndarray,
                    fs: float = TARGET_FS,
                    window_sec: float = 2.0,
                    stride_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    W = int(fs * window_sec)
    S = int(fs * stride_sec)
    windows, starts = [], []
    for start in range(0, len(data) - W + 1, S):
        seg = data[start:start + W]
        seg = seg.T  # (channels, W)
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (
            seg.std(axis=1, keepdims=True) + 1e-6
        )
        windows.append(seg)
        starts.append(start / fs)
    return np.stack(windows), np.array(starts)

# ======== Postprocessing =========
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


def apply_majority_voting(preds: List[int], window: int = 3, threshold: float = 0.6) -> List[int]:
    padded = np.pad(preds, (window, window), 'constant', constant_values=0)
    voted = []
    for i in range(window, len(preds) + window):
        local = padded[i - window:i + window + 1]
        voted.append(1 if np.mean(local) > threshold else 0)
    return voted

# ======== Prediction Interface =========
from typing import Tuple

def predict_labels(
    channels: List[str],
    data: np.ndarray,
    fs: float,
    reference_system: str = None,
    model_path: str = "best_multitask_model.pt"
) -> Dict[str, Any]:
    """
    Run sliding-window inference and postprocess to detect seizure events.

    Returns a dict with:
      seizure_present (0/1), seizure_confidence,
      onset, onset_confidence, offset, offset_confidence
    """
    # Montage extraction
    _, montage_data, _ = get_3montages(channels, data)
    if montage_data.shape[1] != 3:
        montage_data = montage_data.T if montage_data.shape[0] == 3 else None
    if montage_data is None:
        return dict(
            seizure_present=0, seizure_confidence=0.0,
            onset=None, onset_confidence=0.0,
            offset=None, offset_confidence=0.0
        )

    # Resample if needed to TARGET_FS
    if fs != TARGET_FS:
        num_samples = int(montage_data.shape[0] * TARGET_FS / fs)
        montage_data = resample(montage_data, num_samples, axis=0)
        fs = TARGET_FS

    # Preprocess
    raw = apply_filters(montage_data, fs)
    windows, starts = extract_windows(raw, fs)

    # Load model
    model = load_model(model_path)

    # Inference
    all_probs, all_onsets, all_offsets = [], [], []
    with torch.no_grad():
        for i in range(0, len(windows), 64):
            batch = torch.tensor(windows[i:i+64], dtype=torch.float32).to(device)
            logits, regs = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            onsets = regs[:, 0].cpu().numpy()
            offsets = regs[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_onsets.extend(onsets.tolist())
            all_offsets.extend(offsets.tolist())

    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)

    # Postprocess
    voted = apply_majority_voting(all_preds, window=2, threshold=0.6)
    smoothed = smooth_predictions(voted, min_consec=2, max_gap=1)

    # Final decision
    seizure_present = 0
    seizure_confidence = 0.0
    onset = None
    onset_confidence = 0.0
    offset = None
    offset_confidence = 0.0

    MIN_DUR = 1.0  # secs
    CONF_THRESH = 0.6

    if 1 in smoothed:
        idxs = np.where(np.array(smoothed) == 1)[0]
        first, last = idxs[0], idxs[-1]
        dur = (starts[last] + all_offsets[last]) - (starts[first] + all_onsets[first])
        avg_conf = float(all_probs[idxs].mean())
        if dur >= MIN_DUR and avg_conf >= CONF_THRESH:
            seizure_present = 1
            seizure_confidence = avg_conf
            onset = float(starts[first] + all_onsets[first])
            offset = float(starts[last] + all_offsets[last])
            onset_confidence = float(all_probs[first])
            offset_confidence = float(all_probs[last])

    return dict(
        seizure_present=seizure_present,
        seizure_confidence=seizure_confidence,
        onset=onset,
        onset_confidence=onset_confidence,
        offset=offset,
        offset_confidence=offset_confidence
    )
