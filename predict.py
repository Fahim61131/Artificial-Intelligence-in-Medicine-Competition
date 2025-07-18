import os
import json
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import mne
from wettbewerb import get_6montages

WINDOW_SAMPLES = 75000

# -----------------------------------------------------------------------------
# 1) Model definition
# -----------------------------------------------------------------------------
class SeizureModel(nn.Module):
    def __init__(self):
        super(SeizureModel, self).__init__()
        
        # Normalization layer at the start
        self.norm = nn.InstanceNorm1d(6, affine=True)
        
        # Enhanced convolutional layers with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=25, stride=3, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)
        )
        
        # Residual connections
        self.residual1 = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.residual2 = nn.Conv1d(128, 256, kernel_size=1, stride=2)
        self.residual3 = nn.Conv1d(256, 512, kernel_size=1, stride=2)
        
        # LSTM layers with layer normalization
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        
        # Layer normalization after LSTM
        self.ln = nn.LayerNorm(512)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Apply instance normalization
        x = self.norm(x)
        
        # Convolutional layers with residual connections
        x1 = self.conv1(x)
        
        x2 = self.conv2(x1)
        res1 = self.residual1(x1)
        x2 = x2 + res1[:, :, :x2.size(2)]  # Match dimensions
        
        x3 = self.conv3(x2)
        res2 = self.residual2(x2)
        x3 = x3 + res2[:, :, :x3.size(2)]
        
        x4 = self.conv4(x3)
        res3 = self.residual3(x3)
        x4 = x4 + res3[:, :, :x4.size(2)]
        
        # Prepare for LSTM - swap dimensions
        x = x4.permute(0, 2, 1)
        
        # LSTM layers
        x, _ = self.lstm(x)
        
        # Apply layer normalization
        x = self.ln(x)
        
        # Attention mechanism
        attn_weights = self.attention(x)
        x = torch.sum(attn_weights * x, dim=1)
        
        # Classification output
        class_out = self.class_head(x)
        
        # Regression output
        reg_out = self.reg_head(x)
        
        return class_out, reg_out

# -----------------------------------------------------------------------------
# 2) Prediction function
# -----------------------------------------------------------------------------
def predict_labels(channels: List[str],data: np.ndarray,fs: float,reference_system: str,model_name: str = "best_model_2.pth") -> Dict[str, Any]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build 6‑montage and filter
    labels, montage, missing = get_6montages(channels, data)
    for i in range(montage.shape[0]):
        sig = montage[i]
        montage[i] = mne.filter.notch_filter(sig, Fs=fs, freqs=[50.0], verbose=False)
        montage[i] = mne.filter.filter_data(
            montage[i], sfreq=fs, l_freq=0.5, h_freq=70.0, verbose=False
        )

    # ─── pad to ensure at least one full WINDOW_SAMPLES chunk ───
    W = WINDOW_SAMPLES
    if montage.shape[1] < W:
        pad_len = W - montage.shape[1]
        montage = np.pad(montage, ((0,0),(0,pad_len)), mode='constant')

    # now slice into non‑overlapping WINDOW_SAMPLES chunks
    L = montage.shape[1]
    n_windows = L // W

    # load & run model on each chunk
    model = SeizureModel().to(device)
    state = torch.load(model_name, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    probs = []
    onsets_rel = []
    offsets_rel = []
    positions = []

    with torch.no_grad():
        for w in range(n_windows):
            start = w * W
            chunk = montage[:, start:start+W]
            x_chunk = torch.tensor(chunk).unsqueeze(0).float().to(device)
            p_cls, p_reg = model(x_chunk)
            prob = float(torch.sigmoid(p_cls).item())
            onset_rel, offset_rel = p_reg.squeeze().tolist()

            probs.append(prob)
            onsets_rel.append(onset_rel)
            offsets_rel.append(offset_rel)
            positions.append(start)


    first_prob = probs[0]
    seizure_present = int(first_prob > 0.7)

    if not seizure_present:
        return {
            "seizure_present": 0,
            "seizure_confidence": first_prob,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }

    
    first_idx = 0
    last_idx  = len(probs) - 1

    global_onset  = positions[first_idx] / fs + onsets_rel[first_idx]
    global_offset = positions[last_idx]  / fs + offsets_rel[last_idx]
    conf = first_prob

    return {
        "seizure_present":       1,
        "seizure_confidence":    conf,
        "onset":                 float(global_onset),
        "onset_confidence":      1.0,
        "offset":                float(global_offset),
        "offset_confidence":     1.0
    }

