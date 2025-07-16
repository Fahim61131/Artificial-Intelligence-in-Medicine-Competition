import os
import json
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import mne
from wettbewerb import get_6montages

MODEL_PATH = "best_model_2.pth"
WINDOW_SAMPLES = 75000  # first 5 min at 250 Hz

# -----------------------------------------------------------------------------
# 1) Model definition (unchanged)
# -----------------------------------------------------------------------------
class SeizureModel(nn.Module):
    def __init__(self):
        super(SeizureModel, self).__init__()
        self.norm = nn.InstanceNorm1d(6, affine=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 64, 25, stride=3, padding=12),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3), nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 15, stride=2, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 7, stride=2, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, 5, stride=2, padding=2),
            nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.4)
        )
        self.res1 = nn.Conv1d(64, 128, 1, stride=2)
        self.res2 = nn.Conv1d(128, 256, 1, stride=2)
        self.res3 = nn.Conv1d(256, 512, 1, stride=2)
        self.lstm = nn.LSTM(512, 256, num_layers=3,
                            bidirectional=True, batch_first=True, dropout=0.4)
        self.ln = nn.LayerNorm(512)
        self.attn = nn.Sequential(
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 1), nn.Softmax(dim=1)
        )
        self.class_head = nn.Sequential(
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,1), nn.Sigmoid()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,2)
        )

    def forward(self, x):
        x = self.norm(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + self.res1(x1)[:,:,:x2.size(2)]
        x3 = self.conv3(x2) + self.res2(x2)[:,:,:x3.size(2)]
        x4 = self.conv4(x3) + self.res3(x3)[:,:,:x4.size(2)]
        seq,_ = self.lstm(x4.permute(0,2,1))
        seq = self.ln(seq)
        w = self.attn(seq)
        feat = (w * seq).sum(dim=1)
        return self.class_head(feat), self.reg_head(feat)

# -----------------------------------------------------------------------------
# 2) Prediction function (simplified)
# -----------------------------------------------------------------------------
def predict_labels(
    channels: List[str],
    data: np.ndarray,
    fs: float,
    reference_system: str,
    model_name: str = MODEL_PATH
) -> Dict[str, Any]:
    """
    Takes only the first 75 000 samples (5 min) from the montage,
    runs one pass through the network, and returns its classification
    and regression outputs (or zeros if non‐seizure).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build montage
    _, montage, missing = get_6montages(channels, data)
    # filter each channel
    for i in range(montage.shape[0]):
        s = montage[i]
        montage[i] = mne.filter.notch_filter(s, Fs=fs, freqs=[50.0], verbose=False)
        montage[i] = mne.filter.filter_data(
            montage[i], sfreq=fs, l_freq=0.5, h_freq=70.0, verbose=False
        )

    # cut or pad to first WINDOW_SAMPLES
    n = montage.shape[1]
    if n >= WINDOW_SAMPLES:
        win = montage[:, :WINDOW_SAMPLES]
    else:
        pad = WINDOW_SAMPLES - n
        win = np.pad(montage, ((0,0),(0,pad)), mode='constant')

    # turn into tensor
    x = torch.from_numpy(win).unsqueeze(0).float().to(device)

    # load & run model
    model = SeizureModel().to(device)
    state = torch.load(model_name, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        p_cls, p_reg = model(x)
        prob = float(p_cls.item())
        onset, offset = float(p_reg[0,0].item()), float(p_reg[0,1].item())

    # threshold at 0.5
    if prob > 0.5:
        return {
            'seizure_present': True,
            'seizure_confidence': prob,
            'onset': onset,
            'onset_confidence': 1.0,
            'offset': offset,
            'offset_confidence': 1.0
        }
    else:
        return {
            'seizure_present': False,
            'seizure_confidence': prob,
            'onset': 0.0,
            'onset_confidence': 0.0,
            'offset': 0.0,
            'offset_confidence': 0.0
        }
