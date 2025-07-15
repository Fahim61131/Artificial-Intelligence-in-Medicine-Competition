import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from wettbewerb import get_3montages

# ========== Model Definition ==========

class EEGClassifierRegressor(nn.Module):
    def __init__(self):
        super(EEGClassifierRegressor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 30000)
            x = self.conv(dummy_input)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            self.flatten_dim = x[:, -1, :].shape[1]

        self.cls_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        cls_out = self.cls_head(x).squeeze(1)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

# ========== Filtering Utilities ==========

def create_filters(fs):
    b_bp, a_bp = butter(N=4, Wn=[0.5 / (fs / 2), 70.0 / (fs / 2)], btype='band')
    b_notch, a_notch = iirnotch(w0=60.0 / (fs / 2), Q=35.0)
    return b_bp, a_bp, b_notch, a_notch

def apply_filters(signal, b_bp, a_bp, b_notch, a_notch):
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        x = filtfilt(b_bp, a_bp, signal[:, i])
        x = filtfilt(b_notch, a_notch, x)
        filtered[:, i] = x
    return filtered

# ========== Prediction Function Required by Evaluator ==========

def predict_labels(channels, data, fs, reference_system, model_name='model_combined.pth'):
    """
    Predicts seizure presence and onset/offset for 120s EEG window.
    """
    # Load model
    model = EEGClassifierRegressor()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    model.eval()

    # Extract 3 bipolar montages
    _, montage_data, _ = get_3montages(channels, data)
    montage_data = montage_data.T  # shape: (samples, 3)

    # Apply filtering
    b_bp, a_bp, b_notch, a_notch = create_filters(fs)
    filtered = apply_filters(montage_data, b_bp, a_bp, b_notch, a_notch)

    # Take 120s window from start
    fs = int(fs)
    window_samples = 120 * fs
    segment = filtered[:window_samples]

    if segment.shape[0] < window_samples:
        segment = np.pad(segment, ((0, window_samples - segment.shape[0]), (0, 0)), mode='constant')

    # Normalize
    segment = segment.T  # (3, 30000)
    segment = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-6)

    # Inference
    x_input = torch.tensor(segment).unsqueeze(0).float()  # (1, 3, 30000)
    with torch.no_grad():
        cls_out, reg_out = model(x_input)
        seizure_prob = torch.sigmoid(cls_out).item()
        seizure_present = int(seizure_prob > 0.5)

        if seizure_present:
            onset, offset = reg_out.squeeze().tolist()
            prediction = {
                "seizure_present": 1,
                "seizure_confidence": seizure_prob,
                "onset": float(onset),
                "onset_confidence": 1.0,
                "offset": float(offset),
                "offset_confidence": 1.0
            }
        else:
            prediction = {
                "seizure_present": 0,
                "seizure_confidence": seizure_prob,
                "onset": None,
                "onset_confidence": 0.0,
                "offset": None,
                "offset_confidence": 0.0
            }

    return prediction
