import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from wettbewerb import get_3montages

# ========== Model (same as training) ==========
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

# ========== Filtering ==========
def create_filters(fs):
    b_bp, a_bp = butter(4, [0.5/(fs/2), 70.0/(fs/2)], btype='band')
    b_notch, a_notch = iirnotch(60.0/(fs/2), Q=35.0)
    return b_bp, a_bp, b_notch, a_notch

def apply_filters_full(mont, b_bp, a_bp, b_notch, a_notch):
    out = np.zeros_like(mont)
    for ch in range(mont.shape[1]):
        x = filtfilt(b_bp, a_bp, mont[:, ch])
        out[:, ch] = filtfilt(b_notch, a_notch, x)
    return out

# ========== Batched Sliding Prediction ==========
def predict_sliding(channels, data, fs=250,
                    model_path='best_light_convlstm_transformer_model.pth',
                    window_sec=5, stride_sec=2, prob_thresh=0.5,
                    batch_size=64, device='cpu'):
    # load model
    cp = torch.load(model_path, map_location='cpu')
    model = LightConvLSTMTransformer(cp['input_channels']).to(device).eval()
    model.load_state_dict(cp['model_state_dict'])

    # montage + filter once
    _, mont, _ = get_3montages(channels, data)
    mont = mont.T                            # (T, C)
    b_bp, a_bp, b_notch, a_notch = create_filters(fs)
    mont = apply_filters_full(mont, b_bp, a_bp, b_notch, a_notch)

    ws = int(window_sec * fs)
    ss = int(stride_sec * fs)
    total = mont.shape[0]
    starts = np.arange(0, total - ws + 1, ss)

    # build all windows
    windows = np.stack([mont[s:s+ws] for s in starts], axis=0)   # (Nw, ws, C)
    # normalize per-window & transpose to (Nw, C, ws)
    mean = windows.mean(axis=1, keepdims=True)
    std  = windows.std(axis=1, keepdims=True) + 1e-6
    windows = ((windows - mean) / std).transpose(0,2,1)

    # batch inference
    probs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i+batch_size]).float().to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    # threshold, clean spikes, group runs (same as before)
    preds = probs > prob_thresh
    for i in range(1, len(preds)-1):
        if preds[i] and not preds[i-1] and not preds[i+1]:
            preds[i] = False

    events, in_ev = [], False
    for idx, flag in enumerate(preds):
        if flag and not in_ev:
            in_ev, ev0 = True, idx
        elif not flag and in_ev:
            events.append((ev0, idx-1)); in_ev = False
    if in_ev: events.append((ev0, len(preds)-1))

    if not events:
        return {
            "seizure_present": 0,
            "seizure_confidence": float(probs.max()),
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }

    # pick longest
    lengths = [j-i for i,j in events]
    si, ei = events[int(np.argmax(lengths))]
    onset  = starts[si]/fs
    offset = (starts[ei]+ws)/fs
    conf   = float(probs[si:ei+1].mean())

    return {
        "seizure_present": 1,
        "seizure_confidence": conf,
        "onset": onset,
        "onset_confidence": conf,
        "offset": offset,
        "offset_confidence": conf
    }
