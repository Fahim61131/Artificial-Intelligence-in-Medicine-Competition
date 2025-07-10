import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from wettbewerb import get_3montages

========== Model Definition (should match training) ==========
class LightConvLSTMTransformer(nn.Module):
    def __init__(self, in_channels, n_classes=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.fc(x).squeeze(1)
        return x

========== Filter Utilities ==========
def create_filters(fs, lowcut=0.5, highcut=70.0, notch_freq=60.0, Q=35.0):
    b_bp, a_bp = butter(N=4, Wn=[lowcut/(fs/2), highcut/(fs/2)], btype='band')
    b_notch, a_notch = iirnotch(w0=notch_freq/(fs/2), Q=Q)
    return b_bp, a_bp, b_notch, a_notch

def apply_filters(signal, b_bp, a_bp, b_notch, a_notch):
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        x = filtfilt(b_bp, a_bp, signal[:, i])
        filtered[:, i] = filtfilt(b_notch, a_notch, x)
    return filtered

========== Sliding Predict Function ==========
Note: This function assumes a sampling rate fs (e.g., 250 Hz).
def predict_sliding(channels, data, fs=250, reference_system=None,
                    model_path='best_light_convlstm_transformer_model.pth',
                    window_sec=5, stride_sec=2, prob_thresh=0.5):
    """
    Slides a window of length window_sec with stride stride_sec over the signal sampled at fs Hz,
    predicts seizure probability per window, groups consecutive positive windows,
    and returns the most prominent seizure event.

    Args:
        channels: list of channel names
        data: raw EEG data array (channels × time)
        fs: sampling frequency (Hz), default = 250
        reference_system: unused placeholder for API compatibility
        model_path: path to the trained model checkpoint
        window_sec: window length in seconds (default 5)
        stride_sec: stride length in seconds (default 2)
        prob_thresh: probability threshold for binary detection (default 0.5)
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    in_channels = checkpoint.get('input_channels', 3)
    model = LightConvLSTMTransformer(in_channels).eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract 3 bipolar montages
    , mont,  = get_3montages(channels, data)
    mont = mont.T  # shape (T, C)

    # Filters
    b_bp, a_bp, b_notch, a_notch = create_filters(fs)

    # Window params (in samples)
    ws = int(window_sec * fs)    # e.g., 5 sec × 250 Hz = 1250 samples
    ss = int(stride_sec * fs)    # e.g., 2 sec × 250 Hz = 500 samples
    total_samples = mont.shape[0]

    # Slide, filter, normalize, and predict
    probs, positions = [], []
    for start in range(0, total_samples - ws + 1, ss):
        segment = mont[start:start + ws]
        seg_f = apply_filters(segment, b_bp, a_bp, b_notch, a_notch)
        seg_norm = (seg_f - seg_f.mean(axis=0)) / (seg_f.std(axis=0) + 1e-6)
        x = torch.from_numpy(seg_norm.T).unsqueeze(0).float()
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        probs.append(prob)
        positions.append(start)

    probs = np.array(probs)
    preds = probs > prob_thresh

    # Remove isolated spikes
    cleaned = preds.copy()
    for i in range(1, len(preds)-1):
        if preds[i] and not preds[i-1] and not preds[i+1]:
            cleaned[i] = False

    # Find contiguous positive runs
    events, in_event = [], False
    for idx, flag in enumerate(cleaned):
        if flag and not in_event:
            in_event, ev_start = True, idx
        elif not flag and in_event:
            events.append((ev_start, idx-1)); in_event = False
    if in_event:
        events.append((ev_start, len(cleaned)-1))

    # If no seizure detected
    if not events:
        return {'seizure_present': 0,
                'seizure_confidence': float(probs.max()),
                'onset': None, 'onset_confidence': 0.0,
                'offset': None, 'offset_confidence': 0.0}

    # Select the longest event
    lengths = [end-start for start,end in events]
    s_idx, e_idx = events[int(np.argmax(lengths))]
    onset_sec  = positions[s_idx] / fs
    offset_sec = (positions[e_idx] + ws) / fs
    conf       = float(probs[s_idx:e_idx+1].mean())

    return {'seizure_present': 1,
            'seizure_confidence': conf,
            'onset': onset_sec, 'onset_confidence': conf,
            'offset': offset_sec, 'offset_confidence': conf}
