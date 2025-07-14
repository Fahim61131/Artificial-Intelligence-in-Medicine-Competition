import torch
import numpy as np
import mne
from wettbewerb import get_3montages

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model Definition ==========
class SeizureModel(torch.nn.Module):
    def __init__(self):
        super(SeizureModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(3, 32, kernel_size=25, stride=3, padding=12),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            torch.nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        self.lstm = torch.nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        class_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return class_out, reg_out

# ========== Preprocessing Functions ==========
def preprocess_signal(channels, data, fs):
    """Apply montages and filters, keep original sampling frequency"""
    _, montage_data, is_missing = get_3montages(channels, data)
    if is_missing:
        return None

    # Notch + bandpass filtering
    for i in range(montage_data.shape[0]):
        montage_data[i] = mne.filter.notch_filter(
            x=montage_data[i], Fs=fs, freqs=[50.0, 100.0], verbose=False
        )
        montage_data[i] = mne.filter.filter_data(
            data=montage_data[i], sfreq=fs, l_freq=0.5, h_freq=70.0, verbose=False
        )

    return montage_data

def normalize_window(window):
    """Channel-wise normalization"""
    normalized = np.zeros_like(window)
    for i in range(window.shape[0]):
        ch = window[i]
        m, s = ch.mean(), ch.std()
        normalized[i] = (ch - m) / (s + 1e-6) if s > 0 else ch
    return normalized

# ========== Prediction Function ==========
def predict_labels(channels, data, fs, reference_system, model_name='best_model.pth'):
    # Constants
    TARGET_FS = fs  # use original sampling frequency
    WINDOW_DURATION = 300   # 5 minutes
    STRIDE_DURATION = 30    # 30s
    SAMPLES_PER_WINDOW = WINDOW_DURATION * TARGET_FS
    SAMPLES_PER_STRIDE = STRIDE_DURATION * TARGET_FS

    # Preprocess entire signal
    montage_data = preprocess_signal(channels, data, fs)
    if montage_data is None:
        return {"seizure_present":0, "seizure_confidence":0.0,
                "onset":None, "onset_confidence":0.0,
                "offset":None, "offset_confidence":0.0}

    n_samples = montage_data.shape[1]
    original_duration = n_samples / fs

    # Load model
    model = SeizureModel().to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    max_prob = 0.0
    best_onset = best_offset = None

    # Slide window
    for start in range(0, n_samples, SAMPLES_PER_STRIDE):
        end = start + SAMPLES_PER_WINDOW
        if end > n_samples:
            window = montage_data[:, start:]
            pad = end - n_samples
            window = np.pad(window, ((0,0),(0,pad)), mode='constant')
        else:
            window = montage_data[:, start:end]

        win_norm = normalize_window(window)
        inp = torch.tensor(win_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prob, reg = model(inp)
            p = prob.item()
            if p > max_prob:
                max_prob = p
                if p > 0.5:
                    onset_rel, offset_rel = reg.squeeze().cpu().numpy()
                    wstart = start / fs
                    o = np.clip(onset_rel, 0, 1)*WINDOW_DURATION + wstart
                    f = np.clip(offset_rel, 0, 1)*WINDOW_DURATION + wstart
                    best_onset = min(o, original_duration)
                    best_offset = min(f, original_duration)

    if max_prob > 0.5:
        return {"seizure_present":1, "seizure_confidence":max_prob,
                "onset":float(best_onset), "onset_confidence":1.0,
                "offset":float(best_offset), "offset_confidence":1.0}
    else:
        return {"seizure_present":0, "seizure_confidence":max_prob,
                "onset":None,    "onset_confidence":0.0,
                "offset":None,   "offset_confidence":0.0}
