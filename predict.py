import torch
import numpy as np
from scipy.signal import resample_poly
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
    """Apply montages, filters, and resample to 250Hz"""
    _, montage_data, is_missing = get_3montages(channels, data)
    if is_missing:
        return None, None
    
    # Apply filters
    montage_data_filtered = np.zeros_like(montage_data)
    for i in range(3):
        # Notch filter (50Hz and harmonics)
        montage_data_filtered[i] = mne.filter.notch_filter(
            x=montage_data[i], Fs=fs, freqs=[50.0, 100.0], verbose=False
        )
        # Bandpass filter (0.5-70Hz)
        montage_data_filtered[i] = mne.filter.filter_data(
            data=montage_data_filtered[i], sfreq=fs, l_freq=0.5, h_freq=70.0, verbose=False
        )
    
    # Resample to 250Hz if needed
    if int(fs) != 250:
        montage_data_filtered = resample_poly(montage_data_filtered, up=250, down=int(fs), axis=1)
        effective_fs = 250
    else:
        effective_fs = fs
        
    return montage_data_filtered, effective_fs

def normalize_window(window):
    """Channel-wise normalization"""
    normalized = np.zeros_like(window)
    for i in range(3):
        channel = window[i]
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 0:
            normalized[i] = (channel - mean) / std
        else:
            normalized[i] = channel
    return normalized

# ========== Prediction Function ==========
def predict_labels(channels, data, fs, reference_system, model_name='best_model_5.pth'):
    # Constants
    TARGET_FS = 250
    WINDOW_DURATION = 300  # 5 minutes in seconds
    STRIDE_DURATION = 30   # 30 seconds stride
    SAMPLES_PER_WINDOW = WINDOW_DURATION * TARGET_FS
    SAMPLES_PER_STRIDE = STRIDE_DURATION * TARGET_FS
    
    # Preprocess entire signal
    montage_data, effective_fs = preprocess_signal(channels, data, fs)
    if montage_data is None:
        return {
            "seizure_present": 0,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
    
    # Get original recording duration
    original_duration = data.shape[1] / fs
    
    # Initialize tracking variables
    max_prob = 0.0
    best_onset = None
    best_offset = None
    
    # Create windows
    n_samples = montage_data.shape[1]
    starts = range(0, n_samples, SAMPLES_PER_STRIDE)
    
    # Load model
    model = SeizureModel().to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    
    # Process each window
    for start in starts:
        end = start + SAMPLES_PER_WINDOW
        
        # Handle last window
        if end > n_samples:
            window = montage_data[:, start:]
            # Pad with zeros if needed
            if window.shape[1] < SAMPLES_PER_WINDOW:
                pad_width = SAMPLES_PER_WINDOW - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')
        else:
            window = montage_data[:, start:end]
        
        # Normalize window
        window_norm = normalize_window(window)
        
        # Convert to tensor
        window_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            cls_out, reg_out = model(window_tensor)
            prob = cls_out.item()
            
            # Track best prediction
            if prob > max_prob:
                max_prob = prob
                
                if prob > 0.5:
                    onset_rel, offset_rel = reg_out.squeeze().cpu().numpy()
                    
                    # Convert to absolute times (seconds)
                    window_start_sec = start / TARGET_FS
                    onset_abs = max(0, min(onset_rel, WINDOW_DURATION)) + window_start_sec
                    offset_abs = max(0, min(offset_rel, WINDOW_DURATION)) + window_start_sec
                    
                    # Ensure valid seizure segment
                    if onset_abs > offset_abs:
                        offset_abs = onset_abs + 1.0  # minimum 1s duration
                    
                    # Clamp to original recording duration
                    best_onset = min(onset_abs, original_duration)
                    best_offset = min(offset_abs, original_duration)
    
    # Prepare output
    seizure_present = 1 if max_prob > 0.5 else 0
    
    if seizure_present:
        return {
            "seizure_present": 1,
            "seizure_confidence": max_prob,
            "onset": float(best_onset),
            "onset_confidence": 1.0,
            "offset": float(best_offset),
            "offset_confidence": 1.0
        }
    else:
        return {
            "seizure_present": 0,
            "seizure_confidence": max_prob,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
