import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, filtfilt, iirnotch, resample
from wettbewerb import get_3montages

# Constants
TARGET_FS = 250.0  # Target sampling rate in Hz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model Definition ==========
class SeizureModel(nn.Module):
    def __init__(self):
        super(SeizureModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=25, stride=3, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3),
            
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        class_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return class_out, reg_out

# ========== Filtering and Preprocessing ==========
def apply_filters(data: np.ndarray, fs: float) -> np.ndarray:
    """Apply bandpass and notch filters to EEG data"""
    # Bandpass filter (0.5-70 Hz)
    b_bp, a_bp = butter(4, [0.5/(fs/2), 70/(fs/2)], btype='band')
    # Notch filter (50/60 Hz)
    notch_freq = 50.0 if fs < 100 else 60.0
    b_notch, a_notch = iirnotch(notch_freq/(fs/2), Q=35)
    
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        ch_data = data[:, ch]
        # Apply filters
        ch_filtered = filtfilt(b_bp, a_bp, ch_data)
        ch_filtered = filtfilt(b_notch, a_notch, ch_filtered)
        filtered[:, ch] = ch_filtered
    return filtered

def preprocess_segment(segment: np.ndarray) -> np.ndarray:
    """Normalize a segment of EEG data"""
    normalized = np.zeros_like(segment)
    for ch in range(segment.shape[1]):
        channel = segment[:, ch]
        mean = np.mean(channel)
        std = np.std(channel)
        if std < 1e-6:
            std = 1.0
        normalized[:, ch] = (channel - mean) / std
    return normalized

# ========== Prediction Function ==========
def predict_labels(channels: list, data: np.ndarray, fs: float, 
                  reference_system: str, model_name: str = 'best_model.pth') -> dict:
    """
    Predicts seizure presence and onset/offset using sliding window approach
    with enhanced post-processing.
    """
    try:
        # Get bipolar montages
        _, montage_data, _ = get_3montages(channels, data)
        montage_data = montage_data.T  # Shape: (samples, 3)
        
        # Resample to target frequency if needed
        if abs(fs - TARGET_FS) > 1e-3:
            num_samples = int(len(montage_data) * TARGET_FS / fs)
            montage_data = resample(montage_data, num_samples, axis=0)
            fs = TARGET_FS
        
        # Apply filters
        montage_data = apply_filters(montage_data, fs)
        
        # Parameters
        window_size = int(120 * fs)  # 120-second window
        stride = int(10 * fs)         # 10-second stride
        min_seizure_duration = 3      # Minimum consecutive windows for valid seizure
        
        # Handle recordings shorter than window size
        if len(montage_data) < window_size:
            # Pad short recordings
            padding = np.zeros((window_size - len(montage_data), 3))
            montage_data = np.vstack([montage_data, padding])
            num_windows = 1
        else:
            num_windows = (len(montage_data) - window_size) // stride + 1
        
        # Initialize arrays
        seizure_probs = np.zeros(num_windows)
        window_onsets = np.zeros(num_windows)
        window_offsets = np.zeros(num_windows)
        window_times = np.zeros(num_windows)  # Start time of each window
        
        # Load model
        model = SeizureModel()
        model.load_state_dict(torch.load(model_name, map_location=device))
        model.to(device)
        model.eval()
        
        # Process each window
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            segment = montage_data[start:end]
            
            # Preprocess segment
            normalized = preprocess_segment(segment)
            
            # Convert to tensor
            segment_tensor = torch.tensor(normalized.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                cls_out, reg_out = model(segment_tensor)
                seizure_probs[i] = cls_out.item()
                onset, offset = reg_out.squeeze().cpu().numpy()
                
                # Clamp predictions to [0, 120] seconds
                window_onsets[i] = max(0, min(120, onset))
                window_offsets[i] = max(0, min(120, offset))
            
            window_times[i] = start / fs
        
        # Post-processing
        seizure_present = 0
        seizure_confidence = 0.0
        onset = None
        offset = None
        
        # Apply threshold to get binary predictions
        binary_preds = (seizure_probs > 0.4).astype(int)  # Lowered threshold for sensitivity
        
        # Apply morphological operations to clean predictions
        smoothed_preds = binary_closing(binary_preds, structure=np.ones(min_seizure_duration))
        smoothed_preds = binary_opening(smoothed_preds, structure=np.ones(2))
        
        # Find contiguous seizure regions
        diff = np.diff(smoothed_preds, prepend=0, append=0)
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0] - 1
        
        if len(start_indices) > 0:
            # Find the longest contiguous seizure segment
            max_duration = 0
            best_segment_idx = -1
            
            for i in range(len(start_indices)):
                seg_length = end_indices[i] - start_indices[i] + 1
                if seg_length > max_duration:
                    max_duration = seg_length
                    best_segment_idx = i
            
            # Get the best segment
            seg_start = start_indices[best_segment_idx]
            seg_end = end_indices[best_segment_idx]
            
            # Calculate confidence as average probability in segment
            seizure_confidence = np.mean(seizure_probs[seg_start:seg_end+1])
            
            if seizure_confidence > 0.5 and max_duration >= min_seizure_duration:
                seizure_present = 1
                
                # Calculate onset/offset using weighted average
                seg_probs = seizure_probs[seg_start:seg_end+1]
                seg_onsets = window_onsets[seg_start:seg_end+1] + window_times[seg_start:seg_end+1]
                seg_offsets = window_offsets[seg_start:seg_end+1] + window_times[seg_start:seg_end+1]
                
                # Weighted by probability
                onset = np.sum(seg_probs * seg_onsets) / np.sum(seg_probs)
                offset = np.sum(seg_probs * seg_offsets) / np.sum(seg_probs)
                
                # Ensure valid duration
                offset = max(onset + 5, offset)  # Minimum seizure duration of 5 seconds
        
        # Prepare output
        if seizure_present:
            return {
                "seizure_present": 1,
                "seizure_confidence": float(seizure_confidence),
                "onset": float(onset),
                "onset_confidence": 1.0,
                "offset": float(offset),
                "offset_confidence": 1.0
            }
        else:
            # Use max probability as confidence even for non-seizure
            max_prob = np.max(seizure_probs) if num_windows > 0 else 0.0
            return {
                "seizure_present": 0,
                "seizure_confidence": float(max_prob),
                "onset": None,
                "onset_confidence": 0.0,
                "offset": None,
                "offset_confidence": 0.0
            }
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "seizure_present": 0,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
