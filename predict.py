# -*- coding: utf-8 -*-
"""
Enhanced Seizure Detection with Dual-Task Model
Combines classification and regression for precise seizure localization
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from wettbewerb import get_3montages
from typing import List, Dict, Any

# Constants
TARGET_FS = 250.0
WINDOW_SIZE = 5.0  # seconds
STRIDE = 2.0        # seconds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Model Definition ========
class DualTaskEEGModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Shared feature extractor
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64,
                           batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(128)

        # Task-specific heads
        self.class_head = nn.Linear(128, 1)
        self.reg_head = nn.Linear(128, 2)

    def forward(self, x):
        # Shared feature extraction
        x = self.relu(self.conv(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        features = self.norm(x)

        # Task-specific outputs
        class_logits = self.class_head(features).squeeze(1)
        reg_output = self.reg_head(features)
        
        return class_logits, reg_output

# ======== Helper Functions ========
def load_model(model_path: str, in_channels: int = 3) -> DualTaskEEGModel:
    """Load trained model with architecture matching training script"""
    model = DualTaskEEGModel(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def resample_to_250hz(data: np.ndarray, original_fs: float) -> np.ndarray:
    """Resample data to 250 Hz using polyphase resampling"""
    if original_fs == TARGET_FS:
        return data
    
    ratio = TARGET_FS / original_fs
    n_samples = int(len(data) * ratio)
    return resample_poly(data, n_samples, len(data), axis=0)

def apply_filters(data: np.ndarray) -> np.ndarray:
    """Apply bandpass and notch filtering to EEG data"""
    # Bandpass filter (0.5-70 Hz)
    b_bp, a_bp = butter(4, [0.5/(TARGET_FS/2), 70.0/(TARGET_FS/2)], btype='band')
    # Notch filter (50 Hz)
    b_notch, a_notch = iirnotch(50.0/(TARGET_FS/2), Q=35)
    
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        ch_data = data[:, ch]
        x = filtfilt(b_bp, a_bp, ch_data)
        x = filtfilt(b_notch, a_notch, x)
        filtered[:, ch] = x
    return filtered

def extract_windows(data: np.ndarray, window_samples: int, stride_samples: int):
    """Extract overlapping windows from EEG data"""
    n_windows = (data.shape[0] - window_samples) // stride_samples + 1
    windows = []
    starts = []
    
    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples
        window = data[start:end]
        
        # Standardize per channel
        window = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-6)
        windows.append(window.T)  # Transpose to (channels, time)
        starts.append(start / TARGET_FS)  # Start time in seconds
    
    return np.stack(windows), np.array(starts)

def merge_seizure_windows(window_preds, window_probs, window_starts, reg_outputs, 
                         min_seizure_duration=4.0, max_gap=2.0):
    """
    Merge adjacent seizure windows into continuous events using:
    - Classification confidence for detection
    - Regression outputs for precise timing
    """
    # Step 1: Find seizure clusters
    in_seizure = False
    current_start = 0.0
    current_end = 0.0
    current_max_prob = 0.0
    clusters = []
    
    for i, (pred, prob, start) in enumerate(zip(window_preds, window_probs, window_starts)):
        if pred == 1:
            if not in_seizure:
                # Seizure starts
                in_seizure = True
                current_start = start + reg_outputs[i, 0]  # Use regression onset
                current_end = start + reg_outputs[i, 1]    # Use regression offset
                current_max_prob = prob
            else:
                # Extend current seizure
                current_end = start + reg_outputs[i, 1]  # Update offset with current window
                current_max_prob = max(current_max_prob, prob)
        elif in_seizure:
            # Check if gap is small enough to continue
            next_start = window_starts[i+1] if i+1 < len(window_starts) else start + STRIDE
            gap = next_start - current_end
            
            if gap > max_gap:
                # Seizure ends
                in_seizure = False
                # Ensure valid duration
                duration = current_end - current_start
                if duration >= min_seizure_duration:
                    clusters.append((current_start, current_end, current_max_prob))
    
    # Handle seizure continuing to end
    if in_seizure:
        duration = current_end - current_start
        if duration >= min_seizure_duration:
            clusters.append((current_start, current_end, current_max_prob))
    
    return clusters

# ======== Main Prediction Function ========
def predict_labels(
    channels: List[str], 
    data: np.ndarray, 
    fs: float, 
    reference_system: str, 
    model_name: str = "best_dual_task_model.pth"
) -> Dict[str, Any]:
    """
    Main prediction function required by the competition
    Returns dictionary with seizure presence and timing information
    """
    try:
        # ===== 1. Preprocessing =====
        # Convert to 3-montage format
        _, montage_data, _ = get_3montages(channels, data)
        if montage_data is None:
            raise ValueError("Montage conversion failed")
        
        # Resample to 250Hz if needed
        if fs != TARGET_FS:
            montage_data = resample_to_250hz(montage_data, fs)
        
        # Apply filters
        filtered = apply_filters(montage_data)
        
        # ===== 2. Windowing =====
        window_samples = int(WINDOW_SIZE * TARGET_FS)
        stride_samples = int(STRIDE * TARGET_FS)
        windows, starts = extract_windows(filtered, window_samples, stride_samples)
        
        # ===== 3. Model Inference =====
        model = load_model(model_name, in_channels=windows.shape[1])
        batch_size = 128
        all_probs = []
        all_regs = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                logits, regs = model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_regs.extend(regs.cpu().numpy())
        
        probs = np.array(all_probs)
        regs = np.array(all_regs)
        preds = (probs > 0.5).astype(int)
        
        # ===== 4. Post-processing =====
        # Merge adjacent windows into seizure events using regression for timing
        seizures = merge_seizure_windows(preds, probs, starts, regs)
        
        # ===== 5. Prepare Output =====
        if seizures:
            # For competition, we assume single seizure per recording
            # Select seizure with highest confidence
            best_seizure = max(seizures, key=lambda x: x[2])
            onset, offset, confidence = best_seizure
            
            return {
                "seizure_present": 1,
                "seizure_confidence": float(confidence),
                "onset": float(onset),
                "onset_confidence": float(confidence),
                "offset": float(offset),
                "offset_confidence": float(confidence)
            }
        else:
            return {
                "seizure_present": 0,
                "seizure_confidence": 0.0,
                "onset": None,
                "onset_confidence": 0.0,
                "offset": None,
                "offset_confidence": 0.0
            }
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            "seizure_present": 0,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
