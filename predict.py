import numpy as np
import joblib
from typing import List, Dict, Any
from scipy.signal import butter, filtfilt, iirnotch, welch
from wettbewerb import get_6montages

# ========== Configuration ==========
fs = 250
lowcut, highcut = 0.5, 70.0
notch_freq, Q = 60.0, 35.0
bins = [(f, f + 1) for f in range(1, 71)]  # 1Hz bins from 1 to 70

# ========== Load Model ==========
MODEL_PATH = "best_model.pkl"
model = joblib.load(MODEL_PATH)

# ========== Filter Setup ==========
b_bp, a_bp = butter(N=4, Wn=[lowcut, highcut], btype='bandpass', fs=fs)
b_notch, a_notch = iirnotch(notch_freq, Q, fs)

# ========== Helper Functions ==========
def apply_filters(signal: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        x = filtfilt(b_bp, a_bp, signal[:, i])
        x = filtfilt(b_notch, a_notch, x)
        filtered[:, i] = x
    return filtered

def compute_psd_features(signal: np.ndarray, fs: int, bins: List[tuple]) -> np.ndarray:
    features = []
    for ch in range(signal.shape[1]):
        f, Pxx = welch(signal[:, ch], fs=fs, nperseg=512)
        bin_powers = []
        for low, high in bins:
            idx = np.logical_and(f >= low, f < high)
            power = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
            bin_powers.append(power)
        bin_powers = np.array(bin_powers)
        bin_powers /= np.sum(bin_powers) + 1e-9  # normalize
        features.append(bin_powers)
    return np.array(features)  # shape: (6, 70)

# ========== Required Submission Function ==========
def predict_labels(
    channels: List[str],
    data: np.ndarray,
    fs: float,
    reference_system: str,
    model_name: str = MODEL_PATH
) -> Dict[str, Any]:
    try:
        _, montage_data, missing = get_6montages(channels, data)
        if missing:
            return {"label": 0, "confidence": 0.0}

        montages = montage_data.T  # shape: (samples, 6)
        filtered = apply_filters(montages)
        psd = compute_psd_features(filtered, fs, bins)  # shape: (6, 70)
        feature_vector = psd.flatten().reshape(1, -1)  # shape: (1, 420)

        proba = model.predict_proba(feature_vector)[0]
        label = int(np.argmax(proba))
        confidence = float(np.max(proba))

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"Prediction failed: {e}")
        return {"label": 0, "confidence": 0.0}
