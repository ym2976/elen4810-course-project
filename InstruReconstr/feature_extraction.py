from dataclasses import dataclass
from typing import Tuple
import numpy as np
import librosa

@dataclass
class FeatureBundle:
    waveform: np.ndarray
    sample_rate: int
    stft: np.ndarray
    mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    f0_hz: np.ndarray
    f0_times: np.ndarray

def load_mono_audio(path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr

def _smooth_f0(f0_hz: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Median-smooth f0 (ignoring NaNs) to reduce jitter."""
    f0 = f0_hz.copy()
    idx = np.where(~np.isnan(f0))[0]
    if len(idx) == 0:
        return f0

    # Simple median filter on valid region
    valid = f0[idx]
    pad = kernel_size // 2
    padded = np.pad(valid, (pad, pad), mode="edge")
    smoothed = np.array([
        np.nanmedian(padded[i:i+kernel_size]) for i in range(len(valid))
    ])
    f0[idx] = smoothed
    return f0

def _interp_nan(f0_hz: np.ndarray) -> np.ndarray:
    """Interpolate NaNs inside voiced regions (keep leading/trailing NaNs)."""
    f0 = f0_hz.copy()
    n = len(f0)
    x = np.arange(n)
    mask = ~np.isnan(f0)
    if mask.sum() < 2:
        return f0
    f0_interp = np.interp(x, x[mask], f0[mask])
    # keep original NaNs at edges if any
    first, last = x[mask][0], x[mask][-1]
    f0[:first] = np.nan
    f0[last+1:] = np.nan
    # fill middle NaNs
    mid_nan = np.isnan(f0)
    f0[mid_nan] = f0_interp[mid_nan]
    return f0

def extract_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 20,
) -> FeatureBundle:
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spec, ref=np.max),
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    f0_hz, voiced_flag, _ = librosa.pyin(
        waveform,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0_times = librosa.times_like(f0_hz, sr=sample_rate, hop_length=hop_length)

    # Replace unvoiced frames with NaNs
    f0_hz = np.where(voiced_flag, f0_hz, np.nan)

    # NEW: stabilize f0 (less jitter -> better harmonic tracking)
    f0_hz = _smooth_f0(f0_hz, kernel_size=5)
    f0_hz = _interp_nan(f0_hz)

    return FeatureBundle(
        waveform=waveform,
        sample_rate=sample_rate,
        stft=stft,
        mel_spectrogram=mel_spec,
        mfcc=mfcc,
        f0_hz=f0_hz,
        f0_times=f0_times,
    )
