"""
feature_extraction
==================

Feature extraction utilities built on top of librosa for spectral analysis,
MFCCs, fundamental frequency estimation, and reusable preprocessing steps.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import librosa


@dataclass
class FeatureBundle:
    """
    Container for frequently used audio features.

    Attributes:
        waveform (np.ndarray): Mono waveform.
        sample_rate (int): Sample rate of the waveform.
        stft (np.ndarray): Complex STFT matrix.
        mel_spectrogram (np.ndarray): Mel power spectrogram.
        mfcc (np.ndarray): MFCCs derived from the mel spectrogram.
        f0_hz (np.ndarray): Fundamental frequency trajectory (Hz) with NaNs for unvoiced frames.
        f0_times (np.ndarray): Time axis (seconds) for the F0 trajectory.
    """

    waveform: np.ndarray
    sample_rate: int
    stft: np.ndarray
    mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    f0_hz: np.ndarray
    f0_times: np.ndarray


def load_mono_audio(path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Loads a file as mono audio using librosa.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate for loading and resampling.

    Returns:
        A tuple of (waveform, sample_rate).
    """

    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr


def extract_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 20,
) -> FeatureBundle:
    """
    Computes a common set of spectral features for analysis and modeling.

    Args:
        waveform: Mono waveform.
        sample_rate: Sample rate of the waveform.
        n_fft: FFT size for STFT-based operations.
        hop_length: Hop size for STFT.
        n_mels: Number of mel bands for the mel spectrogram.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        FeatureBundle with STFT, mel spectrogram, MFCCs, and F0 trajectory.
    """

    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spec, ref=np.max), sr=sample_rate, n_mfcc=n_mfcc
    )

    f0_hz, voiced_flag, _ = librosa.pyin(
        waveform,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0_times = librosa.times_like(f0_hz, sr=sample_rate, hop_length=hop_length)

    # Replace unvoiced frames with NaNs for downstream masking.
    f0_hz = np.where(voiced_flag, f0_hz, np.nan)

    return FeatureBundle(
        waveform=waveform,
        sample_rate=sample_rate,
        stft=stft,
        mel_spectrogram=mel_spec,
        mfcc=mfcc,
        f0_hz=f0_hz,
        f0_times=f0_times,
    )
