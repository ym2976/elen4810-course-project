"""
evaluation
==========

Evaluation metrics for comparing reconstructed sounds against references.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import librosa


def _align_lengths(reference: np.ndarray, estimate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trims reference and estimate to the same length for fair comparison.
    """

    if len(reference) == len(estimate):
        return reference, estimate
    min_len = min(len(reference), len(estimate))
    return reference[:min_len], estimate[:min_len]


@dataclass
class ReconstructionMetrics:
    """
    Container with commonly reported reconstruction metrics.
    """

    log_spectral_distance: float
    f0_rmse_hz: float
    spectral_convergence: float
    rmse_waveform: float
    rmse_spectrogram: float


def log_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> float:
    """
    Computes Log-Spectral Distance (LSD) between two signals.

    Args:
        reference: Ground-truth waveform.
        estimate: Reconstructed waveform.
        sample_rate: Sample rate shared by both signals.
        n_fft: FFT window size for the STFT.
        hop_length: Hop size for the STFT.

    Returns:
        Scalar LSD value.
    """

    reference, estimate = _align_lengths(reference, estimate)

    ref_mag = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length)) ** 2
    est_mag = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop_length)) ** 2

    ref_db = librosa.power_to_db(ref_mag + 1e-9, ref=np.max)
    est_db = librosa.power_to_db(est_mag + 1e-9, ref=np.max)

    lsd = np.sqrt(np.mean((ref_db - est_db) ** 2))
    return float(lsd)


def f0_rmse(reference_f0: np.ndarray, estimate_f0: np.ndarray) -> float:
    """
    Computes RMSE between two F0 trajectories, ignoring unvoiced frames (NaNs).
    """

    mask = (~np.isnan(reference_f0)) & (~np.isnan(estimate_f0))
    if not np.any(mask):
        return float("nan")

    diff = reference_f0[mask] - estimate_f0[mask]
    return float(np.sqrt(np.mean(diff**2)))


def spectral_convergence(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Computes spectral convergence between two signals.
    """

    reference, estimate = _align_lengths(reference, estimate)

    ref_mag = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length))
    est_mag = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop_length))

    numerator = np.linalg.norm(ref_mag - est_mag)
    denominator = np.linalg.norm(ref_mag) + 1e-9
    return float(numerator / denominator)


def compute_all_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    reference_f0: np.ndarray,
    estimate_f0: np.ndarray,
    sample_rate: int,
) -> ReconstructionMetrics:
    """
    Convenience wrapper returning LSD, F0 RMSE, spectral convergence, waveform RMSE,
    and spectrogram RMSE.
    """

    reference, estimate = _align_lengths(reference, estimate)
    lsd = log_spectral_distance(reference, estimate, sample_rate)
    f0_error = f0_rmse(reference_f0, estimate_f0)
    conv = spectral_convergence(reference, estimate, sample_rate)
    wave_rmse = rmse_waveform(reference, estimate)
    spec_rmse = spectrogram_rmse(reference, estimate, sample_rate)
    return ReconstructionMetrics(
        log_spectral_distance=lsd,
        f0_rmse_hz=f0_error,
        spectral_convergence=conv,
        rmse_waveform=wave_rmse,
        rmse_spectrogram=spec_rmse,
    )


def rmse_waveform(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Root mean squared error between two waveforms.
    """

    reference, estimate = _align_lengths(reference, estimate)
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def spectrogram_rmse(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Root mean squared error between magnitude spectrograms.
    """

    reference, estimate = _align_lengths(reference, estimate)
    ref_mag = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length))
    est_mag = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop_length))
    return float(np.sqrt(np.mean((ref_mag - est_mag) ** 2)))
