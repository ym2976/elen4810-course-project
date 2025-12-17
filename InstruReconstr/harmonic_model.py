"""
harmonic_model
==============

Routines for estimating sparse sinusoidal models from audio and reconstructing
signals using a limited number of partials.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
import librosa
from scipy.signal import find_peaks
from . import synthesis


@dataclass
class HarmonicModelResult:
    """
    Output container for harmonic modeling.

    Attributes:
        partials_hz_amp (list[tuple[float, float]]): Estimated (frequency, amplitude) pairs.
        envelope (np.ndarray): RMS envelope aligned with the input waveform length.
        reconstruction (np.ndarray): Resynthesized waveform using the partials and envelope.
    """

    partials_hz_amp: List[Tuple[float, float]]
    envelope: np.ndarray
    reconstruction: np.ndarray


def _rms_envelope(waveform: np.ndarray, hop_length: int = 512, frame_length: int = 2048) -> np.ndarray:
    """
    Computes an RMS envelope and upsamples it to the waveform length.
    """

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    envelope = np.interp(
        np.linspace(0, len(rms), num=len(waveform)),
        np.arange(len(rms)),
        rms,
    )
    return envelope


def estimate_partials(
    waveform: np.ndarray,
    sample_rate: int,
    n_partials: int = 10,
    n_fft: int = 4096,
    hop_length: int = 512,
    peak_threshold: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Estimates dominant spectral partials using an averaged magnitude spectrum.
    """

    stft_mag = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
    mean_mag = stft_mag.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    peaks, _ = find_peaks(mean_mag, height=peak_threshold * np.max(mean_mag))
    if len(peaks) == 0:
        return []

    sorted_idx = peaks[np.argsort(mean_mag[peaks])][::-1]
    selected_idx = sorted(sorted_idx[:n_partials], key=lambda i: freqs[i])

    partials = [(float(freqs[idx]), float(mean_mag[idx])) for idx in selected_idx]
    # Normalize amplitudes to unit peak
    max_amp = max(amp for _, amp in partials) if partials else 1.0
    partials = [(freq, amp / max_amp) for freq, amp in partials]
    return partials


def resynthesize_with_partials(
    waveform: np.ndarray,
    sample_rate: int,
    partials_hz_amp: Sequence[Tuple[float, float]],
    hop_length: int = 512,
    frame_length: int = 2048,
) -> HarmonicModelResult:
    """
    Resynthesizes a waveform using provided partials and an RMS envelope.
    """

    duration = len(waveform) / sample_rate
    envelope = _rms_envelope(waveform, hop_length=hop_length, frame_length=frame_length)
    synthesis_wave = synthesis.render_sinusoidal_partials(
        partials_hz_amp, duration_seconds=duration, sample_rate=sample_rate, phases=None, normalize=True
    )
    synthesis_wave = synthesis.apply_envelope(synthesis_wave, envelope)
    synthesis_wave = synthesis.normalize(synthesis_wave)

    return HarmonicModelResult(
        partials_hz_amp=list(partials_hz_amp),
        envelope=envelope,
        reconstruction=synthesis_wave,
    )


def fit_and_resynthesize(
    waveform: np.ndarray,
    sample_rate: int,
    n_partials: int = 10,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> HarmonicModelResult:
    """
    Estimates partials from a waveform and resynthesizes it using a sinusoidal model.
    """

    partials = estimate_partials(
        waveform,
        sample_rate=sample_rate,
        n_partials=n_partials,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return resynthesize_with_partials(
        waveform,
        sample_rate=sample_rate,
        partials_hz_amp=partials,
        hop_length=hop_length,
    )
