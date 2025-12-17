
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import librosa

from . import synthesis


# -----------------------------
# Data container
# -----------------------------

@dataclass
class HarmonicModelResult:
    """
    Output container for harmonic modeling.

    Attributes:
        partials_hz_amp: Static partial list for visualization (freq_hz, relative_amp).
        envelope: RMS envelope (upsampled to waveform length).
        reconstruction: Resynthesized waveform (harmonics + envelope + optional residual).
    """
    partials_hz_amp: List[Tuple[float, float]]
    envelope: np.ndarray
    reconstruction: np.ndarray


# -----------------------------
# Helpers
# -----------------------------

def _rms_envelope(
    waveform: np.ndarray,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> np.ndarray:
    """Compute an RMS envelope and upsample it to waveform length."""
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    envelope = np.interp(
        np.linspace(0, len(rms), num=len(waveform)),
        np.arange(len(rms)),
        rms,
    )
    return envelope


def _estimate_harmonic_tracks(
    waveform: np.ndarray,
    sample_rate: int,
    f0_hz: np.ndarray,
    n_partials: int = 12,
    n_fft: int = 4096,
    hop_length: int = 512,
    freq_tolerance_hz: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Estimate time-varying harmonic amplitudes by sampling STFT magnitude
    around harmonic frequencies (k * f0) per frame.

    Returns:
        f0_used: (T,) f0 aligned to STFT frames (NaNs allowed)
        harm_freqs: (K,) harmonic frequencies based on median f0 (for display)
        harm_amps_t: (K, T) time-varying harmonic magnitudes (normalized per-frame)
        partials_hz_amp: list of (freq, amp) summary for plotting (median over voiced)
    """
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)  # (F, T)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    T = mag.shape[1]
    f0 = f0_hz.copy()
    if len(f0) != T:
        f0 = librosa.util.fix_length(f0, size=T)

    voiced = ~np.isnan(f0)
    if voiced.sum() < 3:
        return f0, np.array([]), np.zeros((0, T), dtype=float), []

    f0_med = float(np.nanmedian(f0))
    harm_freqs = f0_med * (np.arange(1, n_partials + 1))

    harm_amps_t = np.zeros((n_partials, T), dtype=float)

    for t in range(T):
        if np.isnan(f0[t]):
            continue
        base = float(f0[t])
        for k in range(1, n_partials + 1):
            target = k * base
            lo = target - freq_tolerance_hz
            hi = target + freq_tolerance_hz
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            if len(idx) == 0:
                continue
            harm_amps_t[k - 1, t] = float(np.max(mag[idx, t]))

    # Normalize per frame to reduce loudness swings (keep relative harmonic shape)
    frame_max = np.max(harm_amps_t, axis=0, keepdims=True) + 1e-12
    harm_amps_t = harm_amps_t / frame_max

    # Static partial summary for plotting (median across voiced frames)
    amps_static = np.nanmedian(harm_amps_t[:, voiced], axis=1)
    if np.max(amps_static) > 0:
        amps_static = amps_static / (np.max(amps_static) + 1e-12)

    partials_hz_amp = [(float(harm_freqs[i]), float(amps_static[i])) for i in range(n_partials)]
    return f0, harm_freqs, harm_amps_t, partials_hz_amp


def _overlap_add_synthesis(
    sample_rate: int,
    f0_hz: np.ndarray,
    harm_amps_t: np.ndarray,
    hop_length: int,
    duration_samples: int,
) -> np.ndarray:
    """
    Frame-wise additive synthesis using harmonic tracks with phase continuity
    and overlap-add (OLA) normalization to avoid amplitude beating.
    """
    K, T = harm_amps_t.shape
    y = np.zeros(duration_samples, dtype=float)

    win_len = hop_length * 2
    win = np.hanning(win_len)
    wsum = np.zeros(duration_samples, dtype=float)

    # Phase accumulator per harmonic (keeps continuity across frames)
    phase = np.zeros(K, dtype=float)
    n = np.arange(win_len) / sample_rate

    for t in range(T):
        if np.isnan(f0_hz[t]):
            continue

        start = t * hop_length
        end = start + win_len
        if start >= duration_samples:
            break

        frame_len = win_len
        if end > duration_samples:
            frame_len = duration_samples - start
            end = duration_samples

        base = float(f0_hz[t])
        frame = np.zeros(frame_len, dtype=float)

        for k in range(1, K + 1):
            A = float(harm_amps_t[k - 1, t])
            f = k * base

            frame += A * np.cos(2 * np.pi * f * n[:frame_len] + phase[k - 1])

            # Advance phase by hop length to preserve continuity between frames
            phase[k - 1] = (phase[k - 1] + 2 * np.pi * f * (hop_length / sample_rate)) % (2 * np.pi)

        w = win[:frame_len]
        y[start:end] += frame * w
        wsum[start:end] += w

    y = y / (wsum + 1e-12)
    y = synthesis.normalize(y)
    return y


# -----------------------------
# Main API
# -----------------------------

def fit_and_resynthesize(
    waveform: np.ndarray,
    sample_rate: int,
    n_partials: int = 12,
    n_fft: int = 4096,
    hop_length: int = 512,
    f0_hz: Optional[np.ndarray] = None,
    residual_mix: float = 0.0,  # 0..1
) -> HarmonicModelResult:
    """
    Harmonic analysis-resynthesis using f0-aligned harmonic tracks.

    Args:
        waveform: Input mono waveform.
        sample_rate: Sample rate of waveform.
        n_partials: Number of harmonics to synthesize.
        n_fft: FFT size for STFT sampling.
        hop_length: Hop size for STFT / OLA.
        f0_hz: Optional f0 track aligned to hop_length. If None, estimates via librosa.pyin.
        residual_mix: Mix back some residual (waveform - harmonic_recon) for realism.

    Returns:
        HarmonicModelResult with partials (for plotting), envelope, and reconstruction.
    """
    if f0_hz is None:
        f0_hz, voiced_flag, _ = librosa.pyin(
            waveform,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        f0_hz = np.where(voiced_flag, f0_hz, np.nan)

    envelope = _rms_envelope(waveform, hop_length=hop_length, frame_length=n_fft)

    f0_used, _, harm_amps_t, partials_hz_amp = _estimate_harmonic_tracks(
        waveform=waveform,
        sample_rate=sample_rate,
        f0_hz=f0_hz,
        n_partials=n_partials,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_tolerance_hz=15.0,
    )

    recon = _overlap_add_synthesis(
        sample_rate=sample_rate,
        f0_hz=f0_used,
        harm_amps_t=harm_amps_t,
        hop_length=hop_length,
        duration_samples=len(waveform),
    )

    recon = synthesis.apply_envelope(recon, envelope)
    recon = synthesis.normalize(recon)

    if residual_mix > 0:
        residual = waveform - recon
        residual = synthesis.normalize(residual) * float(residual_mix)
        recon = synthesis.normalize(recon + residual)

    return HarmonicModelResult(
        partials_hz_amp=partials_hz_amp,
        envelope=envelope,
        reconstruction=recon,
    )
