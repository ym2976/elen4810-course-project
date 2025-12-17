"""
synthesis
=========

Lightweight sinusoidal synthesis helpers for reconstructing signals from
frequency and amplitude estimates.
"""

from typing import Iterable, List, Tuple
import numpy as np


def render_sinusoidal_partials(
    partials_hz_amp: Iterable[Tuple[float, float]],
    duration_seconds: float,
    sample_rate: int,
    phases: Iterable[float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Renders a waveform from (frequency, amplitude) partials.

    Args:
        partials_hz_amp: Iterable of (frequency_hz, amplitude_linear).
        duration_seconds: Output duration in seconds.
        sample_rate: Target sample rate.
        phases: Optional iterable of starting phases (radians). If None, zeros are used.
        normalize: Whether to scale the output to [-1, 1].

    Returns:
        The synthesized mono waveform.
    """

    partials = list(partials_hz_amp)
    if not partials:
        return np.zeros(int(duration_seconds * sample_rate))

    t = np.linspace(0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    if phases is None:
        phases = [0.0 for _ in partials]

    signal = np.zeros_like(t)
    for (freq, amp), phase in zip(partials, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    if normalize:
        max_val = np.max(np.abs(signal)) + 1e-12
        signal = signal / max_val

    return signal


def apply_envelope(signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """
    Applies an amplitude envelope to a signal, stretching or trimming as needed.
    """

    if len(envelope) == len(signal):
        return signal * envelope

    envelope_resampled = np.interp(
        np.linspace(0, 1, num=len(signal)),
        np.linspace(0, 1, num=len(envelope)),
        envelope,
    )
    return signal * envelope_resampled


def normalize(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Scales signal to [-1, 1].
    """

    peak = np.max(np.abs(signal)) + eps
    return signal / peak
