"""
interactive
===========

Command-line workflow for analyzing a single audio file, fitting a sparse
sinusoidal model, and exporting comparative artifacts for listening tests.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from .config_manager import ConfigManager
from . import feature_extraction
from . import harmonic_model
from . import evaluation


def _maybe_play(audio: np.ndarray, sample_rate: int, label: str):
    """
    Plays audio if sounddevice or pyaudio is available; otherwise prints a hint.
    """

    if importlib.util.find_spec("sounddevice"):
        import sounddevice as sd

        sd.play(audio, sample_rate, blocking=True)
        print(f"[info] Played '{label}' using sounddevice.")
        return

    if importlib.util.find_spec("simpleaudio"):
        import simpleaudio as sa

        sa.play_buffer(
            (audio * 32767).astype(np.int16),
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=sample_rate,
        ).wait_done()
        print(f"[info] Played '{label}' using simpleaudio.")
        return

    print(f"[info] No playback backend found. Saved '{label}' for manual listening.")


def run_interactive(
    input_path: Path,
    output_dir: Path,
    n_partials: int = 12,
    sample_rate: int = 22050,
    hop_length: int = 512,
):
    """
    Executes the interactive pipeline: load -> analyze -> resynthesize -> export artifacts.
    """

    waveform, sr = feature_extraction.load_mono_audio(str(input_path), sample_rate=sample_rate)
    features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
    model = harmonic_model.fit_and_resynthesize(
        waveform, sample_rate=sr, n_partials=n_partials, hop_length=hop_length
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / f"{input_path.stem}_original.wav"
    recon_path = output_dir / f"{input_path.stem}_reconstructed.wav"

    sf.write(original_path, waveform, sr)
    sf.write(recon_path, model.reconstruction, sr)

    metrics = evaluation.compute_all_metrics(waveform, model.reconstruction, features.f0_hz, features.f0_hz, sr)

    print(f"[metrics] Log-Spectral Distance : {metrics.log_spectral_distance:.4f}")
    print(f"[metrics] F0 RMSE (Hz)          : {metrics.f0_rmse_hz:.4f}")
    print(f"[metrics] Spectral convergence  : {metrics.spectral_convergence:.4f}")
    print(f"[metrics] RMSE (waveform)       : {metrics.rmse_waveform:.4f}")
    print(f"[metrics] RMSE (spectrogram)    : {metrics.rmse_spectrogram:.4f}")
    print(f"[files] Original saved to     : {original_path}")
    print(f"[files] Reconstruction saved : {recon_path}")

    _maybe_play(waveform, sr, "original")
    _maybe_play(model.reconstruction, sr, "reconstructed")


def main():
    parser = argparse.ArgumentParser(description="Interactive reconstruction and comparison tool.")
    parser.add_argument("input", type=str, help="Path to an input audio file.")
    parser.add_argument(
        "--partials", type=int, default=12, help="Number of sinusoidal components to keep."
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Resample audio to this rate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where artifacts are stored. Defaults to config.PATH_INTERACTIVE_RESULTS.",
    )
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for analysis.")
    args = parser.parse_args()

    cfg = ConfigManager.get_instance().config
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.PATH_INTERACTIVE_RESULTS)

    run_interactive(
        input_path=Path(args.input),
        output_dir=output_dir,
        n_partials=args.partials,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )


if __name__ == "__main__":
    main()
