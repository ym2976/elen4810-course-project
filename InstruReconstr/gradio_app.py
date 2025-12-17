"""
gradio_app
==========

Interactive Gradio demo for uploading audio, viewing original vs. reconstructed
waveforms/spectrograms/envelopes/partials, and inspecting reconstruction metrics.
"""

from __future__ import annotations

from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from . import feature_extraction
from . import harmonic_model
from . import evaluation
from . import visualization


def _waveform_plot(waveform: np.ndarray, sample_rate: int, title: str):
    times = np.arange(len(waveform)) / sample_rate
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(times, waveform, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _spectrogram_plot(waveform: np.ndarray, sample_rate: int, title: str, n_fft: int = 2048, hop_length: int = 512):
    fig, ax = plt.subplots(figsize=(6, 3))
    import librosa
    import librosa.display

    stft = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
    spec_db = librosa.amplitude_to_db(stft, ref=np.max)
    librosa.display.specshow(
        spec_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax,
        cmap="magma",
    )
    ax.set_title(title)
    fig.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def _format_metrics_table(metrics: evaluation.ReconstructionMetrics) -> str:
    rows = [
        ("Log-spectral distance", metrics.log_spectral_distance),
        ("F0 RMSE (Hz)", metrics.f0_rmse_hz),
        ("Spectral convergence", metrics.spectral_convergence),
        ("RMSE waveform", metrics.rmse_waveform),
        ("RMSE spectrogram", metrics.rmse_spectrogram),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    for name, value in rows:
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)


def process(
    audio_path: str,
    n_partials: int = 12,
    hop_length: int = 512,
) -> Tuple[Any, Any, Any, Any, Any, Any, Tuple[int, np.ndarray], Tuple[int, np.ndarray], str]:
    if not audio_path:
        return (None,) * 9

    sample_rate = 22050
    waveform, sr = feature_extraction.load_mono_audio(audio_path, sample_rate=sample_rate)
    features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
    model = harmonic_model.fit_and_resynthesize(
        waveform, sample_rate=sr, n_partials=n_partials, hop_length=hop_length
    )

    metrics = evaluation.compute_all_metrics(waveform, model.reconstruction, features.f0_hz, features.f0_hz, sr)

    orig_wave_fig = _waveform_plot(waveform, sr, "Original waveform")
    orig_spec_fig = _spectrogram_plot(waveform, sr, "Original spectrogram", hop_length=hop_length)

    env_fig, _ = visualization.plot_envelope(model.envelope, sr, title="Envelope")
    partials_fig, _ = visualization.plot_partials(model.partials_hz_amp, title="Estimated partials")

    recon_wave_fig = _waveform_plot(model.reconstruction, sr, "Reconstructed waveform")
    recon_spec_fig = _spectrogram_plot(model.reconstruction, sr, "Reconstructed spectrogram", hop_length=hop_length)

    metrics_table = _format_metrics_table(metrics)

    return (
        orig_wave_fig,
        orig_spec_fig,
        env_fig,
        partials_fig,
        recon_wave_fig,
        recon_spec_fig,
        (sr, waveform),
        (sr, model.reconstruction),
        metrics_table,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="InstruReconstr") as demo:
        gr.Markdown("# InstruReconstr demo\nUpload audio, inspect features, and compare reconstructions.")

        with gr.Row():
            audio_input = gr.Audio(label="Upload audio", type="filepath")
            n_partials = gr.Slider(4, 32, value=12, step=1, label="Number of partials")
            hop_length = gr.Slider(128, 1024, value=512, step=64, label="Hop length")

        with gr.Row():
            with gr.Column():
                orig_wave = gr.Plot(label="Original waveform")
                orig_spec = gr.Plot(label="Original spectrogram")
            with gr.Column():
                envelope_plot = gr.Plot(label="Envelope")
                partials_plot = gr.Plot(label="Partials")
            with gr.Column():
                recon_wave = gr.Plot(label="Reconstructed waveform")
                recon_spec = gr.Plot(label="Reconstructed spectrogram")

        with gr.Row():
            orig_audio = gr.Audio(label="Original audio", type="numpy")
            recon_audio = gr.Audio(label="Reconstructed audio", type="numpy")
        metrics_md = gr.Markdown(label="Metrics")

        for control in (audio_input, n_partials, hop_length):
            control.change(
                process,
                inputs=[audio_input, n_partials, hop_length],
                outputs=[
                    orig_wave,
                    orig_spec,
                    envelope_plot,
                    partials_plot,
                    recon_wave,
                    recon_spec,
                    orig_audio,
                    recon_audio,
                    metrics_md,
                ],
            )
    return demo


def main():
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
