"""
diagnostic
==========

Utilities for diagnosing when harmonic reconstruction performs well or poorly.
The module can batch-process a dataset, compute quality metrics alongside
lightweight descriptors (duration, F0 coverage, zero-crossing rate, etc.), and
emit visualizations/scatter plots to reason about failure modes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from . import dataset_analysis
from . import datasets
from . import evaluation
from . import feature_extraction
from . import harmonic_model


@dataclass
class DescriptorStats:
    """
    Lightweight descriptors that correlate with reconstruction quality.
    """

    duration_sec: float
    mean_f0_hz: float
    f0_std_hz: float
    voiced_ratio: float
    zcr_mean: float
    spectral_centroid_hz: float


@dataclass
class DiagnosticRecord:
    """
    Container with per-file metrics and descriptors.
    """

    path: Path
    label: str
    metrics: evaluation.ReconstructionMetrics
    descriptors: DescriptorStats


def _compute_descriptors(
    waveform: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    hop_length: int,
) -> DescriptorStats:
    voiced = ~np.isnan(f0_hz)
    duration_sec = float(len(waveform) / sr)
    mean_f0 = float(np.nanmean(f0_hz)) if np.any(voiced) else float("nan")
    std_f0 = float(np.nanstd(f0_hz)) if np.any(voiced) else float("nan")
    voiced_ratio = float(np.mean(voiced))

    zcr = librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=hop_length * 2,
        hop_length=hop_length,
    )
    zcr_mean = float(np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        hop_length=hop_length,
    )
    spectral_centroid = float(np.mean(centroid))

    return DescriptorStats(
        duration_sec=duration_sec,
        mean_f0_hz=mean_f0,
        f0_std_hz=std_f0,
        voiced_ratio=voiced_ratio,
        zcr_mean=zcr_mean,
        spectral_centroid_hz=spectral_centroid,
    )


def analyze_file(
    path: Path,
    sample_rate: int = 22050,
    n_partials: int = 12,
    hop_length: int = 512,
    label: str | None = None,
) -> DiagnosticRecord:
    """
    Runs harmonic reconstruction on a single file and returns metrics/descriptors.
    """

    waveform, sr = feature_extraction.load_mono_audio(str(path), sample_rate=sample_rate)
    features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
    model = harmonic_model.fit_and_resynthesize(
        waveform,
        sample_rate=sr,
        n_partials=n_partials,
        hop_length=hop_length,
        f0_hz=features.f0_hz,
    )
    metrics = evaluation.compute_all_metrics(
        waveform,
        model.reconstruction,
        features.f0_hz,
        features.f0_hz,
        sr,
    )
    descriptors = _compute_descriptors(waveform, sr, features.f0_hz, hop_length=hop_length)
    return DiagnosticRecord(
        path=path,
        label=label or path.parent.name,
        metrics=metrics,
        descriptors=descriptors,
    )


def analyze_dataset(
    dataset_path: Path,
    n_partials: int = 12,
    sample_rate: int = 22050,
    hop_length: int = 512,
    limit: Optional[int] = None,
    label_resolver: dataset_analysis.LabelResolver | None = None,
    metadata: Mapping[str, str] | None = None,
) -> List[DiagnosticRecord]:
    """
    Runs diagnostics for every audio file under dataset_path (optionally limited).
    """

    label_resolver = label_resolver or dataset_analysis.default_label_resolver
    audio_files = datasets.list_audio_files(dataset_path)
    if limit:
        audio_files = audio_files[:limit]

    records: List[DiagnosticRecord] = []
    for audio_path in audio_files:
        label = label_resolver(audio_path, metadata)
        record = analyze_file(
            audio_path,
            sample_rate=sample_rate,
            n_partials=n_partials,
            hop_length=hop_length,
            label=label,
        )
        records.append(record)
    return records


def _scatter_metric(
    records: Sequence[DiagnosticRecord],
    metric_attr: str,
    descriptor_attr: str,
    ax,
    descriptor_label: str,
):
    metric_values = np.array([getattr(record.metrics, metric_attr) for record in records], dtype=float)
    descriptor_values = np.array(
        [getattr(record.descriptors, descriptor_attr) for record in records],
        dtype=float,
    )
    mask = np.isfinite(metric_values) & np.isfinite(descriptor_values)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return

    x = descriptor_values[mask]
    y = metric_values[mask]
    ax.scatter(x, y, alpha=0.7, edgecolor="none")

    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), num=100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1)
        corr = np.corrcoef(x, y)[0, 1]
        title = f"{descriptor_label} vs {metric_attr}\nPearson r = {corr:.2f}"
    else:
        title = f"{descriptor_label} vs {metric_attr}"

    ax.set_xlabel(descriptor_label)
    ax.set_ylabel(metric_attr)
    ax.set_title(title)


def plot_diagnostic_correlations(
    records: Sequence[DiagnosticRecord],
    metric_attr: str,
    output_path: Path,
):
    """
    Creates scatter plots comparing a metric to several descriptors.
    """

    if not records:
        return

    descriptor_specs = [
        ("mean_f0_hz", "Mean F0 (Hz)"),
        ("f0_std_hz", "F0 Std (Hz)"),
        ("voiced_ratio", "Voiced ratio"),
        ("duration_sec", "Duration (s)"),
        ("zcr_mean", "Zero-crossing rate"),
        ("spectral_centroid_hz", "Spectral centroid (Hz)"),
    ]

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (attr, label) in zip(axes, descriptor_specs):
        _scatter_metric(records, metric_attr, attr, ax, label)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_record(record: DiagnosticRecord, metric_attr: str) -> str:
    metric_value = getattr(record.metrics, metric_attr)
    desc = record.descriptors
    return (
        f"{record.path.name} (label={record.label}, {metric_attr}={metric_value:.3f}, "
        f"mean_f0={desc.mean_f0_hz:.1f}Hz, voiced={desc.voiced_ratio:.2f}, "
        f"duration={desc.duration_sec:.2f}s)"
    )


def summarize_records(
    records: Sequence[DiagnosticRecord],
    metric_attr: str,
    top_k: int = 5,
) -> str:
    """
    Returns a formatted summary including best/worst examples.
    """

    if not records:
        return "No diagnostic records generated."

    sorted_records = sorted(records, key=lambda item: getattr(item.metrics, metric_attr))
    best = sorted_records[:top_k]
    worst = list(reversed(sorted_records))[:top_k]

    lines = [
        f"Total files analyzed: {len(records)}",
        f"Metric: {metric_attr}",
        f"Mean {metric_attr}: {np.mean([getattr(r.metrics, metric_attr) for r in records]):.4f}",
        f"Median {metric_attr}: {np.median([getattr(r.metrics, metric_attr) for r in records]):.4f}",
        "",
        f"Top {len(best)} best (lower {metric_attr}):",
    ]
    lines.extend(f"  - {_format_record(record, metric_attr)}" for record in best)
    lines.append("")
    lines.append(f"Worst {len(worst)} (higher {metric_attr}):")
    lines.extend(f"  - {_format_record(record, metric_attr)}" for record in worst)
    return "\n".join(lines)


def save_records_to_json(records: Sequence[DiagnosticRecord], path: Path):
    """
    Persists diagnostic data to JSON for further offline analysis.
    """

    payload = [
        {
            "path": str(record.path),
            "label": record.label,
            "metrics": asdict(record.metrics),
            "descriptors": asdict(record.descriptors),
        }
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plot_example_pair(
    path: Path,
    output_path: Path,
    sample_rate: int,
    n_partials: int,
    hop_length: int,
):
    waveform, sr = feature_extraction.load_mono_audio(str(path), sample_rate=sample_rate)
    features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
    model = harmonic_model.fit_and_resynthesize(
        waveform,
        sample_rate=sr,
        n_partials=n_partials,
        hop_length=hop_length,
        f0_hz=features.f0_hz,
    )

    def _spec(ax, signal, title: str):
        spec = np.abs(librosa.stft(signal, n_fft=2048, hop_length=hop_length))
        img = librosa.display.specshow(
            librosa.amplitude_to_db(spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(title)
        return img

    times = np.arange(len(waveform)) / sr
    recon_times = np.arange(len(model.reconstruction)) / sr

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(times, waveform, color="steelblue")
    axes[0, 0].set_title("Original waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[1, 0].plot(recon_times, model.reconstruction, color="darkorange")
    axes[1, 0].set_title("Reconstruction waveform")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")

    img1 = _spec(axes[0, 1], waveform, "Original spectrogram")
    img2 = _spec(axes[1, 1], model.reconstruction, "Reconstruction spectrogram")
    fig.colorbar(img1, ax=axes[0, 1], format="%+2.0f dB")
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

    fig.suptitle(path.name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_example_figures(
    records: Sequence[DiagnosticRecord],
    metric_attr: str,
    output_dir: Path,
    sample_rate: int,
    n_partials: int,
    hop_length: int,
    count: int = 3,
):
    """
    Saves waveform/spectrogram comparisons for the best and worst examples.
    """

    if not records:
        return

    sorted_records = sorted(records, key=lambda item: getattr(item.metrics, metric_attr))
    best = sorted_records[:count]
    worst = list(reversed(sorted_records))[:count]

    for tag, subset in (("best", best), ("worst", worst)):
        for idx, record in enumerate(subset, start=1):
            out_path = output_dir / f"{tag}_{idx}_{record.path.stem}.png"
            _plot_example_pair(
                record.path,
                out_path,
                sample_rate=sample_rate,
                n_partials=n_partials,
                hop_length=hop_length,
            )


def run_cli(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Diagnostic analysis for harmonic reconstructions.")
    parser.add_argument("dataset", type=str, help="Path to dataset root containing audio files.")
    parser.add_argument("--partials", type=int, default=12, help="Number of partials for reconstruction.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Resample audio to this rate.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT/F0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files.")
    parser.add_argument("--metric", type=str, default="log_spectral_distance", help="Metric used for ranking.")
    parser.add_argument("--output-dir", type=str, default="diagnostic_output", help="Directory for plots/data.")
    parser.add_argument("--examples", type=int, default=3, help="How many best/worst examples to export.")
    parsed = parser.parse_args(args=args)

    dataset_path = Path(parsed.dataset)
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = dataset_analysis.load_nsynth_metadata(dataset_path)
    records = analyze_dataset(
        dataset_path=dataset_path,
        n_partials=parsed.partials,
        sample_rate=parsed.sample_rate,
        hop_length=parsed.hop_length,
        limit=parsed.limit,
        metadata=metadata,
    )

    summary_text = summarize_records(records, metric_attr=parsed.metric, top_k=parsed.examples)
    print(summary_text)
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    save_records_to_json(records, output_dir / "diagnostic_records.json")
    plot_diagnostic_correlations(records, parsed.metric, output_dir / "metric_correlations.png")
    export_example_figures(
        records,
        metric_attr=parsed.metric,
        output_dir=output_dir,
        sample_rate=parsed.sample_rate,
        n_partials=parsed.partials,
        hop_length=parsed.hop_length,
        count=parsed.examples,
    )


if __name__ == "__main__":
    run_cli()
