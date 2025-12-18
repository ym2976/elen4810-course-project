"""
data_analysis
=============

Unified dataset reconstruction and diagnostics. This module reconstructs a dataset,
limits samples per instrument with deterministic sampling, computes metrics,
exports best/worst examples, and correlates reconstruction quality with audio
descriptors.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import stats

from . import datasets
from . import evaluation
from . import feature_extraction
from . import harmonic_model


LabelResolver = Callable[[Path, Mapping[str, str] | None], str]
DetailResolver = Callable[[Path, Mapping[str, str] | None], Tuple[str | None, str | None]]


TINYSOL_INSTRUMENTS = {
    "Brass": ["Bass_Tuba", "Horn", "Trombone", "Trumpet_C"],
    "Keyboards": ["Accordion"],
    "Strings": ["Contrabass", "Viola", "Violin", "Violoncello"],
    "Winds": ["Bassoon", "Clarinet_Bb", "Flute", "Oboe", "Sax_Alto"],
}


def _normalize_name_for_lookup(value: str) -> str:
    return value.replace("_", "").replace("-", "").lower()


_TINYSOL_INSTRUMENT_LOOKUP = {
    _normalize_name_for_lookup(instrument): instrument
    for instruments in TINYSOL_INSTRUMENTS.values()
    for instrument in instruments
}


def _looks_like_tinysol_path(path: Path) -> bool:
    return any(part.lower() == "tinysol" for part in path.parts)


def resolve_tinysol_instrument(path: Path) -> str | None:
    if not _looks_like_tinysol_path(path):
        return None

    for part in reversed(path.parts):
        key = _normalize_name_for_lookup(part)
        if key in _TINYSOL_INSTRUMENT_LOOKUP:
            return _TINYSOL_INSTRUMENT_LOOKUP[key]

    stem_prefix = path.stem.split("-")[0]
    key = _normalize_name_for_lookup(stem_prefix)
    return _TINYSOL_INSTRUMENT_LOOKUP.get(key)


@dataclass
class SampleAnalysis:
    """
    Container with per-file reconstruction artifacts, descriptors, and metadata.
    """

    path: Path
    label: str
    instrument: str
    pitch: str | None
    metrics: evaluation.ReconstructionMetrics
    partial_vector: np.ndarray
    partials_hz_amp: List[Tuple[float, float]]
    waveform: np.ndarray
    reconstruction: np.ndarray
    f0_hz: np.ndarray
    recon_f0_hz: np.ndarray
    sample_rate: int
    descriptors: Dict[str, float]


def partials_to_vector(partials: Sequence[Tuple[float, float]], n_partials: int) -> np.ndarray:
    vector = np.zeros((n_partials, 2), dtype=np.float32)
    for idx, (freq, amp) in enumerate(partials[:n_partials]):
        vector[idx, 0] = freq
        vector[idx, 1] = amp
    return vector.flatten()


def load_nsynth_metadata(dataset_root: Path) -> Dict[str, str]:
    candidates = [
        dataset_root / "examples.json",
        dataset_root / "metadata.json",
        dataset_root / "nsynth-test" / "examples.json",
        dataset_root / "nsynth-test" / "metadata.json",
    ] + list(dataset_root.rglob("examples.json"))

    metadata: Dict[str, str] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if isinstance(payload, dict):
            for key, entry in payload.items():
                if not isinstance(entry, dict):
                    continue
                label = (
                    entry.get("instrument_family_str")
                    or entry.get("instrument_str")
                    or str(entry.get("instrument_family", "")).strip()
                )
                if label:
                    metadata[Path(key).stem] = label
        if metadata:
            break
    return metadata


def default_label_resolver(path: Path, metadata: Mapping[str, str] | None = None) -> str:
    if metadata and path.stem in metadata:
        return metadata[path.stem]
    tinysol = resolve_tinysol_instrument(path)
    if tinysol:
        return tinysol
    return path.parent.name


def sample_features(
    waveform: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    hop_length: int = 512,
    n_fft: int = 2048,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Extracts descriptors that help explain reconstruction difficulty.
    """

    x = waveform.astype(float)

    voiced = ~np.isnan(f0_hz)
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else np.nan

    if np.sum(voiced) >= 3:
        f0v = f0_hz[voiced]
        logf0 = np.log(f0v + eps)
        dlogf0 = np.diff(logf0)
        f0_jitter = float(np.std(dlogf0))
        ratios = f0v[1:] / (f0v[:-1] + eps)
        octave_jump_rate = float(np.mean((ratios > 1.8) | (ratios < 0.55)))
        f0_std = float(np.std(f0v))
        f0_med = float(np.median(f0v))
    else:
        f0_jitter = np.nan
        octave_jump_rate = np.nan
        f0_std = np.nan
        f0_med = np.nan

    if len(voiced) > 0:
        changes = np.diff(voiced.astype(int))
        n_runs = int(np.sum(changes == 1) + (1 if voiced[0] else 0))
        voiced_fragmentation = float(n_runs / (np.sum(voiced) + eps))
    else:
        voiced_fragmentation = np.nan

    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) + eps
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0)) if S.shape[1] > 1 else np.array([])
    onset = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(x, hop_length=hop_length)[0]

    H, P = librosa.decompose.hpss(S**2)
    harmonic_energy_ratio = float(np.sum(H) / (np.sum(H) + np.sum(P) + eps))

    rms = librosa.feature.rms(y=x, frame_length=n_fft, hop_length=hop_length)[0]
    rms_dr = float(np.percentile(rms, 95) - np.percentile(rms, 5))

    return {
        "voiced_ratio": voiced_ratio,
        "voiced_fragmentation": voiced_fragmentation,
        "f0_jitter": float(f0_jitter),
        "octave_jump_rate": float(octave_jump_rate),
        "f0_std": float(f0_std),
        "f0_med": float(f0_med),
        "harmonic_energy_ratio": harmonic_energy_ratio,
        "flatness_mean": float(np.mean(flatness)),
        "flatness_p90": float(np.percentile(flatness, 90)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flux_p90": float(np.percentile(flux, 90)) if len(flux) else np.nan,
        "onset_p90": float(np.percentile(onset, 90)) if len(onset) else np.nan,
        "rms_dynamic_range": rms_dr,
    }


def select_audio_files(
    dataset_path: Path,
    label_resolver: LabelResolver,
    metadata: Mapping[str, str] | None,
    per_instrument_limit: Optional[int],
    instrument_limit_map: Mapping[str, int],
    seed: int,
) -> List[Tuple[Path, str]]:
    audio_files = datasets.list_audio_files(dataset_path)
    grouped: Dict[str, List[Path]] = {}
    for audio_path in audio_files:
        label = label_resolver(audio_path, metadata)
        grouped.setdefault(label, []).append(audio_path)

    rng = random.Random(seed)
    selected: List[Tuple[Path, str]] = []
    for instrument, files in grouped.items():
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)
        limit = instrument_limit_map.get(instrument, per_instrument_limit)
        if limit is None or limit <= 0:
            limit = len(files_sorted)
        selected.extend((path, instrument) for path in files_sorted[: min(limit, len(files_sorted))])
    rng.shuffle(selected)
    return selected


def analyze_dataset(
    dataset_path: Path,
    n_partials: int = 12,
    sample_rate: int = 22050,
    hop_length: int = 512,
    per_instrument_limit: Optional[int] = None,
    instrument_limit_map: Mapping[str, int] | None = None,
    seed: int = 123,
    metadata: Mapping[str, str] | None = None,
    label_resolver: LabelResolver | None = None,
    detail_resolver: DetailResolver | None = None,
    limit: Optional[int] = None,
    save_reconstructions: bool = False,
    output_dir: Path | None = None,
) -> List[SampleAnalysis]:
    label_resolver = label_resolver or default_label_resolver
    instrument_limit_map = instrument_limit_map or {}

    selected = select_audio_files(
        dataset_path,
        label_resolver=label_resolver,
        metadata=metadata,
        per_instrument_limit=per_instrument_limit,
        instrument_limit_map=instrument_limit_map,
        seed=seed,
    )
    if limit:
        selected = selected[:limit]

    results: List[SampleAnalysis] = []
    if save_reconstructions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")

    for audio_path, instrument_guess in selected:
        waveform, sr = feature_extraction.load_mono_audio(str(audio_path), sample_rate=sample_rate)
        features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
        model = harmonic_model.fit_and_resynthesize(
            waveform,
            sample_rate=sr,
            n_partials=n_partials,
            hop_length=hop_length,
            f0_hz=features.f0_hz,
            residual_mix=0.15,
        )

        metrics = evaluation.compute_all_metrics(
            waveform,
            model.reconstruction,
            features.f0_hz,
            features.f0_hz,
            sr,
        )
        partial_vector = partials_to_vector(model.partials_hz_amp, n_partials=n_partials)
        descriptors = sample_features(waveform, sr, features.f0_hz, hop_length=hop_length)

        recon_f0, recon_voiced, _ = librosa.pyin(
            model.reconstruction,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
        recon_f0 = np.where(recon_voiced, recon_f0, np.nan)
        recon_f0 = librosa.util.fix_length(recon_f0, size=len(features.f0_hz))

        instrument = instrument_guess
        pitch = None
        if detail_resolver:
            inst_detail, pitch_detail = detail_resolver(audio_path, metadata)
            if inst_detail:
                instrument = inst_detail
            pitch = pitch_detail
        label = instrument

        if save_reconstructions and output_dir:
            recon_path = output_dir / f"{audio_path.stem}_reconstructed.wav"
            sf.write(recon_path, model.reconstruction, sr)

        results.append(
            SampleAnalysis(
                path=audio_path,
                label=label,
                instrument=instrument,
                pitch=pitch,
                metrics=metrics,
                partial_vector=partial_vector,
                partials_hz_amp=model.partials_hz_amp,
                waveform=waveform,
                reconstruction=model.reconstruction,
                f0_hz=features.f0_hz,
                recon_f0_hz=recon_f0,
                sample_rate=sr,
                descriptors=descriptors,
            )
        )

    return results


def summarize_overall(results: Iterable[SampleAnalysis]) -> Dict[str, Tuple[float, float]]:
    accum: Dict[str, List[float]] = {}
    for item in results:
        for name, value in item.metrics.__dict__.items():
            accum.setdefault(name, []).append(float(value))
    return {metric: (float(np.mean(values)), float(np.std(values))) for metric, values in accum.items()}


def summarize_by_instrument(results: Iterable[SampleAnalysis]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for item in results:
        metrics_dict = item.metrics.__dict__
        label_bucket = buckets.setdefault(item.instrument, {})
        for name, value in metrics_dict.items():
            label_bucket.setdefault(name, []).append(float(value))

    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, metrics_map in buckets.items():
        summary[label] = {
            metric: (float(np.mean(values)), float(np.std(values))) for metric, values in metrics_map.items()
        }
    return summary


def format_summary(summary: Mapping[str, Mapping[str, Tuple[float, float]]]) -> str:
    lines: List[str] = []
    for label in sorted(summary.keys()):
        lines.append(f"[{label}]")
        for metric, (mean, std) in summary[label].items():
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
    return "\n".join(lines)


def format_overall(summary: Mapping[str, Tuple[float, float]], title: str = "Overall") -> str:
    lines: List[str] = [f"[{title}]"]
    for metric, (mean, std) in summary.items():
        lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
    return "\n".join(lines)


def rank_records(records: Sequence[SampleAnalysis], metric_attr: str) -> List[SampleAnalysis]:
    return sorted(records, key=lambda item: getattr(item.metrics, metric_attr))


def select_best_worst(
    records: Sequence[SampleAnalysis],
    metric_attr: str,
    best_k: int,
    worst_k: int,
) -> Tuple[List[SampleAnalysis], List[SampleAnalysis]]:
    ranked = rank_records(records, metric_attr)
    best = ranked[: min(best_k, len(ranked))]
    worst = list(reversed(ranked))[: min(worst_k, len(ranked))]
    return best, worst


def select_per_instrument_extremes(
    records: Sequence[SampleAnalysis],
    metric_attr: str,
    best_k: int,
    worst_k: int,
) -> Dict[str, Dict[str, List[SampleAnalysis]]]:
    grouped: Dict[str, List[SampleAnalysis]] = {}
    for record in records:
        grouped.setdefault(record.instrument, []).append(record)

    per_inst: Dict[str, Dict[str, List[SampleAnalysis]]] = {}
    for instrument, items in grouped.items():
        ranked = rank_records(items, metric_attr)
        per_inst[instrument] = {
            "best": ranked[: min(best_k, len(ranked))],
            "worst": list(reversed(ranked))[: min(worst_k, len(ranked))],
        }
    return per_inst


def _plot_wave_and_spec(ax_wave, ax_spec, signal: np.ndarray, sr: int, title: str, hop_length: int):
    times = np.arange(len(signal)) / sr
    ax_wave.plot(times, signal, color="steelblue")
    ax_wave.set_title(f"{title} waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")

    spec = np.abs(librosa.stft(signal, n_fft=2048, hop_length=hop_length))
    img = librosa.display.specshow(
        librosa.amplitude_to_db(spec, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        cmap="magma",
        ax=ax_spec,
    )
    ax_spec.set_title(f"{title} spectrogram")
    return img


def plot_sample_detail(record: SampleAnalysis, output_path: Path, hop_length: int):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ax_wave_orig, ax_spec_orig = axes[0]
    img1 = _plot_wave_and_spec(ax_wave_orig, ax_spec_orig, record.waveform, record.sample_rate, "Original", hop_length)
    ax_wave_rec, ax_spec_rec = axes[1]
    img2 = _plot_wave_and_spec(ax_wave_rec, ax_spec_rec, record.reconstruction, record.sample_rate, "Reconstruction", hop_length)

    fig.colorbar(img1, ax=ax_spec_orig, format="%+2.0f dB")
    fig.colorbar(img2, ax=ax_spec_rec, format="%+2.0f dB")

    time_orig = librosa.times_like(record.f0_hz, sr=record.sample_rate, hop_length=hop_length)
    time_rec = librosa.times_like(record.recon_f0_hz, sr=record.sample_rate, hop_length=hop_length)
    axes[2, 0].plot(time_orig, record.f0_hz, color="green")
    axes[2, 0].set_title("Original pyin f0")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Hz")

    axes[2, 1].plot(time_rec, record.recon_f0_hz, color="darkorange")
    axes[2, 1].set_title("Reconstruction pyin f0")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Hz")

    fig.suptitle(f"{record.path.name} | instrument={record.instrument} | LSD={record.metrics.log_spectral_distance:.3f}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_examples(
    records: Sequence[SampleAnalysis],
    selections: Sequence[SampleAnalysis],
    output_dir: Path,
    tag: str,
    hop_length: int,
):
    for idx, record in enumerate(selections, start=1):
        out_path = output_dir / f"{tag}_{idx}_{record.path.stem}.png"
        plot_sample_detail(record, out_path, hop_length=hop_length)


def compute_feature_correlations(
    records: Sequence[SampleAnalysis],
    metric_attr: str,
) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    metric_values = np.array([getattr(record.metrics, metric_attr) for record in records], dtype=float)
    feature_names = sorted(records[0].descriptors.keys())
    correlations: Dict[str, Dict[str, float]] = {}

    for feature in feature_names:
        feat_values = np.array([record.descriptors.get(feature) for record in records], dtype=float)
        mask = np.isfinite(metric_values) & np.isfinite(feat_values)
        if mask.sum() < 3:
            correlations[feature] = {"spearman_r": np.nan, "p_value": np.nan}
            continue
        r, p = stats.spearmanr(feat_values[mask], metric_values[mask])
        correlations[feature] = {"spearman_r": float(r), "p_value": float(p)}
    return correlations


def plot_feature_correlations(
    records: Sequence[SampleAnalysis],
    metric_attr: str,
    output_path: Path,
):
    if not records:
        return

    features = sorted(records[0].descriptors.keys())
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten()
    metric_values = np.array([getattr(r.metrics, metric_attr) for r in records], dtype=float)

    for ax, feature in zip(axes, features):
        feat_values = np.array([r.descriptors.get(feature) for r in records], dtype=float)
        mask = np.isfinite(metric_values) & np.isfinite(feat_values)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        x = feat_values[mask]
        y = metric_values[mask]
        ax.scatter(x, y, alpha=0.6, edgecolor="none")
        r, p = stats.spearmanr(x, y)
        ax.set_title(f"{feature}\nr={r:.2f}, p={p:.3g}")
        ax.set_xlabel(feature)
        ax.set_ylabel(metric_attr)

    for ax in axes[len(features) :]:
        ax.set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_to_json(
    summary: Mapping[str, Mapping[str, Tuple[float, float]]],
    path: Path,
    overall: Mapping[str, Tuple[float, float]] | None = None,
):
    payload: MutableMapping[str, MutableMapping[str, Dict[str, float]]] = {}
    if overall:
        payload["overall"] = {metric: {"mean": float(m), "std": float(s)} for metric, (m, s) in overall.items()}

    payload["by_label"] = {}
    for label, metrics_map in summary.items():
        payload["by_label"][label] = {}
        for metric, (mean, std) in metrics_map.items():
            payload["by_label"][label][metric] = {"mean": float(mean), "std": float(std)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_records_to_json(records: Sequence[SampleAnalysis], path: Path):
    payload = []
    for record in records:
        payload.append(
            {
                "path": str(record.path),
                "instrument": record.instrument,
                "pitch": record.pitch,
                "metrics": asdict(record.metrics),
                "descriptors": record.descriptors,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_instrument_limit_map(arg: Optional[str]) -> Dict[str, int]:
    if not arg:
        return {}
    candidate = Path(arg)
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(arg)
    if not isinstance(payload, dict):
        raise ValueError("Instrument limit config must be a mapping of instrument -> limit.")
    return {str(k): int(v) for k, v in payload.items()}


def detect_dataset_resolvers(
    dataset_path: Path,
) -> Tuple[LabelResolver | None, DetailResolver | None, Mapping[str, str] | None]:
    probe_files = datasets.list_audio_files(dataset_path)
    is_tinysol = any(datasets.parse_tinysol_sample(path) for path in probe_files[: min(len(probe_files), 50)])
    if is_tinysol:
        return datasets.tinysol_label_resolver, datasets.tinysol_detail_resolver, None
    return None, None, load_nsynth_metadata(dataset_path)


def summarize_records(records: Sequence[SampleAnalysis], metric_attr: str, count: int) -> str:
    best, worst = select_best_worst(records, metric_attr, count, count)
    def _fmt(item: SampleAnalysis) -> str:
        return (
            f"{item.path.name} | {item.instrument} | "
            f"{metric_attr}={getattr(item.metrics, metric_attr):.3f}"
        )

    lines = [
        f"Analyzed files: {len(records)}",
        f"Metric: {metric_attr}",
        "",
        f"Top {len(best)} (lower is better):",
    ]
    lines.extend(f"  - {_fmt(item)}" for item in best)
    lines.append("")
    lines.append(f"Worst {len(worst)}:")
    lines.extend(f"  - {_fmt(item)}" for item in worst)
    return "\n".join(lines)


def run_cli(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Dataset reconstruction with diagnostics.")
    parser.add_argument("dataset", type=str, help="Path to the dataset root containing audio files.")
    parser.add_argument("--partials", type=int, default=12, help="Number of partials to keep.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Resample audio to this rate.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT/F0.")
    parser.add_argument("--per-instrument-limit", type=int, default=None, help="Maximum samples per instrument.")
    parser.add_argument(
        "--instrument-limit-config",
        type=str,
        default=None,
        help="Path or JSON mapping of instrument -> limit (overrides --per-instrument-limit).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed used for sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional overall cap on number of files.")
    parser.add_argument("--metric", type=str, default="log_spectral_distance", help="Metric used for ranking.")
    parser.add_argument("--examples-overall", type=int, default=5, help="Number of best/worst overall examples.")
    parser.add_argument("--examples-per-instrument", type=int, default=3, help="Examples per instrument.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument("--save-recon", action="store_true", help="Save reconstructed WAVs.")
    parser.add_argument("--metadata", type=str, default=None, help="Optional JSON metadata to merge.")
    parsed = parser.parse_args(args=args)

    dataset_path = Path(parsed.dataset)
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit_map = load_instrument_limit_map(parsed.instrument_limit_config)

    label_resolver, detail_resolver, metadata = detect_dataset_resolvers(dataset_path)
    if parsed.metadata:
        custom_meta_path = Path(parsed.metadata)
        if custom_meta_path.exists():
            with open(custom_meta_path, "r", encoding="utf-8") as handle:
                metadata_update = json.load(handle)
            if metadata is None:
                metadata = {}
            if isinstance(metadata_update, dict):
                metadata.update({k: str(v) for k, v in metadata_update.items()})

    records = analyze_dataset(
        dataset_path=dataset_path,
        n_partials=parsed.partials,
        sample_rate=parsed.sample_rate,
        hop_length=parsed.hop_length,
        per_instrument_limit=parsed.per_instrument_limit,
        instrument_limit_map=limit_map,
        seed=parsed.seed,
        metadata=metadata,
        label_resolver=label_resolver,
        detail_resolver=detail_resolver,
        limit=parsed.limit,
        save_reconstructions=parsed.save_recon,
        output_dir=output_dir if parsed.save_recon else None,
    )

    overall_summary = summarize_overall(records)
    instrument_summary = summarize_by_instrument(records)
    print(format_overall(overall_summary))
    print("\nBy instrument:")
    print(format_summary(instrument_summary))

    save_summary_to_json(instrument_summary, output_dir / "metrics_summary.json", overall=overall_summary)
    save_records_to_json(records, output_dir / "analysis_records.json")

    best_overall, worst_overall = select_best_worst(records, parsed.metric, parsed.examples_overall, parsed.examples_overall)
    per_inst = select_per_instrument_extremes(records, parsed.metric, parsed.examples_per_instrument, parsed.examples_per_instrument)

    summary_text = summarize_records(records, parsed.metric, parsed.examples_overall)
    (output_dir / "top_worst_summary.txt").write_text(summary_text, encoding="utf-8")

    examples_dir = output_dir / "examples"
    export_examples(records, best_overall, examples_dir / "overall", "overall_best", parsed.hop_length)
    export_examples(records, worst_overall, examples_dir / "overall", "overall_worst", parsed.hop_length)

    for instrument, payload in per_inst.items():
        inst_dir = examples_dir / instrument
        export_examples(records, payload["best"], inst_dir, "best", parsed.hop_length)
        export_examples(records, payload["worst"], inst_dir, "worst", parsed.hop_length)

    correlations = compute_feature_correlations(records, parsed.metric)
    plot_feature_correlations(records, parsed.metric, output_dir / "feature_correlations.png")
    with open(output_dir / "feature_correlations.json", "w", encoding="utf-8") as handle:
        json.dump(correlations, handle, indent=2)


if __name__ == "__main__":
    run_cli()
