"""
dataset_analysis
================

Batch reconstruction utilities for full datasets. This module reconstructs every audio
file under a root directory, computes quantitative metrics, aggregates them by label,
and derives label-level partial embeddings for visualization (PCA).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from . import datasets
from . import evaluation
from . import feature_extraction
from . import harmonic_model
from . import visualization


LabelResolver = Callable[[Path, Mapping[str, str] | None], str]


TINYSOL_INSTRUMENTS = {
    "Brass": ["Bass_Tuba", "Horn", "Trombone", "Trumpet_C"],
    "Keyboards": ["Accordion"],
    "Strings": ["Contrabass", "Viola", "Violin", "Violoncello"],
    "Winds": ["Bassoon", "Clarinet_Bb", "Flute", "Oboe", "Sax_Alto"],
}


def _normalize_name_for_lookup(value: str) -> str:
    """Normalizes TinySOL tokens so folder/file variations can be matched."""

    return value.replace("_", "").replace("-", "").lower()


_TINYSOL_INSTRUMENT_LOOKUP = {
    _normalize_name_for_lookup(instrument): instrument
    for instruments in TINYSOL_INSTRUMENTS.values()
    for instrument in instruments
}


def _looks_like_tinysol_path(path: Path) -> bool:
    """Returns True if the path is inside a TinySOL dataset tree."""

    return any(part.lower() == "tinysol" for part in path.parts)


def resolve_tinysol_instrument(path: Path) -> str | None:
    """Attempts to recover the TinySOL instrument name from a file path."""

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
class FileAnalysis:
    """
    Container with per-file reconstruction artifacts.
    """

    path: Path
    label: str
    metrics: evaluation.ReconstructionMetrics
    partial_vector: np.ndarray
    partials_hz_amp: List[Tuple[float, float]]
    instrument: str | None = None
    pitch: str | None = None


def partials_to_vector(partials: Sequence[Tuple[float, float]], n_partials: int) -> np.ndarray:
    """
    Converts a list of (freq, amp) partials into a fixed-length vector for PCA.
    """

    vector = np.zeros((n_partials, 2), dtype=np.float32)
    for idx, (freq, amp) in enumerate(partials[:n_partials]):
        vector[idx, 0] = freq
        vector[idx, 1] = amp
    return vector.flatten()


def load_nsynth_metadata(dataset_root: Path) -> Dict[str, str]:
    """
    Loads NSynth metadata mapping audio stems to instrument family strings if available.
    """

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
    """
    Resolves a label for an audio path using metadata if provided, otherwise falling
    back to the parent directory name.
    """

    if metadata and path.stem in metadata:
        return metadata[path.stem]
    tinysol = resolve_tinysol_instrument(path)
    if tinysol:
        return tinysol
    return path.parent.name


def analyze_dataset(
    dataset_path: Path,
    n_partials: int = 12,
    sample_rate: int = 22050,
    hop_length: int = 512,
    label_resolver: LabelResolver | None = None,
    metadata: Mapping[str, str] | None = None,
    limit: Optional[int] = None,
    save_reconstructions: bool = False,
    output_dir: Path | None = None,
    detail_resolver: Callable[[Path, Mapping[str, str] | None], Tuple[str | None, str | None]] | None = None,
) -> List[FileAnalysis]:
    """
    Reconstructs every WAV/FLAC/OGG/MP3 file under `dataset_path` and collects metrics.
    """

    audio_files = datasets.list_audio_files(dataset_path)
    if limit:
        audio_files = audio_files[:limit]

    label_resolver = label_resolver or default_label_resolver
    results: List[FileAnalysis] = []

    if save_reconstructions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        waveform, sr = feature_extraction.load_mono_audio(str(audio_path), sample_rate=sample_rate)
        features = feature_extraction.extract_features(waveform, sr, hop_length=hop_length)
        model = harmonic_model.fit_and_resynthesize(
            waveform, sample_rate=sr, n_partials=n_partials, hop_length=hop_length
        )

        metrics = evaluation.compute_all_metrics(waveform, model.reconstruction, features.f0_hz, features.f0_hz, sr)
        label = label_resolver(audio_path, metadata)
        partial_vector = partials_to_vector(model.partials_hz_amp, n_partials=n_partials)

        instrument = None
        pitch = None
        if detail_resolver:
            instrument, pitch = detail_resolver(audio_path, metadata)
        if instrument is None:
            instrument = label

        if save_reconstructions and output_dir:
            recon_path = output_dir / f"{audio_path.stem}_reconstructed.wav"
            sf.write(recon_path, model.reconstruction, sr)

        results.append(
            FileAnalysis(
                path=audio_path,
                label=label,
                metrics=metrics,
                partial_vector=partial_vector,
                partials_hz_amp=model.partials_hz_amp,
                instrument=instrument,
                pitch=pitch,
            )
        )

    return results


def summarize_by_label(results: Iterable[FileAnalysis]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes mean and std for each metric grouped by label.
    """

    buckets: Dict[str, Dict[str, List[float]]] = {}
    for item in results:
        metrics_dict = item.metrics.__dict__
        label_bucket = buckets.setdefault(item.label, {})
        for name, value in metrics_dict.items():
            label_bucket.setdefault(name, []).append(float(value))

    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, metrics_map in buckets.items():
        summary[label] = {
            metric: (float(np.mean(values)), float(np.std(values))) for metric, values in metrics_map.items()
        }
    return summary


def summarize_by_instrument(results: Iterable[FileAnalysis]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes mean and std for each metric grouped by instrument (falls back to label).
    """

    buckets: Dict[str, Dict[str, List[float]]] = {}
    for item in results:
        label = item.instrument or item.label
        metrics_dict = item.metrics.__dict__
        label_bucket = buckets.setdefault(label, {})
        for name, value in metrics_dict.items():
            label_bucket.setdefault(name, []).append(float(value))

    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, metrics_map in buckets.items():
        summary[label] = {
            metric: (float(np.mean(values)), float(np.std(values))) for metric, values in metrics_map.items()
        }
    return summary


def summarize_overall(results: Iterable[FileAnalysis]) -> Dict[str, Tuple[float, float]]:
    """
    Computes overall mean/std for each metric across all files.
    """

    accum: Dict[str, List[float]] = {}
    for item in results:
        for name, value in item.metrics.__dict__.items():
            accum.setdefault(name, []).append(float(value))
    return {metric: (float(np.mean(values)), float(np.std(values))) for metric, values in accum.items()}


def format_summary(summary: Mapping[str, Mapping[str, Tuple[float, float]]]) -> str:
    """
    Renders a human-readable summary table (mean ± std per label).
    """

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


def label_partial_means(results: Iterable[FileAnalysis], n_partials: int) -> Dict[str, np.ndarray]:
    """
    Returns mean partial vectors for each label.
    """

    accum: Dict[str, List[np.ndarray]] = {}
    for item in results:
        vector = item.partial_vector
        target_size = n_partials * 2
        if vector.size != target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            copy_len = min(target_size, vector.size)
            padded[:copy_len] = vector[:copy_len]
            vector = padded
        accum.setdefault(item.label, []).append(vector)

    return {label: np.vstack(vectors).mean(axis=0) for label, vectors in accum.items() if vectors}


def save_summary_to_json(
    summary: Mapping[str, Mapping[str, Tuple[float, float]]],
    path: Path,
    overall: Mapping[str, Tuple[float, float]] | None = None,
):
    """
    Persists summary metrics to JSON.
    """

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


def run_cli(args: Optional[Sequence[str]] = None):
    """
    Entry point for dataset-wide reconstruction and reporting.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct and score an entire dataset.")
    parser.add_argument("dataset", type=str, help="Path to the dataset root containing audio files.")
    parser.add_argument("--partials", type=int, default=12, help="Number of partials to keep.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Resample audio to this rate.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT/F0.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of files (useful for smoke tests).",
    )
    parser.add_argument(
        "--save-recon",
        action="store_true",
        help="If set, reconstructed WAVs are saved alongside metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store reconstructions, metrics.json, and PCA plot.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional path to a JSON metadata file (e.g., NSynth examples.json).",
    )
    parsed = parser.parse_args(args=args)

    dataset_path = Path(parsed.dataset)
    output_dir = Path(parsed.output_dir) if parsed.output_dir else None

    label_resolver: LabelResolver | None = None
    detail_resolver: Callable[[Path, Mapping[str, str] | None], Tuple[str | None, str | None]] | None = None
    metadata: Mapping[str, str] | None = None

    probe_files = datasets.list_audio_files(dataset_path)
    is_tinysol = any(datasets.parse_tinysol_sample(path) for path in probe_files[: min(len(probe_files), 50)])
    if is_tinysol:
        label_resolver = datasets.tinysol_label_resolver
        detail_resolver = datasets.tinysol_detail_resolver
    else:
        metadata = load_nsynth_metadata(dataset_path)
        if parsed.metadata:
            custom_meta_path = Path(parsed.metadata)
            if custom_meta_path.exists():
                with open(custom_meta_path, "r", encoding="utf-8") as handle:
                    try:
                        user_meta = json.load(handle)
                    except json.JSONDecodeError:
                        user_meta = {}
                if isinstance(user_meta, dict):
                    metadata.update({k: str(v) for k, v in user_meta.items()})

    results = analyze_dataset(
        dataset_path=dataset_path,
        n_partials=parsed.partials,
        sample_rate=parsed.sample_rate,
        hop_length=parsed.hop_length,
        metadata=metadata,
        limit=parsed.limit,
        save_reconstructions=parsed.save_recon and output_dir is not None,
        output_dir=output_dir,
        label_resolver=label_resolver,
        detail_resolver=detail_resolver,
    )

    overall_summary = summarize_overall(results)
    instrument_summary = summarize_by_instrument(results)
    print(format_overall(overall_summary))
    print("\nBy instrument:")
    print(format_summary(instrument_summary))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_summary_to_json(instrument_summary, output_dir / "metrics_summary.json", overall=overall_summary)
        label_vectors = label_partial_means(results, n_partials=parsed.partials)
        if label_vectors:
            fig, _ = visualization.plot_label_partial_pca(label_vectors)
            fig.savefig(output_dir / "label_partial_pca.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    run_cli()
