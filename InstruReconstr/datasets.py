"""
datasets
========

Helpers for organizing public music datasets. This module now focuses on preparing a
curated TinySOL subset tailored for reconstruction experiments. The functions manage
download/extraction and surface lists of audio files for downstream analysis.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import requests

from .config_manager import ConfigManager


DATASET_URLS = {
    "tinysol": "https://zenodo.org/records/3685367/files/TinySOL.tar.gz?download=1",
}

TINYSOL_INSTRUMENTS = {"Contrabass","Bass_Tuba" "Flute", "Alto_Saxophone", "Oboe", "Accordion"}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_root() -> Path:
    """
    Returns the root dataset directory, creating it if needed.
    """

    cfg = ConfigManager.get_instance().config
    return _ensure_dir(Path(cfg.PATH_DATASETS))


def download_file(url: str, target_path: Path, chunk_size: int = 65536) -> Path:
    """
    Streams a file from `url` to `target_path` with chunked downloads.
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(target_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file_handle.write(chunk)
    return target_path


def _extract_archive(archive_path: Path, destination: Path):
    """
    Extracts tar.gz or zip archives to destination.
    """

    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tar_handle:
            tar_handle.extractall(destination)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive type for {archive_path}")


def _pick_archive_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    candidate = Path(parsed.path).name or parsed.netloc or "dataset.tar.gz"
    return candidate


def prepare_dataset(
    name: str,
    destination: Path | None = None,
    url_override: str | None = None,
    local_archive: Path | None = None,
) -> Path:
    """
    Downloads and extracts a dataset archive into a dedicated folder.

    Args:
        name: Dataset key; one of DATASET_URLS keys or a custom label with url_override.
        destination: Optional destination folder; defaults to `<PATH_DATASETS>/<name>`.
        url_override: URL to download instead of the default DATASET_URLS entry.
        local_archive: Optional path to a pre-downloaded archive; if provided, download is skipped.

    Returns:
        Path to the extracted dataset directory.
    """

    dest = destination or dataset_root() / name
    archive_dir = dest / "downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)

    urls = []
    if url_override:
        urls = [url_override]
    else:
        mapped = DATASET_URLS.get(name)
        if mapped is None:
            raise ValueError(
                f"No URL provided for dataset '{name}'. Supply url_override to proceed."
            )
        urls = mapped if isinstance(mapped, list) else [mapped]  # type: ignore[list-item]

    if local_archive:
        archive_path = Path(local_archive)
        if not archive_path.exists():
            raise FileNotFoundError(f"local_archive not found: {archive_path}")
    else:
        archive_path = archive_dir / _pick_archive_name(urls[0])
        if not archive_path.exists():
            last_error: Exception | None = None
            for url in urls:
                try:
                    download_file(url, archive_path)
                    last_error = None
                    break
                except Exception as exc:  # broad to capture HTTPError/Connection issues
                    last_error = exc
            if last_error:
                raise RuntimeError(
                    f"Failed to download dataset '{name}'. Tried URLs: {urls}. "
                    "Provide url_override or local_archive pointing to a valid archive."
                ) from last_error

    if "?" in archive_path.name:
        sanitized_name = archive_path.name.split("?")[0] or "dataset.tar.gz"
        sanitized_path = archive_path.with_name(sanitized_name)
        if not sanitized_path.exists():
            archive_path.rename(sanitized_path)
        archive_path = sanitized_path

    extracted_path = dest / "data"
    if not extracted_path.exists():
        _extract_archive(archive_path, extracted_path)

    return extracted_path


def list_audio_files(root: Path, extensions: Iterable[str] | None = None) -> List[Path]:
    """
    Recursively lists audio files under root.
    """

    extensions = extensions or [".wav", ".mp3", ".flac", ".ogg"]
    paths: List[Path] = []
    for extension in extensions:
        paths.extend(root.rglob(f"*{extension}"))
    return sorted(paths)


class TinySolSample:
    """
    Parsed metadata for a single TinySOL WAV file.
    """

    def __init__(
        self,
        path: Path,
        family: str,
        instrument: str,
        pitch: str,
        dyn: str,
        instance: str,
        misc: str,
    ):
        self.path = path
        self.family = family
        self.instrument = instrument
        self.pitch = pitch
        self.dyn = dyn
        self.instance = instance
        self.misc = misc
        self.midi_pitch = pitch_to_midi(pitch)


_PITCH_PATTERN = re.compile(r"^(?P<note>[A-Ga-g])(?P<accidental>[#b]?)(?P<octave>-?\d+)$")
_NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_to_midi(pitch: str) -> int:
    """
    Converts a pitch label like C4 or F#3 into a MIDI note number.
    """

    match = _PITCH_PATTERN.match(pitch)
    if not match:
        raise ValueError(f"Invalid pitch format: {pitch}")

    note = match.group("note").upper()
    accidental = match.group("accidental")
    octave = int(match.group("octave"))
    semitone = _NOTE_TO_SEMITONE[note]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    return (octave + 1) * 12 + semitone


def parse_tinysol_sample(path: Path) -> TinySolSample | None:
    """
    Parses TinySOL path/filename into a TinySolSample. Returns None if the file
    does not match the expected layout.
    """

    try:
        ord_idx = path.parts.index("ordinario")
    except ValueError:
        return None

    if ord_idx < 1 or len(path.parts) < ord_idx + 1:
        return None

    family = path.parts[ord_idx - 2] if ord_idx >= 2 else ""
    instrument_dir = path.parts[ord_idx - 1]
    stem_parts = path.stem.split("-")
    if len(stem_parts) < 5:
        return None

    instrument_name = stem_parts[0]
    articulation = stem_parts[1]
    pitch = stem_parts[2]
    dyn = stem_parts[3]
    instance = stem_parts[4]
    misc = "-".join(stem_parts[5:]) if len(stem_parts) > 5 else ""

    if articulation.lower() != "ord":
        return None
    if instrument_name.lower() != instrument_dir.lower():
        return None

    try:
        return TinySolSample(
            path=path,
            family=family,
            instrument=instrument_name,
            pitch=pitch,
            dyn=dyn,
            instance=instance,
            misc=misc,
        )
    except ValueError:
        return None


def _find_audio_root(extracted_path: Path) -> Path:
    """
    Returns the directory containing WAV files within the extracted TinySOL archive.
    """

    if any(extracted_path.glob("*.wav")):
        return extracted_path

    for candidate in extracted_path.iterdir():
        if candidate.is_dir() and any(candidate.rglob("*.wav")):
            return candidate
    return extracted_path


def _uniform_sample_by_pitch(samples: Sequence[TinySolSample], count: int) -> List[TinySolSample]:
    """
    Picks `count` samples roughly uniformly across the sorted pitch range.
    """

    ordered = sorted(samples, key=lambda item: (item.midi_pitch, item.path.name))
    if len(ordered) < count:
        raise ValueError(f"Requested {count} samples but only found {len(ordered)} candidates.")

    targets = [int(round(x)) for x in list(np.linspace(0, len(ordered) - 1, num=count))]
    used = set()
    for target in targets:
        target = max(0, min(target, len(ordered) - 1))
        if target not in used:
            used.add(target)
            continue
        offset = 1
        chosen = None
        while chosen is None and (target - offset >= 0 or target + offset < len(ordered)):
            for candidate in (target - offset, target + offset):
                if 0 <= candidate < len(ordered) and candidate not in used:
                    chosen = candidate
                    break
            offset += 1
        if chosen is None:
            break
        used.add(chosen)

    for idx in range(len(ordered)):
        if len(used) >= count:
            break
        if idx not in used:
            used.add(idx)

    indices = sorted(list(used))[:count]
    return [ordered[i] for i in indices]


def collect_tinysol_subset(
    root: Path,
    instruments: Iterable[str] | None = None,
    samples_per_instrument: int = 50,
    dyn: str = "mf",
) -> Dict[str, List[TinySolSample]]:
    """
    Filters TinySOL WAV files and selects a balanced subset across pitch for each instrument.
    """

    instruments = set(instruments or TINYSOL_INSTRUMENTS)
    dyn = dyn.lower()
    audio_root = _find_audio_root(root)
    candidates = []
    for wav_path in list_audio_files(audio_root, extensions=[".wav"]):
        parsed = parse_tinysol_sample(wav_path)
        if not parsed:
            continue
        if parsed.instrument not in instruments:
            continue
        if parsed.dyn.lower() != dyn:
            continue
        candidates.append(parsed)

    grouped: Dict[str, List[TinySolSample]] = {instrument: [] for instrument in instruments}
    for sample in candidates:
        grouped.setdefault(sample.instrument, []).append(sample)

    selected: Dict[str, List[TinySolSample]] = {}
    for instrument, samples in grouped.items():
        if not samples:
            raise ValueError(f"No samples found for instrument '{instrument}' with dyn='{dyn}'.")
        selected[instrument] = _uniform_sample_by_pitch(samples, samples_per_instrument)

    return selected


def materialize_tinysol_subset(subset: Mapping[str, List[TinySolSample]], output_root: Path) -> Path:
    """
    Copies the curated TinySOL subset into `output_root` preserving family/instrument layout.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    for samples in subset.values():
        for sample in samples:
            rel_dir = Path(sample.family) / sample.instrument / "ordinario"
            target_dir = output_root / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / sample.path.name
            if not target_path.exists():
                shutil.copy2(sample.path, target_path)
    return output_root


def prepare_tinysol_subset(
    destination: Path | None = None,
    url: str | None = None,
    samples_per_instrument: int = 50,
    dyn: str = "mf",
) -> Path:
    """
    Downloads TinySOL (if needed), filters to the requested subset, and copies it to a
    curated folder.
    """

    url = url or DATASET_URLS["tinysol"]
    dest = destination or dataset_root() / "tinysol"
    extracted_path = prepare_dataset(
        name="tinysol",
        destination=dest,
        url_override=url,
    )
    subset = collect_tinysol_subset(extracted_path, samples_per_instrument=samples_per_instrument, dyn=dyn)
    curated_root = dest / "curated"
    materialize_tinysol_subset(subset, curated_root)
    return curated_root


def tinysol_label_resolver(path: Path, _: Mapping[str, str] | None = None) -> str:
    parsed = parse_tinysol_sample(path)
    if parsed:
        return parsed.instrument
    return path.parent.name


def tinysol_detail_resolver(path: Path, _: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    parsed = parse_tinysol_sample(path)
    if not parsed:
        return None, None
    return parsed.instrument, parsed.pitch
