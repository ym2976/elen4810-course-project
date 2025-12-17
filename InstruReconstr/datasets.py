"""
datasets
========

Helpers for organizing public music datasets (NSynth, IRMAS, MAESTRO). The functions
are intentionally light-weight: they manage folder structures, optionally download
small subsets, and surface lists of audio files for downstream analysis.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List
import requests
from .config_manager import ConfigManager


DATASET_URLS = {
    # Multiple mirrors for NSynth to avoid intermittent 404/403 responses.
    "nsynth_test_subset": [
        "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
        "https://magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
        "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    ],
    "irmas_test_subset": [
        "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip",
    ],
    "maestro_v301": [
        "https://storage.googleapis.com/magentadata/datasets/maestro/MAESTRO-3.0.0.tar.gz",
    ],
}


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
        urls = mapped if isinstance(mapped, list) else [mapped]

    if local_archive:
        archive_path = Path(local_archive)
        if not archive_path.exists():
            raise FileNotFoundError(f"local_archive not found: {archive_path}")
    else:
        # Use the filename from the first candidate URL.
        archive_path = archive_dir / urls[0].split("/")[-1]
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
