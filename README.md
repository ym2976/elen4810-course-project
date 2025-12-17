# InstruReconstr

Sparse sinusoidal reconstruction and analysis toolkit for musical instrument audio. The library can:

* download and organize public datasets (NSynth/IRMAS/MAESTRO) with resilient mirrors;
* extract STFT, mel, MFCC, F0, envelopes, and partials with `librosa`;
* fit a lightweight harmonic model, resynthesize audio, and report metrics (LSD, F0 RMSE, spectral convergence, waveform RMSE, spectrogram RMSE);
* batch reconstruct entire datasets, aggregate metrics per instrument class, and visualize label-level partials with PCA;
* provide an interactive Gradio demo to compare originals vs. reconstructions with aligned plots and metrics.

Python ≥3.9 is required.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional playback backends
pip install '.[playback]'
```


## Dataset preparation

Use the built-in helper to download NSynth/IRMAS/MAESTRO subsets. NSynth now defaults to the TensorFlow mirror (`http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz`), with fallbacks kept for robustness.

```python
from pathlib import Path
from InstruReconstr import datasets, ConfigManager

cfg = ConfigManager.get_instance().config
root = Path(cfg.PATH_DATASETS)
nsynth = datasets.prepare_dataset("nsynth_test_subset")  # downloads + extracts under PATH_DATASETS
audio_files = datasets.list_audio_files(nsynth)
print(f"Found {len(audio_files)} files")
```

To use your own mirror or a pre-downloaded archive:

```python
custom = datasets.prepare_dataset(
    "nsynth_test_subset",
    url_override="https://your.mirror/nsynth-test.jsonwav.tar.gz",
    # or
    local_archive=Path("/path/to/nsynth-test.jsonwav.tar.gz"),
)
```


## Single-file analysis

```python
from pathlib import Path
from InstruReconstr import feature_extraction, harmonic_model, evaluation, visualization

path = Path("data/example.wav")
waveform, sr = feature_extraction.load_mono_audio(str(path), sample_rate=22050)
features = feature_extraction.extract_features(waveform, sr)
model = harmonic_model.fit_and_resynthesize(waveform, sr, n_partials=12)
metrics = evaluation.compute_all_metrics(waveform, model.reconstruction, features.f0_hz, features.f0_hz, sr)

print(metrics)
visualization.plot_waveform_and_spectrogram(waveform, sr, title="Original")
visualization.plot_envelope(model.envelope, sr, title="Envelope (RMS)")
visualization.plot_partials(model.partials_hz_amp, title="Estimated partials")
```

Run the interactive CLI (prints metrics and exports WAVs) with:

```bash
python -m InstruReconstr.interactive data/example.wav --partials 12 --output-dir results/ab_test
```


## Dataset-wide reconstruction and reporting

Reconstruct **every** audio file in a dataset, aggregate metrics per instrument class, and export a PCA plot of label-level partials:

```bash
python -m InstruReconstr.dataset_analysis \
  /path/to/nsynth/data \
  --partials 12 \
  --output-dir results/nsynth_run \
  --save-recon
```

The script will:

* rebuild each WAV, compute all five metrics, and (optionally) save reconstructions;
* derive labels via NSynth metadata when available or fall back to the parent directory name;
* print mean ± std for every metric per label;
* write `metrics_summary.json` and `label_partial_pca.png` into the output directory.


## Gradio demo

Launch an interactive UI that follows the requested three-row layout (upload → three-column plots → playback & metrics):

```bash
python -m InstruReconstr.gradio_app
```

* **Top row:** upload audio.
* **Middle row (three columns):**
  * Left — original waveform and spectrogram (stacked).
  * Middle — envelope and partials (stacked).
  * Right — reconstructed waveform and spectrogram (stacked).
* **Bottom row:** audio players for original & reconstruction plus a metric table (LSD, F0 RMSE, spectral convergence, waveform RMSE, spectrogram RMSE).


## Paths and configuration

`config.py` defines default directories under your home folder:

* `PATH_DATASETS`: downloaded/extracted datasets (`~/InstruReconstr_datasets`).
* `PATH_INTERACTIVE_RESULTS`: exports from the CLI (`~/InstruReconstr_results/interactive`).
* `PATH_RESULTS`: generic analysis outputs.

Override configuration by providing a custom Python config to `ConfigManager.get_instance(<path>)` if needed.
