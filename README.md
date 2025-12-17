# InstruReconstr

Sparse sinusoidal reconstruction and analysis toolkit for musical instrument audio. The library can:

* download and curate a balanced TinySOL subset (50 mf ordinario samples for six instruments) with preserved family/instrument folder layout;
* extract STFT, mel, MFCC, F0, envelopes, and partials with `librosa`;
* fit a lightweight harmonic model, resynthesize audio, and report metrics (LSD, F0 RMSE, spectral convergence, waveform RMSE, spectrogram RMSE);
* batch reconstruct entire datasets, aggregate metrics per instrument class (overall + per-instrument), and visualize label-level partials with PCA;
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


## Dataset preparation (TinySOL)

The toolkit now focuses on TinySOL. It downloads from the official Zenodo link and curates a balanced subset:

* Instruments: Trumpet, Contrabass, Flute, Clarinet, Oboe, Accordion.
* Articulation: `ordinario` only.
* Dynamics: `mf` only.
* Sampling: 50 WAVs per instrument, pitches evenly spread over C0–B8.

Prepare the dataset (downloads + extracts + curates) with:

```python
from pathlib import Path
from InstruReconstr import datasets, ConfigManager

cfg = ConfigManager.get_instance().config
curated = datasets.prepare_tinysol_subset()  # stored under PATH_DATASETS/tinysol/curated
print(curated)
```

If you already have the archive or want to override the URL:

```python
custom = datasets.prepare_tinysol_subset(
    url="https://zenodo.org/records/3685367/files/TinySOL.tar.gz?download=1",
    samples_per_instrument=50,
    dyn="mf",
)
```

The curated structure keeps the original naming convention and folders (`<FAMILY>/<INSTRUMENT>/ordinario/<INSTR>-ord-<PITCH>-<DYN>-<INSTANCE>-<MISC>.wav`).


## Single-file analysis

```python
from pathlib import Path
from InstruReconstr import feature_extraction, harmonic_model, evaluation, visualization

path = Path("data/example.wav")
waveform, sr = feature_extraction.load_mono_audio(str(path), sample_rate=22050)
features = feature_extraction.extract_features(waveform, sr)
model = harmonic_model.fit_and_resynthesize(
        waveform,
        sample_rate=sr,
        n_partials=n_partials,
        hop_length=hop_length,
        f0_hz=features.f0_hz,
        residual_mix=0.15)
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

Reconstruct **every** audio file in a dataset (TinySOL curated subset is the main target), aggregate metrics overall and per instrument, and export a PCA plot of label-level partials:

```bash
python -m InstruReconstr.dataset_analysis \
  ~/InstruReconstr_datasets/tinysol/curated \
  --partials 12 \
  --output-dir results/tinysol_run \
  --save-recon
```

The script will:

* rebuild each WAV, compute all five metrics, and (optionally) save reconstructions;
* derive instrument and pitch labels from TinySOL filenames (or fall back to directory/metadata for other datasets);
* print overall mean ± std plus per-instrument mean ± std for every metric;
* write `metrics_summary.json` (overall + per instrument) and `label_partial_pca.png` into the output directory.


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
