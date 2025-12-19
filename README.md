# InstruReconstr

Sparse sinusoidal reconstruction and analysis toolkit for musical instrument audio. The library can:

* download and curate a balanced TinySOL subset (50 mf ordinario samples) with preserved family/instrument folder layout;
* extract STFT, mel, MFCC, F0, envelopes, and partials with `librosa`;
* fit a lightweight harmonic model, resynthesize audio, and report metrics (LSD, F0 RMSE, spectral convergence, waveform RMSE, spectrogram RMSE);
* Reconstruct entire datasets, aggregate metrics per instrument class (overall + per-instrument), and analyze the performance;
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
    dyn_filter="mf",
)
```

The curated structure keeps the original naming convention and folders (`<FAMILY>/<INSTRUMENT>/ordinario/<INSTR>-ord-<PITCH>-<DYN>-<INSTANCE>-<MISC>.wav`).



## Dataset-wide reconstruction and diagnostics

Run the unified analysis script to sample each instrument deterministically, rebuild the selected
clips, compute metrics, export best/worst examples, and correlate descriptors with reconstruction
quality:

```bash
python -m InstruReconstr.data_analysis \
  ~/InstruReconstr_datasets/tinysol/curated \
  --partials 16 \
  --per-instrument-limit 20 \
  --examples-overall 5 \
  --examples-per-instrument 3 \
  --output-dir results/tinysol_analysis \
  --save-recon
```

Highlights:

* deterministically sample up to `--per-instrument-limit` clips per instrument (or provide a JSON
  mapping via `--instrument-limit-config`), with an optional global `--limit`;
* compute all metrics (LSD, F0 RMSE, spectral convergence, waveform RMSE, spectrogram RMSE) overall
  and per instrument;
* export best/worst examples overall (default 5) and per instrument (default 3) with waveform,
  spectrogram, and pyin F0 plots for each sample;
* save reconstruction WAVs (optional), full metric/descriptor records, and summary tables;
* compute descriptor correlations (voiced ratio, F0 jitter, harmonic energy ratio, spectral
  flatness, centroid, onset strength, etc.) against LSD with Spearman r / p-values and scatter plots.

Use the generated plots and JSON summaries inside `results/tinysol_analysis` to identify which
instruments or acoustic properties produce accurate reconstructions vs. high-error cases.


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

## Demo for the project

A demo of this project is available via a Google Colab notebook (see link below).

https://colab.research.google.com/drive/1zPCxSBzfziV9gtR9KhxPuvO41wvbKc9Y?usp=sharing
