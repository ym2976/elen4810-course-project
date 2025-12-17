"""
config
======

This configuration file contains constants used throughout the project.

The config file is expected to be in the predefined format, otherwise the program
may not work as expected. In case the user wants to change the default values, it
is recommended to consult the documentation first! This will be the case for any
nontrivial usage of the package.

The documentation can be found in the
GitHub repository of the project: github.com/gilja/instruments_fourier_analysis

Constants:
----------

-   PACKAGE_INSTALLATION_PATH: Path to the package installation folder.
-   PATH_BASE: Base path of the project.
-   PATH_DATA: Path to the data folder.
-   PATH_RESULTS: Path to the results folder.
-   PATH_INSTRUMENT_SAMPLES: Path to the instrument samples folder.

-   FIGURE_WIDTH: Width of the figure.
-   FIGURE_HEIGHT: Height of the figure.
-   FIGURE_HEIGHT_PER_PLOT: Height of each individual plot.
-   HSPACING: Horizontal spacing between subplots.
-   VSPACING: Vertical spacing between subplots.
-   Y_AXIS_MARGIN: Y-axis range margin when exporting individual harmonics to PDF.

-   WAVEFORM_ZOOM_PERCENTAGES: Waveform zoom percentage for each instrument.
-   N_HARMONICS_PER_INSTRUMENT: Number of harmonics to be used in the Fourier analysis for each
    instrument.

-   NOTE_FREQUENCIES: Note corresponding to each frequency.
-   AUDIO_DURATION: Duration of exported individual harmonic audio files.
-   SAMPLE_RATE: Sample rate of exported individual harmonic audio files.

-   PERIOD_BOUNDS: Period bounds for each instrument used in the Fourier analysis. The bounds are
    obtained manually by plotting waveform of each audio file and identifying the periods visually.
    The order of the instruments is the same as in the N_HARMONICS_PER_INSTRUMENT and the
    WAVEFORM_ZOOM_PERCENTAGES.

Notes:
------

Author: Duje Giljanović (giljanovic.duje@gmail.com)
License: MIT License

If you use InstruReconstr in your research or any other publication, please acknowledge it by
citing as follows:

@software{InstruReconstr,
    title = {InstruReconstr: Sparse Sinusoidal Reconstruction of Musical Instruments},
    author = {Duje Giljanović},
    year = {2024},
    url = {github.com/gilja/instruments_fourier_analysis},
}
"""

import os

# Path constants
PACKAGE_INSTALLATION_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_BASE = os.path.expanduser("~")
PATH_RESULTS = os.path.join(PATH_BASE, "InstruReconstr_results", "analyzed")
PATH_INTERACTIVE_RESULTS = os.path.join(PATH_BASE, "InstruReconstr_results", "interactive")
PATH_DATASETS = os.path.join(PATH_BASE, "InstruReconstr_datasets")
PATH_INSTRUMENT_SAMPLES = os.path.join(
    PACKAGE_INSTALLATION_PATH, "InstruReconstr_data", "instrument_samples"
)

# Set figure size for all plots
FIGURE_WIDTH = 1600  # width for the whole figure
FIGURE_HEIGHT = 800
FIGURE_HEIGHT_PER_PLOT = 400  # height for each individual plot

# Set horizontal and vertical spacing between subplots
HSPACING = 0.08
VSPACING = 0.2

# Set the y-axis range margin when exporting individual harmonics to PDF
Y_AXIS_MARGIN = 1.05  # margin for the y-axis range

# Set waveform zoom percentage for each instrument
WAVEFORM_ZOOM_PERCENTAGES = [
    0.008,  # cello
    0.0015,  # clarinet
    0.01,  # double bass
    0.005,  # female vocal
    0.003,  # flute
    0.009,  # nylon string guitar
    0.009,  # oboe
    0.004,  # piano
    0.002,  # piccolo flute
    0.003,  # sax alto
    0.011,  # sax baritone
    0.002,  # sax soprano
    0.007,  # sax tenor
    0.007,  # steel string guitar
    0.009,  # trombone
    0.003,  # trumpet
    0.003,  # violin
]

# Set the number of harmonics to be used in the Fourier analysis for each instrument
N_HARMONICS_PER_INSTRUMENT = [
    50,  # cello
    10,  # clarinet
    45,  # double bass
    20,  # female vocal
    10,  # flute
    20,  # nylon string guitar
    15,  # oboe
    25,  # piano
    10,  # piccolo flute
    10,  # sax alto
    80,  # sax baritone
    10,  # sax soprano
    25,  # sax tenor
    30,  # steel string guitar
    20,  # trombone
    15,  # trumpet
    20,  # violin
]

# Note corresponding to each frequency
NOTE_FREQUENCIES = {
    16.35: "C0",
    17.32: "C#0/Db0",
    18.35: "D0",
    19.45: "D#0/Eb0",
    20.60: "E0",
    21.83: "F0",
    23.12: "F#0/Gb0",
    24.50: "G0",
    25.96: "G#0/Ab0",
    27.50: "A0",
    29.14: "A#0/Bb0",
    30.87: "B0",
    32.70: "C1",
    34.65: "C#1/Db1",
    36.71: "D1",
    38.89: "D#1/Eb1",
    41.20: "E1",
    43.65: "F1",
    46.25: "F#1/Gb1",
    49.00: "G1",
    51.91: "G#1/Ab1",
    55.00: "A1",
    58.27: "A#1/Bb1",
    61.74: "B1",
    65.41: "C2",
    69.30: "C#2/Db2",
    73.42: "D2",
    77.78: "D#2/Eb2",
    82.41: "E2",
    87.31: "F2",
    92.50: "F#2/Gb2",
    98.00: "G2",
    103.83: "G#2/Ab2",
    110.00: "A2",
    116.54: "A#2/Bb2",
    123.47: "B2",
    130.81: "C3",
    138.59: "C#3/Db3",
    146.83: "D3",
    155.56: "D#3/Eb3",
    164.81: "E3",
    174.61: "F3",
    185.00: "F#3/Gb3",
    196.00: "G3",
    207.65: "G#3/Ab3",
    220.00: "A3",
    233.08: "A#3/Bb3",
    246.94: "B3",
    261.63: "C4",
    277.18: "C#4/Db4",
    293.66: "D4",
    311.13: "D#4/Eb4",
    329.63: "E4",
    349.23: "F4",
    369.99: "F#4/Gb4",
    392.00: "G4",
    415.30: "G#4/Ab4",
    440.00: "A4",
    466.16: "A#4/Bb4",
    493.88: "B4",
    523.25: "C5",
    554.37: "C#5/Db5",
    587.33: "D5",
    622.25: "D#5/Eb5",
    659.25: "E5",
    698.46: "F5",
    739.99: "F#5/Gb5",
    783.99: "G5",
    830.61: "G#5/Ab5",
    880.00: "A5",
    932.33: "A#5/Bb5",
    987.77: "B5",
    1046.50: "C6",
    1108.73: "C#6/Db6",
    1174.66: "D6",
    1244.51: "D#6/Eb6",
    1318.51: "E6",
    1396.91: "F6",
    1479.98: "F#6/Gb6",
    1567.98: "G6",
    1661.22: "G#6/Ab6",
    1760.00: "A6",
    1864.66: "A#6/Bb6",
    1975.53: "B6",
    2093.00: "C7",
    2217.46: "C#7/Db7",
    2349.32: "D7",
    2489.02: "D#7/Eb7",
    2637.02: "E7",
    2793.83: "F7",
    2959.96: "F#7/Gb7",
    3135.96: "G7",
    3322.44: "G#7/Ab7",
    3520.00: "A7",
    3729.31: "A#7/Bb7",
    3951.07: "B7",
    4186.01: "C8",
    4434.92: "C#8/Db8",
    4698.63: "D8",
    4978.03: "D#8/Eb8",
    5274.04: "E8",
    5587.65: "F8",
    5919.91: "F#8/Gb8",
    6271.93: "G8",
    6644.88: "G#8/Ab8",
    7040.00: "A8",
    7458.62: "A#8/Bb8",
    7902.13: "B8",
    8000.00: "",  # no note names above B8
}

# set the duration and sample rate for individual harmonic audio files
AUDIO_DURATION = 1.0  # seconds
SAMPLE_RATE = 44100  # Hz

PERIOD_BOUNDS = {
    "cello": [0.8284, 0.83604],
    "clarinet": [2.09145, 2.09334],
    "double_bass": [0.63845, 0.64609],
    "female_vocal": [0.65874, 0.66064],
    "flute": [0.78051, 0.78146],
    "guitar_nylon": [0.441767, 0.44559],
    "oboe": [0.54717, 0.55097],
    "piano": [0.75141, 0.75521],
    "piccolo": [0.69282, 0.69377],
    "sax_alto": [1.2636, 1.2655],
    "sax_baritone": [2.1363, 2.1515],
    "sax_soprano": [1.51283, 1.51472],
    "sax_tenor": [1.08718, 1.09096],
    "guitar_metal": [0.59473, 0.59853],
    "trombone": [0.5417, 0.5455],
    "trumpet": [1.12869, 1.130605],
    "violin": [1.28755, 1.28945],
}
