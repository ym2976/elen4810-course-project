"""
setup
=====

This module sets up the InstruReconstr package for distribution. It includes package
information, versioning, author details, and other metadata necessary for distribution
on PyPI.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="InstruReconstr",
    version="0.1.0",
    author="Duje Giljanović",
    author_email="giljanovic.duje@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/gilja/instruments_fourier_analysis",
    license="LICENSE.txt",
    description="A Python package for reconstructing musical instruments with sparse partials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "librosa>=0.10.0",
        "matplotlib>=3.8.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "scipy>=1.11.4",
        "soundfile>=0.12.1",
        "gradio>=3.50.2",
    ],
    extras_require={
        "playback": ["sounddevice>=0.4.6", "simpleaudio>=1.0.4", "pyaudio>=0.2.13"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    python_requires=">=3.9",
)
