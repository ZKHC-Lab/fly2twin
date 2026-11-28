# SPDX-License-Identifier: Apache-2.0
"""Tempo, beat grid and section energy of a music file (ffmpeg decode, numpy/scipy onset analysis).

Onset strength = positive spectral flux of a log-magnitude STFT (mono 22.05 kHz, hop 512). Tempo is the
autocorrelation peak in 70-160 BPM; the beat phase maximises summed onset strength on that grid.
Prints tempo, beat times and per-second RMS energy (dB) for picking a 60 s excerpt.
Usage: python beats.py track.mp3 [more.mp3 ...]
"""

import subprocess

import numpy as np
from scipy.signal import stft

SR, HOP = 22050, 512


def decode(path):
    from .common import require_ffmpeg

    raw = subprocess.run(
        [require_ffmpeg(), '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.float32)


def analyse(path):
    x = decode(path)
    if len(x) < SR:
        raise ValueError('Beat analysis needs at least one second of audio')
    _, _, Z = stft(x, SR, nperseg=2048, noverlap=2048 - HOP)
    mag = np.log1p(100 * np.abs(Z))
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(0)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fps = SR / HOP
    ac = np.correlate(flux, flux, 'full')[len(flux) - 1 :]
    lags = np.arange(len(ac)) / fps
    ok = (lags > 60 / 160) & (lags < 60 / 70)
    period = lags[ok][np.argmax(ac[ok])]
    n = int(round(period * fps))
    phase = max(range(n), key=lambda p: flux[p::n].sum())
    beats = (np.arange(phase, len(flux), n) / fps).tolist()
    rms = [float(20 * np.log10(np.sqrt(np.mean(x[i : i + SR] ** 2)) + 1e-9)) for i in range(0, len(x) - SR, SR)]
    return dict(
        file=path,
        seconds=len(x) / SR,
        bpm=60 / period,
        beat_period_s=period,
        first_beat_s=beats[0],
        beats=[round(b, 3) for b in beats],
        rms_db_per_s=[round(r, 1) for r in rms],
    )
