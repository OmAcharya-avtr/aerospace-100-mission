"""Seeded synthetic platform-vibration telemetry generator with faults.

Generates a pointing-jitter-like time series as the sum of three
nominal components (all synthetic and idealized — see DATASET_CARD.md):

1. Broadband white base noise (sensor floor).
2. Colored low-frequency noise with a ~1/f^2 power roll-off
   (integrated white noise, high-pass stabilized), mimicking
   structural/thermal drift. Colored-noise shaping by frequency-domain
   filtering follows Kasdin 1995, Proc. IEEE 83(5):802-827.
3. Tonal components at a reaction-wheel-like fundamental and its
   harmonics — RWA disturbance is classically modeled as discrete
   harmonics of wheel speed (Masterson, Miller & Grogan 2002,
   J. Sound Vib. 249(3):575-598).

Injectable fault signatures (each a dict, see ``generate_telemetry``):

- ``new_tone``: a tone at a new frequency appears at ``t_start``.
- ``band_shift``: broadband energy in ``[f_lo, f_hi]`` is scaled by
  ``factor`` from ``t_start`` (band-pass filtered noise added).
- ``transient``: intermittent decaying-sinusoid bursts (impacts /
  micro-events) at random times after ``t_start``.

All randomness flows from a single ``numpy.random.default_rng(seed)``;
the same call is bit-reproducible across runs on the same platform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as _sig

__all__ = ["generate_telemetry"]

_FAULT_KINDS = ("new_tone", "band_shift", "transient")


def _colored_noise(rng: np.random.Generator, n: int, fs: float, f_knee: float) -> np.ndarray:
    """Unit-variance noise with flat PSD below f_knee and ~1/f^2 above.

    Built by frequency-domain shaping of white Gaussian noise
    (Kasdin 1995): amplitude weight 1/sqrt(1 + (f/f_knee)^2).
    """
    white = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    weight = 1.0 / np.sqrt(1.0 + (f / f_knee) ** 2)
    x = np.fft.irfft(spec * weight, n=n)
    std = float(np.std(x))
    return x / std if std > 0 else x


def _validate_fault(fault: dict[str, Any], fs: float, duration_s: float) -> None:
    kind = fault.get("kind")
    if kind not in _FAULT_KINDS:
        raise ValueError(f"fault kind must be one of {_FAULT_KINDS}, got {kind!r}")
    t_start = float(fault.get("t_start", 0.0))
    if not 0.0 <= t_start < duration_s:
        raise ValueError(f"fault t_start must be in [0, {duration_s}) s, got {t_start}")
    if kind == "new_tone":
        f0 = float(fault["freq_hz"])
        if not 0.0 < f0 < fs / 2:
            raise ValueError(f"new_tone freq_hz must be in (0, fs/2), got {f0}")
    if kind == "band_shift":
        lo, hi = float(fault["f_lo"]), float(fault["f_hi"])
        if not 0.0 < lo < hi < fs / 2:
            raise ValueError(f"band_shift needs 0 < f_lo < f_hi < fs/2, got ({lo}, {hi})")
        if float(fault.get("factor", 2.0)) <= 0:
            raise ValueError("band_shift factor must be > 0")


def generate_telemetry(
    duration_s: float = 60.0,
    fs: float = 1000.0,
    seed: int = 0,
    wheel_hz: float = 45.0,
    n_harmonics: int = 3,
    tone_rms: float = 0.5e-6,
    base_rms: float = 0.3e-6,
    colored_rms: float = 0.8e-6,
    faults: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate seeded synthetic pointing-jitter telemetry [rad].

    Parameters
    ----------
    duration_s : float
        Record length [s], > 0.
    fs : float
        Sample rate [Hz], > 0.
    seed : int
        RNG seed; identical arguments + seed give identical output.
    wheel_hz : float
        Reaction-wheel-like fundamental tone frequency [Hz]; harmonics
        ``k * wheel_hz`` (k = 1..n_harmonics) are added with amplitude
        falling as 1/k (idealized; real RWA harmonic coefficients are
        empirical, Masterson et al. 2002).
    n_harmonics : int
        Number of wheel harmonics, >= 1 (harmonics above fs/2 are
        skipped to avoid aliasing).
    tone_rms, base_rms, colored_rms : float
        RMS amplitude [rad] of the fundamental tone, the white base
        noise, and the colored low-frequency noise.
    faults : list of dict, optional
        Fault signatures to inject. Supported ``kind`` values:

        - ``{"kind": "new_tone", "t_start": s, "freq_hz": f, "rms": a}``
        - ``{"kind": "band_shift", "t_start": s, "f_lo": f1, "f_hi": f2,
          "factor": k}`` — adds band-limited noise so band PSD scales
          by ~``factor``.
        - ``{"kind": "transient", "t_start": s, "rate_hz": r,
          "amp": a, "decay_s": tau, "ring_hz": f}`` — Poisson-timed
          decaying sinusoid bursts.

    Returns
    -------
    t : ndarray
        Time stamps [s].
    x : ndarray
        Telemetry samples [rad].
    fault_mask : ndarray of bool
        True where any injected fault is active (label channel for
        detector benchmarking).

    Raises
    ------
    ValueError
        On non-positive duration/fs/amplitudes or malformed faults.
    """
    if duration_s <= 0 or fs <= 0:
        raise ValueError(f"duration_s and fs must be > 0, got {duration_s}, {fs}")
    if min(tone_rms, base_rms, colored_rms) < 0:
        raise ValueError("component RMS amplitudes must be >= 0")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
    if not 0.0 < wheel_hz < fs / 2:
        raise ValueError(f"wheel_hz must be in (0, fs/2), got {wheel_hz}")

    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    x = rng.normal(0.0, base_rms, n)
    x += colored_rms * _colored_noise(rng, n, fs, f_knee=2.0)
    for k in range(1, n_harmonics + 1):
        fk = k * wheel_hz
        if fk >= fs / 2:
            break
        phase = rng.uniform(0.0, 2.0 * np.pi)
        # RMS of A*sin is A/sqrt(2); amplitude falls as 1/k.
        x += (tone_rms * np.sqrt(2.0) / k) * np.sin(2.0 * np.pi * fk * t + phase)

    fault_mask = np.zeros(n, dtype=bool)
    for fault in faults or []:
        _validate_fault(fault, fs, duration_s)
        kind = fault["kind"]
        i0 = int(round(float(fault.get("t_start", 0.0)) * fs))
        active = slice(i0, n)
        if kind == "new_tone":
            a = float(fault.get("rms", tone_rms)) * np.sqrt(2.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            x[active] += a * np.sin(2.0 * np.pi * float(fault["freq_hz"]) * t[active] + phase)
            fault_mask[active] = True
        elif kind == "band_shift":
            lo, hi = float(fault["f_lo"]), float(fault["f_hi"])
            factor = float(fault.get("factor", 2.0))
            sos = _sig.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
            extra = _sig.sosfilt(sos, rng.normal(0.0, 1.0, n - i0))
            std = float(np.std(extra))
            if std > 0:
                # PSD in band scales ~factor when noise power (factor-1)x
                # the nominal in-band base-noise power is added.
                band_power = base_rms**2 * (hi - lo) / (fs / 2)
                extra *= np.sqrt(max(factor - 1.0, 0.0) * band_power) / std
                x[active] += extra
            fault_mask[active] = True
        elif kind == "transient":
            rate = float(fault.get("rate_hz", 0.5))
            amp = float(fault.get("amp", 5.0 * base_rms))
            tau = float(fault.get("decay_s", 0.05))
            ring = float(fault.get("ring_hz", min(0.35 * fs, 150.0)))
            t_ev = float(fault.get("t_start", 0.0))
            while True:
                t_ev += rng.exponential(1.0 / rate)
                if t_ev >= duration_s:
                    break
                j0 = int(round(t_ev * fs))
                j1 = min(n, j0 + int(round(5.0 * tau * fs)))
                tt = t[j0:j1] - t[j0]
                x[j0:j1] += amp * np.exp(-tt / tau) * np.sin(2.0 * np.pi * ring * tt)
                fault_mask[j0:j1] = True
    return t, x, fault_mask
