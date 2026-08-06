# Dataset Card — JitterScope Synthetic Vibration Telemetry

**Dataset:** synthetic platform pointing-jitter telemetry
**Generator:** `src/jitterscope/telemetry.py::generate_telemetry` (committed)
**Version:** 0.1.0 · **Date:** 2026-08-06

## Summary

**This dataset is entirely synthetic and idealized. It contains no real
spacecraft, aircraft, or laboratory measurements.** It exists to exercise and
benchmark the PSD and anomaly-detection code paths, not to represent any real
platform.

No data files are committed. Every record is regenerated deterministically from
a fixed integer seed via a single `numpy.random.default_rng(seed)` stream, so the
benchmark is reproducible from source without storing arrays.

## Composition

Each record is a single-channel pointing-angle time series in radians, sampled at
a user-specified rate (benchmark uses 60 s at 1000 Hz). The nominal signal is the
sum of three components:

| Component | Model | Default RMS | Rationale |
|---|---|---|---|
| Broadband base | White Gaussian noise | 0.3 µrad | Sensor/readout noise floor |
| Colored low-frequency | Frequency-shaped noise, flat below a 2 Hz knee, ~1/f² above | 0.8 µrad | Structural/thermal drift; shaping method after Kasdin 1995, *Proc. IEEE* 83(5):802–827 |
| Tonal | Fundamental at `wheel_hz` (default 45 Hz) plus harmonics `k·wheel_hz`, amplitude ∝ 1/k, random phase | 0.5 µrad (fundamental) | Reaction-wheel disturbance is classically modeled as discrete harmonics of wheel speed — Masterson, Miller & Grogan 2002, *J. Sound Vib.* 249(3):575–598 |

Harmonics above Nyquist are skipped rather than aliased.

## Fault signatures (labels)

Faults are injected on request and returned with a per-sample boolean
`fault_mask` that serves as the ground-truth label channel.

| Kind | Parameters | Physical analogue |
|---|---|---|
| `new_tone` | `t_start`, `freq_hz`, `rms` | A new resonance or a wheel spinning up to a new rate |
| `band_shift` | `t_start`, `f_lo`, `f_hi`, `factor` | Broadband energy growth in a band (bearing degradation, loosened interface); band-pass filtered noise added so in-band PSD scales by ~`factor` |
| `transient` | `t_start`, `rate_hz`, `amp`, `decay_s`, `ring_hz` | Intermittent decaying-sinusoid bursts (thruster firings, mechanism steps, micrometeoroid impacts); event times are Poisson |

**Window labeling convention used in the benchmark:** a 1 s analysis window is
labeled positive if any sample inside it carries the fault mask.

## Benchmark splits

| Split | Records | Seeds | Content |
|---|---|---|---|
| Train (nominal only) | 4 × 60 s | 1000–1003 | No faults; 476 feature windows |
| Test | 24 × 60 s | 2000–2023 | 6 nominal, 6 `new_tone`, 6 `band_shift`, 6 `transient`; 2856 windows (952 faulty) |

Train and test records share no seeds, so test records are entirely unseen.
Fault parameters vary systematically across the 6 records of each class
(tone frequency 120–195 Hz, band edges 150–370 Hz, transient amplitude
3.0–5.5 µrad) to avoid benchmarking against a single operating point.

## Unmodeled effects (what this data is NOT)

The following are real and consequential for flight telemetry and are **absent
from this dataset**. Detector performance reported on it should not be
extrapolated to real telemetry.

- **Nonstationarity:** no slews, no wheel speed sweeps or run-ups (which drag
  harmonics across the spectrum), no thermal snap, no day/night cycling. Both
  detectors assume a stationary nominal.
- **Multi-axis coupling:** single channel only. Real jitter is 2–3 axis with
  cross-correlated structural modes; no cross-spectra, no coherence.
- **Structural dynamics:** no modal transfer function between disturbance source
  and optical line of sight, no damping ratios, no isolator response, no
  gyroscopic stiffening of wheel modes with speed.
- **Sensor artifacts:** no quantization, no bit dropouts, no gaps, no timestamp
  jitter, no saturation, no calibration drift, no aliasing from an imperfect
  anti-alias filter.
- **Non-Gaussian statistics:** noise is exactly Gaussian; real vibration
  telemetry is frequently heavy-tailed and impulsive.
- **Control loop:** no fine-steering-mirror or ADCS closed-loop rejection, which
  strongly shapes the low-frequency PSD of a real pointing error signal.
- **Environment:** no atmospheric turbulence, no launch/ascent loads, no
  microgravity disturbance sources beyond the modeled wheel harmonics.
- **Fault realism:** faults appear as clean step changes at a known instant.
  Real degradation is gradual, often intermittent, and rarely confined to one
  signature at a time.

## Regeneration

```python
from jitterscope import generate_telemetry
t, x, mask = generate_telemetry(
    duration_s=60.0, fs=1000.0, seed=2000,
    faults=[{"kind": "new_tone", "t_start": 30.0, "freq_hz": 120.0, "rms": 0.6e-6}],
)
```

Full benchmark set: `python validation/val_detector.py` (~6 s, 2 CPU cores).

## Licensing and provenance

Generated code and data are original to this package, Apache-2.0,
© 2026 OPTIMA Organisation. No third-party data is included or redistributed.

## Intended use and limits

Intended for software testing, algorithm demonstration, and teaching. **Not
certified for operational flight use.** Do not use this dataset to qualify a
detector for a real mission, to set flight alarm thresholds, or to substantiate
any performance claim about real hardware.
