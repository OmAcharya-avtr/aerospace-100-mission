"""Seeded synthetic weather dataset for the availability model.

ALL DATA PRODUCED HERE IS SYNTHETIC.  No real meteorological observations are
used anywhere in this package; see DATASET_CARD.md.  The generative process is
a plausibility construct, not a validated atmosphere model:

1. Each sample has a calendar month m and a station climatological prior
   p0(m) (clear-sky probability, taken from illustrative station priors).
2. A latent "synoptic state" s ~ N(0, 1) shifts the log-odds of clear sky:
   logit(p_true) = logit(p0) + w_s * s  (fronts make clearing less likely).
3. Observable features are noisy views of s and p0:
   relative humidity, mid-level cloud fraction from a synthetic satellite IR
   proxy, surface pressure anomaly, wind speed, and the month encoded as
   (sin, cos).  Noise is Gaussian with fixed scales.
4. The label y ~ Bernoulli(p_true): 1 = pass succeeded (clear), 0 = clouded
   out.

Because p_true is retained, model calibration can be checked against the
exact generative probability as well as against empirical bin frequencies.
Determinism: fully reproducible from ``seed`` via ``numpy.random.default_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "prior_p_clear",        # climatological monthly prior [0-1]
    "rel_humidity_pct",     # synthetic relative humidity [%]
    "ir_cloud_fraction",    # synthetic IR-derived cloud fraction [0-1]
    "pressure_anom_hpa",    # synthetic surface pressure anomaly [hPa]
    "wind_speed_ms",        # synthetic 10 m wind speed [m/s]
    "month_sin",            # sin(2 pi month/12)
    "month_cos",            # cos(2 pi month/12)
)

# Illustrative monthly priors used only for data generation (synthetic).
_GEN_PRIORS = np.array([
    [0.35, 0.38, 0.42, 0.48, 0.55, 0.62, 0.66, 0.64, 0.56, 0.47, 0.38, 0.34],  # temperate
    [0.72, 0.74, 0.76, 0.78, 0.82, 0.86, 0.88, 0.87, 0.84, 0.80, 0.75, 0.72],  # arid
    [0.55, 0.53, 0.50, 0.46, 0.52, 0.62, 0.68, 0.66, 0.58, 0.50, 0.52, 0.56],  # highland
])

_W_SYNOPTIC = 1.6  # log-odds weight of the latent synoptic state


@dataclass(frozen=True)
class WeatherDataset:
    """Synthetic dataset bundle: features X, labels y, true probabilities."""

    x: np.ndarray            # (n, 7) float features, order = FEATURE_NAMES
    y: np.ndarray            # (n,) int labels, 1 = clear/pass success
    p_true: np.ndarray       # (n,) exact generative probability of y = 1
    feature_names: tuple[str, ...]
    seed: int


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n_samples: int = 6000, seed: int = 0) -> WeatherDataset:
    """Generate the seeded synthetic weather/availability dataset.

    Parameters
    ----------
    n_samples : number of samples, > 0.  Default 6000 trains in seconds on
        2 CPU cores (documented compute budget).
    seed : RNG seed; identical seeds give bit-identical arrays.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    rng = np.random.default_rng(seed)

    site = rng.integers(0, _GEN_PRIORS.shape[0], size=n_samples)
    month = rng.integers(1, 13, size=n_samples)
    p0 = _GEN_PRIORS[site, month - 1]

    s = rng.standard_normal(n_samples)                     # latent synoptic state
    p_true = _sigmoid(_logit(p0) - _W_SYNOPTIC * s)        # s > 0 => cloudier
    y = (rng.random(n_samples) < p_true).astype(int)

    # Noisy observable features correlated with the latent state.
    rel_hum = np.clip(55.0 + 18.0 * s + 8.0 * rng.standard_normal(n_samples), 5.0, 100.0)
    ir_cloud = np.clip(_sigmoid(0.9 * s - _logit(p0) * 0.5)
                       + 0.08 * rng.standard_normal(n_samples), 0.0, 1.0)
    pres_anom = -6.0 * s + 3.0 * rng.standard_normal(n_samples)
    wind = np.clip(6.0 + 2.5 * np.abs(s) + 2.0 * rng.standard_normal(n_samples), 0.0, None)
    month_sin = np.sin(2.0 * np.pi * month / 12.0)
    month_cos = np.cos(2.0 * np.pi * month / 12.0)

    x = np.column_stack([p0, rel_hum, ir_cloud, pres_anom, wind, month_sin, month_cos])
    return WeatherDataset(x=x, y=y, p_true=p_true,
                          feature_names=FEATURE_NAMES, seed=seed)
