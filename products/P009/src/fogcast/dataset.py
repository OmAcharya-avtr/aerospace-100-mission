"""Seeded synthetic dataset generator for FogCast.

IMPORTANT: this dataset is SYNTHETIC and MODEL-DERIVED. The ground-truth attenuation is
the Kim empirical model (Kim et al., Proc. SPIE 4214, 2001) plus physically-motivated
perturbations. It is NOT field measurement data. Any ML "accuracy" reported against this
dataset is accuracy relative to this synthetic generative process, not relative to the
real atmosphere. See DATASET_CARD.md for the full statement of this limitation.

Generative process (all seeded, deterministic)
----------------------------------------------
Features:
- visibility_km : log-uniform on [0.05, 50] km (covers dense fog to clear haze).
- wavelength_nm : 50 % from the FSO telecom set {850, 1310, 1550} nm,
                  50 % uniform on [600, 1700] nm.
- rh_percent    : relative humidity, uniform on [40, 100] %.

Ground truth attenuation (dB/km):
    alpha = alpha_Kim(V, lambda; q_Kim(V) + dq) * m_rh(V, RH) * exp(eps)

- dq ~ N(0, 0.07), clipped so q >= 0: wavelength-dependent perturbation of the
  size-distribution exponent (its effect scales with (lambda/550)^(-dq), i.e. it is
  larger at wavelengths far from 550 nm).
- m_rh = 1 + 0.25 * ((clip(RH - 40, 0, 60)/60)^2) * exp(-0.5 ((ln V - ln 3)/0.8)^2):
  a hygroscopic-growth-inspired humidity multiplier, strongest in the haze regime
  (V ~ 3 km) and at high RH. Qualitatively motivated by aerosol hygroscopic swelling
  (e.g. Haenel 1976, Adv. Geophys. 19); the functional form and coefficients here are
  synthetic choices, not fitted to measurements.
- eps ~ N(0, 0.05): multiplicative (lognormal) measurement noise, ~5 %.

Units: visibility km, wavelength nm, RH percent, attenuation dB/km.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .baselines import (
    DB_PER_NEPER,
    KOSCHMIEDER_2PCT,
    REFERENCE_WAVELENGTH_NM,
    kim_attenuation_db_km,
    kim_q,
)

TELECOM_WAVELENGTHS_NM = (850.0, 1310.0, 1550.0)

#: Fractions of the dataset assigned to train / validation / test splits.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def generate_dataset(n_samples: int = 6000, seed: int = 42) -> dict[str, NDArray[np.float64]]:
    """Generate the seeded synthetic FogCast dataset.

    Parameters
    ----------
    n_samples : number of samples (default 6000; trains in well under 1 minute).
    seed : RNG seed. Same seed => bit-identical dataset.

    Returns
    -------
    dict with float64 arrays of shape (n_samples,):
    ``visibility_km``, ``wavelength_nm``, ``rh_percent``, ``attenuation_db_km``
    (synthetic ground truth), and ``attenuation_kim_db_km`` (unperturbed Kim baseline).
    """
    if n_samples < 10:
        raise ValueError("n_samples must be >= 10.")
    rng = np.random.default_rng(seed)

    visibility = np.exp(rng.uniform(np.log(0.05), np.log(50.0), n_samples))
    use_telecom = rng.random(n_samples) < 0.5
    telecom = rng.choice(np.asarray(TELECOM_WAVELENGTHS_NM), size=n_samples)
    continuous = rng.uniform(600.0, 1700.0, n_samples)
    wavelength = np.where(use_telecom, telecom, continuous)
    rh = rng.uniform(40.0, 100.0, n_samples)

    # Perturbed Kim exponent: wavelength-dependent noise channel.
    q_true = np.clip(np.asarray(kim_q(visibility)) + rng.normal(0.0, 0.07, n_samples), 0.0, None)
    sigma = (KOSCHMIEDER_2PCT / visibility) * (wavelength / REFERENCE_WAVELENGTH_NM) ** (-q_true)
    alpha_clear = DB_PER_NEPER * sigma  # dB/km

    # Hygroscopic humidity multiplier (synthetic, haze-regime bump around V = 3 km).
    rh_norm = np.clip(rh - 40.0, 0.0, 60.0) / 60.0
    haze_weight = np.exp(-0.5 * ((np.log(visibility) - np.log(3.0)) / 0.8) ** 2)
    m_rh = 1.0 + 0.25 * rh_norm**2 * haze_weight

    # ~5 % multiplicative measurement noise.
    noise = np.exp(rng.normal(0.0, 0.05, n_samples))

    alpha = alpha_clear * m_rh * noise

    return {
        "visibility_km": visibility,
        "wavelength_nm": wavelength,
        "rh_percent": rh,
        "attenuation_db_km": alpha,
        "attenuation_kim_db_km": np.asarray(kim_attenuation_db_km(visibility, wavelength)),
    }


def split_indices(
    n_samples: int, seed: int = 42
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Deterministic 70/15/15 train/validation/test index split.

    Uses ``seed + 1`` internally so the split permutation is independent of the
    feature/noise draws in :func:`generate_dataset`.
    """
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_samples)
    n_train = int(round(SPLIT_FRACTIONS[0] * n_samples))
    n_val = int(round(SPLIT_FRACTIONS[1] * n_samples))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def save_dataset_csv(path: str, n_samples: int = 6000, seed: int = 42) -> None:
    """Generate the dataset and save it as CSV (committed-script entry point)."""
    data = generate_dataset(n_samples=n_samples, seed=seed)
    cols = ["visibility_km", "wavelength_nm", "rh_percent", "attenuation_db_km"]
    arr = np.column_stack([data[c] for c in cols])
    np.savetxt(path, arr, delimiter=",", header=",".join(cols), comments="")


def _main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m fogcast.dataset",
        description="Generate the seeded synthetic FogCast dataset as CSV.",
    )
    parser.add_argument("--out", default="fogcast_synthetic.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_dataset_csv(args.out, n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    _main()
