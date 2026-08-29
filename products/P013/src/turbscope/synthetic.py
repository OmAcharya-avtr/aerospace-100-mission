"""Seeded synthetic multi-sensor measurement generator.

READ ``DATASET_CARD.md`` BEFORE USING ANY NUMBER PRODUCED BY THIS MODULE.

Each :class:`Scenario` is a ground-truth path-averaged Cn2 and path length,
observed by one fixed scintillometer + DIMM instrument pair (fixed
wavelengths, subaperture diameter and separation -- see
:data:`SCINT_WAVELENGTH_M` etc.). :func:`synthesize_measurement` runs both
forward models (:mod:`turbscope.scintillometer`, :mod:`turbscope.dimm`) and
adds independent multiplicative noise to emulate finite-averaging-time sensor
noise. Nothing here is a measurement of the real atmosphere; every number
traces back to the closed-form forward models in this package plus a
documented noise model.

Scenario generation deliberately draws the target Rytov variance directly
(log-uniform over a wide range spanning weak through deeply saturated
turbulence) rather than drawing Cn2 and path length independently, so that
the scintillation-saturation transition zone -- the physically interesting
and hardest region -- is well represented rather than a rare tail event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import SPHERICAL_WAVE_RYTOV_COEFF
from .dimm import differential_variance
from .scintillometer import rytov_variance, scintillation_index_full, wave_number

__all__ = [
    "APERTURE_DIAM_M",
    "DIMM_NOISE_STD",
    "DIMM_WAVELENGTH_M",
    "LOG10_RYTOV_RANGE",
    "PATH_LENGTH_RANGE_M",
    "SCINT_NOISE_STD",
    "SCINT_WAVELENGTH_M",
    "SEPARATION_M",
    "WAVE_TYPE",
    "Scenario",
    "Measurement",
    "cn2_from_target_rytov",
    "generate_scenarios",
    "synthesize_measurement",
    "split_indices",
]

# --- fixed instrument suite (one scintillometer + one DIMM, many deployments)
SCINT_WAVELENGTH_M: float = 880e-9
"""Scintillometer source wavelength, m -- a typical near-IR LED source
(representative of commercial displaced-beam scintillometers; not a specific
vendor's spec)."""

WAVE_TYPE: str = "spherical"
"""Wavefront geometry assumed for the scintillometer (point-source /
small-aperture transmitter)."""

DIMM_WAVELENGTH_M: float = 500e-9
"""DIMM observing wavelength, m -- a typical visible-band value."""

APERTURE_DIAM_M: float = 0.14
"""DIMM subaperture diameter, m (the ESO DIMM value quoted by Sarazin &
Roddier 1990)."""

SEPARATION_M: float = 0.20
"""DIMM subaperture centre-to-centre separation, m (the ESO DIMM value quoted
by Sarazin & Roddier 1990)."""

SCINT_NOISE_STD: float = 0.08
"""Relative (multiplicative) 1-sigma noise on the scintillometer's measured
sigma_I^2, representing finite-averaging-time sampling noise. Illustrative
and hand-chosen for this synthetic generator, not a vendor noise-floor
specification -- see ``DATASET_CARD.md``."""

DIMM_NOISE_STD: float = 0.10
"""Relative (multiplicative) 1-sigma noise on each DIMM differential-motion
variance measurement (centroiding + finite-sample noise). Illustrative and
hand-chosen -- see ``DATASET_CARD.md``."""

LOG10_RYTOV_RANGE: tuple[float, float] = (-3.5, 1.85)
"""log10(sigma_R^2) draw range: ~3.2e-4 (deep weak regime) to ~71 (deep
saturation, well past the heuristic curve's asymptote)."""

PATH_LENGTH_RANGE_M: tuple[float, float] = (150.0, 2500.0)
"""Path-length draw range, m -- representative of horizontal scintillometer /
DIMM sightlines (tens of metres to a few km)."""


@dataclass(frozen=True)
class Scenario:
    """One synthetic ground-truth case.

    Attributes
    ----------
    cn2_path : float
        True path-averaged Cn2, m^-2/3 -- the regression target (as log10).
    path_length_m : float
        True path length, m -- a *known* geometry input, not a target.
    rytov_variance_true : float
        True sigma_R^2 implied by ``cn2_path`` and ``path_length_m`` at the
        scintillometer wavelength; kept for diagnostics/plots only (never a
        model input).
    """

    cn2_path: float
    path_length_m: float
    rytov_variance_true: float


@dataclass(frozen=True)
class Measurement:
    """One noisy multi-sensor observation of a :class:`Scenario`.

    Attributes
    ----------
    sigma_i2_scint : float
        Measured scintillometer scintillation index (noisy), dimensionless.
    var_long_dimm, var_trans_dimm : float
        Measured DIMM longitudinal/transverse differential variance (noisy), rad^2.
    path_length_m : float
        Known path length, m (not noisy -- a surveyed geometry input).
    """

    sigma_i2_scint: float
    var_long_dimm: float
    var_trans_dimm: float
    path_length_m: float


def cn2_from_target_rytov(target_rytov_variance: float, path_length_m: float) -> float:
    """Cn2_path that produces a chosen Rytov variance at :data:`SCINT_WAVELENGTH_M`.

    Inverse of :func:`turbscope.scintillometer.rytov_variance` at fixed path
    length and instrument wavelength/wave type.

    Parameters
    ----------
    target_rytov_variance : float
        Desired sigma_R^2 (> 0).
    path_length_m : float
        Path length, m (> 0).

    Returns
    -------
    float
        Cn2_path, m^-2/3.
    """
    x = float(target_rytov_variance)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"target_rytov_variance must be finite and > 0 (got {x!r}).")
    length = float(path_length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"path_length_m must be finite and > 0 (got {path_length_m!r}).")
    k = wave_number(SCINT_WAVELENGTH_M)
    return float(x / (SPHERICAL_WAVE_RYTOV_COEFF * k ** (7.0 / 6.0) * length ** (11.0 / 6.0)))


def generate_scenarios(n_scenarios: int, seed: int = 20260829) -> list[Scenario]:
    """Draw ``n_scenarios`` synthetic ground-truth cases.

    Deterministic given ``seed``: a single ``numpy.random.Generator`` (PCG64)
    is consumed in a fixed order (log-Rytov, then path length), so results are
    bit-identical for the same seed and NumPy version.

    Parameters
    ----------
    n_scenarios : int
        Number of scenarios (>= 1).
    seed : int
        Master seed.

    Returns
    -------
    list of Scenario
    """
    n = int(n_scenarios)
    if n < 1:
        raise ValueError(f"n_scenarios must be >= 1 (got {n_scenarios!r}).")
    rng = np.random.default_rng(seed)
    log_r = rng.uniform(*LOG10_RYTOV_RANGE, size=n)
    lengths = rng.uniform(*PATH_LENGTH_RANGE_M, size=n)
    scenarios = []
    for lr, length in zip(log_r, lengths, strict=True):
        target_r = 10.0**lr
        cn2 = cn2_from_target_rytov(target_r, float(length))
        scenarios.append(
            Scenario(cn2_path=cn2, path_length_m=float(length), rytov_variance_true=target_r)
        )
    return scenarios


def synthesize_measurement(scenario: Scenario, rng: np.random.Generator) -> Measurement:
    """Run both forward models for one scenario and add measurement noise.

    Parameters
    ----------
    scenario : Scenario
        Ground truth.
    rng : numpy.random.Generator
        Source of randomness (caller controls the seed/consumption order so
        that dataset builds are reproducible).

    Returns
    -------
    Measurement
    """
    r_var = rytov_variance(scenario.cn2_path, scenario.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE)
    sigma_i2_true = float(scintillation_index_full(r_var))
    var_long_true = float(
        differential_variance(
            scenario.cn2_path,
            scenario.path_length_m,
            DIMM_WAVELENGTH_M,
            APERTURE_DIAM_M,
            SEPARATION_M,
            component="longitudinal",
        )
    )
    var_trans_true = float(
        differential_variance(
            scenario.cn2_path,
            scenario.path_length_m,
            DIMM_WAVELENGTH_M,
            APERTURE_DIAM_M,
            SEPARATION_M,
            component="transverse",
        )
    )
    scint_meas = sigma_i2_true * (1.0 + rng.normal(0.0, SCINT_NOISE_STD))
    long_meas = var_long_true * (1.0 + rng.normal(0.0, DIMM_NOISE_STD))
    trans_meas = var_trans_true * (1.0 + rng.normal(0.0, DIMM_NOISE_STD))
    # Physical quantities cannot be negative; a noise draw pushing a small true
    # value below zero is floored, not discarded. The floor is set far below
    # any physically produced true value in this generator's range (checked
    # against turbscope.synthetic's own draw ranges) so it only ever catches a
    # rare sign flip from the noise draw and never masks or dominates a
    # genuinely weak true signal -- an earlier, larger floor value doing
    # exactly that was caught and corrected during validation
    # (``validation/VALIDATION.md`` records the check).
    floor = 1e-30
    return Measurement(
        sigma_i2_scint=max(scint_meas, floor),
        var_long_dimm=max(long_meas, floor),
        var_trans_dimm=max(trans_meas, floor),
        path_length_m=scenario.path_length_m,
    )


def split_indices(
    n: int, test_fraction: float = 0.25, seed: int = 4242
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Deterministic shuffle-split of ``range(n)`` into (train, test) indices.

    Parameters
    ----------
    n : int
        Number of items (>= 2).
    test_fraction : float
        Fraction held out for test, in (0, 1).
    seed : int
        Shuffle seed.

    Returns
    -------
    (train_idx, test_idx) : tuple of ndarray
    """
    if int(n) < 2:
        raise ValueError("n must be >= 2.")
    f = float(test_fraction)
    if not 0.0 < f < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1) (got {test_fraction!r}).")
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(n))
    n_test = max(1, int(round(f * n)))
    return order[n_test:], order[:n_test]
