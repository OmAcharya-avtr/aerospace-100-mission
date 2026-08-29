r"""Seeded synthetic dataset: known ``Cn2(z)`` paths and the sensor readings they produce.

**The dataset is entirely synthetic and contains no measurements.**  Every number
comes from the forward models in :mod:`turbscope.scintillation` and
:mod:`turbscope.dimm` plus the sampling model of :mod:`turbscope.measurements`.
Accuracy measured against it is fidelity to a generative process this package
defines, not agreement with a field instrument.  See ``DATASET_CARD.md``.

Scenario model
--------------
Each scenario is one horizontal path observed once:

* path length ``L`` log-uniform in [300, 3000] m;
* wavelength drawn from {850, 1064, 1550} nm (common laser-diode / Nd:YAG /
  telecom lines);
* a level ``C0`` log-uniform in [1e-16, 1e-12] m^(-2/3) -- the plain path average.
  Andrews & Phillips (2005) ch. 12 give 1e-17 m^(-2/3) (weak) to 1e-13 m^(-2/3)
  (strong) as the usual span of near-ground ``Cn2``, with values approaching
  1e-12 m^(-2/3) over strongly heated surfaces; the range here is chosen to put a
  useful number of scenarios in every fluctuation regime rather than to match any
  particular site's climatology, and it is deliberately weighted toward the
  strong end so that the saturation regime is populated;
* a dimensionless shape ``s(u)``, mean 1 over the path, built from a linear tilt,
  a quadratic bow, one sinusoid and (40 % of the time) a localised Gaussian
  "hot patch" in ``log Cn2``.  The shape represents a path crossing surfaces of
  different roughness and heating -- tarmac, grass, water.  **The shape
  coefficients are hand-chosen to give order-of-magnitude along-path variation;
  they are not fitted to any observation.**

The non-uniform shape is the point of the exercise: it makes the
scintillation-kernel and coherence-kernel path averages differ, which is what
gives each sensor "its own bias".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validate import check_count
from .dimm import cn2_average_from_fried, r0_from_dimm_variance
from .geometry import PathGeometry, weighted_path_average
from .measurements import Measurement, SensorSuite, simulate_measurement
from .scintillation import WEAK_REGIME_BETA0_SQ, uniform_cn2_from_beta0_sq

__all__ = [
    "FEATURE_NAMES",
    "features_from_measurement",
    "REGIME_EDGES",
    "REGIME_NAMES",
    "Scenario",
    "SyntheticDataset",
    "generate_dataset",
    "regime_labels",
]

FEATURE_NAMES: tuple[str, ...] = (
    "log10_sigma_i2_point",
    "log10_sigma_i2_aperture",
    "log10_sigma_l2",
    "log10_sigma_t2",
    "sigma_t2_over_l2",
    "log10_path_length_m",
    "log10_wavelength_m",
    "log10_one_plus_aperture_d_sq",
    "log10_cn2_weak_point",
    "log10_cn2_dimm",
    "dimm_valid",
    "log10_n_irradiance_samples",
    "log10_n_dimm_frames",
)

#: beta_0^2 bin edges used for regime-resolved reporting.
REGIME_EDGES: tuple[float, ...] = (0.0, WEAK_REGIME_BETA0_SQ, 1.0, 5.0, np.inf)
REGIME_NAMES: tuple[str, ...] = ("weak", "moderate", "strong", "saturated")

_WAVELENGTHS_M = (850e-9, 1064e-9, 1550e-9)
_RECEIVER_DIAMETERS_M = (0.05, 0.10, 0.15, 0.25)
_N_IRRADIANCE = (500, 1000, 2000)
_N_FRAMES = (200, 500, 1000)
_DIMM_NOISE_ARCSEC = (0.03, 0.05, 0.08)

_CN2_FLOOR = 1e-20  # numerical floor, m^(-2/3), well below any physical value
_SIGMA_FLOOR = 1e-12  # numerical floor for a measured variance


@dataclass(frozen=True)
class Scenario:
    """One synthetic path plus its instrument configuration."""

    path: PathGeometry
    suite: SensorSuite
    z_m: np.ndarray
    cn2: np.ndarray

    @property
    def cn2_scintillation_average(self) -> float:
        """Scintillation-kernel weighted path average, m^(-2/3) (the ML target)."""
        return weighted_path_average(
            self.z_m, self.cn2, kind="scintillation", geometry=self.path.geometry
        )

    @property
    def cn2_coherence_average(self) -> float:
        """Coherence-kernel weighted path average, m^(-2/3)."""
        return weighted_path_average(
            self.z_m, self.cn2, kind="coherence", geometry=self.path.geometry
        )

    @property
    def cn2_plain_average(self) -> float:
        """Unweighted path average ``(1/L) int Cn2 dz``, m^(-2/3)."""
        return float(np.trapezoid(self.cn2, self.z_m) / (self.z_m[-1] - self.z_m[0]))


def _shape(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dimensionless along-path shape with unit mean."""
    g1 = rng.normal(0.0, 0.8)
    g2 = rng.normal(0.0, 0.6)
    g3 = rng.normal(0.0, 0.4)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    log_s = (
        g1 * 2.0 * (u - 0.5)
        + g2 * (4.0 * (u - 0.5) ** 2 - 1.0 / 3.0)
        + g3 * np.sin(2.0 * np.pi * u + phase)
    )
    if rng.uniform() < 0.40:
        amp = rng.uniform(0.5, 2.5)
        centre = rng.uniform(0.05, 0.95)
        width = rng.uniform(0.03, 0.15)
        log_s = log_s + amp * np.exp(-0.5 * ((u - centre) / width) ** 2)
    s = np.exp(log_s)
    return s / float(np.trapezoid(s, u))


def _make_scenario(rng: np.random.Generator, n_grid: int) -> Scenario:
    length = float(10.0 ** rng.uniform(np.log10(300.0), np.log10(3000.0)))
    wavelength = float(_WAVELENGTHS_M[rng.integers(len(_WAVELENGTHS_M))])
    path = PathGeometry(length, wavelength, "spherical")
    suite = SensorSuite(
        receiver_diameter_m=float(_RECEIVER_DIAMETERS_M[rng.integers(len(_RECEIVER_DIAMETERS_M))]),
        dimm_subaperture_m=0.06,
        dimm_baseline_m=0.20,
        n_irradiance_samples=int(_N_IRRADIANCE[rng.integers(len(_N_IRRADIANCE))]),
        n_dimm_frames=int(_N_FRAMES[rng.integers(len(_N_FRAMES))]),
        dimm_noise_arcsec=float(_DIMM_NOISE_ARCSEC[rng.integers(len(_DIMM_NOISE_ARCSEC))]),
    )
    z = np.linspace(0.0, length, n_grid)
    level = float(10.0 ** rng.uniform(np.log10(1e-16), np.log10(1e-12)))
    cn2 = level * _shape(z / length, rng)
    return Scenario(path=path, suite=suite, z_m=z, cn2=cn2)


def features_from_measurement(
    path: PathGeometry, suite: SensorSuite, meas: Measurement
) -> np.ndarray:
    """Build the model feature row (see :data:`FEATURE_NAMES`) for one observation.

    Kept public so that a caller with a real instrument reading can build exactly
    the same feature vector the model was trained on.
    """
    s_point = max(meas.sigma_i2_point, 1e-12)
    s_ap = max(meas.sigma_i2_aperture, 1e-12)
    s_l = max(meas.sigma_l2_rad2, _SIGMA_FLOOR)
    s_t = max(meas.sigma_t2_rad2, _SIGMA_FLOOR)

    cn2_weak = max(uniform_cn2_from_beta0_sq(s_point, path), _CN2_FLOOR)

    corrected = meas.sigma_l2_rad2 - suite.dimm_noise_variance_rad2
    if corrected > 0.0:
        r0 = r0_from_dimm_variance(
            corrected, path.wavelength_m, suite.dimm_subaperture_m, suite.dimm_baseline_m
        )
        cn2_dimm = max(cn2_average_from_fried(r0, path), _CN2_FLOOR)
        dimm_valid = 1.0
    else:
        cn2_dimm = _CN2_FLOOR
        dimm_valid = 0.0

    return np.array(
        [
            np.log10(s_point),
            np.log10(s_ap),
            np.log10(s_l),
            np.log10(s_t),
            s_t / s_l,
            np.log10(path.length_m),
            np.log10(path.wavelength_m),
            np.log10(1.0 + meas.aperture_d_sq),
            np.log10(cn2_weak),
            np.log10(cn2_dimm),
            dimm_valid,
            np.log10(float(suite.n_irradiance_samples)),
            np.log10(float(suite.n_dimm_frames)),
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class SyntheticDataset:
    """Feature matrix, target and per-row metadata."""

    x: np.ndarray
    y: np.ndarray
    beta0_sq: np.ndarray
    cn2_scint: np.ndarray
    cn2_coherence: np.ndarray
    cn2_plain: np.ndarray
    path_length_m: np.ndarray
    wavelength_m: np.ndarray
    aperture_d_sq: np.ndarray
    n_irradiance_samples: np.ndarray
    n_dimm_frames: np.ndarray
    receiver_diameter_m: np.ndarray
    dimm_noise_variance_rad2: np.ndarray
    sigma_i2_point: np.ndarray
    sigma_i2_aperture: np.ndarray
    sigma_l2_rad2: np.ndarray

    def __len__(self) -> int:
        return int(self.y.size)

    def regimes(self) -> np.ndarray:
        """Regime label index per row (see :data:`REGIME_NAMES`)."""
        return regime_labels(self.beta0_sq)

    def take(self, index: np.ndarray) -> SyntheticDataset:
        """Return the sub-dataset selected by ``index`` (an integer or bool array)."""
        idx = np.asarray(index)
        fields = {
            k: getattr(self, k)[idx]
            for k in (
                "beta0_sq", "cn2_scint", "cn2_coherence", "cn2_plain", "path_length_m",
                "wavelength_m", "aperture_d_sq", "n_irradiance_samples", "n_dimm_frames",
                "receiver_diameter_m", "dimm_noise_variance_rad2", "sigma_i2_point",
                "sigma_i2_aperture", "sigma_l2_rad2",
            )
        }
        return SyntheticDataset(x=self.x[idx], y=self.y[idx], **fields)


def regime_labels(beta0_sq: np.ndarray) -> np.ndarray:
    """Map ``beta_0^2`` to an index into :data:`REGIME_NAMES`."""
    b = np.asarray(beta0_sq, dtype=float)
    return np.clip(np.searchsorted(np.array(REGIME_EDGES[1:]), b, side="right"), 0, 3)


def generate_dataset(
    n_scenarios: int = 5000, seed: int = 20260829, n_grid: int = 201
) -> SyntheticDataset:
    """Generate the seeded synthetic dataset.

    Parameters
    ----------
    n_scenarios
        Number of independent path/observation scenarios (one row each).
    seed
        Master seed; the whole dataset is a deterministic function of it.
    n_grid
        Number of path samples per profile (odd, for Simpson's rule).  The
        quadrature error of the scintillation kernel at 201 points is 9.2e-06
        relative -- three orders of magnitude below the measurement noise
        (``validation/validate_forward_models.py``).
    """
    n = check_count("n_scenarios", n_scenarios, minimum=10)
    if n_grid < 3 or n_grid % 2 == 0:
        raise ValueError(f"n_grid must be an odd integer >= 3, got {n_grid}")
    rng = np.random.default_rng(seed)

    rows: list[np.ndarray] = []
    y: list[float] = []
    meta: dict[str, list[float]] = {
        k: []
        for k in (
            "beta0_sq",
            "cn2_scint",
            "cn2_coherence",
            "cn2_plain",
            "path_length_m",
            "wavelength_m",
            "aperture_d_sq",
            "n_irradiance_samples",
            "n_dimm_frames",
            "receiver_diameter_m",
            "dimm_noise_variance_rad2",
            "sigma_i2_point",
            "sigma_i2_aperture",
            "sigma_l2_rad2",
        )
    }
    for _ in range(n):
        scen = _make_scenario(rng, n_grid)
        meas = simulate_measurement(scen.z_m, scen.cn2, scen.path, scen.suite, rng)
        rows.append(features_from_measurement(scen.path, scen.suite, meas))
        y.append(np.log10(scen.cn2_scintillation_average))
        meta["beta0_sq"].append(meas.true_beta0_sq)
        meta["cn2_scint"].append(scen.cn2_scintillation_average)
        meta["cn2_coherence"].append(scen.cn2_coherence_average)
        meta["cn2_plain"].append(scen.cn2_plain_average)
        meta["path_length_m"].append(scen.path.length_m)
        meta["wavelength_m"].append(scen.path.wavelength_m)
        meta["aperture_d_sq"].append(meas.aperture_d_sq)
        meta["n_irradiance_samples"].append(scen.suite.n_irradiance_samples)
        meta["n_dimm_frames"].append(scen.suite.n_dimm_frames)
        meta["receiver_diameter_m"].append(scen.suite.receiver_diameter_m)
        meta["dimm_noise_variance_rad2"].append(scen.suite.dimm_noise_variance_rad2)
        meta["sigma_i2_point"].append(meas.sigma_i2_point)
        meta["sigma_i2_aperture"].append(meas.sigma_i2_aperture)
        meta["sigma_l2_rad2"].append(meas.sigma_l2_rad2)

    return SyntheticDataset(
        x=np.asarray(rows, dtype=float),
        y=np.asarray(y, dtype=float),
        **{k: np.asarray(v, dtype=float) for k, v in meta.items()},
    )
