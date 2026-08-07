"""Seeded synthetic dataset generator for CnCast.

READ ``DATASET_CARD.md`` BEFORE USING ANY NUMBER PRODUCED BY THIS MODULE.

Every profile produced here is generated from the Hufnagel-Valley baseline
(:func:`cncast.baselines.hufnagel_valley`) with parameters driven by surface
meteorology through **heuristic, physically-motivated but unvalidated**
relations, plus smooth random perturbations.  There is no radiosonde,
scintillometer, SCIDAR or thermosonde data anywhere in this product.  A model
trained on this data learns the generator, not the atmosphere.

Generation is fully deterministic given the master seed: the same seed produces
bit-identical scenarios on any machine with the same NumPy version, because all
randomness comes from a single ``numpy.random.Generator`` (PCG64) consumed in a
fixed order.

Units
-----
surface_temp_c          degrees Celsius
surface_wind_m_s        m/s at the standard 10 m anemometer height
relative_humidity_pct   per cent, 0-100
hour_of_day             local solar hours, 0 <= t < 24
day_of_year             1-365 (northern-hemisphere seasonality assumed)
Cn^2                    m^-2/3
altitude                metres above the site
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import hufnagel_valley, rms_high_altitude_wind

__all__ = [
    "ALTITUDE_MAX_M",
    "ALTITUDE_MIN_M",
    "FEATURE_NAMES",
    "Scenario",
    "build_table",
    "default_altitude_grid",
    "generate_scenarios",
    "met_features",
    "profile_cn2",
    "scenario_features",
    "split_scenarios",
]

ALTITUDE_MIN_M: float = 5.0
"""Lowest altitude modelled (m).  Below this the surface layer is not resolved."""

ALTITUDE_MAX_M: float = 20_000.0
"""Highest altitude modelled (m), matching the H-V quoted validity range."""

FEATURE_NAMES: tuple[str, ...] = (
    "surface_temp_c",
    "surface_wind_m_s",
    "relative_humidity_pct",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
    "log10_altitude_m",
)
"""Model input columns, in order."""

_N_NOISE_MODES = 3


@dataclass(frozen=True)
class Scenario:
    """One synthetic atmospheric case.

    The first five fields are the *observable* surface meteorology (the model
    inputs).  The remaining fields are *latent*: they drive the generator but
    are never shown to the learned model, and they are the reason the model
    cannot be perfect and must emit an interval.
    """

    surface_temp_c: float
    surface_wind_m_s: float
    relative_humidity_pct: float
    hour_of_day: float
    day_of_year: int
    # --- latent generator state ---
    ground_cn2: float
    rms_wind_m_s: float
    layer_height_m: float
    layer_strength: float
    layer_width_dex: float
    noise_amp_dex: tuple[float, ...] = field(default=())
    noise_phase_rad: tuple[float, ...] = field(default=())


def _solar_elevation_factor(hour_of_day: ArrayLike) -> NDArray[np.float64]:
    """Crude daytime heating factor: 0 at night, 1 at local noon.

    ``max(0, sin(pi (t - 6) / 12))`` - a half-sine over 06:00-18:00.  This is a
    stand-in for solar elevation with no latitude or date dependence; it is a
    modelling convenience, not a solar-position algorithm.
    """
    t = np.asarray(hour_of_day, dtype=float)
    return np.maximum(0.0, np.sin(np.pi * (t - 6.0) / 12.0))


def _seasonal_warmth(day_of_year: ArrayLike) -> NDArray[np.float64]:
    """+1 near day 183 (boreal summer), -1 near day 1 (boreal winter)."""
    d = np.asarray(day_of_year, dtype=float)
    return np.cos(2.0 * np.pi * (d - 183.0) / 365.0)


def generate_scenarios(n_scenarios: int, seed: int = 20260807) -> list[Scenario]:
    """Draw ``n_scenarios`` synthetic surface-meteorology cases with latent state.

    Parameters
    ----------
    n_scenarios : int
        Number of independent cases (>= 1).
    seed : int
        Master seed for ``numpy.random.default_rng``.  Fixed seed => identical
        output.

    Returns
    -------
    list of Scenario

    Notes
    -----
    Sampling priors (uniform unless stated), chosen to span a temperate
    continental site and NOT derived from any measurement campaign:

    ==========================  ==========================
    surface_temp_c              U(-10, 38)
    surface_wind_m_s            U(0, 14)
    relative_humidity_pct       U(10, 95)
    hour_of_day                 U(0, 24)
    day_of_year                 integer U(1, 365)
    ==========================  ==========================

    Latent mapping (all coefficients are hand-chosen heuristics):

    * ``log10 ground_cn2 = log10(2.0e-15) + 0.60*solar + 0.020*(T - 15)
      + 0.015*(w - 5) - 0.0040*(RH - 50) + 0.15*Z``, ``Z ~ N(0,1)``.
      Rationale: daytime surface heating drives the convective boundary layer
      (the dominant diurnal signal in measured Cn^2, e.g. Beland 1993 Fig. 2-6);
      warmer surfaces sustain larger temperature structure; wind shear adds
      mechanical production; high humidity accompanies cloud and reduced
      surface heating.  Signs follow the literature, magnitudes do not come
      from any fit.
    * ``rms_wind_m_s = rms_high_altitude_wind(w) * (1 - 0.25 * seasonal_warmth)``
      i.e. a 25 % stronger jet stream in winter than in summer, then clipped to
      [8, 45] m/s.
    * One elevated layer: log-uniform height in [800, 8000] m; peak strength
      ``s = exp(N(0, 0.7))`` clipped to [0.1, 6.0] and applied as the multiplier
      ``1 + s`` (Cn^2 at the layer peak is multiplied by 1.1 to 7.0); Gaussian in
      log10 h with width U(0.08, 0.30) dex.  Represents a residual / shear
      layer; real layers are thinner and more numerous.
    * Three smooth multiplicative modes in normalised log-altitude with
      amplitudes N(0, 0.12) dex and uniform phases - the irreducible scatter.

    Raises
    ------
    ValueError
        If ``n_scenarios < 1``.
    """
    n = int(n_scenarios)
    if n < 1:
        raise ValueError(f"n_scenarios must be >= 1 (got {n_scenarios!r}).")
    rng = np.random.default_rng(int(seed))

    temp = rng.uniform(-10.0, 38.0, n)
    wind = rng.uniform(0.0, 14.0, n)
    rh = rng.uniform(10.0, 95.0, n)
    hour = rng.uniform(0.0, 24.0, n)
    doy = rng.integers(1, 366, n)

    solar = _solar_elevation_factor(hour)
    warmth = _seasonal_warmth(doy)

    z = rng.standard_normal(n)
    log_a = (
        np.log10(2.0e-15)
        + 0.60 * solar
        + 0.020 * (temp - 15.0)
        + 0.015 * (wind - 5.0)
        - 0.0040 * (rh - 50.0)
        + 0.15 * z
    )
    ground_cn2 = 10.0**log_a

    rms_wind = np.array([rms_high_altitude_wind(float(w)) for w in wind])
    rms_wind = np.clip(rms_wind * (1.0 - 0.25 * warmth), 8.0, 45.0)

    layer_h = 10.0 ** rng.uniform(np.log10(800.0), np.log10(8000.0), n)
    layer_strength = np.expm1(rng.normal(0.0, 0.7, n)) + 1.0
    layer_strength = np.clip(layer_strength, 0.1, 6.0)
    layer_width = rng.uniform(0.08, 0.30, n)

    noise_amp = rng.normal(0.0, 0.12, (n, _N_NOISE_MODES))
    noise_phase = rng.uniform(0.0, 2.0 * np.pi, (n, _N_NOISE_MODES))

    return [
        Scenario(
            surface_temp_c=float(temp[i]),
            surface_wind_m_s=float(wind[i]),
            relative_humidity_pct=float(rh[i]),
            hour_of_day=float(hour[i]),
            day_of_year=int(doy[i]),
            ground_cn2=float(ground_cn2[i]),
            rms_wind_m_s=float(rms_wind[i]),
            layer_height_m=float(layer_h[i]),
            layer_strength=float(layer_strength[i]),
            layer_width_dex=float(layer_width[i]),
            noise_amp_dex=tuple(float(x) for x in noise_amp[i]),
            noise_phase_rad=tuple(float(x) for x in noise_phase[i]),
        )
        for i in range(n)
    ]


def profile_cn2(scenario: Scenario, h_m: ArrayLike) -> NDArray[np.float64]:
    r"""Ground-truth Cn^2 profile of one synthetic scenario.

    .. math::

        C_n^2(h) = \mathrm{HV}(h; v, A)\;
                   \bigl[1 + s\,e^{-\frac{1}{2}((\log_{10}h - \log_{10}h_L)/\sigma)^2}\bigr]\;
                   10^{\sum_k a_k \sin(2\pi (k{+}1) u + \phi_k)}

    with ``u`` the altitude normalised onto [0, 1] in log10 between
    ``ALTITUDE_MIN_M`` and ``ALTITUDE_MAX_M``.

    Parameters
    ----------
    scenario : Scenario
        Case, including latent state.
    h_m : array_like
        Altitudes in metres (> 0).

    Returns
    -------
    ndarray
        Cn^2 in m^-2/3.

    Notes
    -----
    This is the *definition* of truth in this product.  It is a generative
    process, not a measurement.  Accuracy against it says nothing about
    accuracy against the sky.
    """
    h = np.asarray(h_m, dtype=float)
    if np.any(h <= 0.0) or not np.all(np.isfinite(h)):
        raise ValueError("h_m must be finite and > 0 m for the log-altitude parameterisation.")
    base = hufnagel_valley(h, scenario.rms_wind_m_s, scenario.ground_cn2)
    logh = np.log10(h)
    layer = 1.0 + scenario.layer_strength * np.exp(
        -0.5 * ((logh - np.log10(scenario.layer_height_m)) / scenario.layer_width_dex) ** 2
    )
    span = np.log10(ALTITUDE_MAX_M) - np.log10(ALTITUDE_MIN_M)
    u = (logh - np.log10(ALTITUDE_MIN_M)) / span
    wiggle = np.zeros_like(h)
    modes = zip(scenario.noise_amp_dex, scenario.noise_phase_rad, strict=True)
    for k, (amp, phase) in enumerate(modes):
        wiggle = wiggle + amp * np.sin(2.0 * np.pi * (k + 1) * u + phase)
    return np.asarray(base * layer * 10.0**wiggle, dtype=float)


def met_features(
    surface_temp_c: float,
    surface_wind_m_s: float,
    relative_humidity_pct: float,
    hour_of_day: float,
    day_of_year: int,
    h_m: ArrayLike,
) -> NDArray[np.float64]:
    """Feature matrix from raw surface meteorology over altitudes ``h_m``.

    Parameters
    ----------
    surface_temp_c : float
        Surface air temperature, degrees Celsius (-90 to 60 accepted).
    surface_wind_m_s : float
        Surface wind speed, m/s (>= 0).
    relative_humidity_pct : float
        Relative humidity, per cent (0-100).
    hour_of_day : float
        Local solar time, hours, 0 <= t < 24.
    day_of_year : int
        1-365.
    h_m : array_like
        Altitudes in metres (> 0).

    Returns
    -------
    ndarray, shape (len(h_m), 8)
        Columns in the order of :data:`FEATURE_NAMES`.

    Raises
    ------
    ValueError
        On any physically impossible input.  These bounds are physical limits,
        not the training domain; see :meth:`cncast.model.CnCastModel.predict`
        for the extrapolation flag.
    """
    t = float(surface_temp_c)
    w = float(surface_wind_m_s)
    rh = float(relative_humidity_pct)
    hour = float(hour_of_day)
    doy = int(day_of_year)
    if not np.isfinite(t) or not (-90.0 <= t <= 60.0):
        raise ValueError(f"surface_temp_c must be finite and within [-90, 60] C (got {t!r}).")
    if not np.isfinite(w) or w < 0.0:
        raise ValueError(f"surface_wind_m_s must be finite and >= 0 m/s (got {w!r}).")
    if not np.isfinite(rh) or not (0.0 <= rh <= 100.0):
        raise ValueError(f"relative_humidity_pct must be within [0, 100] % (got {rh!r}).")
    if not np.isfinite(hour) or not (0.0 <= hour < 24.0):
        raise ValueError(f"hour_of_day must satisfy 0 <= t < 24 (got {hour!r}).")
    if not (1 <= doy <= 365):
        raise ValueError(f"day_of_year must be within [1, 365] (got {day_of_year!r}).")

    h = np.atleast_1d(np.asarray(h_m, dtype=float))
    if h.size == 0:
        raise ValueError("h_m must contain at least one altitude.")
    if np.any(h <= 0.0) or not np.all(np.isfinite(h)):
        raise ValueError("h_m must be finite and > 0 m.")
    n = h.size
    hour_ang = 2.0 * np.pi * hour / 24.0
    doy_ang = 2.0 * np.pi * doy / 365.0
    return np.column_stack(
        [
            np.full(n, t),
            np.full(n, w),
            np.full(n, rh),
            np.full(n, np.sin(hour_ang)),
            np.full(n, np.cos(hour_ang)),
            np.full(n, np.sin(doy_ang)),
            np.full(n, np.cos(doy_ang)),
            np.log10(h),
        ]
    )


def scenario_features(scenario: Scenario, h_m: ArrayLike) -> NDArray[np.float64]:
    """Build the model feature matrix for one scenario over altitudes ``h_m``.

    Returns
    -------
    ndarray, shape (len(h_m), 8)
        Columns in the order of :data:`FEATURE_NAMES`.
    """
    return met_features(
        scenario.surface_temp_c,
        scenario.surface_wind_m_s,
        scenario.relative_humidity_pct,
        scenario.hour_of_day,
        scenario.day_of_year,
        h_m,
    )


def default_altitude_grid(n_points: int = 24) -> NDArray[np.float64]:
    """Log-spaced evaluation grid from ``ALTITUDE_MIN_M`` to ``ALTITUDE_MAX_M``.

    Parameters
    ----------
    n_points : int
        Number of grid points (>= 2).
    """
    if int(n_points) < 2:
        raise ValueError("n_points must be >= 2.")
    return np.geomspace(ALTITUDE_MIN_M, ALTITUDE_MAX_M, int(n_points))


def build_table(
    scenarios: list[Scenario],
    n_altitudes: int = 24,
    seed: int | None = None,
    grid: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Flatten scenarios into a (features, target, group) training table.

    Parameters
    ----------
    scenarios : list of Scenario
    n_altitudes : int
        Altitudes sampled per scenario.
    seed : int or None
        If given, altitudes are drawn log-uniform per scenario with this seed
        (recommended for training: it prevents the tree ensemble from learning a
        staircase locked to a fixed grid).  If ``None`` and ``grid`` is ``None``,
        the shared :func:`default_altitude_grid` is used.
    grid : array_like or None
        Explicit shared altitude grid; overrides ``seed``.

    Returns
    -------
    x : ndarray (n_rows, 8)
    y : ndarray (n_rows,)
        Target = ``log10(Cn^2)``.  Working in log10 is essential: Cn^2 spans
        seven decades, so a linear-space loss would only ever fit the ground.
    groups : ndarray (n_rows,)
        Scenario index of each row - use it to split by scenario, never by row.
    """
    if not scenarios:
        raise ValueError("scenarios must be a non-empty list.")
    if int(n_altitudes) < 1:
        raise ValueError("n_altitudes must be >= 1.")
    rng = None if seed is None else np.random.default_rng(int(seed))
    shared = None
    if grid is not None:
        shared = np.atleast_1d(np.asarray(grid, dtype=float))
    elif rng is None:
        shared = default_altitude_grid(int(n_altitudes))

    xs, ys, gs = [], [], []
    lo, hi = np.log10(ALTITUDE_MIN_M), np.log10(ALTITUDE_MAX_M)
    for i, sc in enumerate(scenarios):
        h = shared if shared is not None else 10.0 ** rng.uniform(lo, hi, int(n_altitudes))
        xs.append(scenario_features(sc, h))
        ys.append(np.log10(profile_cn2(sc, h)))
        gs.append(np.full(h.size, i, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(gs)


def split_scenarios(
    n_scenarios: int, test_fraction: float = 0.25, seed: int = 4242
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Scenario-level (grouped) train/test split.

    Splitting by *scenario* rather than by row is mandatory here: rows from one
    profile share all five meteorological features and the same latent layer, so
    a row-level split would leak the answer across the boundary and inflate
    every metric.

    Parameters
    ----------
    n_scenarios : int
    test_fraction : float
        In (0, 1).
    seed : int
        Permutation seed.

    Returns
    -------
    (train_idx, test_idx) : tuple of ndarray
    """
    n = int(n_scenarios)
    if n < 2:
        raise ValueError("Need at least 2 scenarios to split.")
    f = float(test_fraction)
    if not 0.0 < f < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1) (got {test_fraction!r}).")
    perm = np.random.default_rng(int(seed)).permutation(n)
    n_test = max(1, int(round(f * n)))
    if n_test >= n:
        raise ValueError("test_fraction leaves no training scenarios.")
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])
