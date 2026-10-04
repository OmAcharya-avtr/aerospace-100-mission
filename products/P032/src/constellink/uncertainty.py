"""Uncertainty analysis for contact windows and link capacity (Level 3).

Three distinct error sources are treated separately, because they do not
combine and reporting a single "accuracy" figure would hide all three.

1. Propagation (input) uncertainty
----------------------------------
A TLE carries no covariance.  The honest way to quantify its effect on a
contact window is to perturb the element set by a stated amount and see how
much the window edges move.  :func:`window_edge_sensitivity` perturbs the
mean anomaly -- the along-track direction, which dominates TLE error growth --
by an angle equivalent to a stated along-track position error, and reports the
resulting shift in the window edges.  The relation used to convert a position
error to a mean-anomaly perturbation is the circular-orbit arc length,
``dM = ds / a`` [rad], exact for a circular orbit (Vallado 2013,
"Fundamentals of Astrodynamics and Applications", 4th ed., Ch. 2).

The along-track error itself is an INPUT, not a result of this package.  The
caller must supply a figure appropriate to their element set's age; published
discussions of TLE error growth are Vallado & Cefola 2012, IAC-12.C1.6.7, and
Kelso 2007, AAS 07-127.  This package asserts no error magnitude of its own.

2. Discretisation (method) uncertainty
--------------------------------------
Window edges are bisected to ``refine_tol_s``, so the edge error against the
propagator is bounded by that tolerance.  The grid step is a separate matter:
a window can be MISSED entirely if it is shorter than the step.
:func:`grid_step_convergence` recomputes the windows of one link at several
grid steps and reports the window count and total duration at each, which
exposes both effects as data rather than as an assertion.

3. Parameter uncertainty in the link budget
-------------------------------------------
:func:`monte_carlo_capacity` propagates stated distributions over terminal
parameters through the capacity model and returns the rate percentiles.  This
is plain Monte Carlo with a fixed seed; the number of draws needed for a given
percentile precision follows the usual order-statistics argument, and the
standard error of the mean is reported alongside so the draw count can be
judged.

4. Verification-statistic uncertainty
-------------------------------------
For the predictors, :func:`~constellink.metrics.bootstrap_ci` gives percentile
intervals on the Brier score and its decomposition terms, and the ensemble
spread of :class:`~constellink.availability.LinkAvailabilityModel` is the
epistemic component.  See that module for what the spread does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sgp4.api import WGS72, Satrec

from .capacity import OpticalTerminal, RfTerminal, optical_link, rf_link
from .constellation import Constellation, GroundStation, Satellite
from .contacts import contact_windows_ground, contact_windows_isl

__all__ = [
    "EdgeSensitivity",
    "window_edge_sensitivity",
    "GridConvergence",
    "grid_step_convergence",
    "CapacityDistribution",
    "monte_carlo_capacity",
    "perturb_mean_anomaly",
]


def perturb_mean_anomaly(sat: Satellite, delta_m_rad: float) -> Satellite:
    """Return a copy of ``sat`` with its mean anomaly shifted by ``delta_m_rad``.

    Rebuilds the ``Satrec`` through ``sgp4init`` from the original satellite's
    own mean elements, so no element is altered except the mean anomaly.  The
    gravity model is fixed to WGS72, which is the model SGP4 element sets are
    defined against (Vallado et al. 2006, AIAA 2006-6753).
    """
    src = sat.satrec
    out = Satrec()
    out.sgp4init(
        WGS72, "i", src.satnum,
        (src.jdsatepoch - 2433281.5) + src.jdsatepochF,
        src.bstar, src.ndot, src.nddot, src.ecco, src.argpo, src.inclo,
        float((src.mo + delta_m_rad) % (2.0 * np.pi)),
        src.no_kozai, src.nodeo,
    )
    return Satellite(name=f"{sat.name}+dM", satrec=out,
                     max_epoch_age_days=sat.max_epoch_age_days)


@dataclass(frozen=True)
class EdgeSensitivity:
    """Window-edge shift under an along-track perturbation.

    Attributes
    ----------
    along_track_error_km : the applied along-track error [km].
    delta_m_rad : the equivalent mean-anomaly perturbation [rad].
    n_windows_nominal, n_windows_perturbed : window counts.
    d_t_open_s, d_t_close_s : signed edge shifts of the matched windows [s].
    d_duration_s : signed duration changes [s].
    """

    along_track_error_km: float
    delta_m_rad: float
    n_windows_nominal: int
    n_windows_perturbed: int
    d_t_open_s: np.ndarray
    d_t_close_s: np.ndarray
    d_duration_s: np.ndarray

    @property
    def max_abs_edge_shift_s(self) -> float:
        """Largest absolute edge shift over the matched windows [s]."""
        if self.d_t_open_s.size == 0:
            return float("nan")
        return float(max(np.abs(self.d_t_open_s).max(), np.abs(self.d_t_close_s).max()))


def window_edge_sensitivity(const: Constellation, sat_name: str,
                            station: GroundStation,
                            t0: datetime, t1: datetime, step_s: float,
                            along_track_error_km: float,
                            max_range_km: float = 3000.0) -> EdgeSensitivity:
    """Shift of a ground link's window edges under an along-track error.

    Parameters
    ----------
    const : the constellation (must contain ``sat_name``).
    sat_name : the satellite to perturb.
    station : the ground station.
    t0, t1, step_s : horizon and grid step for the contact scan.
    along_track_error_km : along-track position error to apply [km], > 0.
    max_range_km : terminal range limit passed to the contact scan [km].

    Windows are matched between the nominal and perturbed runs by nearest
    mid-time; unmatched windows are reported through the two counts and are
    excluded from the shift arrays.
    """
    if along_track_error_km <= 0.0:
        raise ValueError(
            f"along_track_error_km must be > 0, got {along_track_error_km}")
    sat = const.satellite(sat_name)
    a_km = _semi_major_axis_km(sat)
    delta_m = along_track_error_km / a_km
    pert = perturb_mean_anomaly(sat, delta_m)

    eph_nom = Constellation([sat]).ephemeris(t0, t1, step_s)
    eph_per = Constellation([pert]).ephemeris(t0, t1, step_s)
    w_nom = contact_windows_ground(eph_nom, sat, station, max_range_km=max_range_km)
    w_per = contact_windows_ground(eph_per, pert, station, max_range_km=max_range_km)

    d_open, d_close, d_dur = [], [], []
    used: set[int] = set()
    for wn in w_nom:
        best, best_gap = -1, None
        for j, wp in enumerate(w_per):
            if j in used:
                continue
            gap = abs((wp.mid_time - wn.mid_time).total_seconds())
            if best_gap is None or gap < best_gap:
                best, best_gap = j, gap
        if best < 0 or best_gap is None or best_gap > 0.5 * wn.duration_s + step_s:
            continue
        used.add(best)
        wp = w_per[best]
        d_open.append((wp.t_open - wn.t_open).total_seconds())
        d_close.append((wp.t_close - wn.t_close).total_seconds())
        d_dur.append(wp.duration_s - wn.duration_s)
    return EdgeSensitivity(
        along_track_error_km=along_track_error_km, delta_m_rad=float(delta_m),
        n_windows_nominal=len(w_nom), n_windows_perturbed=len(w_per),
        d_t_open_s=np.asarray(d_open), d_t_close_s=np.asarray(d_close),
        d_duration_s=np.asarray(d_dur))


def _semi_major_axis_km(sat: Satellite) -> float:
    """Semi-major axis [km] from the Satrec's own mean elements.

    ``Satrec.a`` is in Earth radii in the SGP4 internal unit system; converting
    with the WGS72 Earth radius of 6378.135 km that SGP4 itself uses (Vallado
    et al. 2006) keeps this consistent with the propagator.
    """
    return float(sat.satrec.a) * 6378.135


@dataclass(frozen=True)
class GridConvergence:
    """Window count and total duration against contact-scan grid step."""

    step_s: np.ndarray
    n_windows: np.ndarray
    total_duration_s: np.ndarray

    def format_table(self) -> str:
        """Fixed-width table (str)."""
        lines = [f"{'step [s]':>10}{'n windows':>12}{'total [s]':>14}", "-" * 36]
        for s, n, d in zip(self.step_s, self.n_windows,
                           self.total_duration_s, strict=True):
            lines.append(f"{s:>10.1f}{int(n):>12d}{d:>14.3f}")
        return "\n".join(lines)


def grid_step_convergence(const: Constellation, sat_a: str, sat_b: str,
                          t0: datetime, t1: datetime,
                          steps_s: list[float],
                          max_range_km: float = 5000.0) -> GridConvergence:
    """Recompute one ISL's windows at several grid steps.

    A step that misses short windows shows up as a lower count and a shorter
    total duration.  ``steps_s`` must be positive and is used as given (not
    sorted), so the output order matches the input.
    """
    if not steps_s:
        raise ValueError("steps_s must be non-empty")
    if any(s <= 0.0 for s in steps_s):
        raise ValueError("every entry of steps_s must be > 0")
    a = const.satellite(sat_a)
    b = const.satellite(sat_b)
    counts, totals = [], []
    for step in steps_s:
        eph = Constellation([a, b]).ephemeris(t0, t1, step)
        ws = contact_windows_isl(eph, a, b, max_range_km=max_range_km)
        counts.append(len(ws))
        totals.append(sum(w.duration_s for w in ws))
    return GridConvergence(step_s=np.asarray(steps_s, dtype=float),
                           n_windows=np.asarray(counts, dtype=int),
                           total_duration_s=np.asarray(totals, dtype=float))


@dataclass(frozen=True)
class CapacityDistribution:
    """Monte Carlo distribution of an achievable rate.

    Attributes
    ----------
    samples_bps : the draws [bit/s].
    mean_bps, std_bps : sample mean and standard deviation [bit/s].
    sem_bps : standard error of the mean [bit/s], ``std / sqrt(n)``.
    percentiles : mapping of percentile -> value [bit/s].
    n_draws, seed : provenance.
    """

    samples_bps: np.ndarray
    mean_bps: float
    std_bps: float
    sem_bps: float
    percentiles: dict[float, float]
    n_draws: int
    seed: int

    def format_summary(self) -> str:
        """Human-readable summary (str)."""
        lines = [f"Monte Carlo capacity: n={self.n_draws}, seed={self.seed}",
                 (f"  mean  {self.mean_bps / 1e6:12.3f} Mbit/s "
                  f"(sem {self.sem_bps / 1e6:.3f})"),
                 f"  stdev {self.std_bps / 1e6:12.3f} Mbit/s"]
        for q in sorted(self.percentiles):
            lines.append(f"  p{q:<5g}{self.percentiles[q] / 1e6:12.3f} Mbit/s")
        return "\n".join(lines)


def monte_carlo_capacity(range_km: float,
                         terminal: RfTerminal | OpticalTerminal,
                         sigmas: dict[str, float],
                         n_draws: int = 4000, seed: int = 0,
                         percentiles: tuple[float, ...] = (1.0, 5.0, 50.0, 95.0, 99.0),
                         ) -> CapacityDistribution:
    """Propagate parameter uncertainty through a capacity model by Monte Carlo.

    Parameters
    ----------
    range_km : nominal range [km], > 0.
    terminal : an :class:`~constellink.capacity.RfTerminal` or
        :class:`~constellink.capacity.OpticalTerminal`.
    sigmas : per-field 1-sigma Gaussian standard deviations, keyed by the
        terminal's field names, in that field's own units.  Fields absent from
        the dict are held at their nominal value.  Every sigma must be >= 0.
        Draws that violate a field's domain (for example a negative loss) are
        reflected to the domain boundary, and the count of reflections is NOT
        hidden: it is reported in the returned ``samples_bps`` only implicitly,
        so keep sigmas small relative to the nominal values.
    n_draws : number of draws, >= 100.
    seed : master seed.
    percentiles : percentiles to report, each in (0, 100).

    All draws are independent; correlations between terminal parameters are
    NOT modelled, which will generally make the spread wider than reality for
    positively correlated parameters and narrower for negatively correlated
    ones.
    """
    if range_km <= 0.0:
        raise ValueError(f"range_km must be > 0, got {range_km}")
    if n_draws < 100:
        raise ValueError(f"n_draws must be >= 100, got {n_draws}")
    if any(v < 0.0 for v in sigmas.values()):
        raise ValueError("every sigma must be >= 0")
    valid_fields = set(vars(terminal).keys())
    unknown = set(sigmas) - valid_fields
    if unknown:
        raise KeyError(
            f"unknown terminal fields in sigmas: {sorted(unknown)}; "
            f"available: {sorted(valid_fields)}")
    rng = np.random.default_rng(seed)
    is_rf = isinstance(terminal, RfTerminal)
    nominal = dict(vars(terminal))
    positive_only = {"tx_power_w", "wavelength_m", "beam_divergence_full_rad",
                     "rx_aperture_diameter_m", "photons_per_bit",
                     "frequency_hz", "bandwidth_hz"}
    non_negative = {"tx_loss_db", "other_loss_db", "margin_db",
                    "pointing_error_rad", "tx_optics_loss_db", "rx_optics_loss_db"}
    samples = np.empty(n_draws)
    for i in range(n_draws):
        kwargs = dict(nominal)
        for field, sigma in sigmas.items():
            if sigma == 0.0:
                continue
            val = nominal[field] + rng.normal(0.0, sigma)
            if field in positive_only:
                val = abs(val) if val != 0.0 else nominal[field]
            elif field in non_negative:
                val = abs(val)
            kwargs[field] = float(val)
        if is_rf:
            samples[i] = rf_link(range_km, RfTerminal(**kwargs)).achievable_rate_bps
        else:
            samples[i] = optical_link(range_km,
                                      OpticalTerminal(**kwargs)).achievable_rate_bps
    std = float(samples.std(ddof=1))
    return CapacityDistribution(
        samples_bps=samples, mean_bps=float(samples.mean()), std_bps=std,
        sem_bps=std / np.sqrt(n_draws),
        percentiles={float(q): float(np.percentile(samples, q)) for q in percentiles},
        n_draws=n_draws, seed=seed)
