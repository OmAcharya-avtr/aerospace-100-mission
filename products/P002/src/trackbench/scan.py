"""Acquisition scan-pattern generators and acquisition-time statistics.

Scope
-----
Open-loop spatial acquisition for free-space optical links: the transmitter
sweeps its beam over an angular uncertainty region until the receiver falls
inside the beam footprint. Spiral and raster scans are the standard patterns
used in laser-communication pointing-acquisition-tracking (PAT) systems; see
the survey by Kaymak et al. (2018), "A Survey on Acquisition, Tracking, and
Pointing Mechanisms for Mobile Free-Space Optical Communications", IEEE
Communications Surveys & Tutorials 20(2), and Hemmati (ed., 2006), "Deep Space
Optical Communications", ch. on acquisition/tracking, for the concepts used
here (spiral scan of a Gaussian uncertainty cone, track-spacing overlap,
per-dwell detection probability).

Conventions and units
---------------------
- All angles are in radians (angular offsets from boresight, small-angle
  approximation: the 2-D scan plane is the tangent plane of the pointing
  sphere; valid for uncertainty half-cones < ~0.05 rad).
- The pointing uncertainty of the target is an isotropic 2-D Gaussian with
  standard deviation ``sigma`` per axis, so the radial offset is
  Rayleigh-distributed (textbook result, e.g. Papoulis & Pillai 2002,
  "Probability, Random Variables and Stochastic Processes").
- The beam footprint is a disc of angular radius ``beam_radius`` (half-angle).
  Detection occurs on a dwell if the target lies inside the footprint and an
  independent Bernoulli trial with probability ``p_dwell`` succeeds.

Analytic expected-acquisition-time expressions here use the uniform-coverage
approximation (scan sweeps area at constant rate ``track_spacing *
scan_speed``); the derivations are internal (documented in the docstrings and
in validation/VALIDATION.md) and are checked against Monte Carlo in
``validation/v2_acquisition_time.py``. Comparable spiral-scan acquisition
statistics appear throughout the laser-comm PAT literature (see the Kaymak
et al. 2018 survey, sec. on acquisition); no page-specific formula is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "GaussianUncertainty",
    "ScanPattern",
    "spiral_scan",
    "raster_scan",
    "coverage_fraction",
    "simulate_acquisition",
    "expected_acquisition_time_spiral",
]


class GaussianUncertainty:
    """Isotropic 2-D Gaussian angular uncertainty region.

    Parameters
    ----------
    sigma : float
        Per-axis standard deviation of the target angular offset [rad].
        Must be > 0. Small-angle (flat tangent plane) assumption: valid for
        sigma < ~0.01 rad.

    Notes
    -----
    Radial offset r = sqrt(x^2 + y^2) is Rayleigh(sigma):
    CDF  P(r <= R) = 1 - exp(-R^2 / (2 sigma^2))     [standard result]
    Quantile r(p)  = sigma * sqrt(-2 ln(1 - p))
    """

    def __init__(self, sigma: float) -> None:
        if not (isinstance(sigma, (int, float)) and math.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be a finite positive number [rad], got {sigma!r}")
        self.sigma = float(sigma)

    def prob_within(self, radius: float) -> float:
        """Probability that the target lies within ``radius`` [rad] of the mean."""
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius!r}")
        return 1.0 - math.exp(-(radius**2) / (2.0 * self.sigma**2))

    def containment_radius(self, p: float) -> float:
        """Radius [rad] containing probability mass ``p`` (0 < p < 1)."""
        if not 0.0 < p < 1.0:
            raise ValueError(f"containment probability must be in (0, 1), got {p!r}")
        return self.sigma * math.sqrt(-2.0 * math.log(1.0 - p))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` target offsets, shape (n, 2) [rad]."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n!r}")
        return rng.normal(0.0, self.sigma, size=(n, 2))


@dataclass
class ScanPattern:
    """A dwell-sampled scan pattern.

    Attributes
    ----------
    points : np.ndarray
        Dwell centres, shape (N, 2) [rad].
    dwell_time : float
        Dwell duration per point [s].
    track_spacing : float
        Cross-track spacing between adjacent scan tracks [rad].
    beam_radius : float
        Beam footprint angular radius used in the design [rad].
    max_radius : float
        Design radius of the covered region [rad].
    kind : str
        "spiral" or "raster".
    """

    points: np.ndarray
    dwell_time: float
    track_spacing: float
    beam_radius: float
    max_radius: float
    kind: str
    times: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.times = (np.arange(len(self.points)) + 1.0) * self.dwell_time

    @property
    def n_points(self) -> int:
        """Number of dwell points."""
        return len(self.points)

    @property
    def scan_time(self) -> float:
        """Total single-pass scan duration [s] = n_points * dwell_time."""
        return self.n_points * self.dwell_time

    @property
    def scan_speed(self) -> float:
        """Mean along-track angular speed [rad/s] (arc step / dwell time)."""
        if self.n_points < 2:
            return 0.0
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return float(np.mean(steps)) / self.dwell_time


def _check_scan_args(
    beam_radius: float, overlap: float, dwell_time: float, step_fraction: float
) -> None:
    if not (math.isfinite(beam_radius) and beam_radius > 0):
        raise ValueError(f"beam_radius must be > 0 [rad], got {beam_radius!r}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
    if not (math.isfinite(dwell_time) and dwell_time > 0):
        raise ValueError(f"dwell_time must be > 0 [s], got {dwell_time!r}")
    if not 0.0 < step_fraction <= 1.0:
        raise ValueError(f"step_fraction must be in (0, 1], got {step_fraction!r}")


def track_spacing(beam_radius: float, overlap: float) -> float:
    """Cross-track spacing s = 2 * beam_radius * (1 - overlap) [rad].

    Adjacent tracks of a scan leave no gap iff s <= 2*beam_radius; the
    overlap factor (0 <= overlap < 1) is margin against along-track motion
    and pointing jitter (standard PAT design practice; see Kaymak et al.
    2018 survey).
    """
    _check_scan_args(beam_radius, overlap, 1.0, 0.5)
    return 2.0 * beam_radius * (1.0 - overlap)


def spiral_scan(
    uncertainty: GaussianUncertainty,
    beam_radius: float,
    overlap: float = 0.25,
    containment: float = 0.995,
    dwell_time: float = 1e-3,
    step_fraction: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
) -> ScanPattern:
    """Archimedean spiral scan covering the uncertainty region.

    The spiral r(phi) = a * phi with a = s / (2*pi) has constant radial pitch
    s = 2*beam_radius*(1-overlap) between successive turns (Archimedean
    property), guaranteeing cross-track coverage. Dwell points are placed at
    constant arc-length steps dl = step_fraction * beam_radius using
    ds = sqrt(r^2 + a^2) dphi (arc-length element of the Archimedean spiral,
    standard calculus result).

    Parameters
    ----------
    uncertainty : GaussianUncertainty
        Target uncertainty region.
    beam_radius : float
        Beam footprint angular radius [rad], > 0.
    overlap : float
        Track overlap factor in [0, 1). Default 0.25.
    containment : float
        Probability mass to cover; sets max radius = Rayleigh quantile.
    dwell_time : float
        Dwell per point [s], > 0.
    step_fraction : float
        Along-track step as a fraction of beam_radius, in (0, 1].
    center : tuple of float
        Scan centre offset [rad].

    Returns
    -------
    ScanPattern
    """
    _check_scan_args(beam_radius, overlap, dwell_time, step_fraction)
    if not 0.0 < containment < 1.0:
        raise ValueError(f"containment must be in (0, 1), got {containment!r}")

    s = track_spacing(beam_radius, overlap)
    a = s / (2.0 * math.pi)
    r_max = uncertainty.containment_radius(containment)
    dl = step_fraction * beam_radius

    phis = [0.0]
    phi = 0.0
    # incremental arc-length stepping: dphi = dl / sqrt(r^2 + a^2)
    while a * phi < r_max:
        phi += dl / math.hypot(a * phi, a)
        phis.append(phi)
    phi_arr = np.asarray(phis)
    r = a * phi_arr
    pts = np.column_stack((r * np.cos(phi_arr), r * np.sin(phi_arr)))
    pts += np.asarray(center, dtype=float)
    return ScanPattern(
        points=pts,
        dwell_time=dwell_time,
        track_spacing=s,
        beam_radius=beam_radius,
        max_radius=r_max,
        kind="spiral",
    )


def raster_scan(
    uncertainty: GaussianUncertainty,
    beam_radius: float,
    overlap: float = 0.25,
    containment: float = 0.995,
    dwell_time: float = 1e-3,
    step_fraction: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
) -> ScanPattern:
    """Serpentine (boustrophedon) raster scan over the square bounding the cone.

    Rows are spaced by s = 2*beam_radius*(1-overlap) and span
    [-r_max, r_max] in both axes, where r_max is the Rayleigh containment
    radius. The raster covers the full square, hence the full uncertainty
    disc, at the cost of scanning low-probability corners (this is the
    standard trade against spiral scans; see Kaymak et al. 2018 survey).
    """
    _check_scan_args(beam_radius, overlap, dwell_time, step_fraction)
    if not 0.0 < containment < 1.0:
        raise ValueError(f"containment must be in (0, 1), got {containment!r}")

    s = track_spacing(beam_radius, overlap)
    r_max = uncertainty.containment_radius(containment)
    dl = step_fraction * beam_radius
    n_rows = int(math.ceil(2.0 * r_max / s)) + 1
    ys = -r_max + s * np.arange(n_rows)
    xs = np.arange(-r_max, r_max + dl, dl)
    rows = []
    for i, y in enumerate(ys):
        x_row = xs if i % 2 == 0 else xs[::-1]
        rows.append(np.column_stack((x_row, np.full_like(x_row, y))))
    pts = np.vstack(rows) + np.asarray(center, dtype=float)
    return ScanPattern(
        points=pts,
        dwell_time=dwell_time,
        track_spacing=s,
        beam_radius=beam_radius,
        max_radius=r_max,
        kind="raster",
    )


def coverage_fraction(
    pattern: ScanPattern,
    uncertainty: GaussianUncertainty,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo probability that a Gaussian-distributed target is covered.

    A target is covered if it lies within ``pattern.beam_radius`` of at least
    one dwell point. Uses a k-d tree nearest-neighbour query.

    Returns the covered probability mass in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    targets = uncertainty.sample(n_samples, rng)
    tree = cKDTree(pattern.points)
    dist, _ = tree.query(targets, k=1)
    return float(np.mean(dist <= pattern.beam_radius))


def simulate_acquisition(
    pattern: ScanPattern,
    target: np.ndarray,
    p_dwell: float = 1.0,
    rng: np.random.Generator | None = None,
    max_passes: int = 5,
) -> float | None:
    """Simulate dwell-by-dwell acquisition of a static target.

    On each dwell the target is detected iff it lies within the beam
    footprint AND an independent Bernoulli(p_dwell) trial succeeds. If a
    full pass fails, the scan repeats (up to ``max_passes`` passes).

    Parameters
    ----------
    pattern : ScanPattern
    target : array-like, shape (2,)
        True target offset [rad].
    p_dwell : float
        Per-dwell detection probability in (0, 1].
    rng : np.random.Generator, optional
    max_passes : int
        Maximum scan repetitions.

    Returns
    -------
    float or None
        Time of detection [s] from scan start, or None if never detected.
    """
    if not 0.0 < p_dwell <= 1.0:
        raise ValueError(f"p_dwell must be in (0, 1], got {p_dwell!r}")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,):
        raise ValueError(f"target must have shape (2,), got {target.shape}")
    if rng is None:
        rng = np.random.default_rng(0)
    in_beam = np.linalg.norm(pattern.points - target, axis=1) <= pattern.beam_radius
    idx = np.flatnonzero(in_beam)
    if idx.size == 0:
        return None
    for p in range(max_passes):
        hits = rng.random(idx.size) < p_dwell
        if hits.any():
            first = idx[int(np.argmax(hits))]
            return p * pattern.scan_time + (first + 1) * pattern.dwell_time
    return None


def expected_acquisition_time_spiral(
    uncertainty: GaussianUncertainty,
    beam_radius: float,
    overlap: float,
    scan_speed: float,
    containment: float = 0.995,
    p_pass: float = 1.0,
) -> float:
    """Expected acquisition time under the uniform-coverage approximation.

    Model (internal derivation; see VALIDATION.md sec. 2)
    -----------------------------------------------------
    A spiral with track spacing s scanned at along-track speed v covers area
    at rate s*v, so the time to first reach radius r is approximately
        t(r) = pi r^2 / (s v)                                        [s]
    (area of the disc of radius r divided by the coverage rate; neglects the
    discrete inner turns, so it is an approximation, best for r >> s).
    Averaging over the Rayleigh radial density f(r) = (r/sigma^2)
    exp(-r^2/(2 sigma^2)) conditioned on r <= r_max gives the single-pass
    conditional expectation; for r_max -> inf and p_pass = 1 it reduces to
        E[T] = 2 pi sigma^2 / (s v).
    With per-pass detection probability p_pass < 1 the number of extra full
    passes is geometric, adding (1/p_pass - 1) * T_full where
    T_full = pi r_max^2/(s v).

    Parameters
    ----------
    scan_speed : float
        Along-track angular speed v [rad/s], > 0.
    p_pass : float
        Probability the target is detected on a pass that CROSSES it, (0, 1].
        This is a per-crossing probability, not a per-dwell probability. A
        crossing of the footprint contains roughly
        ``n_dwell = 2 * beam_radius / (step_fraction * beam_radius)`` dwells,
        so for independent per-dwell detections
        ``p_pass = 1 - (1 - p_dwell)**n_dwell``. Passing ``p_dwell`` here
        directly overestimates the acquisition time; validation script
        ``validation/v2_acquisition_time.py`` quantifies that error
        (-38 % vs -0.6 % relative deviation for p_dwell = 0.9).

    Returns
    -------
    float
        Expected acquisition time [s], conditioned on the target lying
        within the containment radius.
    """
    if not (math.isfinite(scan_speed) and scan_speed > 0):
        raise ValueError(f"scan_speed must be > 0 [rad/s], got {scan_speed!r}")
    if not 0.0 < p_pass <= 1.0:
        raise ValueError(f"p_pass must be in (0, 1], got {p_pass!r}")
    s = track_spacing(beam_radius, overlap)
    sigma = uncertainty.sigma
    r_max = uncertainty.containment_radius(containment)
    rate = s * scan_speed
    # E[t(r) | r <= r_max] = (pi/(s v)) E[r^2 | r <= r_max], Rayleigh truncated moment
    r = np.linspace(0.0, r_max, 20001)
    f = (r / sigma**2) * np.exp(-(r**2) / (2.0 * sigma**2))
    num = np.trapezoid(math.pi * r**2 / rate * f, r)
    den = np.trapezoid(f, r)
    t_single = num / den
    t_full = math.pi * r_max**2 / rate
    return float(t_single + (1.0 / p_pass - 1.0) * t_full)
