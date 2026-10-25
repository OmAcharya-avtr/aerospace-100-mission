"""Divergence curves: the two baselines, and the lower bound they support.

What is measured, stated once and precisely
-------------------------------------------
This package has no precision ephemerides and no two-line element archive. It
therefore **cannot** measure the error of a TLE against truth. What it
measures is the separation between two propagations, and the statement that
can be made about it is a triangle inequality and nothing more:

    if two propagators disagree by ``D`` at some time, and ``e1`` and ``e2``
    are their unknown errors against truth, then ``|e1| + |e2| >= D``, so
    ``max(|e1|, |e2|) >= D / 2``.

So ``D/2`` is a rigorous lower bound on *the larger of the two models'*
position errors. It is **not** a lower bound on the error of either one
individually: a perfect propagator compared against a bad one produces a large
``D`` while its own error is zero. ``D`` is also not an estimate of the error:
two models can agree closely and both be wrong, which is exactly what two
analytic theories sharing an atmosphere model would do.

:func:`larger_error_lower_bound_km` returns that bound and nothing else, and
``validation/validate_lower_bound.py`` checks the inequality against a case
where truth is known in closed form -- a two-body orbit, where
:mod:`orbitdrift.kepler` is truth and two deliberately wrong propagations
stand in for the two models. That is a check against an independently derived
bound rather than a comparison of two implementations, and it is the check
class that catches an inequality written the wrong way round.

The two baselines
-----------------
``baseline 1``
    :func:`self_consistency_divergence`. Propagate the earlier of two element
    sets for the same object forward and compare against the later one. This
    is what operators actually do, it needs no second dynamical model, and it
    is limited by the fact that both sides are SGP4 -- any error common to the
    theory cancels out of it entirely.

``baseline 2``
    :func:`model_divergence`. Compare SGP4 against the independent numerical
    force model of :mod:`orbitdrift.forces`, started from the SGP4 state at
    epoch so that the curve begins at exactly zero and every later value is
    accumulated disagreement rather than an initialisation offset.

The drift metric
----------------
The instantaneous separation of two propagations oscillates at the orbital
period: the radial and cross-track components are periodic, and only the
in-track component grows secularly. Sampling that at an interval not
commensurate with the period aliases the oscillation into the growth curve and
produces a ragged, seed-dependent result. :func:`window_rms` therefore reduces
the fine separation to its root-mean-square over a window of one orbital
period centred on each reported age. The reported drift is that RMS, in km,
and the README says so wherever a number from it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import SatrecArray

from .constants import SECONDS_PER_DAY
from .errors import PropagationError
from .features import epoch_jd, period_s_from_satrec
from .forces import ForceModel, cd_area_over_mass_m2_kg
from .frames import to_ric
from .integrate import propagate


@dataclass(frozen=True)
class DivergenceCurve:
    """A drift-against-epoch-age curve for one object.

    Attributes
    ----------
    ages_days
        Reported epoch ages, shape ``(k,)``.
    drift_km
        Window-RMS separation at each age, km, shape ``(k,)``.
    drift_ric_km
        Window-RMS of the radial, in-track and cross-track components,
        shape ``(k, 3)``. These do not add in quadrature to ``drift_km``
        exactly, because each is an RMS over the same window of a different
        signal; the quadrature sum of the three RMS values does equal the RMS
        of the magnitude, and ``tests/test_divergence.py`` checks that
        identity to tolerance.
    source
        ``"sgp4-vs-numerical"`` or ``"sgp4-self-consistency"``.
    detail
        Free text naming the force model or the element-set pair used.
    """

    ages_days: np.ndarray
    drift_km: np.ndarray
    drift_ric_km: np.ndarray
    source: str
    detail: str

    def at(self, age_days: float) -> float:
        """Linearly interpolated drift at ``age_days``, km."""
        return float(np.interp(age_days, self.ages_days, self.drift_km))


def larger_error_lower_bound_km(drift_km: np.ndarray | float) -> np.ndarray:
    """``drift_km / 2``: a lower bound on the larger of the two model errors.

    See the module docstring for the derivation and for what this is not.
    """
    return 0.5 * np.asarray(drift_km, dtype=float)


def sgp4_states(
    satrec, times_s: np.ndarray, *, jd_whole: float | None = None, fr: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """SGP4 TEME states at ``times_s`` seconds after the element-set epoch.

    The Julian date is kept split into a whole part and a fraction, which is
    why ``sgp4`` exposes ``jdsatepoch`` and ``jdsatepochF`` separately: adding
    them before passing them in throws away about 1e-8 day of resolution.

    Raises
    ------
    PropagationError
        If SGP4 returns a non-zero code at any requested time, naming the code
        and the time. A silently NaN row would otherwise propagate into a
        divergence curve as a NaN drift.
    """
    whole = float(satrec.jdsatepoch) if jd_whole is None else float(jd_whole)
    fraction = float(satrec.jdsatepochF) if fr is None else float(fr)
    times = np.asarray(times_s, dtype=float)
    jd = np.full(times.shape, whole)
    frac = fraction + times / SECONDS_PER_DAY
    codes, positions, velocities = SatrecArray([satrec]).sgp4(jd, frac)
    bad = np.flatnonzero(codes[0] != 0)
    if bad.size:
        first = int(bad[0])
        raise PropagationError(
            f"SGP4 returned error code {int(codes[0][first])} for satellite "
            f"{satrec.satnum} at {times[first] / SECONDS_PER_DAY:.4f} days after epoch; "
            f"{bad.size} of {times.size} requested times failed"
        )
    return positions[0], velocities[0]


def window_rms(
    times_s: np.ndarray,
    series: np.ndarray,
    window_s: float,
    out_times_s: np.ndarray,
) -> np.ndarray:
    """Root-mean-square of ``series`` over a window of ``window_s`` about each output time.

    ``series`` may be ``(m,)`` or ``(m, c)``; the result is ``(k,)`` or
    ``(k, c)``. Windows are clipped to the available span, so the first and
    last output times use a half window; the number of samples actually used
    is not returned, which means an output time within half a window of the
    start is a less converged RMS than the others. That is why
    :func:`model_divergence` never reports an age inside half a window of
    zero.
    """
    t = np.asarray(times_s, dtype=float)
    y = np.asarray(series, dtype=float)
    out_t = np.atleast_1d(np.asarray(out_times_s, dtype=float))
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    half = 0.5 * float(window_s)
    result = np.empty((out_t.size, y.shape[1]))
    for index, centre in enumerate(out_t):
        mask = (t >= centre - half) & (t <= centre + half)
        if not np.any(mask):
            mask = np.zeros(t.shape, dtype=bool)
            mask[int(np.argmin(np.abs(t - centre)))] = True
        result[index] = np.sqrt(np.mean(y[mask] ** 2, axis=0))
    return result[:, 0] if squeeze else result


def model_divergence(
    satrec,
    *,
    span_days: float = 7.0,
    steps_per_rev: int = 360,
    report_every_hours: float = 6.0,
    force: ForceModel | None = None,
    step_s: float | None = None,
    sample_every: int = 4,
) -> DivergenceCurve:
    """Baseline 2: SGP4 against the independent numerical force model.

    The numerical propagation is started from the SGP4 state at the element-set
    epoch, so the curve is zero at zero age by construction and no part of it
    is an initialisation offset.

    Parameters
    ----------
    satrec
        An initialised ``sgp4.api.Satrec``.
    span_days
        Propagation span.
    steps_per_rev
        Integration steps per orbital revolution, used only when ``step_s`` is
        not given. The truncation error this buys is measured in
        ``validation/validate_step_refinement.py``; 360 is the value every
        number in this repository was produced with.
    report_every_hours
        Spacing of the reported ages.
    force
        Force model. Default: every term on.
    step_s
        Explicit integration step, which overrides ``steps_per_rev``. Used by
        the step-refinement study.
    sample_every
        Record the integrated state every this many steps. Must divide the
        step count.
    """
    force = force or ForceModel()
    period_s = period_s_from_satrec(satrec)
    h = float(step_s) if step_s is not None else period_s / float(steps_per_rev)
    n_steps = int(round(span_days * SECONDS_PER_DAY / h))
    n_steps -= n_steps % sample_every
    if n_steps < sample_every:
        raise PropagationError(
            f"span {span_days} days at step {h:.3f} s gives {n_steps} usable steps"
        )
    code, r0, v0 = satrec.sgp4(float(satrec.jdsatepoch), float(satrec.jdsatepochF))
    if code != 0:
        raise PropagationError(
            f"SGP4 returned code {code} at the epoch of satellite {satrec.satnum}; "
            "the element set cannot be used as an initial condition"
        )
    coefficient = np.atleast_1d(cd_area_over_mass_m2_kg(float(satrec.bstar)))
    trajectory = propagate(
        np.asarray(r0)[None, :],
        np.asarray(v0)[None, :],
        epoch_jd(satrec),
        coefficient,
        n_steps=n_steps,
        step_s=h,
        force=force,
        sample_every=sample_every,
    )
    reference_r, reference_v = sgp4_states(satrec, trajectory.times_s)
    delta = trajectory.positions_km[0] - reference_r
    magnitude = np.linalg.norm(delta, axis=-1)
    ric = to_ric(delta, reference_r, reference_v)

    half_window = 0.5 * period_s
    ages = np.arange(
        half_window,
        trajectory.times_s[-1] - half_window + 1.0,
        report_every_hours * 3600.0,
    )
    if ages.size == 0:
        ages = np.array([trajectory.times_s[-1] - half_window])
    return DivergenceCurve(
        ages_days=ages / SECONDS_PER_DAY,
        drift_km=window_rms(trajectory.times_s, magnitude, period_s, ages),
        drift_ric_km=window_rms(trajectory.times_s, np.abs(ric), period_s, ages),
        source="sgp4-vs-numerical",
        detail=f"{force.label()}, step {h:.4f} s ({period_s / h:.1f} per rev)",
    )


def self_consistency_divergence(
    satrec_early,
    satrec_late,
    *,
    report_every_hours: float = 6.0,
    samples_per_rev: int = 32,
) -> DivergenceCurve:
    """Baseline 1: the same object's earlier element set against its later one.

    Both element sets are propagated to a common grid of absolute times
    spanning from the later epoch to the later epoch plus the epoch gap, and
    the separation is reported against the **age of the earlier element set**,
    which is the quantity an operator holding a stale TLE actually has.

    This needs no second dynamical model and is the standard practical proxy.
    Its limitation is structural and is stated in README.md: both sides are
    SGP4, so any error common to the theory -- the atmosphere model, the
    truncation of the Brouwer-Lyddane expansion, the absence of solar
    radiation pressure -- cancels out of the comparison and is invisible to
    it. It measures the part of the error that came from the *fit*, not the
    part that came from the *model*.

    Raises
    ------
    PropagationError
        If the two element sets are for different catalogue numbers, or the
        later epoch is not after the earlier one.
    """
    if int(satrec_early.satnum) != int(satrec_late.satnum):
        raise PropagationError(
            f"self-consistency needs two element sets for the same object; got "
            f"catalogue numbers {satrec_early.satnum} and {satrec_late.satnum}"
        )
    jd_early = epoch_jd(satrec_early)
    jd_late = epoch_jd(satrec_late)
    gap_days = jd_late - jd_early
    if gap_days <= 0.0:
        raise PropagationError(
            f"the second element set must be later; the epoch gap is {gap_days:.6f} days"
        )
    period_s = period_s_from_satrec(satrec_late)
    step_s = period_s / float(samples_per_rev)
    span_s = gap_days * SECONDS_PER_DAY
    times_from_late = np.arange(0.0, span_s + step_s, step_s)

    r_late, v_late = sgp4_states(satrec_late, times_from_late)
    r_early, _ = sgp4_states(satrec_early, times_from_late + span_s)
    delta = r_early - r_late
    magnitude = np.linalg.norm(delta, axis=-1)
    ric = to_ric(delta, r_late, v_late)

    age_s = times_from_late + span_s
    half_window = 0.5 * period_s
    ages = np.arange(
        age_s[0] + half_window, age_s[-1] - half_window + 1.0, report_every_hours * 3600.0
    )
    if ages.size == 0:
        ages = np.array([0.5 * (age_s[0] + age_s[-1])])
    return DivergenceCurve(
        ages_days=ages / SECONDS_PER_DAY,
        drift_km=window_rms(age_s, magnitude, period_s, ages),
        drift_ric_km=window_rms(age_s, np.abs(ric), period_s, ages),
        source="sgp4-self-consistency",
        detail=(
            f"catalogue {satrec_late.satnum}, epoch gap {gap_days:.4f} days, "
            f"{times_from_late.size} samples"
        ),
    )
