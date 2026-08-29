r"""The bench: run the same truth through several estimators and score them.

Scoring is deliberately *two*-dimensional. Accuracy alone (RMSE) rewards a
filter that is lucky; consistency alone (NEES/NIS) rewards a filter that is
merely humble. A filter is only good if it is accurate **and** credible, so
every result carries both.

Definitions
-----------
* ``rmse_position`` — root mean square position error over runs and steps [m].
* ``anees`` — NEES averaged over runs and steps, divided by the state
  dimension. 1.0 nominal; > 1 optimistic, < 1 conservative.
* ``anis`` — same for the innovations, divided by the measurement dimension.
* ``nees_report`` / ``nis_report`` — full :class:`navbench.consistency.
  ConsistencyReport` including the chi-squared acceptance region actually used.

Monte Carlo power: every function takes an explicit ``n_runs`` and every result
records it, because the width of the acceptance region — and therefore the
meaning of "passed" — depends on it (see :mod:`navbench.consistency`).

References: Bar-Shalom, Rong Li & Kirubarajan (2001), *Estimation with
Applications to Tracking and Navigation*, §5.4 (consistency), §6.2 (the CV
model), §10.3 (EKF for nonlinear measurements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .adaptive import FixedQ, QAdapter
from .consistency import ConsistencyReport, assess, nees_series
from .kf import KalmanFilter
from .models import ConstantVelocity

__all__ = ["BenchResult", "run_linear_mc", "tune_fixed_scale", "position_rmse"]


def position_rmse(errors: NDArray[np.float64], dim: int) -> float:
    """RMS position error [m] from a ``(M, K, n)`` error array of a CV state."""
    idx = np.arange(0, 2 * dim, 2)
    return float(np.sqrt(np.mean(errors[..., idx] ** 2) * dim))


@dataclass(frozen=True)
class BenchResult:
    """Outcome of one Monte Carlo bench run.

    Attributes
    ----------
    label : str
        Estimator/adapter name.
    n_runs, n_steps : int
        Monte Carlo size.
    rmse_position : float
        RMS position error [m].
    anees, anis : float
        Normalised average NEES / NIS (1.0 nominal).
    nees_report, nis_report : ConsistencyReport
        Full chi-squared assessments.
    mean_scale, final_scale_std : float
        Mean applied process-noise multiplier over the run, and the standard
        deviation of the final multiplier across runs.
    seconds : float
        Wall-clock time for the whole Monte Carlo [s].
    """

    label: str
    n_runs: int
    n_steps: int
    rmse_position: float
    anees: float
    anis: float
    nees_report: ConsistencyReport
    nis_report: ConsistencyReport
    mean_scale: float
    final_scale_std: float
    seconds: float

    def summary(self) -> str:
        """Compact one-line summary."""
        return (
            f"{self.label:<22s} RMSE={self.rmse_position:8.3f} m  ANEES={self.anees:6.3f}  "
            f"ANIS={self.anis:6.3f}  s̄={self.mean_scale:7.3f}  "
            f"NEES {'PASS' if self.nees_report.passed else 'FAIL'}  "
            f"NIS {'PASS' if self.nis_report.passed else 'FAIL'}"
        )


def run_linear_mc(
    model: ConstantVelocity,
    *,
    adapter_factory: Callable[[], QAdapter] | None = None,
    n_runs: int = 100,
    n_steps: int = 160,
    seed: int = 0,
    q_true_scale: float | Sequence[float] = 1.0,
    burn_in: int = 20,
    label: str = "kf",
    x0: NDArray[np.float64] | None = None,
    p0_diag: Sequence[float] | None = None,
) -> BenchResult:
    r"""Monte Carlo a linear KF on the constant-velocity model.

    Parameters
    ----------
    model : ConstantVelocity
        Nominal model. ``model.q(1.0)`` is the nominal ``Q₀``.
    adapter_factory : callable or None
        Returns a fresh :class:`~navbench.adaptive.QAdapter` per run. ``None``
        means a fixed unit scale.
    n_runs, n_steps : int
        Monte Carlo size.
    seed : int
        Base seed; run ``i`` uses ``seed + 7919·i`` (a prime stride, so the
        streams of different runs and different studies do not overlap).
    q_true_scale : float or sequence of float
        Multiplier applied to the *truth* process noise. A sequence is cycled
        over runs, which is how the mis-specification sweep is driven.
    burn_in : int
        Steps discarded before scoring, so transient initialisation error does
        not dominate the consistency statistics.
    label : str
        Name carried into the result.
    x0 : ndarray or None
        Truth initial state; defaults to ``[0, 10, 0, −5]`` for ``dim = 2``.
    p0_diag : sequence of float or None
        Initial covariance diagonal; defaults to ``[100, 25] · dim``. The filter
        is initialised at the truth mean with this covariance, which makes the
        initial NEES exactly consistent by construction.

    Returns
    -------
    BenchResult
    """
    import time

    if n_runs < 1 or n_steps < burn_in + 2:
        raise ValueError(f"need n_runs >= 1 and n_steps >= burn_in+2, got {n_runs}, {n_steps}")
    dim = model.dim
    n = model.n
    if x0 is None:
        base = np.zeros(n)
        base[0::2] = 0.0
        base[1::2] = np.linspace(10.0, -5.0, dim)
        x0 = base
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    p0 = np.diag(np.asarray(p0_diag, dtype=float)) if p0_diag is not None else np.diag(
        np.array([100.0, 25.0] * dim)
    )
    scales_seq = np.atleast_1d(np.asarray(q_true_scale, dtype=float))

    f, h, r = model.f(), model.h(), model.r()
    q0 = model.q(1.0)
    keep = n_steps - burn_in
    nees = np.zeros((n_runs, keep))
    nis = np.zeros((n_runs, keep))
    errs = np.zeros((n_runs, keep, n))
    scale_trace = np.zeros((n_runs, keep))
    final_scales = np.zeros(n_runs)

    t0 = time.perf_counter()
    for i in range(n_runs):
        rng = np.random.default_rng(seed + 7919 * i)
        s_true = float(scales_seq[i % scales_seq.size])
        xs, zs = model.simulate(x0, n_steps, rng, q_true_scale=s_true)
        adapter = adapter_factory() if adapter_factory is not None else FixedQ(1.0)
        adapter.reset()
        kf = KalmanFilter(
            f=f, q=q0 * adapter.scale, h=h, r=r,
            x=x0 + np.linalg.cholesky(p0) @ rng.standard_normal(n), p=p0.copy(),
        )
        for k in range(n_steps):
            kf.predict(q=q0 * adapter.scale)
            info = kf.update(zs[k])
            p_post = kf.p.copy()
            adapter.observe(
                info.innovation, info.innovation_cov, info.gain,
                f=f, h=h, p_post=p_post, q0=q0, r=r,
            )
            if k >= burn_in:
                j = k - burn_in
                e = xs[k + 1] - kf.x
                errs[i, j] = e
                nees[i, j] = nees_series(e[None, :], p_post[None, :, :])[0]
                nis[i, j] = info.nis
                scale_trace[i, j] = adapter.scale
        final_scales[i] = adapter.scale
    seconds = time.perf_counter() - t0

    nees_rep = assess(nees, dof=n, label=f"NEES[{label}]")
    nis_rep = assess(nis, dof=model.m, label=f"NIS[{label}]")
    return BenchResult(
        label=label,
        n_runs=n_runs,
        n_steps=keep,
        rmse_position=position_rmse(errs, dim),
        anees=nees_rep.normalised_mean,
        anis=nis_rep.normalised_mean,
        nees_report=nees_rep,
        nis_report=nis_rep,
        mean_scale=float(np.mean(scale_trace)),
        final_scale_std=float(np.std(final_scales)),
        seconds=seconds,
    )


def tune_fixed_scale(
    model: ConstantVelocity,
    *,
    candidate_scales: Sequence[float],
    train_true_scales: Sequence[float],
    n_runs: int = 30,
    n_steps: int = 160,
    seed: int = 12345,
    objective: str = "rmse",
) -> tuple[float, dict[float, float]]:
    """Grid-search the best *fixed* process-noise multiplier on a training set.

    This is the honest version of "hand-tuned Q": a single scale chosen offline
    on trajectories the evaluation never sees (different seed stream), then
    frozen. It is the baseline the adaptive schemes must beat.

    Parameters
    ----------
    objective : {"rmse", "anees"}
        ``"rmse"`` minimises position RMSE; ``"anees"`` minimises
        ``|ANEES − 1|`` — the consistency-optimal choice, which is generally a
        *different* scale, and the benchmark reports both.

    Returns
    -------
    best : float
        Winning scale.
    table : dict
        Objective value for every candidate.
    """
    if objective not in ("rmse", "anees"):
        raise ValueError(f"objective must be 'rmse' or 'anees', got {objective!r}")
    table: dict[float, float] = {}
    for s in candidate_scales:
        res = run_linear_mc(
            model,
            adapter_factory=lambda s=s: FixedQ(s),
            n_runs=n_runs,
            n_steps=n_steps,
            seed=seed,
            q_true_scale=list(train_true_scales),
            label=f"fixed[{s:g}]",
        )
        table[float(s)] = (
            res.rmse_position if objective == "rmse" else abs(res.anees - 1.0)
        )
    best = min(table, key=lambda k: table[k])
    return best, table
