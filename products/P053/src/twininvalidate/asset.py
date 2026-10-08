"""Asset-side changes and the seeded residual streams they produce.

The twin is declared and fixed. The *asset* is whatever the measurements
actually come from, and this module injects changes into it, then runs the
unchanged fixed-gain residual generator of :mod:`twininvalidate.twin` over the
result. Everything is generated from an explicit seed, so every number in
``validation/`` is reproducible from the commands in the README.

Change kinds
------------
``none``
    The asset obeys the declared model. Normalised residuals are i.i.d.
    N(0, 1) (the innovations property, Kailath 1968); this is the in-control
    stream every threshold is calibrated on.
``parameter_step``
    The asset's actuator gain steps from 1 to ``1 + magnitude`` at sample
    ``onset``: the asset's input matrix becomes ``(1 + magnitude) B`` while the
    twin keeps ``B``. A loss of actuator effectiveness, or equally a twin whose
    control effectiveness was declared wrong. Under the constant part of the
    reference excitation this produces a *constant* shift in the residual mean,
    which is the change a CUSUM is the optimal test for.
``slow_ramp``
    The same actuator gain ramps linearly over ``ramp_samples`` to the same
    final value. The gradual case, which a step-optimal detector is not
    designed for.
``noise_variance``
    The asset's process-noise covariance is multiplied by ``magnitude`` from
    ``onset``. The residual mean stays zero and only its spread changes, so
    every mean-shift statistic in this package is mis-specified for it by
    construction. That is the honest failure mode, measured in
    ``validation/validate_arl_curve.py``.
``sensor_offset``
    A constant offset of ``magnitude`` rad appears in the measurement path
    from ``onset``. Used only by :mod:`twininvalidate.ambiguity`, because it is
    the mechanism for which twin-side and asset-side causes can be shown to
    produce *bit-identical* residuals.

Units
-----
``magnitude`` is a dimensionless relative actuator-gain change for
``parameter_step`` and ``slow_ramp`` (negative for effectiveness loss), a
dimensionless variance multiplier for ``noise_variance``, and radians for
``sensor_offset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .twin import (
    REFERENCE_OMEGA_N,
    REFERENCE_ZETA,
    LinearGaussianTwin,
    SteadyStateFilter,
    attitude_channel,
    reference_excitation,
)

CHANGE_KINDS = ("none", "parameter_step", "slow_ramp", "noise_variance", "sensor_offset")
"""The change kinds this package injects."""

DEFAULT_BURN_IN = 200
"""Samples run before sample 0 of the returned stream, to reach steady state."""


@dataclass(frozen=True)
class AssetChange:
    """A declared asset-side change.

    Parameters
    ----------
    kind:
        One of :data:`CHANGE_KINDS`.
    onset:
        Sample index, relative to sample 0 of the returned stream, at which the
        change begins. Must be non-negative.
    magnitude:
        See the module docstring for the units of each kind.
    ramp_samples:
        Length of the linear ramp in samples; used by ``slow_ramp`` only.
    """

    kind: str = "none"
    onset: int = 0
    magnitude: float = 0.0
    ramp_samples: int = 200

    def __post_init__(self) -> None:
        if self.kind not in CHANGE_KINDS:
            raise ValueError(f"kind must be one of {CHANGE_KINDS}, got {self.kind!r}")
        if self.onset < 0:
            raise ValueError(f"onset must be non-negative, got {self.onset}")
        if self.ramp_samples <= 0:
            raise ValueError(f"ramp_samples must be positive, got {self.ramp_samples}")
        if self.kind == "noise_variance" and self.magnitude <= 0.0:
            raise ValueError(
                f"noise_variance magnitude is a variance multiplier and must be "
                f"positive, got {self.magnitude}"
            )
        if self.kind in ("parameter_step", "slow_ramp") and self.magnitude <= -1.0:
            raise ValueError(
                f"a relative gain change of {self.magnitude} would reverse or "
                f"null the actuator; the model is not valid there"
            )

    @property
    def is_change(self) -> bool:
        """True unless the kind is ``none``."""
        return self.kind != "none"


def no_change() -> AssetChange:
    """The in-control change spec."""
    return AssetChange(kind="none")


def _parameter_trajectory(change: AssetChange, n_samples: int) -> np.ndarray:
    """Relative actuator-gain change at each sample, shape ``(n_samples,)``."""
    traj = np.zeros(n_samples)
    if change.kind == "parameter_step":
        traj[change.onset :] = change.magnitude
    elif change.kind == "slow_ramp":
        k = np.arange(n_samples, dtype=float)
        frac = np.clip((k - change.onset) / float(change.ramp_samples), 0.0, 1.0)
        traj = change.magnitude * frac
    return traj


def _variance_trajectory(change: AssetChange, n_samples: int) -> np.ndarray:
    """Process-noise variance multiplier at each sample, shape ``(n_samples,)``."""
    traj = np.ones(n_samples)
    if change.kind == "noise_variance":
        traj[change.onset :] = change.magnitude
    return traj


def _offset_trajectory(change: AssetChange, n_samples: int) -> np.ndarray:
    """Asset measurement offset at each sample in rad, shape ``(n_samples,)``."""
    traj = np.zeros(n_samples)
    if change.kind == "sensor_offset":
        traj[change.onset :] = change.magnitude
    return traj


@dataclass(frozen=True)
class StreamSpec:
    """Everything needed to regenerate one batch of residual streams.

    Parameters
    ----------
    change:
        The injected asset-side change.
    n_runs:
        Number of independent noise realisations.
    n_samples:
        Length of each returned stream, in samples.
    seed:
        Seed for ``numpy.random.default_rng``.
    burn_in:
        Samples discarded before sample 0, to let the filter error reach
        steady state.
    omega_n, zeta:
        Nominal channel parameters, shared by twin and asset before the change.
    twin_offset:
        Declared measurement offset held by the twin, rad.
    """

    change: AssetChange
    n_runs: int = 200
    n_samples: int = 2000
    seed: int = 53000
    burn_in: int = DEFAULT_BURN_IN
    omega_n: float = REFERENCE_OMEGA_N
    zeta: float = REFERENCE_ZETA
    twin_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.n_runs <= 0:
            raise ValueError(f"n_runs must be positive, got {self.n_runs}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {self.burn_in}")
        if self.change.onset >= self.n_samples:
            raise ValueError(
                f"onset {self.change.onset} is at or past the end of a "
                f"{self.n_samples}-sample stream"
            )


def simulate_residuals(spec: StreamSpec) -> np.ndarray:
    """Generate normalised residual streams, shape ``(n_runs, n_samples)``.

    The twin and its fixed gain are built once from ``spec.omega_n`` and
    ``spec.zeta``; the asset is stepped forward with the parameter, variance and
    offset trajectories of the requested change. The returned array is the
    normalised residual ``z = e / sqrt(S)`` of equation (6) in
    :mod:`twininvalidate.twin`, which is dimensionless and i.i.d. N(0, 1)
    whenever ``spec.change.kind == "none"``.

    Determinism: identical ``spec`` values give bit-identical output.
    """
    twin = attitude_channel(omega_n=spec.omega_n, zeta=spec.zeta, offset=spec.twin_offset)
    filt = twin.steady_state()
    return simulate_residuals_with(twin, filt, spec)


def simulate_residuals_with(
    twin: LinearGaussianTwin, filt: SteadyStateFilter, spec: StreamSpec
) -> np.ndarray:
    """As :func:`simulate_residuals`, with a caller-supplied twin and gain.

    Separated so that :mod:`twininvalidate.ambiguity` can hold the asset fixed
    and move the twin, which is the whole point of that module.
    """
    total = spec.burn_in + spec.n_samples
    rng = np.random.default_rng(spec.seed)
    u = reference_excitation(total, dt=twin.dt)

    pad_zeros = np.zeros(spec.burn_in)
    param = np.concatenate([pad_zeros, _parameter_trajectory(spec.change, spec.n_samples)])
    varmul = np.concatenate(
        [np.ones(spec.burn_in), _variance_trajectory(spec.change, spec.n_samples)]
    )
    aoff = np.concatenate([pad_zeros, _offset_trajectory(spec.change, spec.n_samples)])

    n = twin.n_states
    chol_q = np.linalg.cholesky(twin.Q)
    sigma_v = float(np.sqrt(twin.R[0, 0]))
    sqrt_s = float(np.sqrt(filt.S))
    a_f, b_f, c, k_gain = twin.A, twin.B, twin.C, filt.K
    c_row = c[0]
    k_col = k_gain[:, 0]

    # The asset's input matrix is (1 + param[t]) * B; its state matrix equals
    # the twin's at all times, because the injected changes act on the actuator
    # path, the noise path and the measurement path, not on the dynamics.
    gain = 1.0 + param

    x = np.zeros((spec.n_runs, n))
    xh = np.zeros((spec.n_runs, n))
    z = np.empty((spec.n_runs, spec.n_samples))

    for t in range(total):
        y = x @ c_row + aoff[t] + sigma_v * rng.standard_normal(spec.n_runs)
        e = y - (xh @ c_row + twin.offset)
        if t >= spec.burn_in:
            z[:, t - spec.burn_in] = e / sqrt_s
        w = (rng.standard_normal((spec.n_runs, n)) @ chol_q.T) * np.sqrt(varmul[t])
        x = x @ a_f.T + (gain[t] * u[t]) @ b_f.T + w
        xh = xh @ a_f.T + u[t] @ b_f.T + np.outer(e, k_col)
    return z
