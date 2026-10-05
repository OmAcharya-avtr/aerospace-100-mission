"""Deterministic synthetic housekeeping telemetry, in monitoring-window blocks.

Every array produced here is a block of shape ``(n_windows, window_length,
n_channels)`` of *standardised* deviates: nominal channels are marginally
``Normal(0, 1)``.  Working in sigma units means the control-chart designs in
:mod:`telemetryool.arl` apply without a scale conversion, and it makes the
"shift in sigma" axis of the detection-delay curves unambiguous.

Why synthetic and not real telemetry: the whole point of this package is the
*designed* false-alarm rate, and a designed rate can only be checked against a
known nominal hypothesis.  Real housekeeping telemetry has no ground-truth
label for "nominal", so a measured false-alarm rate on it would be an estimate
of an unknown quantity.  The limitation this buys is stated plainly in
``DATASET_CARD.md``: results here characterise the detectors under the generator's
model, not under any real spacecraft's telemetry.

Nominal model
-------------
``x_t = L e_t`` where ``e_t`` is an ``n_channels``-vector AR(1) process

    ``e_t = rho e_{t-1} + sqrt(1 - rho^2) n_t``,  ``n_t ~ Normal(0, I)``

so each component is marginally unit variance for any ``rho`` in ``[0, 1)``, and
``L`` is a Cholesky factor of the target correlation matrix, rescaled so the
marginals stay unit variance.  The AR(1) state is burned in for
``burn_in`` samples before the window starts, so no window has a transient.

Anomaly models
--------------
``step``
    ``+delta`` added to the affected channels from ``onset`` onwards.
``drift``
    ``+slope * (t - onset)`` added from ``onset`` onwards (a ramp).
``stuck``
    affected channels hold their value at ``onset`` for the rest of the window.
``spike``
    a single sample at ``onset`` offset by ``+delta`` on the affected channels.
``decorrelate``
    affected channels are replaced from ``onset`` by independent unit-variance
    noise, leaving every marginal distribution unchanged and breaking only the
    cross-channel correlation.  A univariate monitor on any single channel is
    blind to this by construction; it exists to show where a multivariate
    detector is the only thing that works.

References
----------
No literature equation is used.  The AR(1) unit-variance parameterisation and
the Cholesky correlation construction are standard; see e.g. Box, Jenkins and
Reinsel, *Time Series Analysis: Forecasting and Control*, 4th ed., Wiley 2008,
ch. 3, for the AR(1) variance relation ``var = sigma_n^2 / (1 - rho^2)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NominalModel",
    "Anomaly",
    "ANOMALY_KINDS",
    "generate_nominal",
    "apply_anomaly",
    "generate_anomalous",
    "equicorrelation",
]

#: Anomaly kinds understood by :func:`apply_anomaly`.
ANOMALY_KINDS = ("step", "drift", "stuck", "spike", "decorrelate")


def equicorrelation(n_channels: int, rho_cross: float) -> NDArray[np.float64]:
    """Equicorrelation matrix: 1 on the diagonal, ``rho_cross`` off it.

    Positive definite for ``rho_cross`` in ``(-1 / (n_channels - 1), 1)``.
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels!r}")
    if n_channels > 1 and not (-1.0 / (n_channels - 1) < rho_cross < 1.0):
        raise ValueError(
            f"rho_cross={rho_cross} gives a non-positive-definite equicorrelation "
            f"matrix for n_channels={n_channels}; it must lie in "
            f"(-1/{n_channels - 1}, 1)"
        )
    mat = np.full((n_channels, n_channels), float(rho_cross))
    np.fill_diagonal(mat, 1.0)
    return mat


@dataclass(frozen=True)
class NominalModel:
    """The in-control telemetry model.

    Parameters
    ----------
    n_channels
        Number of channels, >= 1.
    rho_time
        AR(1) coefficient in ``[0, 1)``, dimensionless.  ``0`` gives independent
        samples, which is the hypothesis the chart designs in
        :mod:`telemetryool.arl` assume.
    correlation
        Cross-channel correlation matrix, shape ``(n_channels, n_channels)``.
        ``None`` means the identity.  Must be symmetric positive definite.
    burn_in
        AR(1) samples discarded before each window, so the first sample of a
        window is already in the stationary distribution.  Ignored when
        ``rho_time == 0``.
    """

    n_channels: int
    rho_time: float = 0.0
    correlation: NDArray[np.float64] | None = None
    burn_in: int = 200
    _chol: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_channels, (int, np.integer)) or self.n_channels < 1:
            raise ValueError(f"n_channels must be an integer >= 1, got {self.n_channels!r}")
        if not (0.0 <= float(self.rho_time) < 1.0):
            raise ValueError(f"rho_time must lie in [0, 1), got {self.rho_time!r}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {self.burn_in!r}")
        if self.correlation is None:
            chol = np.eye(self.n_channels)
        else:
            corr = np.asarray(self.correlation, dtype=float)
            if corr.shape != (self.n_channels, self.n_channels):
                raise ValueError(
                    f"correlation must have shape "
                    f"({self.n_channels}, {self.n_channels}), got {corr.shape}"
                )
            if not np.allclose(corr, corr.T, atol=1e-12):
                raise ValueError("correlation matrix must be symmetric")
            if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
                raise ValueError("correlation matrix must have unit diagonal")
            try:
                chol = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "correlation matrix is not positive definite"
                ) from exc
        object.__setattr__(self, "_chol", chol)

    @property
    def cholesky(self) -> NDArray[np.float64]:
        """Lower Cholesky factor of the correlation matrix."""
        return self._chol


def generate_nominal(
    model: NominalModel,
    n_windows: int,
    window_length: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Nominal telemetry block, shape ``(n_windows, window_length, n_channels)``.

    Marginally ``Normal(0, 1)`` per channel for any ``rho_time``; the empirical
    standard deviation of the returned block is checked in
    ``tests/test_synthetic.py``.
    """
    if n_windows < 1 or window_length < 1:
        raise ValueError(
            f"n_windows and window_length must both be >= 1, "
            f"got {n_windows!r} and {window_length!r}"
        )
    c = model.n_channels
    rho = float(model.rho_time)
    if rho == 0.0:
        noise = rng.standard_normal((n_windows, window_length, c))
    else:
        total = model.burn_in + window_length
        innov = rng.standard_normal((n_windows, total, c))
        scale = np.sqrt(1.0 - rho * rho)
        state = innov[:, 0, :]
        out = np.empty((n_windows, total, c))
        out[:, 0, :] = state
        for t in range(1, total):
            state = rho * state + scale * innov[:, t, :]
            out[:, t, :] = state
        noise = out[:, model.burn_in :, :]
    return noise @ model.cholesky.T


@dataclass(frozen=True)
class Anomaly:
    """An injected anomaly.

    Parameters
    ----------
    kind
        One of :data:`ANOMALY_KINDS`.
    onset
        Zero-based sample index at which the anomaly starts, in
        ``[0, window_length)``.
    magnitude
        ``step`` and ``spike``: offset in sigma units.
        ``drift``: slope in sigma per sample.
        ``stuck`` and ``decorrelate``: ignored (must be 0).
    channels
        Channel indices affected.  ``None`` means channel 0 only.
    """

    kind: str
    onset: int
    magnitude: float = 0.0
    channels: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ANOMALY_KINDS:
            raise ValueError(f"kind must be one of {ANOMALY_KINDS}, got {self.kind!r}")
        if self.onset < 0:
            raise ValueError(f"onset must be >= 0, got {self.onset!r}")
        if self.kind in ("stuck", "decorrelate") and float(self.magnitude) != 0.0:
            raise ValueError(f"{self.kind!r} takes no magnitude, got {self.magnitude!r}")
        if self.kind in ("step", "spike", "drift") and float(self.magnitude) == 0.0:
            raise ValueError(f"{self.kind!r} requires a non-zero magnitude")

    def channel_indices(self, n_channels: int) -> NDArray[np.int64]:
        """Resolved channel indices, validated against ``n_channels``."""
        idx = np.array([0] if self.channels is None else list(self.channels), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("channels must name at least one channel")
        if idx.min() < 0 or idx.max() >= n_channels:
            raise ValueError(
                f"channel indices {idx.tolist()} out of range for n_channels={n_channels}"
            )
        return idx


def apply_anomaly(
    block: NDArray[np.float64],
    anomaly: Anomaly,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return a copy of ``block`` with ``anomaly`` injected.

    Parameters
    ----------
    block
        Shape ``(n_windows, window_length, n_channels)``, standardised deviates.
    anomaly
        The anomaly to inject; see :class:`Anomaly`.
    rng
        Generator, used only by the ``decorrelate`` kind.

    Returns
    -------
    numpy.ndarray
        Same shape as ``block``.
    """
    arr = np.asarray(block, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"block must be 3-D (windows, length, channels), got {arr.shape}")
    n_win, length, n_ch = arr.shape
    if anomaly.onset >= length:
        raise ValueError(f"onset {anomaly.onset} is outside a window of length {length}")
    out = arr.copy()
    ch = anomaly.channel_indices(n_ch)
    t0 = int(anomaly.onset)
    mag = float(anomaly.magnitude)

    if anomaly.kind == "step":
        out[:, t0:, :][:, :, ch] = arr[:, t0:, :][:, :, ch] + mag
    elif anomaly.kind == "spike":
        out[:, t0, :][:, ch] = arr[:, t0, :][:, ch] + mag
    elif anomaly.kind == "drift":
        ramp = (np.arange(length) - t0).clip(min=0).astype(float) * mag
        out[:, :, :][:, :, ch] = arr[:, :, :][:, :, ch] + ramp[None, :, None]
    elif anomaly.kind == "stuck":
        held = arr[:, t0, :][:, ch]
        out[:, t0:, :][:, :, ch] = np.repeat(held[:, None, :], length - t0, axis=1)
    elif anomaly.kind == "decorrelate":
        fresh = rng.standard_normal((n_win, length - t0, ch.size))
        out[:, t0:, :][:, :, ch] = fresh
    return out


def generate_anomalous(
    model: NominalModel,
    anomaly: Anomaly,
    n_windows: int,
    window_length: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Nominal block with ``anomaly`` injected, shape ``(n_windows, W, n_channels)``."""
    base = generate_nominal(model, n_windows, window_length, rng)
    return apply_anomaly(base, anomaly, rng)
