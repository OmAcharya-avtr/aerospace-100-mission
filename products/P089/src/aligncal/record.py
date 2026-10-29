"""The alignment record: what two or more rate sensors measured, and what you declared.

Measurement model
-----------------
One sensor is the **reference**. Each other sensor is a **secondary** with a
declared nominal mounting rotation ``C_j`` from the reference frame to its own
frame. The true mounting carries an unknown misalignment ``theta_j`` expressed
in the reference frame:

    (4)   C_j^true = C_j exp(-[theta_j]x)

so that de-rotating the secondary measurement by the declared mounting gives,
in the reference frame,

    (5)   y_j(t) = C_j^T w_j(t) = exp(-[theta_j]x) w(t) + v_j(t)

with ``w(t)`` the true body rate in rad/s and ``v_j`` the secondary's noise
rotated into the reference frame, which leaves its covariance ``sigma_j^2 I``
unchanged because ``C_j`` is orthogonal. The reference measures

    (6)   w_A(t) = w(t) + n_A(t),   n_A ~ N(0, sigma_A^2 I)

Subtracting (6) from (5) and linearising ``exp(-[theta]x) ~ I - [theta]x``:

    (7)   r_j(t) = y_j(t) - w_A(t) = [w(t)]x theta_j + e_j(t)
          e_j ~ N(0, (sigma_A^2 + sigma_j^2) I)

using ``-theta x w = [w]x theta`` from the identity (3) of
:mod:`aligncal.geometry`. Equation (7) is the whole product: the sensitivity
matrix at time ``t`` is the skew of the body rate, and the skew of a vector has
that vector in its null space.

Units and ranges
----------------
``times`` in seconds, strictly increasing. Rates in rad/s. ``noise_sigma`` is
the per-axis one-sigma white-noise amplitude of a single rate sample, in rad/s;
it is the angular-random-walk-free part of the error budget and is treated as
declared, not estimated, which is stated in README.md under Limitations.
Misalignments ``theta`` in radians.

Sources
-------
Equation (4)-(7): the standard relative-alignment formulation, e.g. Markley,
F. L. and Crassidis, J. L. (2014), *Fundamentals of Spacecraft Attitude
Determination and Control*, Springer, section 6.2 (gyro and attitude-sensor
alignment) and section 2.9 (the small-rotation parameterisation). The
formulation of a misalignment as a rotation vector added multiplicatively on
the right of a declared mounting is the convention of that text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .geometry import is_rotation

__all__ = ["AlignmentRecord", "SensorSpec", "residuals"]


@dataclass(frozen=True)
class SensorSpec:
    """Declared properties of one rate sensor.

    Parameters
    ----------
    name:
        Identifier used in every report.
    noise_sigma:
        Per-axis one-sigma white noise of a single rate sample, rad/s. Must be
        strictly positive: a sensor declared noiseless makes the weighted
        normal equations singular and the Cramer-Rao bound zero, which is a
        statement no instrument supports.
    mounting:
        Declared nominal rotation ``C_j`` from the reference frame to this
        sensor's frame, dimensionless ``(3, 3)``. The reference sensor carries
        the identity. Checked for orthogonality and ``det = +1``.
    """

    name: str
    noise_sigma: float
    mounting: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SensorSpec.name must be a non-empty string")
        sigma = float(self.noise_sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"SensorSpec {self.name!r}: noise_sigma must be finite and > 0 rad/s, "
                f"got {self.noise_sigma!r}"
            )
        matrix = np.asarray(self.mounting, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"SensorSpec {self.name!r}: mounting must be (3, 3), got {matrix.shape}"
            )
        if not is_rotation(matrix, tolerance=1.0e-8):
            raise ValueError(
                f"SensorSpec {self.name!r}: mounting is not a rotation matrix "
                "(needs R^T R = I and det = +1 to 1e-8)"
            )
        object.__setattr__(self, "noise_sigma", sigma)
        object.__setattr__(self, "mounting", matrix)


@dataclass(frozen=True)
class AlignmentRecord:
    """A time series of rates from a reference sensor and one or more secondaries.

    Parameters
    ----------
    times:
        Shape ``(n,)``, seconds, strictly increasing, finite.
    reference:
        :class:`SensorSpec` of the reference sensor. Its mounting must be the
        identity by construction: the reference frame is defined by it.
    reference_rates:
        Shape ``(n, 3)``, rad/s, as measured by the reference sensor in its own
        frame.
    secondaries:
        Mapping from sensor name to :class:`SensorSpec`. At least one.
    secondary_rates:
        Mapping from the same names to ``(n, 3)`` arrays in rad/s, as measured
        by each secondary **in its own frame** (not de-rotated).
    provenance:
        Free text recorded in every report. Say where the data came from.
    """

    times: np.ndarray
    reference: SensorSpec
    reference_rates: np.ndarray
    secondaries: Mapping[str, SensorSpec]
    secondary_rates: Mapping[str, np.ndarray]
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size < 2:
            raise ValueError(f"AlignmentRecord needs at least 2 samples, got {times.size}")
        if not np.all(np.isfinite(times)):
            raise ValueError("AlignmentRecord.times must be finite")
        if not np.all(np.diff(times) > 0.0):
            bad = int(np.argmin(np.diff(times)))
            raise ValueError(
                "AlignmentRecord.times must be strictly increasing; the smallest "
                f"interval is {float(np.min(np.diff(times))):.6g} s at index {bad}"
            )
        if not is_rotation(np.asarray(self.reference.mounting, dtype=float), tolerance=1.0e-12):
            raise ValueError("the reference sensor defines the frame; its mounting must be I")
        if float(np.max(np.abs(np.asarray(self.reference.mounting) - np.eye(3)))) > 1.0e-12:
            raise ValueError(
                "the reference sensor defines the frame; its mounting must be the identity, "
                "and a non-identity reference mounting is silently absorbed into every "
                "misalignment estimate if it is allowed through"
            )
        rates = _as_rate_block(self.reference_rates, times.size, "reference_rates")
        if not self.secondaries:
            raise ValueError("AlignmentRecord needs at least one secondary sensor")
        if set(self.secondaries) != set(self.secondary_rates):
            missing = sorted(set(self.secondaries) ^ set(self.secondary_rates))
            raise ValueError(
                f"secondaries and secondary_rates must have the same names; differ on {missing}"
            )
        if self.reference.name in self.secondaries:
            raise ValueError(
                f"sensor name {self.reference.name!r} is used for both the reference and a "
                "secondary; every name must be unique"
            )
        checked = {
            name: _as_rate_block(block, times.size, f"secondary_rates[{name!r}]")
            for name, block in self.secondary_rates.items()
        }
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "reference_rates", rates)
        object.__setattr__(self, "secondaries", dict(self.secondaries))
        object.__setattr__(self, "secondary_rates", checked)

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def names(self) -> tuple[str, ...]:
        """Secondary sensor names, in sorted order, so every report is stable."""
        return tuple(sorted(self.secondaries))

    @property
    def duration_s(self) -> float:
        """Record span in seconds, ``times[-1] - times[0]``."""
        return float(self.times[-1] - self.times[0])

    @property
    def mean_interval_s(self) -> float:
        """Mean sampling interval in seconds."""
        return self.duration_s / max(len(self) - 1, 1)

    def pair_sigma(self, name: str) -> float:
        """Combined one-sigma of the difference (7) for secondary ``name``, rad/s.

        ``sqrt(sigma_A^2 + sigma_j^2)``, from equation (7).
        """
        if name not in self.secondaries:
            raise KeyError(
                f"unknown secondary {name!r}; this record carries {list(self.names)}"
            )
        return float(
            np.hypot(self.reference.noise_sigma, self.secondaries[name].noise_sigma)
        )

    def derotated(self, name: str) -> np.ndarray:
        """Secondary ``name`` expressed in the reference frame, ``y_j`` of (5), rad/s."""
        if name not in self.secondaries:
            raise KeyError(
                f"unknown secondary {name!r}; this record carries {list(self.names)}"
            )
        mounting = np.asarray(self.secondaries[name].mounting, dtype=float)
        return np.asarray(self.secondary_rates[name], dtype=float) @ mounting

    def subrecord(self, stop: int, start: int = 0) -> AlignmentRecord:
        """A copy covering samples ``[start, stop)``, for record-length sweeps."""
        if stop <= start:
            raise ValueError(f"subrecord() needs stop > start, got {start} and {stop}")
        window = slice(start, stop)
        return AlignmentRecord(
            times=self.times[window],
            reference=self.reference,
            reference_rates=self.reference_rates[window],
            secondaries=self.secondaries,
            secondary_rates={n: b[window] for n, b in self.secondary_rates.items()},
            provenance=f"{self.provenance} [samples {start}:{stop}]",
        )


def _as_rate_block(block: np.ndarray, samples: int, label: str) -> np.ndarray:
    array = np.asarray(block, dtype=float)
    if array.shape != (samples, 3):
        raise ValueError(f"{label} must have shape ({samples}, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        count = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"{label} carries {count} non-finite values; a gap must be removed from the "
            "record rather than carried as nan, because the Gramian of (8) would "
            "silently become nan and every verdict with it"
        )
    return array


def residuals(record: AlignmentRecord, name: str) -> np.ndarray:
    """Residual ``r_j(t)`` of equation (7) for secondary ``name``, rad/s.

    Shape ``(n, 3)``. This is the de-rotated secondary rate minus the reference
    rate, and it is what every estimator in this package consumes.
    """
    return record.derotated(name) - record.reference_rates
