"""Baseline 2: the recursive estimator, giving the same answer online.

A multiplicative extended Kalman filter on the misalignment of one secondary.
It is scored against the batch solution of :mod:`aligncal.batch`, which is the
reference, and it also takes the observability verdict as a required argument.

The filter
----------
The state is a three-component local error rotation ``d_theta`` about the
current estimate, held with the estimate as a rotation matrix ``D_hat``
approximating ``exp(-[theta]x)``. With the local error state reset to zero
after every update, the propagation is the identity: a mounting misalignment is
a constant, so

    (19)  d_theta_(k|k-1) = 0,   P_(k|k-1) = P_(k-1|k-1) + Q dt

with ``Q`` a per-second random-walk density in rad^2/s. ``Q = 0`` is the
default and is the correct model for a rigid mounting; a non-zero ``Q`` is
offered because a thermally distorting bench is a real case, and it is the one
knob that makes this filter forget.

The measurement is the innovation against the current estimate. From (5),
``y = exp(-[theta]x) w``, and writing ``exp(-[theta]x) = exp(-[d_theta]x)
D_hat``,

    (20)  y_k - D_hat w_k = [ D_hat w_k ]x d_theta + e_k + O(|d_theta|^2)

so the measurement Jacobian is the skew of the **predicted** secondary rate,
not of the reference rate. The standard Kalman update follows with
``R = s_j^2 I``:

    (21)  K = P H^T ( H P H^T + R )^-1
          d_theta_hat = K ( y_k - D_hat w_k )
          P <- ( I - K H ) P ( I - K H )^T + K R K^T

(21) is written in Joseph form because the short-form update ``(I - KH) P``
loses symmetry and positive-definiteness over the thousands of updates a rate
record contains, and a filter whose covariance has gone indefinite reports a
confident wrong answer. The reset is

    (22)  D_hat <- exp(-[d_theta_hat]x) D_hat,   d_theta_hat <- 0

which is what makes this a multiplicative EKF rather than an additive one: the
estimate stays an exact rotation at every step and the linearisation of (20) is
always taken about the current best estimate, so the filter does not inherit
the ``O(|theta|^2)`` truncation of a single linearisation about the identity.

Sources: Lefferts, E. J., Markley, F. L. and Shuster, M. D. (1982), Kalman
filtering for spacecraft attitude estimation, *Journal of Guidance, Control,
and Dynamics* 5(5), 417-429, for the multiplicative error-state formulation and
the reset (22); Markley, F. L. and Crassidis, J. L. (2014), *Fundamentals of
Spacecraft Attitude Determination and Control*, Springer, section 6.2, for the
alignment state; Bar-Shalom, Y., Li, X.-R. and Kirubarajan, T. (2001),
*Estimation with Applications to Tracking and Navigation*, Wiley, section 5.2,
for the Joseph-form update of (21).

What the filter says about an unobservable degree of freedom
------------------------------------------------------------
The initial covariance ``P_0 = p0^2 I`` is a prior, so this filter is a
recursive MAP estimator and not the unbiased batch solution. In a direction the
record does not excite, ``H`` has no component, the gain is zero there, and
**the covariance stays at ``p0^2`` while the estimate stays at zero**. The
filter therefore reports its own refusal through ``P``, and
:meth:`RecursiveSolution.inflation_ratio` is that statement as a number. This
is the behaviour the learned estimator of :mod:`aligncal.learned` is measured
against.

Agreement with the batch solution on the observable subspace is a shrinkage
question and not an accident: the prior contributes ``1 / p0^2`` to the
information, so the recursive estimate is the batch estimate scaled by
``lambda_i / (lambda_i + 1 / p0^2)`` along degree of freedom ``i``, to first
order. With the default ``p0 = 0.1`` rad against alignment eigenvalues of order
1e7 and above, the predicted shrinkage is below 1e-5 relative; the measured
agreement is in ``validation/validate_recursive_agreement.py`` and the
predicted bound is checked against it there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import rotation_from_vector, rotation_to_vector, skew
from .observability import ObservabilityReport, SensorObservability
from .record import AlignmentRecord

__all__ = ["DEFAULT_INITIAL_SIGMA_RAD", "RecursiveSolution", "run"]

#: Default prior one-sigma per component, radians. 0.1 rad is 5.7 degrees, far
#: larger than any mounting misalignment a declared geometry would miss, so the
#: prior is weak by construction. The shrinkage it causes is computed and
#: reported rather than assumed negligible.
DEFAULT_INITIAL_SIGMA_RAD: float = 0.1


@dataclass(frozen=True)
class RecursiveSolution:
    """The recursive estimate for one secondary, with its observability verdict first.

    Attributes
    ----------
    observability:
        The verdict that gated this estimate. First field, deliberately.
    theta_rad:
        Final estimate, radians, in the reference frame.
    covariance:
        Final ``P``, rad^2, shape ``(3, 3)``.
    history:
        ``(n, 3)`` estimate after every update, radians, for convergence plots.
    sigma_history:
        ``(n, 3)`` per-canonical-axis one-sigma after every update, radians.
    initial_sigma_rad:
        The prior one-sigma per component used.
    joseph_symmetry:
        Largest absolute asymmetry of ``P`` over the whole run, rad^2. A
        non-zero value here is the reason (21) is written in Joseph form.
    min_eigenvalue:
        Smallest eigenvalue of ``P`` over the whole run, rad^2. A negative
        value would mean the filter had gone indefinite; it is reported rather
        than asserted.
    """

    observability: SensorObservability
    name: str
    theta_rad: np.ndarray
    covariance: np.ndarray
    history: np.ndarray
    sigma_history: np.ndarray
    initial_sigma_rad: float
    joseph_symmetry: float
    min_eigenvalue: float
    process_noise: float

    @property
    def theta_mrad(self) -> np.ndarray:
        """The estimate in milliradians."""
        return self.theta_rad * 1.0e3

    def sigma_rad_in_direction(self, direction: np.ndarray) -> float:
        """Filter one-sigma along ``direction``, radians: ``sqrt(u^T P u)``."""
        unit = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(unit))
        if norm == 0.0:
            raise ValueError("sigma_rad_in_direction() needs a non-zero direction")
        unit = unit / norm
        return float(np.sqrt(max(unit @ self.covariance @ unit, 0.0)))

    def inflation_ratio(self) -> float:
        """Ratio of the filter's worst-direction sigma to its best, dimensionless.

        For a record with an unobservable degree of freedom this is of order
        ``p0 * sqrt(lambda_best)``, a large number, because the filter never
        reduced the prior in that direction. It is the recursive estimator's
        own statement that it learned nothing there, and it is what the learned
        estimator is compared against.
        """
        dofs = self.observability.degrees_of_freedom
        sigmas = [self.sigma_rad_in_direction(d.direction) for d in dofs]
        best = min(sigmas)
        if best <= 0.0:
            return float("inf")
        return float(max(sigmas) / best)

    def prior_retention(self) -> np.ndarray:
        """Final sigma divided by the prior sigma along each degree of freedom.

        A value near 1 means the record taught the filter nothing about that
        degree of freedom. Dimensionless, shape ``(3,)``, in report order.
        """
        return np.array(
            [
                self.sigma_rad_in_direction(d.direction) / self.initial_sigma_rad
                for d in self.observability.degrees_of_freedom
            ]
        )

    def summary(self) -> list[str]:
        """Verdict first, then the estimate, then the prior retention."""
        lines = list(self.observability.summary())
        lines.append(f"RECURSIVE (MEKF) ESTIMATE for {self.name} (equations 19-22)")
        lines.append(
            "  theta = ["
            + ", ".join(f"{c:+.6f}" for c in self.theta_mrad)
            + "] mrad"
        )
        retention = self.prior_retention()
        for dof in self.observability.degrees_of_freedom:
            component = float(np.dot(self.theta_rad, dof.direction))
            sigma = self.sigma_rad_in_direction(dof.direction)
            lines.append(
                f"    dof {dof.index} ({dof.verdict}): {component * 206264.806247:+.3f} "
                f"+/- {sigma * 206264.806247:.3f} arcsec (1 sigma, filter P); "
                f"prior retained {retention[dof.index]:.6f}"
            )
        lines.append(
            f"  prior one sigma {self.initial_sigma_rad:g} rad, process noise "
            f"{self.process_noise:g} rad^2/s; worst/best sigma ratio "
            f"{self.inflation_ratio():.6g}"
        )
        lines.append(
            f"  P symmetry worst {self.joseph_symmetry:.6g} rad^2, smallest eigenvalue "
            f"{self.min_eigenvalue:.6g} rad^2"
        )
        return lines


def run(
    record: AlignmentRecord,
    observability: ObservabilityReport,
    name: str,
    *,
    initial_sigma_rad: float = DEFAULT_INITIAL_SIGMA_RAD,
    process_noise: float = 0.0,
    store_history: bool = True,
) -> RecursiveSolution:
    """Run the MEKF of equations (19)-(22) over the record, one sample at a time.

    Parameters
    ----------
    record:
        The measurements.
    observability:
        The verdict from :func:`aligncal.observability.assess`, computed first.
        This estimator does not restrict itself to the observable subspace --
        it does not need to, because its covariance reports the refusal -- but
        the verdict is required so that no estimate in this package can be
        produced without it.
    name:
        Secondary sensor name.
    initial_sigma_rad:
        Prior one-sigma per component, radians. Must be finite and positive.
    process_noise:
        Random-walk density on the misalignment, rad^2/s. Zero models a rigid
        mounting and is the default.
    store_history:
        Keep the per-sample estimate and sigma. Costs ``6 n`` floats.

    Returns
    -------
    RecursiveSolution
    """
    if not np.isfinite(initial_sigma_rad) or initial_sigma_rad <= 0.0:
        raise ValueError(
            f"initial_sigma_rad must be finite and > 0 rad, got {initial_sigma_rad!r}"
        )
    if not np.isfinite(process_noise) or process_noise < 0.0:
        raise ValueError(f"process_noise must be finite and >= 0 rad^2/s, got {process_noise!r}")
    sensor = observability[name]
    rates = record.reference_rates
    measured = record.derotated(name)
    variance = record.pair_sigma(name) ** 2
    noise = variance * np.eye(3)
    samples = len(record)
    intervals = np.diff(record.times, prepend=record.times[0])

    covariance = (initial_sigma_rad**2) * np.eye(3)
    correction = np.eye(3)
    identity = np.eye(3)
    history = np.zeros((samples, 3)) if store_history else np.zeros((0, 3))
    sigma_history = np.zeros((samples, 3)) if store_history else np.zeros((0, 3))
    worst_asymmetry = 0.0
    smallest = float(np.min(np.linalg.eigvalsh(covariance)))

    for index in range(samples):
        if process_noise > 0.0:
            covariance = covariance + process_noise * float(intervals[index]) * identity
        predicted = correction @ rates[index]
        jacobian = skew(predicted)
        innovation = measured[index] - predicted
        gain_denominator = jacobian @ covariance @ jacobian.T + noise
        gain = np.linalg.solve(gain_denominator.T, (covariance @ jacobian.T).T).T
        step = gain @ innovation
        closed = identity - gain @ jacobian
        covariance = closed @ covariance @ closed.T + gain @ noise @ gain.T
        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        worst_asymmetry = max(worst_asymmetry, asymmetry)
        covariance = 0.5 * (covariance + covariance.T)
        smallest = min(smallest, float(np.min(np.linalg.eigvalsh(covariance))))
        correction = rotation_from_vector(-step) @ correction
        if store_history:
            history[index] = -rotation_to_vector(correction)
            sigma_history[index] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    theta = -rotation_to_vector(correction)
    return RecursiveSolution(
        observability=sensor,
        name=name,
        theta_rad=theta,
        covariance=covariance,
        history=history,
        sigma_history=sigma_history,
        initial_sigma_rad=float(initial_sigma_rad),
        joseph_symmetry=worst_asymmetry,
        min_eigenvalue=smallest,
        process_noise=float(process_noise),
    )
