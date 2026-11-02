"""Baseline 1: a closed-form certificate that a deadline miss cannot have mattered.

This module is the whole point of the product, and it was written and
validated before any learned component existed. It answers one question in
closed form, with no simulation of the miss at all:

    given the declared plant, the declared controller, the declared divergence
    metric and two numbers read off the run -- the overrun ``tau`` and the
    input step ``du`` it holds through -- what is the largest the trajectory
    could possibly have moved?

If that number is below the declared tolerance, the miss **cannot** have
mattered. That is a proof about the declared linearisation, not an estimate,
and no re-simulation of that miss can change it.

The derivation
--------------
Write ``e(t) = x_with-miss(t) - x_reference(t)`` for the difference between the
run with exactly one miss injected and the miss-free reference trajectory.
Both trajectories start at the same state at the sample instant ``t_k`` where
the overrun begins, so ``e(t_k) = 0``.

**Step 1 -- the hold injects the error, open loop.** Over ``[t_k, t_k + tau)``
the perturbed branch holds the stale command. The reference branch applies its
own commands, computed from the reference trajectory, which is a KNOWN
sequence; so the input difference over the hold is an exogenous signal, not a
feedback of ``e``, and the error obeys the open-loop equation
``e_dot = A e + B du(t)`` with ``|du(t)| <= du_max``. By Duhamel's formula and
the logarithmic norm (equation 8),

```
mu_2(M)      = lambda_max( (M + M^T)/2 ),    || exp(M t) ||_2 <= exp(mu_2(M) t)  (8)
phi(mu, t)   = ( exp(mu t) - 1 ) / mu,       phi(0, t) = t                       (9)
|| e(t_k + tau) ||_S <= || S B ||_2 phi(mu_S, tau) du_max                        (10)
```

Equation (8) is Dahlquist's inequality: Dahlquist (1958), *Stability and error
bounds in the numerical integration of ordinary differential equations*,
Transactions of the Royal Institute of Technology Stockholm 130; Coppel
(1965), *Stability and Asymptotic Behavior of Differential Equations*, Heath,
chapter 2; reviewed with the modern proof in Soderlind (2006), "The
logarithmic norm. History and modern theory", *BIT Numerical Mathematics*
46(3), 631-652. It is an equality for ``t = 0`` and is sharp for normal ``M``;
for non-normal ``M`` it is conservative, which is where most of this
certificate's looseness comes from and is measured rather than asserted.
``mu_S`` is the logarithmic norm of ``S A S^-1``, the Jacobian expressed in the
units of the divergence metric, so the certificate is a statement in the same
metric as the tolerance.

**Step 2 -- the rest of the period settles, still open loop.** From
``t_k + tau`` to the next sample instant the freshest command has landed and is
the same in both branches up to the feedback on ``e``, which has not yet been
sampled; the error evolves freely for at most one period:

```
c_free = max( 1, exp(mu_S T) )                                                  (11)
```

**Step 3 -- the closed loop propagates it.** From the first sample instant
after the overrun the controller sees the perturbed state, so the error maps
through the closed-loop one-period matrix ``Phi = A_d + B_d K`` once per
period. Over a horizon of ``N`` remaining sample instants the largest possible
amplification is

```
gamma_m      = || S Phi^m S^-1 ||_2 ,     gamma_sup(N) = max_{0<=m<=N} gamma_m  (12)
```

which is an exact maximum over a finite set, not a grid approximation of a
continuum: the divergence metric is read at sample instants, and there are
``N + 1`` of them.

**The certificate.** Composing the three steps,

```
Bound(tau, du, N) = c_free * gamma_sup(N) * || S B ||_2 * phi(mu_S, tau) * du   (13)
```

and because (13) is strictly increasing in ``tau`` it inverts in closed form
into the largest overrun that is certainly harmless:

```
tau*(du, N, tol) = (1/mu_S) * ln( 1 + mu_S * tol / (c_free gamma_sup ||SB||_2 du) )
                                                                               (14)
```

with the ``mu_S -> 0`` limit ``tol / (c_free gamma_sup ||SB||_2 du)``.
Equation (14) is the headline number of this product: **one scalar, computed
once per run, below which no miss needs looking at.**

For several misses in one run the linearised responses superpose, so the
aggregate is bounded by the sum:

```
Aggregate(run) <= sum_i Bound(tau_i, du_i, N_i)                                (15)
```

(15) is a bound on the combined effect and is the only statement this module
makes about miss interactions; the per-miss numbers are one-at-a-time.

What is proved and what is not
------------------------------
(13) is a theorem about the declared **linearisation** at the declared
operating point. It is not a theorem about the real plant, and this package
never claims it is. The gap is settled by measurement, not assertion:
``validation/validate_certificate.py`` evaluates (13) against the nonlinear
re-simulation of :mod:`hilverdict.resimulate` over the whole miss population
and reports the number of violations and the worst ratio of true divergence to
bound; ``validation/validate_independent_bound.py`` checks (12) against a
Gronwall bound derived from different mathematics; and the amplitude at which
the linearisation stops covering the plant is swept and declared.

Three conditions are checked and **refused** rather than extrapolated through,
because each one breaks a step of the derivation:

- actuator saturation anywhere in the reference run, which makes ``B`` wrong;
- reference incidence beyond :data:`hilverdict.plant.ALPHA_VALID_RAD`, where
  the declared cubic term is an extrapolation;
- mean utilisation at or above 1, where the non-cascading miss model does not
  describe the run at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plant import ALPHA_VALID_RAD, LoopModel
from .timeline import MissEvent, RunLog, miss_events, utilisation

__all__ = [
    "CertificateModel",
    "MissCertificate",
    "RunCertificate",
    "certified_overrun_s",
    "certify_miss",
    "certify_run",
    "divergence_bound",
    "exponential_integral",
    "gronwall_horizon_factor",
    "horizon_gains",
    "log_norm",
]

#: Verdict for a miss whose bound falls below tolerance: it cannot have
#: mattered, for the declared linearisation.
DISMISSED = "DISMISSED"

#: Verdict for a miss whose bound does not fall below tolerance. This is NOT a
#: claim that the miss mattered; it is a statement that the certificate cannot
#: settle it and re-simulation must.
UNDECIDED = "UNDECIDED"


def log_norm(matrix: np.ndarray) -> float:
    """Logarithmic norm ``mu_2`` of equation (8), units of ``matrix``.

    ``mu_2(M) = lambda_max((M + M^T)/2)``, the largest eigenvalue of the
    symmetric part. It may be positive even when every eigenvalue of ``M`` has
    a negative real part, and then (8) is a growth bound over short times and a
    useless one over long times. That is a property of the inequality and is
    why equation (12) carries the long-horizon behaviour instead.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"matrix must be square, got shape {array.shape}")
    symmetric = 0.5 * (array + array.T)
    return float(np.linalg.eigvalsh(symmetric)[-1])


def exponential_integral(mu: float, duration_s: float) -> float:
    """Equation (9): ``(exp(mu t) - 1)/mu``, with the removable singularity handled.

    Units: seconds, for ``mu`` in 1/s. Monotonically increasing in
    ``duration_s`` for every real ``mu``, which is what makes equation (14)
    invertible.

    For ``mu > 0`` the result is at least ``duration_s`` and for ``mu < 0`` at
    most ``duration_s``, which is the sign statement equation (13) relies on.

    **Two numerical cases, both found by property testing during this product's
    build and both recorded rather than silently patched.** The naive
    ``expm1(mu t)/mu`` is correct for ordinary arguments and wrong at the ends:

    - at ``mu = -1.1125369292536007e-308`` and ``t = 0.015045340851328677`` it
      returns a value one unit in the last place ABOVE ``t``, breaking the sign
      statement;
    - at ``mu = 5e-324`` (the smallest subnormal) and ``t = 0.03125``,
      ``expm1(mu t)`` underflows to exactly zero and the quotient returns
      ``0.0`` instead of ``t`` -- a complete loss, not a rounding error.

    Both are avoided by taking the first-order series ``t (1 + mu t / 2)``
    whenever ``|mu t| < 1e-8``, where the neglected term is of relative order
    ``(mu t)^2 / 6 < 2e-17``, below one unit in the last place. Both cases are
    pinned as named regression tests.
    """
    if duration_s < 0.0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    value = float(mu)
    product = value * float(duration_s)
    if abs(product) < 1.0e-8:
        return float(duration_s * (1.0 + 0.5 * product))
    return float(np.expm1(product) / value)


def horizon_gains(scaled_monodromy: np.ndarray, steps: int) -> np.ndarray:
    """Equation (12): ``gamma_m`` for ``m = 0 .. steps``, dimensionless.

    Computed by repeated multiplication and a 2-norm per power, so the maximum
    returned by :func:`numpy.max` over the result is exact over the finite set
    of sample instants rather than a sample of a continuum.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    matrix = np.asarray(scaled_monodromy, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"scaled_monodromy must be square, got {matrix.shape}")
    power = np.eye(matrix.shape[0])
    gains = np.empty(steps + 1, dtype=float)
    for index in range(steps + 1):
        gains[index] = np.linalg.norm(power, 2)
        power = power @ matrix
    return gains


def gronwall_horizon_factor(model: LoopModel, steps: int) -> float:
    """An independent horizon factor from Gronwall, for cross-checking (12).

    Treats the closed loop as the open loop plus a state-dependent input,
    ``e_dot = A e + B K e``, and bounds it with the logarithmic norm of the
    open-loop Jacobian plus the induced norm of the feedback term:

    ```
    gamma_gronwall(N) = exp( (mu_S + || S B K S^-1 ||_2) N T )                 (16)
    ```

    This uses the sub-additivity of the logarithmic norm rather than the
    discrete monodromy of (12), so it is a bound derived from mathematics the
    implementation does not use. It is much looser -- that is the point: a
    ``gamma_sup`` above it would mean (12) is computing the wrong thing.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    feedback = (
        model.metric.scaling
        @ model.plant.input_matrix
        @ model.controller.gain_row
        @ model.metric.inverse_scaling
    )
    rate = log_norm(model.scaled_jacobian()) + float(np.linalg.norm(feedback, 2))
    return float(math.exp(rate * steps * model.sampling.period_s))


@dataclass(frozen=True)
class CertificateModel:
    """The three model constants of equations (10)-(13), computed once per loop.

    Attributes
    ----------
    model
        The declared loop.
    mu_s
        ``mu_S``, the logarithmic norm of ``S A S^-1``, 1/s.
    input_gain
        ``|| S B ||_2``, the scaled input Jacobian's induced norm, 1/s per rad
        of command.
    free_factor
        ``c_free`` of equation (11), dimensionless.
    gains
        ``gamma_m`` of equation (12) for ``m = 0 .. horizon``, dimensionless.
    """

    model: LoopModel
    mu_s: float
    input_gain: float
    free_factor: float
    gains: np.ndarray

    @classmethod
    def build(cls, model: LoopModel, horizon_steps: int) -> CertificateModel:
        """Compute the constants for ``model`` over a horizon of ``horizon_steps``."""
        if horizon_steps < 0:
            raise ValueError(f"horizon_steps must be non-negative, got {horizon_steps}")
        mu_s = log_norm(model.scaled_jacobian())
        input_gain = float(np.linalg.norm(model.scaled_input_matrix(), 2))
        free_factor = max(1.0, math.exp(mu_s * model.sampling.period_s))
        gains = horizon_gains(model.scaled_monodromy(), horizon_steps)
        return cls(
            model=model,
            mu_s=mu_s,
            input_gain=input_gain,
            free_factor=free_factor,
            gains=gains,
        )

    @property
    def horizon_steps(self) -> int:
        """Largest horizon the constants were built for."""
        return int(self.gains.shape[0] - 1)

    def gamma_sup(self, steps_remaining: int) -> float:
        """``gamma_sup(N)`` of equation (12), clipped to the built horizon."""
        if steps_remaining < 0:
            raise ValueError(
                f"steps_remaining must be non-negative, got {steps_remaining}"
            )
        if steps_remaining > self.horizon_steps:
            raise ValueError(
                f"constants were built for a horizon of {self.horizon_steps} steps "
                f"but {steps_remaining} were requested; rebuild with "
                "CertificateModel.build(model, steps_remaining)"
            )
        return float(np.max(self.gains[: steps_remaining + 1]))

    def prefactor(self, steps_remaining: int) -> float:
        """Everything in (13) except ``phi(mu_S, tau) * du``; units 1/s per rad."""
        return self.free_factor * self.gamma_sup(steps_remaining) * self.input_gain


def divergence_bound(
    constants: CertificateModel,
    overrun_s: float,
    input_step_rad: float,
    steps_remaining: int,
) -> float:
    """Equation (13): the certified upper bound on the divergence, dimensionless.

    Parameters
    ----------
    constants
        Model constants from :meth:`CertificateModel.build`.
    overrun_s
        ``tau``, s. Must be non-negative.
    input_step_rad
        ``du_max``, rad. Must be non-negative.
    steps_remaining
        ``N``, sample instants left after the overrun.

    Returns
    -------
    float
        The bound in the units of the declared divergence metric, so a value
        below 1 at the declared tolerance means the miss cannot have mattered.
        Zero overrun or zero input step gives exactly zero, which is correct:
        holding a command that was not going to change changes nothing.
    """
    if input_step_rad < 0.0:
        raise ValueError(f"input_step_rad must be non-negative, got {input_step_rad}")
    return float(
        constants.prefactor(steps_remaining)
        * exponential_integral(constants.mu_s, overrun_s)
        * float(input_step_rad)
    )


def certified_overrun_s(
    constants: CertificateModel,
    input_step_rad: float,
    steps_remaining: int,
    tolerance: float = 1.0,
) -> float:
    """Equation (14): the largest overrun that is certainly harmless, s.

    Any miss with ``tau <= tau*`` and an input step no larger than
    ``input_step_rad`` is dismissed without simulation. Returns ``inf`` when
    the input step is zero (nothing to inject) and ``0.0`` when even an
    infinitesimal overrun could exceed the tolerance, which happens only if
    the tolerance is non-positive.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if input_step_rad < 0.0:
        raise ValueError(f"input_step_rad must be non-negative, got {input_step_rad}")
    scale = constants.prefactor(steps_remaining) * float(input_step_rad)
    if scale == 0.0:
        return float("inf")
    target = float(tolerance) / scale
    if constants.mu_s == 0.0:
        return target
    argument = 1.0 + constants.mu_s * target
    if argument <= 0.0:
        # mu_S < 0 and the tolerance exceeds the asymptote -1/mu_S: the bound
        # can never reach the tolerance, however long the hold.
        return float("inf")
    return float(math.log(argument) / constants.mu_s)


@dataclass(frozen=True)
class MissCertificate:
    """The certificate for one miss.

    Attributes
    ----------
    event
        The miss it is about.
    bound
        Equation (13), in metric units.
    tolerance
        The declared tolerance it was compared against.
    verdict
        ``DISMISSED`` or ``UNDECIDED``.
    margin
        ``tolerance - bound``. Positive for a dismissal; its size says how much
        room the proof had, which is the honest way to read a dismissal.
    """

    event: MissEvent
    bound: float
    tolerance: float
    verdict: str
    margin: float

    @property
    def dismissed(self) -> bool:
        """True when the miss is certified harmless."""
        return self.verdict == DISMISSED


def certify_miss(
    constants: CertificateModel, event: MissEvent, tolerance: float = 1.0
) -> MissCertificate:
    """Apply equation (13) to one miss and return its certificate."""
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    bound = divergence_bound(
        constants, event.overrun_s, event.input_step_rad, event.steps_remaining
    )
    verdict = DISMISSED if bound < tolerance else UNDECIDED
    return MissCertificate(
        event=event,
        bound=bound,
        tolerance=float(tolerance),
        verdict=verdict,
        margin=float(tolerance) - bound,
    )


@dataclass(frozen=True)
class RunCertificate:
    """The certificate for a whole run.

    Attributes
    ----------
    label
        The run's declared label.
    certificates
        One :class:`MissCertificate` per recorded miss, in iteration order.
    refusals
        Reasons the declared derivation does not apply to this run. Empty means
        admissible. A non-empty tuple makes every per-miss verdict void and the
        run undecidable, which is reported rather than worked around.
    aggregate_bound
        Equation (15): the sum of the per-miss bounds, in metric units.
    tolerance
        The declared tolerance.
    certified_overrun_s
        Equation (14) evaluated at the run's largest input step and full
        horizon: the single conservative threshold below which no miss in a run
        like this needs looking at, s.
    """

    label: str
    certificates: tuple[MissCertificate, ...]
    refusals: tuple[str, ...]
    aggregate_bound: float
    tolerance: float
    certified_overrun_s: float

    @property
    def admissible(self) -> bool:
        """True when no declared condition of the derivation is violated."""
        return not self.refusals

    @property
    def dismissed(self) -> tuple[MissCertificate, ...]:
        """Misses the certificate disposes of."""
        return tuple(c for c in self.certificates if c.dismissed)

    @property
    def undecided(self) -> tuple[MissCertificate, ...]:
        """Misses the certificate leaves for re-simulation."""
        return tuple(c for c in self.certificates if not c.dismissed)

    @property
    def dismissal_rate(self) -> float:
        """Fraction of recorded misses disposed of in closed form."""
        if not self.certificates:
            return 1.0
        return len(self.dismissed) / len(self.certificates)


def certify_run(log: RunLog, tolerance: float = 1.0) -> RunCertificate:
    """Certify every miss in ``log``, and check the derivation's own conditions.

    The refusals are checked first and reported in the result; they do not
    raise, because a batch of runs is normally certified in a loop and a
    refusal is a finding about one run rather than an error in the caller.
    Use :func:`require_admissible` to turn one into an exception.
    """
    refusals: list[str] = []
    if log.saturated_steps:
        refusals.append(
            f"the reference run saturates the actuator on {log.saturated_steps} of "
            f"{log.steps} iterations; equation (13) rests on a linear input map "
            "and does not describe a saturated actuator"
        )
    if log.peak_alpha_rad > ALPHA_VALID_RAD:
        refusals.append(
            f"reference incidence reaches {math.degrees(log.peak_alpha_rad):.4f} deg, "
            f"beyond the declared validity limit of "
            f"{math.degrees(ALPHA_VALID_RAD):.4f} deg; the cubic pitching-moment term "
            "is an extrapolation there"
        )
    used = utilisation(log)
    if used >= 1.0:
        refusals.append(
            f"mean utilisation is {used:.4f}; at or above 1 the loop cannot keep up "
            "and the non-cascading miss model does not describe the run"
        )
    events = miss_events(log)
    constants = CertificateModel.build(log.model, log.steps)
    certificates = tuple(certify_miss(constants, event, tolerance) for event in events)
    aggregate = float(sum(c.bound for c in certificates))
    largest_step = max((e.input_step_rad for e in events), default=0.0)
    threshold = certified_overrun_s(constants, largest_step, log.steps, tolerance)
    return RunCertificate(
        label=log.label,
        certificates=certificates,
        refusals=tuple(refusals),
        aggregate_bound=aggregate,
        tolerance=float(tolerance),
        certified_overrun_s=threshold,
    )


def require_admissible(certificate: RunCertificate) -> None:
    """Raise :class:`ValueError` when a run certificate carries any refusal."""
    if certificate.refusals:
        joined = "; ".join(certificate.refusals)
        raise ValueError(
            f"run {certificate.label!r} is not admissible to the declared "
            f"derivation: {joined}"
        )
