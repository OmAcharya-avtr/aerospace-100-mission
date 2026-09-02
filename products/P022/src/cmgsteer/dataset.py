"""Seeded manoeuvre suites and the lookahead-oracle dataset for the learned policy.

Nothing here is stored on disk: the generators are committed and every array is
reproduced bit-for-bit from an integer seed through ``numpy.random.default_rng``.

A **manoeuvre** is a concatenation of rest-to-rest torque pulses about random
body axes.  A single pulse returns the array to its starting momentum, so a
sequence of them is what actually drives the gimbals into awkward corners of
the configuration space: the momentum comes back to zero, the gimbal angles do
not.  That is the regime in which reconfiguring the gimbals is supposed to pay.

The **policy dataset** labels states with the null-motion coefficient a
short-horizon oracle would choose.  For a state visited during an SR-inverse
run, each candidate coefficient is held constant over ``horizon`` steps of
simulated steering and scored by the total momentum error it accumulates; the
best candidate, refined by a three-point parabola, is the label.  The oracle
sees the future commanded torque and the trained policy does not, so the
oracle is an upper bound on what the policy can reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arrays import CMGArray
from .ml import DEFAULT_MAX_NULL_RATE, policy_features
from ._fast import FastStepper
from .nullmotion import unit_null_vector
from .simulate import TorqueProfile, rest_to_rest_profile
from .steering import DEFAULT_LAM0, DEFAULT_MU, sr_inverse_steer

__all__ = [
    "ManoeuvreSuite",
    "PolicyDataset",
    "generate_policy_dataset",
    "manoeuvre_suite",
    "rollout_score",
]


@dataclass(frozen=True)
class ManoeuvreSuite:
    """A reproducible set of manoeuvres.

    Attributes
    ----------
    profiles
        One :class:`~cmgsteer.simulate.TorqueProfile` per manoeuvre.
    initial_deltas
        ``(n_manoeuvres, n)`` starting gimbal angles [rad].
    seed
        The integer seed that generated the suite.
    """

    profiles: tuple[TorqueProfile, ...]
    initial_deltas: NDArray[np.float64]
    seed: int

    def __len__(self) -> int:
        return len(self.profiles)

    def __iter__(self):
        return iter(zip(self.profiles, self.initial_deltas))

    @property
    def n_steps(self) -> int:
        """Total number of steering steps in the whole suite."""
        return int(sum(p.n_steps for p in self.profiles))


def manoeuvre_suite(
    array: CMGArray,
    n_manoeuvres: int,
    seed: int,
    n_segments: int = 4,
    segment_duration: float = 6.0,
    dt: float = 0.02,
    momentum_fraction: tuple[float, float] = (0.35, 0.65),
    initial_spread: float = 0.15,
    shape: str = "sine",
) -> ManoeuvreSuite:
    """Build a seeded suite of multi-segment manoeuvres.

    Parameters
    ----------
    array
        The CMG array; only its momentum capacity and size are used.
    n_manoeuvres
        Number of manoeuvres, >= 1.
    seed
        Integer seed.
    n_segments
        Rest-to-rest pulses per manoeuvre, >= 1.
    segment_duration
        Duration of each pulse [s].
    dt
        Step length [s].
    momentum_fraction
        Peak stored momentum of each pulse, as a fraction of the array's total
        momentum capacity.  ``(0.35, 0.65)`` of a four-CMG unit array means
        1.4 to 2.6 N*m*s.
    initial_spread
        Starting gimbal angles are drawn uniformly from
        ``[-initial_spread, initial_spread]`` [rad].
    shape
        Pulse shape, ``"sine"`` or ``"bang-bang"``.
    """
    if n_manoeuvres < 1:
        raise ValueError(f"n_manoeuvres must be >= 1, got {n_manoeuvres}")
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    lo, hi = momentum_fraction
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(
            f"momentum_fraction must satisfy 0 < lo <= hi < 1, got {momentum_fraction}"
        )
    rng = np.random.default_rng(seed)
    cap = array.total_momentum_capacity
    profiles: list[TorqueProfile] = []
    starts = np.empty((n_manoeuvres, array.n_cmgs))
    for i in range(n_manoeuvres):
        rows = []
        for _ in range(n_segments):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            dh = float(rng.uniform(lo, hi)) * cap
            rows.append(
                rest_to_rest_profile(axis, dh, segment_duration, dt, shape=shape).torques
            )
        profiles.append(TorqueProfile(np.vstack(rows), dt, name=f"manoeuvre{i + 1}"))
        starts[i] = rng.uniform(-initial_spread, initial_spread, array.n_cmgs)
    return ManoeuvreSuite(tuple(profiles), starts, int(seed))


def rollout_score(
    array: CMGArray,
    deltas: ArrayLike,
    torques: ArrayLike,
    dt: float,
    coefficient: float,
    max_null_rate: float = DEFAULT_MAX_NULL_RATE,
    lam0: float = DEFAULT_LAM0,
    mu: float = DEFAULT_MU,
    max_gimbal_rate: float | None = None,
    fast: bool = True,
) -> float:
    """Momentum error accumulated by holding one null coefficient over a horizon.

    Returns the path-length sum ``sum_k |(-tau_k dt) - (h_{k+1} - h_k)|``
    [N*m*s] produced by SR-inverse steering with a constant null-motion
    coefficient.  Lower is better; this is the oracle's objective.

    ``fast=True`` routes the step through :class:`cmgsteer._fast.FastStepper`,
    which fuses the three singular value decompositions of the public path into
    one.  ``fast=False`` uses the public
    :func:`cmgsteer.steering.sr_inverse_steer` path; the two agree to round-off
    and ``tests/test_dataset.py`` pins that.
    """
    d = np.asarray(deltas, dtype=float).reshape(-1).copy()
    taus = np.atleast_2d(np.asarray(torques, dtype=float))
    k = float(coefficient)
    total = 0.0
    if fast:
        stepper = FastStepper(array, lam0, mu, max_gimbal_rate)
        h_prev = stepper.momentum(d)
        for tau in taus:
            d = d + stepper.step(d, tau, k, max_null_rate) * dt
            h_now = stepper.momentum(d)
            total += float(np.linalg.norm((-tau * dt) - (h_now - h_prev)))
            h_prev = h_now
        return total
    h_prev = array.momentum(d)
    for tau in taus:
        null = None
        if k != 0.0:
            try:
                null = max_null_rate * k * unit_null_vector(array, d)
            except ValueError:
                null = None
        res = sr_inverse_steer(
            array,
            d,
            tau,
            lam0=lam0,
            mu=mu,
            null_rates=null,
            max_gimbal_rate=max_gimbal_rate,
        )
        d = d + array.expand_rates(res.gimbal_rates) * dt
        h_now = array.momentum(d)
        total += float(np.linalg.norm((-tau * dt) - (h_now - h_prev)))
        h_prev = h_now
    return total


@dataclass(frozen=True)
class PolicyDataset:
    """Features and oracle labels for the learned null-motion policy.

    Attributes
    ----------
    features
        ``(n_samples, n_features)`` from :func:`cmgsteer.ml.policy_features`.
    coefficients
        ``(n_samples,)`` oracle-optimal null coefficients in ``[-1, 1]``.
    candidate_scores
        ``(n_samples, n_candidates)`` horizon momentum error [N*m*s] per
        candidate coefficient.
    candidates
        ``(n_candidates,)`` the grid the oracle searched.
    gradient_scores
        ``(n_samples,)`` horizon score of the classical gradient policy at the
        same states, for reference.
    seed, horizon, dt, max_null_rate
        Generation settings.
    """

    features: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    candidate_scores: NDArray[np.float64]
    candidates: NDArray[np.float64]
    gradient_scores: NDArray[np.float64]
    seed: int
    horizon: int
    dt: float
    max_null_rate: float

    @property
    def n_samples(self) -> int:
        """Number of labelled states."""
        return int(self.features.shape[0])

    @property
    def zero_scores(self) -> NDArray[np.float64]:
        """Horizon score of ``k = 0`` (plain SR inverse) at each state [N*m*s]."""
        j = int(np.argmin(np.abs(self.candidates)))
        return self.candidate_scores[:, j]

    @property
    def best_scores(self) -> NDArray[np.float64]:
        """Horizon score of the best candidate at each state [N*m*s]."""
        return self.candidate_scores.min(axis=1)


def _refine(candidates: NDArray[np.float64], scores: NDArray[np.float64]) -> float:
    """Three-point parabolic refinement of the discrete argmin."""
    j = int(np.argmin(scores))
    if j in (0, scores.size - 1):
        return float(candidates[j])
    y0, y1, y2 = scores[j - 1], scores[j], scores[j + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom <= 0.0:
        return float(candidates[j])
    step = 0.5 * (y0 - y2) / denom
    step = float(np.clip(step, -1.0, 1.0))
    spacing = float(candidates[j + 1] - candidates[j])
    return float(np.clip(candidates[j] + step * spacing, candidates[0], candidates[-1]))


def generate_policy_dataset(
    array: CMGArray,
    n_samples: int,
    seed: int,
    horizon: int = 25,
    n_candidates: int = 9,
    max_null_rate: float = DEFAULT_MAX_NULL_RATE,
    stride: int = 17,
    suite: ManoeuvreSuite | None = None,
    n_manoeuvres: int = 24,
    lam0: float = DEFAULT_LAM0,
    mu: float = DEFAULT_MU,
    max_gimbal_rate: float | None = 2.0,
    gradient_gain: float = 1.0,
) -> PolicyDataset:
    """Generate the lookahead-oracle dataset.

    States are visited by running plain SR-inverse steering over a seeded
    manoeuvre suite and sampling every ``stride`` steps.  At each sampled state
    every candidate coefficient in ``linspace(-1, 1, n_candidates)`` is held
    constant for ``horizon`` steps and scored by :func:`rollout_score`; the
    label is the parabola-refined argmin.

    Parameters
    ----------
    n_samples
        Number of labelled states to collect, >= 1.
    seed
        Integer seed for the manoeuvre suite (and therefore for the states).
    horizon
        Lookahead length in steps.
    n_candidates
        Size of the coefficient grid, >= 3 and odd so that ``k = 0`` is on it.
    stride
        Sample one state in every ``stride`` steering steps.
    suite
        Use this suite instead of generating one from ``seed``.
    max_gimbal_rate
        Rate limit used in both the state-visiting run and the rollouts
        [rad/s].

    Notes
    -----
    Cost is ``n_samples * (n_candidates + 1) * horizon`` simulated steering
    steps plus the states' own run; with the defaults that is about 260 000
    steps and roughly a minute on two cores.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_candidates < 3 or n_candidates % 2 == 0:
        raise ValueError(f"n_candidates must be odd and >= 3, got {n_candidates}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    suite = suite or manoeuvre_suite(array, n_manoeuvres, seed)
    candidates = np.linspace(-1.0, 1.0, n_candidates)
    stepper = FastStepper(array, lam0, mu, max_gimbal_rate)

    feats: list[NDArray[np.float64]] = []
    labels: list[float] = []
    scores: list[NDArray[np.float64]] = []
    grad_scores: list[float] = []

    for profile, start in suite:
        d = np.asarray(start, dtype=float).copy()
        taus = profile.torques
        n = profile.n_steps
        for step in range(n):
            tau = taus[step]
            if step % stride == 0 and len(feats) < n_samples and step + horizon <= n:
                window = taus[step : step + horizon]
                row = np.array(
                    [
                        rollout_score(
                            array,
                            d,
                            window,
                            profile.dt,
                            c,
                            max_null_rate=max_null_rate,
                            lam0=lam0,
                            mu=mu,
                            max_gimbal_rate=max_gimbal_rate,
                        )
                        for c in candidates
                    ]
                )
                feats.append(policy_features(array, d, tau))
                scores.append(row)
                labels.append(_refine(candidates, row))
                grad_scores.append(
                    _gradient_rollout_score(
                        array, d, window, profile.dt, stepper, gradient_gain, max_null_rate
                    )
                )
            d = d + stepper.step(d, tau, 0.0, max_null_rate) * profile.dt
        if len(feats) >= n_samples:
            break

    if not feats:
        raise ValueError(
            "no states were sampled; increase n_manoeuvres or reduce stride/horizon"
        )
    return PolicyDataset(
        features=np.array(feats),
        coefficients=np.array(labels),
        candidate_scores=np.array(scores),
        candidates=candidates,
        gradient_scores=np.array(grad_scores),
        seed=int(seed),
        horizon=int(horizon),
        dt=float(suite.profiles[0].dt),
        max_null_rate=float(max_null_rate),
    )


def _gradient_rollout_score(
    array: CMGArray,
    deltas: NDArray[np.float64],
    torques: NDArray[np.float64],
    dt: float,
    stepper: FastStepper,
    gradient_gain: float,
    max_null_rate: float,
) -> float:
    """Horizon momentum error of the classical gradient policy from one state."""
    d = np.asarray(deltas, dtype=float).copy()
    total = 0.0
    h_prev = stepper.momentum(d)
    for tau in torques:
        d = d + stepper.step(d, tau, 0.0, max_null_rate, gradient_gain=gradient_gain) * dt
        h_now = stepper.momentum(d)
        total += float(np.linalg.norm((-tau * dt) - (h_now - h_prev)))
        h_prev = h_now
    return total
