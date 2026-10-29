"""Seeded synthetic rate records with complete ground truth. No flight data anywhere.

Every number in this repository comes from this module. The closed forms are
printed here so that a reader can see exactly what the measured scores are
scores of, and so that the limits of the generator are visible rather than
implied. ``DATASET_CARD.md`` states those limits in prose.

The motion
----------
The body rate is a sum of sinusoids on a declared set of axes:

    (23)  w(t) = sum_k a_k sin( 2 pi f_k t + phi_k ) u_k      rad/s

with ``u_k`` unit axes. The **excitation profile** chooses the axes and
amplitudes, and it is the single most important control in this package because
by equation (10) the excitation second moment ``M`` determines which alignment
degrees of freedom are observable at all. The five declared profiles are:

``single_axis``
    One axis, one sinusoid. ``M`` has ``p = (1, 0, 0)`` in exact arithmetic, so
    the alignment component **about that axis is unobservable at any data
    volume**. This is the record the specification asks for and it is the
    centre of ``validation/validate_unobservable.py``.
``near_single_axis``
    The same, plus a transverse component of ``leak`` times the amplitude on a
    second axis. ``p1 = 1 - O(leak^2)``, so the third degree of freedom is
    observable in principle and poorly conditioned in practice. This is the
    case that separates "rank deficient" from "useless", and it is why the
    verdict in :mod:`aligncal.observability` is three-valued.
``two_axis``
    Two orthogonal axes at different frequencies. ``p = (1/2, 1/2, 0)``, which
    by (10) makes **all three** degrees of freedom observable: rotation about
    two independent axes is sufficient, and that is a result worth stating
    because it is easy to assume three are needed.
``three_axis``
    Three axes at incommensurate frequencies and unequal amplitudes.
``isotropic``
    Band-limited random rates with no preferred axis, as an excitation-rich
    upper reference.

The axes are rotated by a per-seed random rotation so that no result in this
repository depends on the excitation happening to lie along a body axis, which
would make the "name the unobservable direction" claim trivially easy.

The instruments
---------------
Each sensor adds independent zero-mean Gaussian white noise of declared
one-sigma. That is the whole error model: there is **no** bias, no scale-factor
error, no angular random walk, no g-sensitivity and no quantisation, and none
of the scores in this repository therefore say anything about those. Stated
here and in ``DATASET_CARD.md`` rather than left to be discovered.

The misalignment of secondary ``j`` is drawn uniformly on the sphere with
magnitude uniform in ``[0, theta_max]``, and the generated secondary rate is
the **exact** rotation of equation (4), not its linearisation, so the
linearisation error of every estimator here is measured rather than cancelled.

Determinism
-----------
Every function takes an explicit integer seed and uses
``numpy.random.default_rng``, so a given seed reproduces a given record
bit-for-bit on a given numpy. Nothing is cached and nothing is written to disk:
the generated arrays for the largest configuration used anywhere in this
repository are under 2 MB, so no data file is committed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import rotation_from_vector
from .record import AlignmentRecord, SensorSpec

__all__ = [
    "PROFILES",
    "GeneratedRecord",
    "generate",
    "rate_profile",
    "training_population",
]

#: The declared excitation profiles, in increasing richness. The value is the
#: number of independent rotation axes the profile excites in exact arithmetic,
#: which by equation (10) is the number of alignment degrees of freedom that
#: are *not* suppressed.
PROFILES: dict[str, int] = {
    "single_axis": 1,
    "near_single_axis": 2,
    "two_axis": 2,
    "three_axis": 3,
    "isotropic": 3,
}


def rate_profile(
    profile: str,
    times: np.ndarray,
    *,
    amplitude: float = 0.05,
    seed: int = 0,
    leak: float = 0.02,
) -> np.ndarray:
    """Body rate ``w(t)`` of equation (23) for a declared profile, rad/s.

    Parameters
    ----------
    profile:
        One of :data:`PROFILES`.
    times:
        Shape ``(n,)``, seconds.
    amplitude:
        Peak rate of the dominant axis, rad/s. Must be positive.
    seed:
        Integer seed for the axis rotation and the phases.
    leak:
        Transverse amplitude fraction for ``near_single_axis``, dimensionless.
        Must be in ``(0, 1]``.

    Returns
    -------
    ndarray
        Shape ``(n, 3)``, rad/s.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; declared profiles are {sorted(PROFILES)}")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(f"amplitude must be finite and > 0 rad/s, got {amplitude!r}")
    if not np.isfinite(leak) or not 0.0 < leak <= 1.0:
        raise ValueError(f"leak must be in (0, 1], got {leak!r}")
    stamp = np.asarray(times, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))
    # A random but seeded rotation of the excitation axes, so that no result
    # depends on the motion lying along a body axis.
    frame = rotation_from_vector(rng.normal(size=3) * 0.7)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    base = np.array([0.31, 0.17, 0.23])
    rate = np.zeros((stamp.size, 3))
    if profile == "single_axis":
        rate[:, 0] = amplitude * np.sin(2.0 * np.pi * base[0] * stamp + phases[0])
    elif profile == "near_single_axis":
        rate[:, 0] = amplitude * np.sin(2.0 * np.pi * base[0] * stamp + phases[0])
        rate[:, 1] = leak * amplitude * np.sin(2.0 * np.pi * base[1] * stamp + phases[1])
    elif profile == "two_axis":
        rate[:, 0] = amplitude * np.sin(2.0 * np.pi * base[0] * stamp + phases[0])
        rate[:, 1] = amplitude * np.sin(2.0 * np.pi * base[1] * stamp + phases[1])
    elif profile == "three_axis":
        rate[:, 0] = amplitude * np.sin(2.0 * np.pi * base[0] * stamp + phases[0])
        rate[:, 1] = 0.8 * amplitude * np.sin(2.0 * np.pi * base[1] * stamp + phases[1])
        rate[:, 2] = 0.6 * amplitude * np.sin(2.0 * np.pi * base[2] * stamp + phases[2])
    else:  # isotropic
        raw = rng.normal(size=(stamp.size, 3))
        # A 5-sample moving average band-limits the sequence without needing a
        # filter design; the resulting rate is not white and that is the point.
        kernel = np.ones(5) / 5.0
        smoothed = np.vstack(
            [np.convolve(raw[:, axis], kernel, mode="same") for axis in range(3)]
        ).T
        scale = float(np.max(np.abs(smoothed))) or 1.0
        rate = amplitude * smoothed / scale
    return rate @ frame.T


@dataclass(frozen=True)
class GeneratedRecord:
    """A generated record with its complete ground truth.

    Attributes
    ----------
    record:
        The :class:`aligncal.record.AlignmentRecord` an estimator sees.
    truth:
        Mapping from secondary name to the injected misalignment vector in
        radians, in the reference frame.
    true_rates:
        The noiseless body rate of (23), rad/s, shape ``(n, 3)``. Not available
        to any estimator; used only to measure what the noise in the Gramian
        costs.
    profile, amplitude, noise_sigma, seed, leak:
        The generating configuration, carried so that every figure can label
        itself.
    """

    record: AlignmentRecord
    truth: dict[str, np.ndarray]
    true_rates: np.ndarray
    profile: str
    amplitude: float
    noise_sigma: float
    seed: int
    leak: float

    def truth_component(self, name: str, direction: np.ndarray) -> float:
        """Injected misalignment of ``name`` along ``direction``, radians."""
        unit = np.asarray(direction, dtype=float).reshape(3)
        unit = unit / float(np.linalg.norm(unit))
        return float(np.dot(self.truth[name], unit))


def generate(
    *,
    profile: str = "three_axis",
    samples: int = 2000,
    rate_hz: float = 100.0,
    amplitude: float = 0.05,
    noise_sigma: float = 2.0e-4,
    theta_max_rad: float = 3.0e-3,
    n_secondaries: int = 1,
    seed: int = 20261029,
    leak: float = 0.02,
    theta_rad: np.ndarray | None = None,
) -> GeneratedRecord:
    """Generate one record with a known injected misalignment per secondary.

    Parameters
    ----------
    profile:
        Excitation profile, one of :data:`PROFILES`.
    samples:
        Number of rate samples. At least 2.
    rate_hz:
        Sampling rate, Hz. Must be positive.
    amplitude:
        Peak body rate, rad/s.
    noise_sigma:
        Per-axis one-sigma white noise of every sensor, rad/s.
    theta_max_rad:
        Upper bound on the injected misalignment magnitude, radians. The
        magnitude is uniform in ``[0, theta_max_rad]`` and the direction is
        uniform on the sphere.
    n_secondaries:
        How many secondary sensors to generate, each with its own declared
        mounting and its own injected misalignment. At least 1.
    seed:
        Integer seed.
    leak:
        Transverse fraction for ``near_single_axis``.
    theta_rad:
        Optional explicit misalignment to inject into **every** secondary,
        radians, shape ``(3,)``. When given, ``theta_max_rad`` is not used and
        the misalignment is not random. This exists so that a negative control
        can be anchored on a chosen truth -- for instance a misalignment lying
        entirely along the record's own rotation axis, which is the direction
        the record cannot support -- rather than on whatever a seed produced.

    Returns
    -------
    GeneratedRecord
    """
    if samples < 2:
        raise ValueError(f"generate() needs at least 2 samples, got {samples}")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be finite and > 0, got {rate_hz!r}")
    if not np.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError(f"noise_sigma must be finite and > 0 rad/s, got {noise_sigma!r}")
    if not np.isfinite(theta_max_rad) or theta_max_rad <= 0.0:
        raise ValueError(f"theta_max_rad must be finite and > 0 rad, got {theta_max_rad!r}")
    if n_secondaries < 1:
        raise ValueError(f"generate() needs at least 1 secondary, got {n_secondaries}")
    injected: np.ndarray | None = None
    if theta_rad is not None:
        injected = np.asarray(theta_rad, dtype=float).reshape(-1)
        if injected.size != 3:
            raise ValueError(f"theta_rad must have 3 components, got {injected.size}")
        if not np.all(np.isfinite(injected)):
            raise ValueError("theta_rad must be finite")

    rng = np.random.default_rng(int(seed))
    times = np.arange(int(samples), dtype=float) / float(rate_hz)
    rates = rate_profile(profile, times, amplitude=amplitude, seed=int(seed), leak=leak)

    reference = SensorSpec("reference", noise_sigma)
    measured_reference = rates + rng.normal(scale=noise_sigma, size=rates.shape)

    secondaries: dict[str, SensorSpec] = {}
    blocks: dict[str, np.ndarray] = {}
    truth: dict[str, np.ndarray] = {}
    for index in range(int(n_secondaries)):
        name = f"secondary_{index + 1}"
        mounting = rotation_from_vector(rng.normal(size=3) * 0.9)
        direction = rng.normal(size=3)
        direction = direction / float(np.linalg.norm(direction))
        theta = direction * float(rng.uniform(0.0, theta_max_rad))
        if injected is not None:
            theta = injected.copy()
        # The EXACT rotation of equation (4), not its linearisation.
        true_mounting = mounting @ rotation_from_vector(-theta)
        # Row-wise application: ``rates @ A.T`` has rows ``A w_t``, so this is
        # ``C_j^true w(t)`` of equation (4). The transpose was omitted in the
        # first draft of this module, which generated ``(C^true)^T w`` instead
        # and made every batch estimate on a non-identity mounting diverge; it
        # is pinned as a regression test.
        clean = rates @ true_mounting.T
        secondaries[name] = SensorSpec(name, noise_sigma, mounting)
        blocks[name] = clean + rng.normal(scale=noise_sigma, size=clean.shape)
        truth[name] = theta

    record = AlignmentRecord(
        times=times,
        reference=reference,
        reference_rates=measured_reference,
        secondaries=secondaries,
        secondary_rates=blocks,
        provenance=(
            f"aligncal.dataset.generate(profile={profile!r}, samples={samples}, "
            f"rate_hz={rate_hz:g}, amplitude={amplitude:g}, noise_sigma={noise_sigma:g}, "
            f"seed={seed}"
            + ("" if injected is None else ", theta_rad=declared")
            + ")"
        ),
    )
    return GeneratedRecord(
        record=record,
        truth=truth,
        true_rates=rates,
        profile=profile,
        amplitude=amplitude,
        noise_sigma=noise_sigma,
        seed=int(seed),
        leak=leak,
    )


def training_population(
    *,
    count: int,
    samples: int = 150,
    rate_hz: float = 100.0,
    seed: int,
    profiles: tuple[str, ...] = ("single_axis", "near_single_axis", "two_axis", "three_axis"),
    amplitude_range: tuple[float, float] = (0.01, 0.08),
    noise_range: tuple[float, float] = (5.0e-5, 8.0e-4),
    theta_max_rad: float = 3.0e-3,
) -> list[GeneratedRecord]:
    """A population of **short** records, which is the case the learned estimator is for.

    Each record draws its profile uniformly from ``profiles``, its amplitude
    log-uniformly from ``amplitude_range`` in rad/s and its noise one-sigma
    log-uniformly from ``noise_range`` in rad/s. ``count`` records of ``samples``
    samples each. The population deliberately includes ``single_axis`` records,
    so the learned estimator is trained on data that contains unobservable
    degrees of freedom and has every opportunity to learn to refuse.

    Compute: 4000 records of 150 samples is about 2 MB of float64 and generates
    in a few seconds on two cores. Nothing is written to disk.
    """
    if count < 1:
        raise ValueError(f"training_population() needs count >= 1, got {count}")
    if not profiles:
        raise ValueError("training_population() needs at least one profile")
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles {unknown}; declared profiles are {sorted(PROFILES)}")
    low_amplitude, high_amplitude = amplitude_range
    low_noise, high_noise = noise_range
    if not 0.0 < low_amplitude <= high_amplitude:
        raise ValueError(f"amplitude_range must be positive and ordered, got {amplitude_range!r}")
    if not 0.0 < low_noise <= high_noise:
        raise ValueError(f"noise_range must be positive and ordered, got {noise_range!r}")
    rng = np.random.default_rng(int(seed))
    population: list[GeneratedRecord] = []
    for index in range(int(count)):
        profile = str(rng.choice(np.asarray(profiles, dtype=object)))
        amplitude = float(
            np.exp(rng.uniform(np.log(low_amplitude), np.log(high_amplitude)))
        )
        noise = float(np.exp(rng.uniform(np.log(low_noise), np.log(high_noise))))
        population.append(
            generate(
                profile=profile,
                samples=samples,
                rate_hz=rate_hz,
                amplitude=amplitude,
                noise_sigma=noise,
                theta_max_rad=theta_max_rad,
                n_secondaries=1,
                seed=int(seed) + 1 + index,
            )
        )
    return population
