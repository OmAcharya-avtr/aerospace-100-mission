"""Telemetry sets: which channels exist, at what noise level, over which campaign.

A telemetry set is the whole input to an identifiability question. "Is ``J``
identifiable?" has no answer; "is ``J`` identifiable from a 200 s rate record
at 0.02 rad/s sensor noise?" has one, and it is no. This module is where the
question gets its subject.

The design matrix
-----------------
For a telemetry set with channels ``c`` and declared noise standard deviations
``sigma_c``, the scaled sensitivity matrix is the vertical stack

    equation (8)    A = [ S_c / sigma_c ]_c          shape (sum_c N_c, 5)

where ``S_c`` is the channel's sensitivity with respect to the logarithmic
parameters, equation (2). Scaling each row by its own noise standard deviation
is what makes ``A`` the matrix whose Gramian is the Fisher information of a
Gaussian likelihood with independent errors,

    equation (9)    F = A^T A,

and it is also what makes ``A`` dimensionless, so that its singular values can
be compared with each other and its condition number means something. Both
properties matter; see Bard 1974, "Nonlinear Parameter Estimation", Academic
Press, chapters 4 and 7, for the Gaussian-likelihood Fisher information of a
nonlinear regression model and for the scaling argument.

Noise model, assumptions and validity
-------------------------------------
- Additive, zero-mean, Gaussian, independent between samples and between
  channels, with a KNOWN standard deviation per channel. Independence between
  samples is an assumption about the sensor, not a derived property: a
  tachometer that differentiates an encoder count produces correlated rate
  noise, and the correlation would change every interval width reported here.
  Named in README.md under Limitations.
- The declared standard deviations below are declared, not measured:
  0.02 rad/s on the rate channel and 2.0e-4 N m on the reaction-torque
  channel. They set the scale of every Cramer-Rao bound and confidence region
  in this package, and the record-length sweep in
  ``validation/validate_identifiability.py`` shows which conclusions depend on
  them (the practical ones) and which do not (the structural ones).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .plant import (
    CAMPAIGNS,
    Campaign,
    Trajectory,
    WheelParameters,
    simulate,
    simulate_with_sensitivities,
)

__all__ = [
    "RATE_SIGMA_RAD_S",
    "TELEMETRY_SETS",
    "TORQUE_SIGMA_NM",
    "TelemetrySet",
    "design_matrix",
    "measure",
    "residual_vector",
]

#: Declared rate-sensor noise standard deviation, rad/s. Declared, not measured.
RATE_SIGMA_RAD_S: Final[float] = 2.0e-2

#: Declared reaction-torque-sensor noise standard deviation, N m. Declared.
TORQUE_SIGMA_NM: Final[float] = 2.0e-4


@dataclass(frozen=True)
class TelemetrySet:
    """A campaign plus the channels recorded during it and their noise levels.

    ``channels`` is an ordered tuple drawn from ``("rate", "torque")``. The
    order fixes the row order of :func:`design_matrix` and of
    :func:`residual_vector`, and the two must agree, so both read it from here.
    """

    name: str
    campaign: Campaign
    channels: tuple[str, ...] = ("rate",)
    sigmas: dict[str, float] = field(
        default_factory=lambda: {
            "rate": RATE_SIGMA_RAD_S,
            "torque": TORQUE_SIGMA_NM,
        }
    )
    note: str = ""

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a telemetry set must record at least one channel")
        for channel in self.channels:
            if channel not in ("rate", "torque"):
                raise ValueError(
                    f"unknown channel {channel!r}; this package records 'rate' "
                    "(a tachometer) and 'torque' (a reaction-torque sensor between "
                    "the wheel housing and its mount)"
                )
            if self.sigmas.get(channel, 0.0) <= 0.0:
                raise ValueError(
                    f"channel {channel!r} needs a strictly positive noise standard "
                    f"deviation, got {self.sigmas.get(channel)!r}"
                )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels must be distinct, got {self.channels!r}")

    @property
    def rows(self) -> int:
        """Number of scalar measurements in the set."""
        return len(self.channels) * self.campaign.samples


#: The four declared telemetry sets. Their measured identifiability structure is
#: the headline table of ``validation/validate_identifiability_output.txt``.
TELEMETRY_SETS: Final[dict[str, TelemetrySet]] = {
    "rate_full": TelemetrySet(
        "rate_full",
        CAMPAIGNS["full"],
        ("rate",),
        note="200 s rate-only record, 40 s driven. The reference set: rank 4 of "
        "5, the single null direction being the exact scale direction of "
        "equation (6).",
    ),
    "rate_torque_full": TelemetrySet(
        "rate_torque_full",
        CAMPAIGNS["full"],
        ("rate", "torque"),
        note="The same campaign with a reaction-torque sensor added. Full rank, "
        "because the scale direction's sensitivity in the torque channel is "
        "exactly the torque signal, equation (7).",
    ),
    "rate_short": TelemetrySet(
        "rate_short",
        CAMPAIGNS["short"],
        ("rate",),
        note="20 s rate-only record. Structurally identical to rate_full -- rank "
        "4, same null direction -- and PRACTICALLY far worse, which is the "
        "distinction this set exists to make.",
    ),
    "rate_torque_short": TelemetrySet(
        "rate_torque_short",
        CAMPAIGNS["short"],
        ("rate", "torque"),
        note="20 s record with both channels. Structurally full rank -- the "
        "torque channel removes the scale degeneracy whatever the record "
        "length -- and practically unable to determine the drag coefficient at "
        "all. This is the set on which the WEAK verdict appears.",
    ),
    "rate_coast": TelemetrySet(
        "rate_coast",
        CAMPAIGNS["coast"],
        ("rate",),
        note="60 s undriven coast-down. Rank 3 of 5: the scale direction plus "
        "k_t, which a zero current never excites.",
    ),
}


def design_matrix(
    telemetry: TelemetrySet,
    parameters: WheelParameters,
    trajectory: Trajectory | None = None,
) -> tuple[np.ndarray, Trajectory]:
    """Equation (8): the noise-scaled sensitivity matrix, shape ``(rows, 5)``.

    ``trajectory`` may be supplied to avoid re-integrating; it must carry
    sensitivities and have been produced for the same campaign, which is
    checked rather than assumed.
    """
    if trajectory is None:
        trajectory = simulate_with_sensitivities(parameters, telemetry.campaign)
    if trajectory.rate_sensitivity is None or trajectory.torque_sensitivity is None:
        raise ValueError(
            "design_matrix needs a trajectory carrying sensitivities; call "
            "plant.simulate_with_sensitivities, not plant.simulate"
        )
    if trajectory.campaign.name != telemetry.campaign.name:
        raise ValueError(
            f"trajectory was produced for campaign {trajectory.campaign.name!r} but "
            f"telemetry set {telemetry.name!r} uses {telemetry.campaign.name!r}"
        )
    blocks = []
    for channel in telemetry.channels:
        sensitivity = (
            trajectory.rate_sensitivity
            if channel == "rate"
            else trajectory.torque_sensitivity
        )
        blocks.append(sensitivity / telemetry.sigmas[channel])
    return np.vstack(blocks), trajectory


def measure(
    telemetry: TelemetrySet,
    parameters: WheelParameters,
    seed: int,
) -> tuple[np.ndarray, Trajectory]:
    """Noisy telemetry from ``parameters``, stacked in ``telemetry.channels`` order.

    The returned vector is in PHYSICAL units per block (rad/s then N m), not
    scaled by the noise; :func:`residual_vector` does the scaling. ``seed`` is
    the only source of randomness and the generator is constructed from it
    here, so a repeated call with the same seed returns the same bits.
    """
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed!r}")
    truth = simulate(parameters, telemetry.campaign)
    generator = np.random.default_rng(seed)
    blocks = []
    for channel in telemetry.channels:
        clean = truth.rate_rad_s if channel == "rate" else truth.torque_nm
        blocks.append(
            clean + generator.normal(0.0, telemetry.sigmas[channel], size=clean.shape)
        )
    return np.concatenate(blocks), truth


def residual_vector(
    telemetry: TelemetrySet,
    observed: np.ndarray,
    twin: Trajectory,
) -> np.ndarray:
    """The noise-scaled twin-versus-reality residual, shape ``(rows,)``.

    ``(observed - twin) / sigma`` per channel, in the same row order as
    :func:`design_matrix`, so that ``A @ delta`` and this vector are directly
    comparable and ``|r|^2`` is a chi-square statistic.
    """
    array = np.asarray(observed, dtype=float)
    if array.shape != (telemetry.rows,):
        raise ValueError(
            f"telemetry set {telemetry.name!r} has {telemetry.rows} rows but the "
            f"observation has shape {array.shape}"
        )
    if twin.campaign.name != telemetry.campaign.name:
        raise ValueError(
            f"twin was simulated for campaign {twin.campaign.name!r} but telemetry "
            f"set {telemetry.name!r} uses {telemetry.campaign.name!r}"
        )
    blocks = []
    offset = 0
    width = telemetry.campaign.samples
    for channel in telemetry.channels:
        clean = twin.rate_rad_s if channel == "rate" else twin.torque_nm
        blocks.append((array[offset : offset + width] - clean) / telemetry.sigmas[channel])
        offset += width
    return np.concatenate(blocks)
