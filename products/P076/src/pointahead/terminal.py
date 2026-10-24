"""Terminal states and the crosslink they form.

A :class:`TerminalState` is a position, a velocity and an epoch in one
inertial frame. A :class:`Crosslink` pairs a local terminal -- the one doing the
pointing -- with a remote terminal, and attaches a trajectory model to each.

The local terminal's velocity is not optional and is not a refinement. It
enters the point-ahead angle at first order through the aberration of the
received beam and of the transmitted beam (see :mod:`pointahead.aberration`),
and omitting it would change a low Earth orbit answer by tens of
microradians.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import C_LIGHT, as_vector, norm, unit
from .kepler import MU_EARTH, KeplerTrajectory, LinearTrajectory, Trajectory

TRAJECTORY_MODELS = ("kepler", "linear")


@dataclass(frozen=True)
class TerminalState:
    """Inertial state of one optical terminal.

    Attributes
    ----------
    position_m
        Position in the chosen inertial frame, metres.
    velocity_mps
        Velocity in the same frame, m/s.
    epoch_s
        Absolute epoch of the state, seconds. All epochs in one crosslink must
        share an origin; the value of the origin is never used.
    name
        Label carried into reports.
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray
    epoch_s: float = 0.0
    name: str = "terminal"

    def __post_init__(self) -> None:
        position = as_vector(self.position_m, f"{self.name}.position_m")
        velocity = as_vector(self.velocity_mps, f"{self.name}.velocity_mps")
        if float(np.linalg.norm(position)) == 0.0:
            raise ValueError(f"{self.name}.position_m must be non-zero")
        speed = float(np.linalg.norm(velocity))
        if speed >= C_LIGHT:
            raise ValueError(
                f"{self.name}.velocity_mps has speed {speed} m/s, which is not below c; "
                "check the units, this field is metres per second"
            )
        if not np.isfinite(float(self.epoch_s)):
            raise ValueError(f"{self.name}.epoch_s must be finite, got {self.epoch_s!r}")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", velocity)

    @property
    def speed_mps(self) -> float:
        """Speed in m/s."""
        return norm(self.velocity_mps)

    @property
    def beta(self) -> np.ndarray:
        """Dimensionless velocity ``v / c``."""
        return self.velocity_mps / C_LIGHT

    def trajectory(self, model: str = "kepler", mu: float = MU_EARTH) -> Trajectory:
        """Build a trajectory for this state under the named model."""
        if model == "kepler":
            return KeplerTrajectory(self.position_m, self.velocity_mps, self.epoch_s, mu)
        if model == "linear":
            return LinearTrajectory(self.position_m, self.velocity_mps, self.epoch_s)
        raise ValueError(f"unknown trajectory model {model!r}; expected one of {TRAJECTORY_MODELS}")


@dataclass(frozen=True)
class Crosslink:
    """A local terminal pointing at a remote terminal.

    Attributes
    ----------
    local
        The terminal whose transmit direction is being computed.
    remote
        The far terminal.
    trajectory_model
        ``"kepler"`` or ``"linear"``. Applied to both terminals. The remote
        trajectory is what the light-time solution queries at the retarded and
        advanced epochs; the local trajectory is used only for the ephemeris
        staleness term.
    mu
        Gravitational parameter for the Kepler model, m^3/s^2.
    label
        Free text carried into reports.
    """

    local: TerminalState
    remote: TerminalState
    trajectory_model: str = "kepler"
    mu: float = MU_EARTH
    label: str = "crosslink"
    frame: str = "unspecified inertial"

    def __post_init__(self) -> None:
        if self.trajectory_model not in TRAJECTORY_MODELS:
            raise ValueError(
                f"trajectory_model must be one of {TRAJECTORY_MODELS}, "
                f"got {self.trajectory_model!r}"
            )
        if abs(self.local.epoch_s - self.remote.epoch_s) > 0.0:
            raise ValueError(
                "local and remote states must share an epoch; propagate one of them first "
                f"(local {self.local.epoch_s}, remote {self.remote.epoch_s})"
            )
        if self.range_m == 0.0:
            raise ValueError("local and remote terminals are co-located, so there is no link")

    @property
    def epoch_s(self) -> float:
        """Common epoch of both terminal states, seconds."""
        return float(self.local.epoch_s)

    @property
    def separation_m(self) -> np.ndarray:
        """Vector from local to remote at the common epoch, metres."""
        return self.remote.position_m - self.local.position_m

    @property
    def range_m(self) -> float:
        """Instantaneous geometric range at the common epoch, metres."""
        return float(np.linalg.norm(self.separation_m))

    @property
    def line_of_sight(self) -> np.ndarray:
        """Unit vector from local to remote at the common epoch."""
        return unit(self.separation_m, "separation")

    @property
    def relative_velocity_mps(self) -> np.ndarray:
        """``v_remote - v_local`` in m/s."""
        return self.remote.velocity_mps - self.local.velocity_mps

    @property
    def nominal_light_time_s(self) -> float:
        """``range / c``, the zeroth iterate of both light-time solutions."""
        return self.range_m / C_LIGHT

    def trajectories(self) -> tuple[Trajectory, Trajectory]:
        """``(local, remote)`` trajectories under the declared model."""
        return (
            self.local.trajectory(self.trajectory_model, self.mu),
            self.remote.trajectory(self.trajectory_model, self.mu),
        )

    def reversed(self) -> Crosslink:
        """The same link seen from the far terminal.

        Used by the sign-reversal invariant: the point-ahead offset of the
        reverse link is the negative of the forward one in a common basis,
        to first order.
        """
        return Crosslink(
            local=self.remote,
            remote=self.local,
            trajectory_model=self.trajectory_model,
            mu=self.mu,
            label=f"{self.label} (reversed)",
            frame=self.frame,
        )

    def boosted(self, delta_velocity_mps: object) -> Crosslink:
        """The same link with a constant velocity added to both terminals.

        A constant boost changes the inertial frame. The *total* point-ahead
        angle must be invariant under it to first order, while the split
        between the light-time lead and the aberration terms must not be. Both
        are measured in ``validation/validate_frames.py``.

        The Kepler model is not boost-invariant, because adding a velocity to
        an orbit changes the orbit. This method therefore forces the linear
        trajectory model, which is the only self-consistent choice, and says so
        rather than returning a quietly wrong object.
        """
        delta = as_vector(delta_velocity_mps, "delta_velocity_mps")
        return Crosslink(
            local=TerminalState(
                self.local.position_m, self.local.velocity_mps + delta, self.epoch_s,
                self.local.name,
            ),
            remote=TerminalState(
                self.remote.position_m, self.remote.velocity_mps + delta, self.epoch_s,
                self.remote.name,
            ),
            trajectory_model="linear",
            mu=self.mu,
            label=f"{self.label} (boosted)",
            frame=f"{self.frame} boosted by a constant velocity",
        )


@dataclass(frozen=True)
class BeamContext:
    """The transmit beam the point-ahead error has to be judged against.

    A point-ahead error in radians means nothing on its own. The same 5
    microradian error is 10 per cent of a 50 microradian beam and the whole of a
    5 microradian beam, so every budget this package produces carries the beam
    it was judged against.

    Attributes
    ----------
    wavelength_m
        Optical wavelength, metres.
    waist_m
        Gaussian beam waist radius ``w0`` at the transmit aperture, metres,
        defined at the 1/e^2 intensity radius.
    label
        Free text.
    """

    wavelength_m: float
    waist_m: float
    label: str = "beam"
    _checked: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, value in (("wavelength_m", self.wavelength_m), ("waist_m", self.waist_m)):
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        if float(self.waist_m) < float(self.wavelength_m):
            raise ValueError(
                f"waist_m {self.waist_m!r} is below wavelength_m {self.wavelength_m!r}; "
                "the paraxial Gaussian far-field divergence is not valid there"
            )
