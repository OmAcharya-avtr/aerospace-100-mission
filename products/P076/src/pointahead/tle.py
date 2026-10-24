"""Terminal states from two-line element sets, via ``sgp4``.

``sgp4`` is the reference implementation of the SGP4/SDP4 propagator and this
module does not reimplement any part of it. It converts an SGP4 output into the
:class:`~pointahead.terminal.TerminalState` this package works in, and it is
separated into its own module so that the rest of the package has no dependency
on an orbit propagator.

Two caveats that matter and are not hidden:

1. **Frame.** SGP4 returns position and velocity in TEME, the true equator and
   mean equinox of date. TEME is not J2000 and it is not ITRF. This module
   labels the frame ``TEME of date`` and passes it through; it performs no frame
   rotation. For a point-ahead angle the frame choice is almost irrelevant,
   because the total depends on the relative transverse velocity and is
   invariant under a change of inertial frame to first order -- but the
   *attribution* between the light-time lead and the aberration terms is not
   invariant, so an attributed budget computed in TEME is an attribution in
   TEME and says so.

2. **Trajectory model.** The light-time solution needs the remote terminal's
   position at the retarded and advanced epochs. Those are tens of milliseconds
   away, over which SGP4 and a two-body propagation of the same state are
   indistinguishable at the angular resolution of this package, so this module
   returns a state and lets the caller choose a trajectory model rather than
   driving SGP4 inside the light-time iteration. An ephemeris staleness term
   built on a two-body propagation of an SGP4 state is a statement about the
   two-body model, not about SGP4, and the README says so under Limitations.

``sgp4`` version 2.27 is the version every number in this repository was
measured against and the dependency is bounded on both sides in
``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec

from .terminal import Crosslink, TerminalState

TEME_FRAME = "TEME of date"


@dataclass(frozen=True)
class TleTerminal:
    """One terminal's SGP4 record with the label to carry into reports."""

    name: str
    satellite: Satrec

    @classmethod
    def from_lines(cls, name: str, line1: str, line2: str) -> TleTerminal:
        """Build from the two element-set lines.

        Raises
        ------
        ValueError
            If ``sgp4`` rejects the element set.
        """
        try:
            satellite = Satrec.twoline2rv(str(line1), str(line2))
        except Exception as exc:  # noqa: BLE001 - sgp4 raises several types
            raise ValueError(
                f"sgp4 could not parse the element set for {name!r}: {exc}"
            ) from exc
        if satellite.error != 0:
            raise ValueError(
                f"sgp4 rejected the element set for {name!r} with error "
                f"{satellite.error}: {SGP4_ERRORS.get(satellite.error, 'unknown error')}"
            )
        return cls(name=str(name), satellite=satellite)

    def state(self, jd: float, fr: float) -> TerminalState:
        """State at Julian date ``jd + fr``, converted to metres and m/s.

        ``sgp4`` returns kilometres and kilometres per second; both are scaled
        here, which is the only unit conversion in this module.
        """
        error, position_km, velocity_km_s = self.satellite.sgp4(float(jd), float(fr))
        if error != 0:
            raise ValueError(
                f"sgp4 propagation of {self.name!r} failed at jd={jd!r}+{fr!r} "
                f"with error code {error}: {SGP4_ERRORS.get(error, 'unknown error')}"
            )
        return TerminalState(
            position_m=np.asarray(position_km, dtype=np.float64) * 1.0e3,
            velocity_mps=np.asarray(velocity_km_s, dtype=np.float64) * 1.0e3,
            epoch_s=0.0,
            name=self.name,
        )


def crosslink_from_tles(
    local: TleTerminal,
    remote: TleTerminal,
    jd: float,
    fr: float = 0.0,
    trajectory_model: str = "kepler",
    label: str = "tle_crosslink",
) -> Crosslink:
    """A crosslink between two SGP4-propagated terminals at one Julian date.

    The epoch of both states is set to zero seconds, because the point-ahead
    angle is defined at one event and only the offsets from that event matter.
    """
    return Crosslink(
        local=local.state(jd, fr),
        remote=remote.state(jd, fr),
        trajectory_model=trajectory_model,
        label=label,
        frame=TEME_FRAME,
    )
