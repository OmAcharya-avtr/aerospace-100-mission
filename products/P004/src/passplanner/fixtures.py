"""Public TLE test fixtures (real, historic element sets).

Both TLEs are widely published historic element sets, reproduced verbatim;
checksums are validated at import of :class:`passplanner.passes.TLE`.
They are YEARS out of date -- use them only for testing/demonstration, never
for finding the satellites today (SGP4 accuracy degrades km/day-class away
from epoch; Vallado et al. 2006, AIAA 2006-6753, "Revisiting Spacetrack
Report #3").
"""

from __future__ import annotations

from .passes import TLE

#: ISS (ZARYA), epoch 2008-09-20 12:25:40 UTC (day 264.51782528 of 2008).
#: Source: the canonical example TLE from the python-sgp4 documentation
#: (Rhodes, https://pypi.org/project/sgp4/), originally distributed with the
#: Vallado et al. 2006 SGP4 verification materials.
ISS_2008 = TLE(
    "ISS (ZARYA)",
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)

#: NOAA 14, epoch 1997-11-16 21:49:37 UTC (day 320.90946019 of 1997).
#: Source: the worked example in T.S. Kelso's Celestrak column
#: "FAQ: Two-Line Element Set Format" (Satellite Times, 1998; celestrak.org).
NOAA14_1997 = TLE(
    "NOAA 14",
    "1 23455U 94089A   97320.90946019  .00000140  00000-0  10191-3 0  2621",
    "2 23455  99.0090 272.6745 0008546 223.1686 136.8816 14.11711747148495",
)

ALL_FIXTURES: tuple[TLE, ...] = (ISS_2008, NOAA14_1997)
