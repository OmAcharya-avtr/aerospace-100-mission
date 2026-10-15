"""Physical constants used by the twin models, with their sources.

Every value here is a defined or measured physical constant, not a tuned model
parameter. Model parameters live in :mod:`twincalib.scenarios`.
"""

from __future__ import annotations

STEFAN_BOLTZMANN: float = 5.670374419184431e-8
"""Stefan-Boltzmann constant, W m^-2 K^-4.

Source: CODATA/SI 2019 revision, an exact consequence of the defined values of
h, k_B and c: sigma = 2 pi^5 k_B^4 / (15 h^3 c^2). ``scipy.constants`` carries
more digits of the same number; ``tests/test_model.py`` checks agreement to a
relative tolerance of 1e-12 rather than bit-exactly, because the two literals
are truncations of the same irrational value at different places.
"""

T_DEEP_SPACE: float = 2.72548
"""Cosmic microwave background temperature, K.

Source: Fixsen, D. J. (2009), "The temperature of the cosmic microwave
background", Astrophysical Journal 707(2), 916-920: T = 2.72548 +/- 0.00057 K.
Used as the radiative sink temperature for a surface with an unobstructed view
of deep space. Assumption: no planetary or solar infrared input to the radiator;
valid for a radiator pointed away from the Earth limb and the Sun. Contributes
h_rad * T_DEEP_SPACE**4, which is of order 1e-6 W for the shipped designs and is
therefore negligible but retained for correctness.
"""

ZERO_CELSIUS: float = 273.15
"""Ice point, K. Source: definition of the Celsius scale, SI Brochure 9th ed."""
