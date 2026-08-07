"""Numerical coefficients of the turbulence integrals, with their derivations.

Every coefficient below is either (a) a value quoted in the standard
literature, or (b) derived here from such values by algebra that is shown in
full. No coefficient in this module was invented or fitted.

Primary sources (work-level citations; see README "Citation policy" — no page,
equation or table numbers are quoted anywhere in this package because they were
not verified against physical copies during the build):

* D. L. Fried, "Optical Resolution Through a Randomly Inhomogeneous Medium for
  Very Long and Very Short Exposures", J. Opt. Soc. Am. 56(10), 1372-1379,
  1966.
* D. P. Greenwood, "Bandwidth specification for adaptive optics systems",
  J. Opt. Soc. Am. 67(3), 390-393, 1977.
* D. L. Fried, "Anisoplanatism in adaptive optics", J. Opt. Soc. Am. 72(1),
  52-61, 1982.
* L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005.
* J. W. Hardy, "Adaptive Optics for Astronomical Telescopes", Oxford
  University Press, 1998.
* F. Roddier, "The Effects of Atmospheric Turbulence in Optical Astronomy",
  Progress in Optics XIX, 281-376, 1981.
"""

from __future__ import annotations

from math import gamma as _gamma

__all__ = [
    "C_STRUCTURE_PLANE",
    "C_FRIED_DEFINITION",
    "C_FRIED",
    "C_FRIED_EXACT",
    "C_ISOPLANATIC",
    "C_GREENWOOD",
    "C_GREENWOOD_LAMBDA",
    "C_RYTOV",
    "C_THETA0_OVER_R0",
    "EXPONENT_SEC_ZENITH",
    "EXPONENT_WAVELENGTH",
]

# ---------------------------------------------------------------------------
# Structure-function coefficients
# ---------------------------------------------------------------------------

#: Coefficient in the plane-wave phase structure function for a Kolmogorov
#: spectrum,  D_phi(r) = C * k^2 * r^(5/3) * integral Cn^2(z) dz.
#: Quoted as 2.914 by Fried (1966), Roddier (1981), Andrews & Phillips (2005)
#: and Hardy (1998).  Dimensionless.
C_STRUCTURE_PLANE: float = 2.914

#: Coefficient in Fried's *definition* of r0 via D_phi(r) = C * (r/r0)^(5/3).
#: The exact analytic value is 2 * (24/5 * Gamma(6/5))^(5/6); evaluating it
#: here (rather than hard-coding 6.88) documents the derivation.
C_FRIED_DEFINITION: float = 2.0 * (24.0 / 5.0 * _gamma(6.0 / 5.0)) ** (5.0 / 6.0)
# -> 6.883877 ... , the familiar "6.88".

#: Fried-parameter coefficient:  r0 = [C * k^2 * sec(zeta) * mu_0]^(-3/5).
#: Derivation: equate the two structure-function forms above,
#:     C_STRUCTURE_PLANE * k^2 * r^(5/3) * mu_0 = C_FRIED_DEFINITION * (r/r0)^(5/3)
#: =>  r0 = [ (C_STRUCTURE_PLANE / C_FRIED_DEFINITION) * k^2 * mu_0 ]^(-3/5)
#: with C_STRUCTURE_PLANE / C_FRIED_DEFINITION = 0.42331...
#: The literature universally quotes the 3-significant-figure value 0.423
#: (Fried 1966; Andrews & Phillips 2005; Hardy 1998), which is what this
#: package uses so that its numbers reproduce published ones exactly.
C_FRIED_EXACT: float = C_STRUCTURE_PLANE / C_FRIED_DEFINITION  # 0.4233...
C_FRIED: float = 0.423
# Using 0.423 instead of C_FRIED_EXACT changes r0 by
#   (0.423 / 0.423305)^(-3/5) - 1 = +4.3e-4  (i.e. +0.043 %),
# which is far below the accuracy of any Cn^2 profile model.

#: Isoplanatic-angle coefficient:
#:     theta0 = [C * k^2 * sec(zeta)^(8/3) * mu_(5/3)]^(-3/5).
#: Numerically equal to C_STRUCTURE_PLANE; quoted as 2.914 by Fried (1982),
#: Roddier (1981) and Andrews & Phillips (2005).
C_ISOPLANATIC: float = 2.914

#: Ratio theta0 / (r0 / h_bar) implied by the two coefficients above:
#:     theta0 / r0 = (C_FRIED / C_ISOPLANATIC)^(3/5) / h_bar
#: with h_bar = [mu_(5/3) / mu_0]^(3/5).  Evaluates to 0.31409..., i.e. the
#: familiar theta0 = 0.314 r0 / h_bar (Roddier 1981; Hardy 1998).
C_THETA0_OVER_R0: float = (C_FRIED / C_ISOPLANATIC) ** 0.6

#: Greenwood-frequency coefficient:
#:     f_G = [C * k^2 * sec(zeta) * integral Cn^2(h) v(h)^(5/3) dh]^(3/5).
#: Quoted as 0.102 by Greenwood (1977) and Hardy (1998).  Units of the bracket
#: are s^(-5/3), so f_G is in Hz.
C_GREENWOOD: float = 0.102

#: The equivalent wavelength form,  f_G = C * lambda^(-6/5) * [sec(zeta) *
#: integral Cn^2 v^(5/3) dh]^(3/5), with C = C_GREENWOOD^(3/5) * (2 pi)^(6/5).
#: Evaluates to 2.3067, i.e. the "2.31" quoted by Greenwood (1977) and Hardy
#: (1998).  Derived here, not independently asserted.
C_GREENWOOD_LAMBDA: float = C_GREENWOOD**0.6 * (2.0 * 3.141592653589793) ** 1.2

#: Rytov-variance coefficient:
#:     sigma_R^2 = C * k^(7/6) * sec(zeta)^(11/6) * integral Cn^2(h) W(h) dh.
#: Quoted as 2.25 by Andrews & Phillips (2005) for slant paths.  Consistency
#: check performed in validation/: for a homogeneous horizontal path of length
#: L the plane-wave weight h^(5/6) integrates to (6/11) L^(11/6), giving
#: 2.25 * 6/11 = 1.227, i.e. the textbook 1.23 Cn^2 k^(7/6) L^(11/6).
C_RYTOV: float = 2.25

# ---------------------------------------------------------------------------
# Analytic scaling exponents (used by the docs, the CLI and the tests)
# ---------------------------------------------------------------------------

#: Exponent p in  Q(zeta) = Q(0) * sec(zeta)^p  for each quantity, in the
#: plane-parallel (flat-Earth) atmosphere approximation.  Derivations are in
#: the docstring of each metric in :mod:`atmoprofile.metrics`.
EXPONENT_SEC_ZENITH: dict[str, float] = {
    "r0": -3.0 / 5.0,
    "theta0": -8.0 / 5.0,
    "f_greenwood": 3.0 / 5.0,
    "rytov_plane": 11.0 / 6.0,
    "rytov_spherical": 11.0 / 6.0,
    "scintillation_index": 11.0 / 6.0,
}

#: Exponent q in  Q(lambda) = Q(lambda_ref) * (lambda / lambda_ref)^q.
EXPONENT_WAVELENGTH: dict[str, float] = {
    "r0": 6.0 / 5.0,
    "theta0": 6.0 / 5.0,
    "f_greenwood": -6.0 / 5.0,
    "rytov_plane": -7.0 / 6.0,
    "rytov_spherical": -7.0 / 6.0,
    "scintillation_index": -7.0 / 6.0,
}
