"""Baseline 1: the same object's earlier element set against its later one.

This is the standard practical proxy and it needs no second dynamical model.
Its structural limitation is that both sides are SGP4, so any error common to
the theory -- the atmosphere model, the truncation of the Brouwer-Lyddane
expansion, the absence of solar radiation pressure -- cancels out of the
comparison entirely. It measures the part of the error that came from the
*fit*, not the part that came from the *model*.

This package has no two-line element archive, so there is exactly one real
same-object pair available in this environment, and it is used and labelled as
what it is. The machinery is then validated on synthetic pairs built from
explicitly chosen mean elements, where the expected separation follows from
geometry and can be compared against the measurement:

- A pair differing only in **mean anomaly** by ``dM`` has a separation that is
  constant in time and equal to the chord ``2 a sin(dM/2)``, which for a small
  offset is ``a dM``.
- A pair differing only in **mean motion** by ``dn``, with the phases arranged
  to agree at the later epoch, has an along-track separation that grows as
  ``a dn t``.

Both are independently derived predictions about the geometry, not second
implementations of the measurement, so agreement is evidence about the
measurement rather than about itself.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines, is_fast

bootstrap()

import numpy as np  # noqa: E402
from sgp4.api import WGS72 as SGP4_WGS72  # noqa: E402
from sgp4.api import Satrec  # noqa: E402

from orbitdrift.constants import WGS72  # noqa: E402
from orbitdrift.divergence import self_consistency_divergence  # noqa: E402
from orbitdrift.errors import PropagationError  # noqa: E402
from orbitdrift.features import epoch_jd  # noqa: E402

# Catalogue 5, Vanguard 1. Two real published element sets twenty years apart.
# The 2000 pair is line 1 and 2 of SGP4-VER.TLE, which ships inside the
# installed sgp4 distribution as the first of its 33 verification cases. The
# 2020 pair is the element set the same distribution carries in its test module
# as VANGUARD_TLE, sourced there from celestrak.com/satcat/tle.php?CATNR=5.
# Both are transcribed here rather than imported, so that this script does not
# depend on a private module of another package; the checksums are recomputed
# and verified below, which is what makes the transcription checkable.
VANGUARD_2000 = (
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
)
VANGUARD_2020 = (
    "1 00005U 58002B   20287.20333880 -.00000016  00000-0 -22483-4 0  9998",
    "2 00005  34.2443 225.5254 1845686 162.2516 205.2356 10.84869164218149",
)

EPOCH_1949_DAYS = 27774.0
"""Days from 1949 December 31 for 2026 January 15, the synthetic earlier epoch."""


def _tle_checksum(line: str) -> int:
    """Modulo-ten checksum of a two-line element set line, digits plus minus signs."""
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def _synthetic_pair(
    *,
    gap_days: float,
    mean_anomaly_offset_rad: float = 0.0,
    mean_motion_offset_rad_min: float = 0.0,
    semi_major_axis_km: float = 6878.135,
    eccentricity: float = 0.001,
) -> tuple[Satrec, Satrec, float]:
    """Two element sets for one object, differing by a declared amount.

    With both offsets zero the two sets describe the **same orbit at the same
    phase**, and the separation between them is zero to the propagator's own
    rounding. Getting that right is the whole difficulty of the construction
    and it is worth stating how.

    Advancing the later set's mean anomaly by the Kozai mean motion times the
    gap does **not** work: SGP4 propagates the mean anomaly, the argument of
    perigee and the right ascension at secular rates that include the ``J2``
    corrections, and those rates are not the mean motion. A first version did
    exactly that and the identical pair came out 6882 km apart, which is the
    orbit radius -- the two sets were simply at different places.

    ``Satrec`` exposes the three secular rates as ``mdot``, ``argpdot`` and
    ``nodedot``, so the later set's angles are advanced by those rates over
    the gap. With ``B*`` set to zero the drag-driven higher-order terms in
    SGP4's mean anomaly vanish and the match is exact; the identical-pair
    control below measures what "exact" turns out to mean.

    Returns ``(early, late, semi_major_axis_km)``.
    """
    mean_motion = np.sqrt(WGS72.mu_km3_s2 / semi_major_axis_km**3) * 60.0
    mean_anomaly = 0.3
    arg_perigee = 0.7
    node = 0.4
    early = Satrec()
    early.sgp4init(
        SGP4_WGS72, "i", 90001, EPOCH_1949_DAYS,
        0.0, 0.0, 0.0,
        eccentricity, arg_perigee, float(np.deg2rad(51.6)), mean_anomaly,
        mean_motion, node,
    )
    gap_minutes = gap_days * 1440.0
    late = Satrec()
    late.sgp4init(
        SGP4_WGS72, "i", 90001, EPOCH_1949_DAYS + gap_days,
        0.0, 0.0, 0.0,
        eccentricity,
        arg_perigee + float(early.argpdot) * gap_minutes,
        float(np.deg2rad(51.6)),
        mean_anomaly + float(early.mdot) * gap_minutes + mean_anomaly_offset_rad,
        mean_motion + mean_motion_offset_rad_min,
        node + float(early.nodedot) * gap_minutes,
    )
    return early, late, float(semi_major_axis_km)


def main() -> int:
    out = Tee("validate_self_consistency_output.txt")
    out.rule("orbitdrift -- baseline 1, SGP4 self-consistency between two element sets")
    for line in environment_lines():
        out(line)

    out.rule("what this baseline can and cannot see")
    out("  Both sides are SGP4. Any error common to the theory cancels out of the")
    out("  comparison: the atmosphere model, the truncation of the Brouwer-Lyddane")
    out("  expansion, the absence of solar radiation pressure. What is left is the")
    out("  part of the error that came from the fit, which is a real and useful")
    out("  quantity and is not the whole error.")

    out.rule("the one real pair available in this environment")
    for name, pair in (("2000-179", VANGUARD_2000), ("2020-287", VANGUARD_2020)):
        for index, line in enumerate(pair, start=1):
            computed = _tle_checksum(line)
            printed = int(line[68])
            out(f"  {name} line {index}: checksum computed {computed}, printed {printed}")
            out.check(
                f"the transcribed {name} line {index} has a valid checksum",
                computed == printed,
                f"{computed} against {printed}",
            )
    early = Satrec.twoline2rv(*VANGUARD_2000)
    late = Satrec.twoline2rv(*VANGUARD_2020)
    gap = epoch_jd(late) - epoch_jd(early)
    out("")
    out(f"  catalogue                     : {int(early.satnum)} (Vanguard 1)")
    out(f"  earlier epoch                 : JD {epoch_jd(early):.6f}")
    out(f"  later epoch                   : JD {epoch_jd(late):.6f}")
    out(f"  epoch gap                     : {gap:.4f} days = {gap / 365.25:.2f} years")
    out(f"  mean motion, earlier          : {float(early.no_kozai):.12f} rad/min")
    out(f"  mean motion, later            : {float(late.no_kozai):.12f} rad/min")
    out(f"  relative change in mean motion: {float(late.no_kozai / early.no_kozai - 1.0):.6e}")
    out("")
    out("  The gap is twenty years. The magnitude below is therefore MEANINGLESS as")
    out("  a staleness figure and is not quoted as one anywhere: it is here because")
    out("  it is the only real same-object element-set pair in this environment,")
    out("  and running the code path on real data is worth more than not running")
    out("  it. A staleness measurement needs pairs days apart, which requires an")
    out("  element-set archive this product does not have. README.md says so.")
    try:
        curve = self_consistency_divergence(
            early,
            late,
            report_every_hours=720.0,
            samples_per_rev=2 if is_fast() else 32,
        )
        out("")
        out("  epoch age of the earlier set, days      separation km")
        for age, drift in zip(curve.ages_days[:6], curve.drift_km[:6], strict=False):
            out(f"  {age:38.3f}      {drift:14.3f}")
        out(f"  ... {curve.ages_days.size} reported ages in all")
        out(f"  detail                        : {curve.detail}")
        out.check(
            "the real pair produces a finite, positive separation curve",
            bool(np.all(np.isfinite(curve.drift_km)) and np.all(curve.drift_km > 0.0)),
            f"{curve.drift_km.size} ages, range {float(curve.drift_km.min()):.1f} to "
            f"{float(curve.drift_km.max()):.1f} km",
        )
    except PropagationError as error:
        out(f"  the real pair could not be propagated over its whole gap: {error}")
        out.check("the real pair produces a separation curve", False, str(error))

    out.rule("synthetic pair 1: a pure mean-anomaly offset, separation constant in time")
    out("  Predicted separation: the chord of the orbit subtended by dM, which is")
    out("  2 a sin(dM/2). For a near-circular orbit and a small offset that is")
    out("  a dM. The prediction comes from the geometry of a circle and shares no")
    out("  expression with the measurement.")
    out("")
    out("  dM rad      a dM km    chord km    measured km   measured/chord   spread")
    for offset in ((1e-3,) if is_fast() else (1e-5, 1e-4, 1e-3, 1e-2)):
        early, late, semi_major = _synthetic_pair(
            gap_days=3.0, mean_anomaly_offset_rad=offset
        )
        curve = self_consistency_divergence(early, late, report_every_hours=6.0)
        chord = 2.0 * semi_major * np.sin(0.5 * offset)
        measured = float(np.median(curve.drift_km))
        spread = float(np.ptp(curve.drift_km) / measured)
        out(
            f"  {offset:.1e}   {semi_major * offset:9.5f}   {chord:9.5f}   "
            f"{measured:11.5f}   {measured / chord:14.5f}   {spread:7.4f}"
        )
        out.check(
            f"dM = {offset:.1e}: measured separation within 2 per cent of the chord",
            abs(measured / chord - 1.0) < 0.02,
            f"ratio {measured / chord:.5f}",
        )
        out.check(
            f"dM = {offset:.1e}: the separation is constant in time to 5 per cent",
            spread < 0.05,
            f"relative spread {spread:.4f} over {curve.drift_km.size} ages",
        )

    out.rule("synthetic pair 2: a pure mean-motion offset, separation linear in time")
    out("  Predicted along-track separation: a * (mdot_late - mdot_early) * t, with")
    out("  the rates in rad/min and t in minutes. The rates are the propagator's")
    out("  own exposed secular rates; turning a phase difference into an arc length")
    out("  is the geometric step, and that is what is being checked.")
    out("")
    out("  dn rad/min   fitted slope km/day   predicted km/day   ratio   r^2")
    for offset in ((1e-8,) if is_fast() else (1e-9, 1e-8, 1e-7)):
        early, late, semi_major = _synthetic_pair(
            gap_days=1.0, mean_motion_offset_rad_min=offset
        )
        curve = self_consistency_divergence(early, late, report_every_hours=2.0)
        # The separation is measured against the EARLIER set's age, which starts
        # at the epoch gap; the elapsed time since the later epoch is what drives
        # the growth, so the abscissa is shifted by the gap.
        elapsed_days = curve.ages_days - (epoch_jd(late) - epoch_jd(early))
        design = np.column_stack([np.ones_like(elapsed_days), elapsed_days])
        solution, *_ = np.linalg.lstsq(design, curve.drift_km, rcond=None)
        slope = float(solution[1])
        # The phase rate difference is taken from the propagator's own exposed
        # secular rates; turning a phase difference into an arc length is the
        # geometric step and is this script's, not SGP4's.
        rate_difference = float(late.mdot) - float(early.mdot)
        predicted = semi_major * rate_difference * 1440.0
        residual = curve.drift_km - design @ solution
        total = curve.drift_km - curve.drift_km.mean()
        r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
        out(
            f"  {offset:.1e}    {slope:19.6f}   {predicted:16.6f}   "
            f"{slope / predicted:5.4f}   {r_squared:.6f}"
        )
        out.check(
            f"dn = {offset:.1e}: fitted growth rate within 5 per cent of a dn",
            abs(slope / predicted - 1.0) < 0.05,
            f"ratio {slope / predicted:.5f}",
        )
        out.check(
            f"dn = {offset:.1e}: the growth is linear in time to r^2 > 0.999",
            r_squared > 0.999,
            f"r^2 {r_squared:.6f}",
        )

    out.rule("negative control: an identical pair")
    out("  Two element sets with both offsets zero describe the same orbit at the")
    out("  same phase, so the separation must be at the propagator's own rounding")
    out("  level rather than a physical magnitude.")
    early, late, semi_major = _synthetic_pair(gap_days=3.0)
    curve = self_consistency_divergence(early, late, report_every_hours=6.0)
    worst = float(curve.drift_km.max())
    out(f"  worst separation              : {worst:.6e} km")
    out(f"  as a fraction of the orbit    : {worst / semi_major:.3e}")
    out.check(
        "an identical pair separates by less than a metre",
        worst < 1e-3,
        f"{worst:.6e} km",
    )

    out.rule("refusals")
    other = Satrec.twoline2rv(
        "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
        "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
    )
    for label, call in (
        (
            "two different objects",
            lambda: self_consistency_divergence(Satrec.twoline2rv(*VANGUARD_2000), other),
        ),
        (
            "the later set given first",
            lambda: self_consistency_divergence(
                Satrec.twoline2rv(*VANGUARD_2020), Satrec.twoline2rv(*VANGUARD_2000)
            ),
        ),
    ):
        try:
            call()
            out(f"  [NOT REFUSED] {label}")
            out.check(f"{label} is refused", False, "it returned")
        except PropagationError as error:
            out(f"  refused: {label} -- {str(error)[:70]}")
            out.check(f"{label} is refused", True, "")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
