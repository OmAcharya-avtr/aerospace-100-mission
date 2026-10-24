"""Re-measure every defect this build found, so each number has an output file.

``validation/VALIDATION.md`` tabulates ten defects. Six were found by
Hypothesis, three by the numerical sensitivity check in
:func:`pointahead.budget.verify_sensitivities`, and one by a validation script
disagreeing with its own prediction. All ten are pinned in
``tests/test_regressions.py`` with literal inputs, because ``.hypothesis/`` is
gitignored and a counterexample that is not pinned is lost on a fresh clone.

This script exists so that every number quoted in the defect table traces to a
file produced by a script that was executed, rather than to a test assertion a
reader would have to go and read. It re-measures each one from the literal
inputs and prints it.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.aberration import aberrate, aberration_angle  # noqa: E402
from pointahead.angle import (  # noqa: E402
    closed_form_point_ahead,
    light_time_pair,
    offset_difference,
    reference_point_ahead,
)
from pointahead.budget import (  # noqa: E402
    _stale_link,
    direction_noise_floor_rad,
    relative_acceleration,
    relative_jerk,
    staleness_position_error_m,
    staleness_velocity_error_mps,
)
from pointahead.geometry import (  # noqa: E402
    C_LIGHT,
    norm,
    orthonormal_basis,
    rotation_matrix,
    transverse,
    unit,
)
from pointahead.kepler import propagate_two_body  # noqa: E402
from pointahead.scenario import leo_crossing_pair, leo_trailing_pair  # noqa: E402
from pointahead.terminal import Crosslink, TerminalState  # noqa: E402

MU = 3.986004418e14
EPS = float(np.finfo(np.float64).eps)


def property_link(radius: float, first: float, second: float, velocity: np.ndarray) -> Crosslink:
    """Rebuild the configuration the property-test strategies produced."""
    local = radius * np.array([float(np.cos(first)), float(np.sin(first)), 0.0])
    remote = radius * np.array(
        [
            float(np.cos(second)),
            float(np.sin(second)) * 0.6,
            float(np.sin(second)) * 0.8,
        ]
    )
    return Crosslink(
        local=TerminalState(local, np.zeros(3), 0.0, "local"),
        remote=TerminalState(remote, velocity, 0.0, "remote"),
        trajectory_model="linear",
        label="regression",
        frame="regression frame",
    )


def main() -> int:
    tee = Tee("validate_regressions_output.txt")
    header(tee, "every defect this build found, re-measured from its literal inputs")
    tee(f"  float64 machine epsilon            : {EPS!r}")

    tee.rule("defect 1 (code, Hypothesis): Kepler iteration over many revolutions")
    tee("  Falsifying example: radius=400000.0 m, 0.6 of circular speed,")
    tee("  dt=3354.9375 s. Before the fix the universal-anomaly Newton iteration")
    tee("  reported 'did not converge in 60 iterations'.")
    radius = 400000.0
    circular = float(np.sqrt(MU / radius))
    r0 = np.array([radius, 0.0, 0.0])
    v0 = 0.6 * circular * np.array([0.0, 1.0, 0.0])
    energy = 0.5 * float(np.dot(v0, v0)) - MU / radius
    semi_major = -MU / (2.0 * energy)
    period = 2.0 * float(np.pi) * float(np.sqrt(semi_major**3 / MU))
    step = propagate_two_body(r0, v0, 3354.9375, MU)
    after = 0.5 * float(np.dot(step.velocity_mps, step.velocity_mps)) - MU / float(
        np.linalg.norm(step.position_m)
    )
    tee(f"  semi-major axis                    : {semi_major:.6f} m")
    tee(f"  orbital period                     : {period:.6f} s")
    tee(f"  revolutions in 3354.9375 s         : {3354.9375 / period:.6f}")
    tee(f"  revolutions removed by the fix     : {step.revolutions_removed!r}")
    tee(f"  Newton iterations after the fix    : {step.iterations}")
    tee(f"  relative energy drift              : {abs(after - energy) / abs(energy):.6e}")
    tee.check("the propagation converges and conserves energy", step.iterations <= 10)

    tee.rule("defect 2 (code, sensitivity check): the missing jerk term")
    link = leo_crossing_pair()
    n = link.line_of_sight
    acceleration = relative_acceleration(link)
    jerk = relative_jerk(link)
    acceleration_perp = norm(transverse(acceleration, n))
    jerk_perp = norm(transverse(jerk, n))
    age = 30.0
    first_order_only = 2.0 * acceleration_perp * age / C_LIGHT
    with_jerk = 2.0 * staleness_velocity_error_mps(link, age) / C_LIGHT
    measured = offset_difference(
        reference_point_ahead(_stale_link(link, age, "velocity")), reference_point_ahead(link)
    )
    tee(f"  geometry                           : {link.label}, equal radius")
    tee(f"  |a_rel|                            : {norm(acceleration):.9f} m/s^2")
    tee(f"  |a_rel_perp|                       : {acceleration_perp:.6e} m/s^2")
    tee(f"  transverse fraction of a_rel       : {acceleration_perp / norm(acceleration):.6e}")
    tee(f"  |adot_rel_perp|                    : {jerk_perp:.9f} m/s^3")
    tee(f"  ephemeris age                      : {age:.1f} s")
    tee(f"  FIRST FORMULATION, 2 a_perp dt / c : {first_order_only!r} rad")
    tee(f"  with the jerk term                 : {with_jerk!r} rad")
    tee(f"  numerically propagated             : {measured!r} rad")
    tee(f"  factor the first formulation was out by: "
        f"{measured / max(first_order_only, 1.0e-300):.6e}")
    tee.check(
        "the corrected line reproduces the propagation to 1 per cent",
        abs(with_jerk - measured) / measured < 1.0e-2,
        f"relative difference {abs(with_jerk - measured) / measured:.6e}",
    )

    tee.rule("defect 3 (code, sensitivity check): the cubic coefficient")
    correct = staleness_position_error_m(link, age)
    wrong = 0.5 * acceleration_perp * age * age + jerk_perp * age**3 / 6.0
    bound = 2.0 * norm(link.relative_velocity_mps) * correct / (C_LIGHT * link.range_m)
    wrong_bound = 2.0 * norm(link.relative_velocity_mps) * wrong / (C_LIGHT * link.range_m)
    measured = offset_difference(
        reference_point_ahead(_stale_link(link, age, "position")), reference_point_ahead(link)
    )
    tee("  r_stale - r = -a dt^2 / 2 + adot dt^3 / 3, not adot dt^3 / 6:")
    tee("  the dt^3 / 3 is -1/6 from the position expansion plus +1/2 from")
    tee("  carrying the stale velocity forward.")
    tee(f"  transverse position error, dt^3/3   : {correct:.9f} m")
    tee(f"  transverse position error, dt^3/6   : {wrong:.9f} m")
    tee(f"  ratio                               : {correct / wrong:.9f}")
    tee(f"  bound from the correct coefficient  : {bound!r} rad")
    tee(f"  bound from the wrong coefficient    : {wrong_bound!r} rad")
    tee(f"  numerically propagated              : {measured!r} rad")
    tee.check("the correct bound holds", measured <= bound * 1.01)
    tee.check("the wrong bound does not hold", measured > wrong_bound)

    tee.rule("defect 4 (criterion, validation script): the float64 floor constant")
    trailing = leo_trailing_pair()
    scale = max(norm(trailing.local.position_m), norm(trailing.remote.position_m))
    old_floor = 4.0 * EPS * scale / trailing.range_m
    new_floor = direction_noise_floor_rad(trailing)
    truncated = offset_difference(
        reference_point_ahead(trailing, max_iter=0), reference_point_ahead(trailing)
    )
    tee(f"  geometry                           : {trailing.label}")
    tee(f"  FIRST FORMULATION, 4 eps R / rho   : {old_floor!r} rad")
    tee(f"  corrected, 8 sqrt(3) eps R / rho   : {new_floor!r} rad")
    tee(f"  ratio                              : {new_floor / old_floor:.9f} "
        f"(2 sqrt(3) = {2.0 * float(np.sqrt(3.0)):.9f})")
    tee(f"  measured at iteration cap zero     : {truncated!r} rad")
    tee.check("the measurement exceeds the first floor", truncated > old_floor)
    tee.check("the measurement is below the corrected floor", truncated < new_floor)

    tee.rule("defect 5 (prediction, validation script): remote vs relative radial velocity")
    for item in (leo_crossing_pair(), leo_trailing_pair()):
        receive, transmit = light_time_pair(item)
        gap = transmit.tau_s - receive.tau_s
        remote_radial = float(np.dot(item.remote.velocity_mps, item.line_of_sight))
        relative_radial = float(np.dot(item.relative_velocity_mps, item.line_of_sight))
        good = 2.0 * remote_radial * item.nominal_light_time_s / C_LIGHT
        bad = 2.0 * relative_radial * item.nominal_light_time_s / C_LIGHT
        tee("")
        tee(f"  {item.label}")
        tee(f"    v_remote . n                     : {remote_radial:.9f} m/s")
        tee(f"    v_rel . n                        : {relative_radial:.6e} m/s")
        tee(f"    measured tau_tx - tau_rx         : {gap!r} s")
        tee(f"    predicted from v_remote . n      : {good!r} s")
        tee(f"    FIRST FORMULATION, v_rel . n     : {bad!r} s")
        tee.check(
            f"{item.label}: the remote-velocity prediction holds to 1 per cent",
            abs(gap - good) / abs(good) < 1.0e-2,
            f"relative difference {abs(gap - good) / abs(good):.6e}",
        )

    tee.rule("defect 6 (documentation, pytest): the hand-written literal")
    exact = 15000.0 / C_LIGHT
    tee(f"  2 * 7500 / c                        : {exact!r} rad")
    tee(f"  FIRST FORMULATION in the test       : {5.003461712e-05!r} rad")
    tee(f"  absolute difference                 : {abs(exact - 5.003461712e-05)!r} rad")
    hand = Crosslink(
        local=TerminalState([1.0e7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "local"),
        remote=TerminalState([1.0e7, 2.0e6, 0.0], [7500.0, 0.0, 0.0], 0.0, "remote"),
        trajectory_model="linear",
        label="hand",
        frame="hand frame",
    )
    result = closed_form_point_ahead(hand)
    offset_norm = float(np.linalg.norm(result.offset_rad))
    tee(f"  closed-form offset norm             : {offset_norm!r} rad")
    tee(f"  closed-form total_rad               : {result.total_rad!r} rad")
    tee.check(
        "the closed-form offset reproduces the arithmetic exactly",
        abs(offset_norm - exact) / exact < 1.0e-15,
        f"relative difference {abs(offset_norm - exact) / exact:.6e}",
    )

    tee.rule("defect 11 (criterion, this script): the pytest.approx absolute floor")
    tee("  pytest.approx(expected, rel=r) carries a DEFAULT absolute tolerance of")
    tee("  1e-12 and accepts a value within EITHER bound. Every angle this")
    tee("  package produces is of order 1e-05 rad, so that default dominates any")
    tee("  relative tolerance tighter than 2e-08: an assertion written rel=1e-12")
    tee("  is in fact a 2e-08 relative comparison.")
    tee("")
    tee("  It mattered. total_rad is the angle between two unit vectors offset")
    tee("  either side of the line of sight, so it is 2 arctan(theta / 2) =")
    tee("  theta - theta^3 / 12 and NOT theta. The known-answer test asserted it")
    tee("  equal to theta at rel=1e-12 and passed.")
    deficit = (exact - result.total_rad) / exact
    tee("")
    tee(f"  theta = 2 v_perp / c                : {exact!r} rad")
    tee(f"  2 arctan(theta / 2)                 : "
        f"{2.0 * float(np.arctan(0.5 * exact))!r} rad")
    tee(f"  total_rad                           : {result.total_rad!r} rad")
    tee(f"  relative deficit                    : {deficit!r}")
    tee(f"  theta^2 / 12                        : {exact**2 / 12.0!r}")
    tee(f"  ratio of the two                    : {deficit / (exact**2 / 12.0):.9f}")
    tee("")
    tee("  Re-running the whole suite with every rel= call given abs=0.0 found")
    tee("  exactly three assertions of 271 that depended on the default. All")
    tee("  three are now explicit and the whole suite passes with abs=0.0")
    tee("  everywhere.")
    tee.check(
        "total_rad equals 2 arctan(theta / 2) to rounding",
        abs(result.total_rad - 2.0 * float(np.arctan(0.5 * exact))) / exact < 1.0e-15,
    )
    tee.check(
        "the relative deficit is theta^2 / 12",
        abs(deficit - exact**2 / 12.0) / (exact**2 / 12.0) < 1.0e-6,
        f"measured {deficit:.9e}, predicted {exact**2 / 12.0:.9e}",
    )

    tee.rule("defect 7 (documentation, pytest): the first-order error order")
    tee("  The module docstring claimed O(beta^2). For a direction exactly")
    tee("  transverse to the boost the second-order term vanishes and the error is")
    tee("  arcsin(beta) - arctan(beta) = beta^3 / 2.")
    tee("")
    tee(f"  {'beta':>10s} {'transverse error':>18s} {'ratio to prev':>14s} "
        f"{'oblique error':>16s} {'ratio to prev':>14s}")
    tee(f"  {'-' * 10} {'-' * 18} {'-' * 14} {'-' * 16} {'-' * 14}")
    oblique = unit([1.0, 1.0, 0.0])
    previous_t = previous_o = None
    for magnitude in (1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        t_error = abs(
            aberration_angle([0.0, 1.0, 0.0], beta, "first_order")
            - aberration_angle([0.0, 1.0, 0.0], beta, "momentum_boost")
        )
        o_error = float(
            np.linalg.norm(
                aberrate(oblique, beta, "first_order") - aberrate(oblique, beta, "momentum_boost")
            )
        )
        tee(
            f"  {magnitude:>10.0e} {t_error:>18.6e} "
            f"{(t_error / previous_t if previous_t else float('nan')):>14.3f} "
            f"{o_error:>16.6e} "
            f"{(o_error / previous_o if previous_o else float('nan')):>14.3f}"
        )
        previous_t, previous_o = t_error, o_error

    tee.rule("defect 8 (criterion, Hypothesis): arcsin(beta) at beta = 1e-12")
    axis = unit([1.0, 1.0, 1.0])
    magnitude = 1.0e-12
    _, e1, _ = orthonormal_basis(axis)
    expected = float(np.arcsin(magnitude))
    tee("  axis                               : [1.0, 1.0, 1.0] normalised")
    tee(f"  |beta|                             : {magnitude!r}")
    tee(f"  arcsin(beta)                       : {expected!r} rad")
    tee(f"  float64 unit-vector resolution     : {EPS!r}")
    for route in ("angle_relation", "momentum_boost", "event_worldline"):
        measured = aberration_angle(e1, axis * magnitude, route)
        residual = abs(measured - expected)
        tee(f"  {route:<34s}: residual {residual!r} rad, "
            f"{residual / expected:.6e} relative")
    tee.check(
        "the residual is below the float64 unit-vector resolution",
        abs(aberration_angle(e1, axis * magnitude, "momentum_boost") - expected) < EPS,
    )

    tee.rule("defect 9 (criterion, Hypothesis): three relative-only tolerances")
    cases = (
        ("closed form vs 2 v_perp / c", 400000.0, 0.0, 1.0, np.array([0.0, 0.0, 1.0])),
        ("rotation invariance", 400000.0, 0.0, 1.0, np.array([0.0, 0.0, 1.0])),
    )
    item = property_link(*cases[0][1:])
    expected = 2.0 * norm(transverse(item.relative_velocity_mps, item.line_of_sight)) / C_LIGHT
    measured = closed_form_point_ahead(item).total_rad
    tee("  closed form vs 2 v_perp / c, radius 400000.0 m")
    tee(f"    expected                         : {expected!r} rad")
    tee(f"    measured                         : {measured!r} rad")
    tee(f"    absolute residual                : {abs(measured - expected)!r} rad")
    tee(f"    relative residual                : {abs(measured - expected) / expected:.6e}")
    matrix = rotation_matrix([0.0, 0.0, 1.0], 1.0)
    rotated = Crosslink(
        local=TerminalState(
            matrix @ item.local.position_m, matrix @ item.local.velocity_mps, 0.0, "local"
        ),
        remote=TerminalState(
            matrix @ item.remote.position_m, matrix @ item.remote.velocity_mps, 0.0, "remote"
        ),
        trajectory_model="linear",
        label="rotated",
        frame=item.frame,
    )
    before = closed_form_point_ahead(item).total_rad
    after = closed_form_point_ahead(rotated).total_rad
    tee("")
    tee("  rotation invariance, same geometry, quarter-turn about +z by 1.0 rad")
    tee(f"    absolute residual                : {abs(after - before)!r} rad")
    tee(f"    relative residual                : {abs(after - before) / before:.6e}")
    radial_base = property_link(738448.0, 0.0, 0.125, np.zeros(3))
    radial = Crosslink(
        local=radial_base.local,
        remote=TerminalState(
            radial_base.remote.position_m, 2.0 * radial_base.line_of_sight, 0.0, "remote"
        ),
        trajectory_model="linear",
        label="radial",
        frame=radial_base.frame,
    )
    total = reference_point_ahead(radial).total_rad
    tee("")
    tee(f"  radial motion, radius 738448.0 m, range {radial.range_m / 1.0e3:.6f} km")
    tee(f"    reference point-ahead angle      : {total!r} rad")
    tee(f"    FIRST CRITERION, fixed 1e-15 rad : {'passes' if total < 1.0e-15 else 'FAILS'}")
    tee(f"    float64 direction resolution     : {direction_noise_floor_rad(radial)!r} rad")
    tee.check(
        "the radial residual is below the geometry's own floor",
        total <= direction_noise_floor_rad(radial),
    )

    tee.rule("defect 10 (criterion, Hypothesis): reference vs closed form at a 12 km range")
    item = property_link(1037906.0, 0.0, 0.01171875, np.array([1.0, 0.0, 0.0]))
    baseline = closed_form_point_ahead(item)
    reference = reference_point_ahead(item)
    difference = offset_difference(baseline, reference)
    floor = direction_noise_floor_rad(item)
    tee("  radius                             : 1037906.0 m")
    tee(f"  range                              : {item.range_m:.6f} m")
    tee(f"  point-ahead angle                  : {baseline.total_rad!r} rad")
    tee(f"  offset difference                  : {difference!r} rad")
    tee(f"  as a fraction of the angle         : {difference / baseline.total_rad:.6e}")
    tee(f"  float64 direction resolution       : {floor!r} rad")
    tee(f"  ratio of the floor to the residual : {floor / difference:.6f}")
    tee.check("the residual is below the geometry's own floor", difference < floor)

    tee.rule("orbital beta, quoted in several places")
    speed = norm(leo_crossing_pair().local.velocity_mps)
    tee(f"  600 km circular orbital speed      : {speed:.9f} m/s")
    tee(f"  beta                               : {speed / C_LIGHT!r}")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
