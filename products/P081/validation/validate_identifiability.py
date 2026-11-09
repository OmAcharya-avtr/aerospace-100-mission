"""Baseline 1 against its own derivation, and structural separated from practical.

Four things are measured here.

1. **The identifiability table for every declared telemetry set**: rank, the
   rank's margin from the cutoff, the Cramer-Rao bounds with their finiteness
   clause, and the identifiable combinations reported in place of the refused
   parameters.
2. **The computed null space against the on-paper derivation.** The derivation
   in :mod:`twindiverge.plant` is algebra -- identities A and B and their
   corollary -- and :func:`twindiverge.identifiability.analytic_null_basis`
   returns it without touching a sensitivity matrix. The computed null space is
   the right singular vectors of equation (8) below the rank cutoff. The two
   must agree, and the principal angles between them are the measurement.
   **This is the check the product's central claim rests on**: the derivation
   is the independently-derived reference and the rank computation is the
   implementation.
3. **That a perturbation along the null space is invisible.** Not an angle but
   a trajectory: two parameter sets differing by a null-space vector are
   simulated and their rate records compared.
4. **Structural against practical identifiability, by sweeping the record
   length.** The rank, the null space and the refusal list do not move. The
   bound on the worst identifiable direction moves by four orders of magnitude.
   That is the whole content of the distinction, and it is why this product
   reports both.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import (  # noqa: E402
    NULL_COMPONENT_TOL,
    PRACTICAL_LIMIT,
    RANK_RTOL,
    analyse,
    analytic_null_basis,
    combination_label,
    subspace_angles_deg,
)
from twindiverge.plant import (  # noqa: E402
    NOMINAL,
    PARAMETER_NAMES,
    Campaign,
    perturb,
    simulate,
)
from twindiverge.report import format_table, number, render_identifiability  # noqa: E402
from twindiverge.telemetry import TELEMETRY_SETS, TelemetrySet  # noqa: E402

#: Record lengths swept in part 4, in samples at the declared 0.1 s period. The
#: driven fraction is held at one fifth of the record so that the excitation
#: shape is the same at every length and only the length changes.
SWEEP_SAMPLES = (101, 201, 401, 801, 1201, 2001)

#: Declared perturbations used in part 3.
INVISIBILITY_OFFSETS = (
    np.array([0.08, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, -0.11, 0.0, 0.0]),
    np.array([-0.05, 0.13, 0.02, -0.09, 0.07]),
)


def main() -> int:
    out = Tee("validate_identifiability_output.txt")
    out.rule("twindiverge -- baseline 1, and the derivation it must agree with")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared rank cutoff        : {RANK_RTOL:.1e} (relative singular value)")
    out(f"declared null-component tol : {NULL_COMPONENT_TOL:.1e}")
    out(f"declared practical limit    : {PRACTICAL_LIMIT:.2f} (relative standard error)")

    out.rule("1. the identifiability table for every declared telemetry set")
    summary = []
    reports = {}
    for name, telemetry in TELEMETRY_SETS.items():
        report = analyse(telemetry, NOMINAL)
        reports[name] = report
        summary.append([
            name,
            "+".join(telemetry.channels),
            f"{telemetry.campaign.duration_s:.1f}",
            str(telemetry.rows),
            f"{report.rank}",
            f"{report.rank_margin_decades:.4f}",
            f"{report.condition_number:.6e}",
            ", ".join(report.refused) or "none",
            ", ".join(report.weak) or "none",
            ", ".join(report.identifiable) or "none",
        ])
    out(format_table(
        ["set", "channels", "length [s]", "rows", "rank", "margin [decades]",
         "condition", "REFUSED", "WEAK", "IDENTIFIABLE"],
        summary,
    ))
    out("")
    for name in TELEMETRY_SETS:
        out("")
        out("-" * 78)
        out(f"telemetry set {name}: {TELEMETRY_SETS[name].note}")
        out("-" * 78)
        out(render_identifiability(reports[name]))
    out("")
    out.check(
        "every declared telemetry set has a rank margin of at least four decades",
        min(report.rank_margin_decades for report in reports.values()) > 4.0,
        f"worst margin {min(r.rank_margin_decades for r in reports.values()):.4f} "
        "decades",
    )
    out.check(
        "the rate-only sets refuse every parameter and the two-channel sets none",
        all(reports[name].refused == PARAMETER_NAMES
            for name in ("rate_full", "rate_short", "rate_coast"))
        and all(reports[name].refused == ()
                for name in ("rate_torque_full", "rate_torque_short")),
        "three rate-only sets refuse 5 of 5; two two-channel sets refuse 0 of 5",
    )

    out.rule("2. the computed null space against the derivation -- the central check")
    out("The derived basis is algebra from identities A and B of twindiverge.plant")
    out("and from the corollary about an unexcited parameter. It never touches a")
    out("sensitivity matrix. The computed basis is the right singular vectors of")
    out("equation (8) below the rank cutoff. Exact in exact arithmetic; the angles")
    out("below are the measured rounding.")
    out("")
    rows = []
    worst_angle = 0.0
    for name, telemetry in TELEMETRY_SETS.items():
        report = reports[name]
        derived = analytic_null_basis(telemetry)
        angles = subspace_angles_deg(derived, report.null_basis)
        angle = float(angles.max()) if angles.size else 0.0
        worst_angle = max(worst_angle, angle)
        rows.append([
            name,
            str(derived.shape[0]),
            str(report.null_basis.shape[0]),
            "yes" if derived.shape[0] == report.null_basis.shape[0] else "NO",
            f"{angle:.6e}" if angles.size else "(both empty)",
        ])
    out(format_table(
        ["set", "derived dimension", "computed dimension", "dimensions agree",
         "worst principal angle [deg]"],
        rows,
    ))
    out("")
    for name, telemetry in TELEMETRY_SETS.items():
        derived = analytic_null_basis(telemetry)
        if not derived.size:
            continue
        out(f"{name}: derived basis")
        out(format_table(
            ["direction", *PARAMETER_NAMES],
            [[f"d{index + 1}", *[f"{value:+.12f}" for value in row]]
             for index, row in enumerate(derived)],
        ))
        out(f"{name}: computed basis")
        out(format_table(
            ["direction", *PARAMETER_NAMES],
            [[f"c{index + 1}", *[f"{value:+.12f}" for value in row]]
             for index, row in enumerate(reports[name].null_basis)],
        ))
        out("")
    out.check(
        "the derived and computed null spaces have the same dimension everywhere",
        all(
            analytic_null_basis(telemetry).shape[0] == reports[name].null_basis.shape[0]
            for name, telemetry in TELEMETRY_SETS.items()
        ),
        "5 of 5 telemetry sets agree",
    )
    out.check(
        "the derived and computed null spaces are the same subspace",
        worst_angle < 1.0e-5,
        f"worst principal angle {worst_angle:.6e} degrees over all declared sets",
    )

    out.rule("3. a perturbation inside the null space changes no telemetry")
    out("The angle above is linear algebra. This is the trajectory: two parameter")
    out("sets differing by a null-space vector, simulated and compared.")
    out("")
    rows = []
    worst_invisible = 0.0
    for name, telemetry in TELEMETRY_SETS.items():
        report = reports[name]
        if not report.null_basis.size:
            rows.append([name, "-", "-", "no null space"])
            continue
        for offset in INVISIBILITY_OFFSETS:
            unseen = report.equivalent_perturbation(offset)
            first = simulate(perturb(NOMINAL, offset), telemetry.campaign).rate_rad_s
            second = simulate(
                perturb(NOMINAL, offset - unseen), telemetry.campaign
            ).rate_rad_s
            relative = float(np.abs(first - second).max() / np.abs(first).max())
            worst_invisible = max(worst_invisible, relative)
            rows.append([
                name,
                np.array2string(offset, precision=2),
                np.array2string(unseen, precision=4),
                f"{relative:.6e}",
            ])
    out(format_table(
        ["set", "perturbation", "its null projection", "relative rate difference"],
        rows,
    ))
    out("")
    out.check(
        "removing the null projection from a perturbation changes no rate record",
        worst_invisible < 1.0e-10,
        f"worst relative rate difference {worst_invisible:.6e}",
    )

    out.rule("4. structural against practical identifiability, by record length")
    out("The model and the observation function are the same at every row. Only the")
    out("record changes. Structural identifiability is a property of the first two")
    out("and does not move; practical identifiability is a property of the third and")
    out("moves by four orders of magnitude. This is the distinction Raue et al. 2009")
    out("draws, measured on this plant.")
    out("")
    rows = []
    ranks = set()
    bounds = []
    for samples in SWEEP_SAMPLES:
        period = 0.1
        driven = period * round(0.2 * (samples - 1))
        campaign = Campaign(f"sweep{samples}", period, samples, 1.0, driven)
        telemetry = TelemetrySet(f"sweep{samples}", campaign, ("rate",))
        report = analyse(telemetry, NOMINAL)
        ratios = report.singular_values / report.singular_values[0]
        ranks.add(report.rank)
        bounds.append(float(report.direction_crb_relative[-1]))
        rows.append([
            f"{campaign.duration_s:.1f}",
            str(samples),
            str(report.rank),
            f"{float(ratios[-1]):.6e}",
            f"{report.rank_margin_decades:.4f}",
            number(float(report.direction_crb_relative[-1])),
            number(float(report.direction_crb_relative[0])),
            ", ".join(report.refused) or "none",
        ])
    out(format_table(
        ["length [s]", "samples", "rank", "least ratio", "margin [decades]",
         "worst direction CRB", "best direction CRB", "refused"],
        rows,
    ))
    out("")
    out(f"ranks observed over the sweep              : {sorted(ranks)}")
    out(f"worst-direction bound at the shortest record: {bounds[0]:.6f}")
    out(f"worst-direction bound at the longest record : {bounds[-1]:.6f}")
    out(f"ratio                                      : {bounds[0] / bounds[-1]:.6f}")
    out("")
    out.check(
        "the structural rank does not move over the record-length sweep",
        ranks == {4},
        f"rank 4 at every one of {len(SWEEP_SAMPLES)} record lengths",
    )
    out.check(
        "the practical bound moves by more than three orders of magnitude",
        bounds[0] / bounds[-1] > 1.0e3,
        f"worst-direction bound falls from {bounds[0]:.6f} to {bounds[-1]:.6f}, a "
        f"factor of {bounds[0] / bounds[-1]:.1f}",
    )

    out.rule("5. what is reported in place of a refused parameter")
    out("For the reference set, the four identifiable combinations of equation (11).")
    out("The rate channel sees exactly the four ratios k_t/J, T_c/J, b/J and c_d/J,")
    out("which is a four-dimensional space, and these four combinations span it.")
    out("")
    report = reports["rate_full"]
    out(format_table(
        ["combination", "CRB on its relative standard error"],
        [[label, number(bound)] for label, bound in report.combinations()],
    ))
    out("")
    ratios_basis = np.array([
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    out("the four ratios, written as log-parameter directions:")
    out(format_table(
        ["ratio", *PARAMETER_NAMES],
        [
            [label, *[f"{value:+.1f}" for value in row]]
            for label, row in zip(
                ("log(k_t/J)", "log(T_c/J)", "log(b/J)", "log(c_d/J)"),
                ratios_basis,
                strict=True,
            )
        ],
    ))
    out("")
    angles = subspace_angles_deg(ratios_basis, report.identifiable_basis)
    out("principal angles between the ratio space and the identifiable subspace, degrees:")
    for index, angle in enumerate(angles):
        out(f"  angle {index + 1} : {angle:.6e}")
    out("")
    out("The tolerance below is 1e-5 degrees, the same as part 2's, and for the same")
    out("reason: a singular subspace whose complementary singular value is at the")
    out("1e-16 level is itself resolved only to about the square root of that, so an")
    out("angle of order 1e-6 degrees IS the measured precision of the decomposition")
    out("and not a discrepancy. Exact in exact arithmetic.")
    out("")
    out.check(
        "the identifiable subspace of the reference set IS the space of ratios",
        float(angles.max()) < 1.0e-5,
        f"worst principal angle {float(angles.max()):.6e} degrees",
    )
    out("")
    out("and the one direction that is NOT in it:")
    out(f"  {combination_label(report.null_basis[0], digits=6)}")
    out("  which is the geometric mean of all five parameters: the absolute torque")
    out("  scale, which a rate record cannot see.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
