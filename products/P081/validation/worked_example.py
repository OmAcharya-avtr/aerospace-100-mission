"""The worked example from README.md, executed, with its output committed.

Every line of the README's worked example is produced here and nowhere else, so
the README cannot drift from the code: the numbers it quotes are copied from
this script's committed output.

The story is one engineer's afternoon. A wheel twin disagrees with a 200 s
telemetry record. The engineer wants to know which parameter drifted. The answer
is that this record cannot say -- and the product's job is to say THAT, with the
four combinations it can say something about, and the family of parameter sets
it cannot distinguish, before it says anything else.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from twindiverge.attribution import attribute  # noqa: E402
from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.plant import (  # noqa: E402
    NOMINAL,
    PARAMETER_NAMES,
    PARAMETER_UNITS,
    perturb,
    simulate,
    simulate_with_sensitivities,
)
from twindiverge.report import (  # noqa: E402
    format_table,
    number,
    render_attribution,
    render_identifiability,
)
from twindiverge.telemetry import TELEMETRY_SETS, measure, residual_vector  # noqa: E402

#: The declared truth of this example: the inertia is 8 per cent high in
#: logarithm and nothing else has moved. The engineer does not know this.
TRUTH = np.array([0.08, 0.0, 0.0, 0.0, 0.0])

#: The declared noise seed.
SEED = 90001


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("twindiverge -- the worked example of README.md")
    for line in environment_lines():
        out(line)

    out.rule("the situation")
    telemetry = TELEMETRY_SETS["rate_full"]
    out(f"telemetry set : {telemetry.name}")
    out(f"  {telemetry.note}")
    out(f"campaign      : {telemetry.campaign.duration_s:.1f} s at "
        f"{telemetry.campaign.sample_period_s:.1f} s, "
        f"{telemetry.campaign.samples} samples, "
        f"{telemetry.campaign.drive_current_a:+.2f} A for the first "
        f"{telemetry.campaign.drive_duration_s:.1f} s")
    out(f"channels      : {', '.join(telemetry.channels)}")
    out(f"noise         : {telemetry.sigmas['rate']} rad/s on the rate channel")
    out("")
    out("the twin's parameters:")
    out("")
    out(format_table(
        ["parameter", "twin value", "unit"],
        [
            [name, number(value), unit]
            for name, value, unit in zip(
                PARAMETER_NAMES, NOMINAL.as_array(), PARAMETER_UNITS, strict=True
            )
        ],
    ))
    out("")
    out("and reality, which the engineer does not know:")
    out("")
    truth = perturb(NOMINAL, TRUTH)
    out(format_table(
        ["parameter", "true value", "relative change", "in logarithm"],
        [
            [
                name,
                number(float(truth.as_array()[index])),
                f"{float(truth.as_array()[index] / NOMINAL.as_array()[index] - 1.0):+.6f}",
                f"{float(TRUTH[index]):+.6f}",
            ]
            for index, name in enumerate(PARAMETER_NAMES)
        ],
    ))

    out.rule("step 0: the residual")
    observed, _ = measure(telemetry, truth, seed=SEED)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    residual = residual_vector(telemetry, observed, twin)
    raw = observed - twin.rate_rad_s
    out(f"peak residual            : {float(np.abs(raw).max()):.6f} rad/s")
    out(f"root-mean-square residual: {float(np.sqrt((raw**2).mean())):.6f} rad/s")
    out(f"declared sensor noise    : {telemetry.sigmas['rate']:.6f} rad/s")
    out(f"ratio                    : "
        f"{float(np.sqrt((raw**2).mean())) / telemetry.sigmas['rate']:.6f}")
    out("")
    out("So there is a real discrepancy, about 24 times the sensor noise. The")
    out("question is what to blame.")

    out.rule("step 1: baseline 1, which runs first and is binding")
    report = analyse(telemetry, NOMINAL)
    out(render_identifiability(report))
    out("")
    out("Every one of the five parameters is REFUSED. The reason is one exact")
    out("identity: the rate channel's five sensitivities sum to zero at every")
    out("sample, so scaling all five parameters by the same factor changes nothing")
    out("the sensor can see. No residual can tell J from the other four.")

    out.rule("step 2: baseline 2, on the identifiable subspace only")
    result = attribute(telemetry, report, residual, twin=twin)
    out(render_attribution(result))

    out.rule("step 3: what the engineer is told")
    out("")
    out("  A real discrepancy is present:")
    out(f"    chi-square {result.residual_chi2:.6f} on {result.residual_rows} rows "
        f"against a threshold of {result.detection_threshold:.6f}.")
    out("")
    out("  No per-parameter attribution is possible from this telemetry set.")
    out(f"    refused: {', '.join(result.refused)}")
    out("")
    out("  What DID change, in the combinations this record can see:")
    for line in result.combination_answer:
        out(f"    {line}")
    out("")
    out("  Under the additional assumption that exactly one parameter moved -- which")
    out("  this telemetry set cannot verify -- the best single-parameter explanation")
    out(f"  is {result.constrained_choice} by "
        f"{float(result.constrained_estimate[PARAMETER_NAMES.index(result.constrained_choice)]):+.6f}")
    out(f"  in logarithm, with relative weight "
        f"{result.constrained_confidence:.6f} against the other four.")
    out("")
    out("  And the alternatives that cannot be distinguished from it:")
    family = result.equivalence_family()
    out("")
    out(format_table(
        ["alternative", *PARAMETER_NAMES],
        [
            [
                "chosen" if index == 0 else f"equivalent {index}",
                *[f"{value:+.6f}" for value in row],
            ]
            for index, row in enumerate(family)
        ],
    ))
    out("")
    worst = 0.0
    for row in family[1:]:
        first = simulate(perturb(NOMINAL, family[0]), telemetry.campaign).rate_rad_s
        second = simulate(perturb(NOMINAL, row), telemetry.campaign).rate_rad_s
        worst = max(worst, float(np.abs(first - second).max()))
    out(f"  worst rate difference between the alternatives: {worst:.6e} rad/s,")
    out(f"  against a sensor noise of {telemetry.sigmas['rate']:.6f} rad/s.")
    out("")
    out("  How close is the estimate? Equation (12) is a LINEAR estimator of a")
    out("  nonlinear model, so at an eight per cent perturbation it carries a")
    out("  second-order error, and that error is larger than the statistical")
    out("  interval in the two best-determined directions. Both are printed:")
    out("")
    comparison = []
    for direction in result.directions:
        projection = float(direction.weights @ TRUTH)
        gap = direction.estimate - projection
        comparison.append([
            direction.label,
            f"{direction.estimate:+.6f}",
            f"{projection:+.6f}",
            f"{gap:+.6e}",
            number(direction.standard_deviation),
            f"{gap / direction.standard_deviation:+.2f}",
        ])
    out(format_table(
        ["combination", "estimate", "truth's projection", "difference",
         "standard error", "difference in standard errors"],
        comparison,
    ))
    out("")
    out("  That is not a defect to be hidden behind a wider interval: it is what a")
    out("  first-order attribution costs at this perturbation size, and it is")
    out("  measured below rather than asserted. The error must be SECOND order in")
    out("  the perturbation, so halving the perturbation must quarter it.")
    out("")
    out("  The measurement below uses NOISELESS telemetry. With noise, the")
    out("  statistical error of 4.677542e-06 in the best direction is comparable to")
    out("  the second-order term once the perturbation falls below about one per")
    out("  cent, the two partly cancel, and the measured order rises above 2 for a")
    out("  reason that has nothing to do with the linearisation: measured with noise")
    out("  at this seed, the orders were 2.0178, 2.0895, 2.4366 and 3.2620.")
    out("")
    magnitudes = (0.08, 0.04, 0.02, 0.01, 0.005)
    errors = []
    for magnitude in magnitudes:
        offset = np.zeros(5)
        offset[0] = magnitude
        local_observed = simulate(
            perturb(NOMINAL, offset), telemetry.campaign
        ).rate_rad_s
        local = attribute(
            telemetry,
            report,
            residual_vector(telemetry, local_observed, twin),
            twin=twin,
        )
        # The best-determined direction, where the statistical error is 4.7e-06
        # and the second-order term dominates it by orders of magnitude.
        direction = local.directions[0]
        errors.append(
            abs(direction.estimate - float(direction.weights @ offset))
        )
    orders = [
        float(np.log2(errors[index] / errors[index + 1]))
        for index in range(len(magnitudes) - 1)
    ]
    out(format_table(
        ["perturbation on log J", "error in the best direction", "halving order"],
        [
            [
                f"{magnitude:.3f}",
                f"{errors[index]:.6e}",
                f"{orders[index - 1]:.4f}" if index else "-",
            ]
            for index, magnitude in enumerate(magnitudes)
        ],
    ))
    out("")
    out.check(
        "the linearisation error of equation (12) is second order in the perturbation",
        1.7 < min(orders) and max(orders) < 2.3,
        f"measured halving orders from {min(orders):.4f} to {max(orders):.4f}, "
        "against the 2 that a second-order term requires",
    )
    out.check(
        "the single-parameter route names the parameter that actually moved",
        result.constrained_choice == "J",
        f"chose {result.constrained_choice} under its declared and unverifiable "
        "sparsity assumption",
    )
    out.check(
        "the alternatives it cannot distinguish produce the same rate record",
        worst / telemetry.sigmas["rate"] < 1.0e-10,
        f"worst difference {worst:.6e} rad/s, which is "
        f"{worst / telemetry.sigmas['rate']:.3e} of the sensor noise",
    )

    out.rule("step 4: what a second sensor would buy")
    out("The same campaign with a reaction-torque sensor between the wheel housing")
    out("and its mount. The scale direction's sensitivity in that channel is exactly")
    out("the torque signal, equation (7), so the degeneracy disappears.")
    out("")
    richer = TELEMETRY_SETS["rate_torque_full"]
    richer_report = analyse(richer, NOMINAL)
    richer_observed, _ = measure(richer, truth, seed=SEED)
    richer_twin = simulate_with_sensitivities(NOMINAL, richer.campaign)
    richer_result = attribute(
        richer,
        richer_report,
        residual_vector(richer, richer_observed, richer_twin),
        twin=richer_twin,
    )
    out(format_table(
        ["parameter", "verdict", "estimate", "95 per cent half-width", "truth"],
        [
            [
                name,
                richer_report.verdict[index],
                f"{float(richer_result.parameter_estimate[index]):+.6f}",
                number(float(richer_result.parameter_half_width[index])),
                f"{float(TRUTH[index]):+.6f}",
            ]
            for index, name in enumerate(PARAMETER_NAMES)
        ],
    ))
    out("")
    out(f"equivalent alternatives : "
        f"{richer_result.equivalence_family().shape[0] - 1}")
    out("")
    out.check(
        "adding the torque channel removes every refusal",
        richer_report.refused == (),
        f"refused {len(richer_report.refused)} of 5",
    )
    out.check(
        "and the inertia estimate then covers the truth",
        abs(float(richer_result.parameter_estimate[0]) - float(TRUTH[0]))
        < float(richer_result.parameter_half_width[0])
        + 0.05 * abs(float(TRUTH[0])),
        f"estimate {float(richer_result.parameter_estimate[0]):+.6f} against a truth "
        f"of {float(TRUTH[0]):+.6f}, half-width "
        f"{float(richer_result.parameter_half_width[0]):.6f} plus the five per cent "
        "second-order allowance",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
