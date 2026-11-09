"""Negative controls: every check in this repository, deliberately broken.

A check that cannot fail is not a check. Each control below injects a violation
into a quantity whose TRUE value is known from the derivation -- never by
offsetting an output, because an output already sits inside the band a check
allows and only an injection anchored on the truth is guaranteed to leave it --
and then asserts that the corresponding check FIRES. Four controls are declared
in the other direction and must NOT fire, because a criterion that fires on
everything is as useless as one that fires on nothing.

Where a control's criterion involves a confidence level, the criterion is the
DECLARED four standard errors (``_bootstrap.CONTROL_SIGMA``) and not the nominal
level: a control whose criterion is a 95 per cent interval is expected to fire
once in twenty by construction, and a predecessor in this mission corrected such
a control rather than changing its seed until it passed.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from _bootstrap import (  # noqa: E402
    BOUND_SEEDS,
    CONTROL_SIGMA,
    COVERAGE_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from twindiverge.attribution import (  # noqa: E402
    CONFIDENCE_LEVEL,
    DETECTION_LEVEL,
    attribute,
)
from twindiverge.identifiability import (  # noqa: E402
    NULL_COMPONENT_TOL,
    RANK_RTOL,
    analyse,
    analytic_null_basis,
    subspace_angles_deg,
)
from twindiverge.learned import ResidualClassifier  # noqa: E402
from twindiverge.pipeline import analyse_case, montecarlo_directions  # noqa: E402
from twindiverge.plant import (  # noqa: E402
    NOMINAL,
    SCALE_DIRECTION,
    WheelParameters,
    perturb,
    simulate,
    simulate_with_sensitivities,
)
from twindiverge.profile import FLAT_TOLERANCE, profile_direction  # noqa: E402
from twindiverge.reference import exact_campaign  # noqa: E402
from twindiverge.report import format_table  # noqa: E402
from twindiverge.telemetry import (  # noqa: E402
    TELEMETRY_SETS,
    design_matrix,
    measure,
    residual_vector,
)

#: Declared angle, in degrees, above which the null-space agreement check fires.
#: Part 2 of validate_identifiability.py uses 1e-5; the controls use the same
#: value so that a control fires against the same criterion the check uses.
ANGLE_LIMIT = 1.0e-5

#: Declared relative difference above which the closed-form agreement fires.
CLOSED_FORM_LIMIT = 1.0e-13

#: Declared relative residual above which identity A fires.
IDENTITY_LIMIT = 1.0e-11


def main() -> int:
    out = Tee("validate_negative_controls_output.txt")
    out.rule("twindiverge -- every check, deliberately broken")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared control sigma : {CONTROL_SIGMA:.1f}")
    out(f"declared angle limit   : {ANGLE_LIMIT:.1e} degrees")
    out(f"declared rank cutoff   : {RANK_RTOL:.1e}")

    telemetry = TELEMETRY_SETS["rate_full"]
    report = analyse(telemetry, NOMINAL)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    matrix, _ = design_matrix(telemetry, NOMINAL, trajectory=twin)

    out.rule("A. the null space against its derivation")
    out("The truth is the DERIVED basis, which is algebra. Each control replaces it")
    out("with a wrong subspace of the same dimension and measures the angle the")
    out("check would see.")
    out("")
    derived = analytic_null_basis(telemetry)
    rows = []
    cases = (
        ("the derivation itself", derived, False),
        ("a single coordinate axis instead", np.eye(5)[:1], True),
        ("the scale direction with one sign flipped",
         (SCALE_DIRECTION * np.array([1.0, 1.0, 1.0, 1.0, -1.0]))[None, :]
         / np.linalg.norm(SCALE_DIRECTION * np.array([1.0, 1.0, 1.0, 1.0, -1.0])),
         True),
        ("the best-determined identifiable direction", report.identifiable_basis[:1], True),
    )
    for label, basis, must_fire in cases:
        angles = subspace_angles_deg(basis, report.null_basis)
        angle = float(angles.max()) if angles.size else 0.0
        rows.append([label, f"{angle:.6e}", "yes" if angle > ANGLE_LIMIT else "no"])
        out.control(
            f"null-space agreement: {label}",
            angle > ANGLE_LIMIT,
            must_fire,
            f"worst principal angle {angle:.6e} degrees against a limit of "
            f"{ANGLE_LIMIT:.1e}",
        )
    out("")
    out(format_table(["injected subspace", "worst angle [deg]", "fires"], rows))

    out.rule("B. identity A, equation (6)")
    out("The truth is that the five rate sensitivities sum to zero. The control adds")
    out("a column of ones scaled to a declared fraction of the largest sensitivity,")
    out("which breaks the sum by exactly that fraction and nothing else.")
    out("")
    sensitivity = twin.rate_sensitivity
    assert sensitivity is not None
    scale = float(np.abs(sensitivity).max())
    rows = []
    for fraction, must_fire in ((0.0, False), (1.0e-14, False), (1.0e-6, True), (1.0e-2, True)):
        broken = sensitivity.copy()
        broken[:, 0] += fraction * scale
        residual = float(np.abs(broken.sum(axis=1)).max()) / scale
        rows.append([
            f"{fraction:.1e}", f"{residual:.6e}",
            "yes" if residual > IDENTITY_LIMIT else "no",
        ])
        out.control(
            f"identity A with a {fraction:.1e} relative offset injected",
            residual > IDENTITY_LIMIT,
            must_fire,
            f"relative residual {residual:.6e} against a limit of "
            f"{IDENTITY_LIMIT:.1e}",
        )
    out("")
    out(format_table(
        ["injected fraction", "relative residual", "fires"], rows
    ))
    out("")
    out("The 1e-14 row is the control that must NOT fire: an injection below the")
    out("check's own rounding floor is indistinguishable from no injection, and a")
    out("check that fired on it would be reporting noise.")

    out.rule("C. the closed-form reference, equations (20) and (21)")
    out("The truth is the closed form. The control perturbs the PARAMETERS the")
    out("closed form is evaluated at, by a declared relative amount, so that the")
    out("integrator and the reference are solving different problems.")
    out("")
    campaign = telemetry.campaign
    integrated = simulate(NOMINAL, campaign).rate_rad_s
    rows = []
    for fraction, must_fire in ((0.0, False), (1.0e-12, True), (1.0e-6, True)):
        values = NOMINAL.as_array().copy()
        values[3] *= 1.0 + fraction
        wrong = exact_campaign(WheelParameters.from_array(values), campaign)
        relative = float(np.abs((integrated - wrong) / wrong).max())
        rows.append([
            f"{fraction:.1e}", f"{relative:.6e}",
            "yes" if relative > CLOSED_FORM_LIMIT else "no",
        ])
        out.control(
            f"closed form with the viscous coefficient moved by {fraction:.1e}",
            relative > CLOSED_FORM_LIMIT,
            must_fire,
            f"worst relative difference {relative:.6e} against a limit of "
            f"{CLOSED_FORM_LIMIT:.1e}",
        )
    out("")
    out(format_table(
        ["injected fraction on b", "worst relative difference", "fires"], rows
    ))

    out.rule("D. the detection test")
    out("The truth is that a pure scale change produces no residual and an")
    out("identifiable change produces one. Both directions are controlled.")
    out("")
    rows = []
    for label, offset, must_fire in (
        ("nothing moved", np.zeros(5), False),
        ("pure scale, 0.15", 0.15 * SCALE_DIRECTION * np.sqrt(5.0), False),
        ("pure scale, 1.00", 1.00 * SCALE_DIRECTION * np.sqrt(5.0), False),
        ("identifiable, 0.002 on J", np.array([2.0e-3, 0.0, 0.0, 0.0, 0.0]), True),
    ):
        detections = 0
        chi2 = []
        for seed in TEST_SEEDS[:40]:
            observed, _ = measure(telemetry, perturb(NOMINAL, offset), seed=seed)
            result = attribute(
                telemetry, report, residual_vector(telemetry, observed, twin), twin=twin
            )
            detections += int(result.detected)
            chi2.append(result.residual_chi2)
        rate = detections / 40
        expected = 1.0 - DETECTION_LEVEL
        error = float(np.sqrt(expected * (1.0 - expected) / 40))
        fires = rate > expected + CONTROL_SIGMA * error
        rows.append([
            label, f"{rate:.6f}", f"{float(np.median(chi2)):.4f}",
            "yes" if fires else "no",
        ])
        out.control(
            f"detection with {label}",
            fires,
            must_fire,
            f"detection rate {rate:.6f} against a declared false-positive rate of "
            f"{expected:.6f} plus {CONTROL_SIGMA:.1f} standard errors "
            f"({expected + CONTROL_SIGMA * error:.6f})",
        )
    out("")
    out(format_table(
        ["injected perturbation", "detection rate", "median chi2", "fires"], rows
    ))
    out("")
    out("The 'pure scale, 1.00' row is the strongest form of the product's claim: a")
    out("parameter set whose every member is multiplied by e = 2.718 produces")
    out("telemetry the detection test cannot distinguish from no change at all.")

    out.rule("E. the pseudo-inverse diagonal, tested as if it were a bound")
    out("The truth is the Cramer-Rao bound, which is INFINITE for every parameter on")
    out("this set. The control takes the pseudo-inverse diagonal -- the number a")
    out("naive implementation reports instead -- and tests it against the measured")
    out("spread of repeated fits, exactly as validate_independent_bound.py tests the")
    out("real bound. It must fail, because it is not a bound.")
    out("")
    amplitudes, bound = montecarlo_directions(telemetry, NOMINAL, BOUND_SEEDS)
    basis = report.identifiable_basis
    parameter_estimates = amplitudes @ basis
    spread = parameter_estimates.std(axis=0, ddof=1)
    claimed = report.pinv_diagonal_not_a_bound
    out(format_table(
        ["parameter", "pinv diagonal (claimed)", "measured sd of the minimum-norm "
         "estimate", "sd / claimed"],
        [
            [
                name,
                f"{float(claimed[index]):.6e}",
                f"{float(spread[index]):.6e}",
                f"{float(spread[index] / claimed[index]):.6f}",
            ]
            for index, name in enumerate(("J", "k_t", "T_c", "b", "c_d"))
        ],
    ))
    out("")
    out("The ratios are close to one, and that is the point: the pseudo-inverse")
    out("diagonal is a CORRECT number about the WRONG quantity. It describes the")
    out("scatter of the minimum-norm solution, which is one arbitrary member of an")
    out("infinite family, and says nothing about the parameter.")
    out("")
    out("The control makes that concrete. Starting from the fitted offset, a multiple")
    out("of the null direction is added. Every member of that family fits the data")
    out("equally well -- the chi-square change is the measurement -- while the")
    out("estimate of J moves by an arbitrary amount. No finite bound can cover a")
    out("family of unbounded diameter, which is why equation (10)'s finiteness clause")
    out("is not optional.")
    out("")
    centre = amplitudes[0] @ basis
    observed, _ = measure(telemetry, NOMINAL, seed=BOUND_SEEDS[0])
    from twindiverge.profile import chi_square

    base_chi2 = chi_square(telemetry, observed, centre)
    rows = []
    fired = 0
    for multiple in (0.0, 1.0, 10.0, 100.0):
        moved = centre + multiple * report.null_basis[0]
        try:
            moved_chi2 = chi_square(telemetry, observed, moved)
            reachable = True
        except ValueError:
            moved_chi2 = float("nan")
            reachable = False
        shift = abs(float(moved[0] - centre[0]))
        widths = shift / float(claimed[0])
        if reachable and multiple > 0.0:
            fired += int(
                abs(moved_chi2 - base_chi2) < 1.0e-6 and widths > CONTROL_SIGMA
            )
        rows.append([
            f"{multiple:.1f}",
            f"{moved_chi2:.9f}" if reachable else "outside the validity window",
            f"{abs(moved_chi2 - base_chi2):.6e}" if reachable else "-",
            f"{shift:.6f}",
            f"{widths:.3f}",
        ])
    out(format_table(
        ["multiple of the null direction", "chi-square", "change in chi-square",
         "shift in log J", "shift, in claimed standard errors"],
        rows,
    ))
    out("")
    out.control(
        "the family of equally good fits is unbounded in the claimed error bar's units",
        fired >= 2,
        True,
        f"{fired} of 3 non-zero multiples move log J by more than "
        f"{CONTROL_SIGMA:.1f} claimed standard errors at an unchanged chi-square",
    )
    out.check(
        "the measured spread of the minimum-norm estimate matches the pinv diagonal",
        bool(np.all(np.abs(spread / claimed - 1.0) < 0.5)),
        f"ratios from {float((spread / claimed).min()):.6f} to "
        f"{float((spread / claimed).max()):.6f}, so the diagonal describes the "
        "minimum-norm solution correctly and the parameter not at all",
    )
    _ = bound

    out.rule("F. the confidence intervals of equation (13)")
    out("The truth is the declared 95 per cent level. The control rescales each")
    out("interval by a declared factor and measures the coverage, which must fall")
    out("outside four standard errors of the nominal level.")
    out("")
    offset = np.array([2.0e-3, 0.0, 0.0, 0.0, 0.0])
    truth = perturb(NOMINAL, offset)
    short = TELEMETRY_SETS["rate_short"]
    short_report = analyse(short, NOMINAL)
    short_twin = simulate_with_sensitivities(NOMINAL, short.campaign)
    collected: list[list[float]] = []
    targets: list[float] = []
    for seed in COVERAGE_SEEDS:
        observed, _ = measure(short, truth, seed=seed)
        result = attribute(
            short, short_report, residual_vector(short, observed, short_twin),
            twin=short_twin,
        )
        if not targets:
            targets = [
                float(direction.weights @ offset) for direction in result.directions
            ]
        collected.append([
            abs(direction.estimate - targets[index]) / direction.half_width
            for index, direction in enumerate(result.directions)
        ])
    distances = np.array(collected)
    error = float(
        np.sqrt(CONFIDENCE_LEVEL * (1.0 - CONFIDENCE_LEVEL) / len(COVERAGE_SEEDS))
    )
    out("Coverage is MARGINAL, one direction at a time, because that is what a 95")
    out("per cent interval on a single direction promises. A joint statement over")
    out("four directions has a nominal level near 0.95 to the fourth power and")
    out("comparing it against 0.95 would make the correct intervals look broken --")
    out("which is how the first version of this control failed.")
    out("")
    rows = []
    for factor, must_fire in ((1.0, False), (0.25, True), (4.0, True)):
        coverage = (distances <= factor).mean(axis=0)
        deviations = np.abs(coverage - CONFIDENCE_LEVEL) / error
        fires = bool(np.any(deviations > CONTROL_SIGMA))
        rows.append([
            f"{factor:.2f}",
            " ".join(f"{value:.6f}" for value in coverage),
            f"{float(deviations.max()):.4f}",
            "yes" if fires else "no",
        ])
        out.control(
            f"intervals rescaled by {factor:.2f}",
            fires,
            must_fire,
            "marginal coverages "
            + ", ".join(f"{value:.6f}" for value in coverage)
            + f" against a nominal {CONFIDENCE_LEVEL:.2f}; worst deviation "
            f"{float(deviations.max()):.4f} standard errors of {error:.6f}",
        )
    out("")
    out(format_table(
        ["interval width factor", "marginal coverage per direction",
         "worst sigmas from nominal", "fires"],
        rows,
    ))
    out("")
    out(f"realisations : {len(COVERAGE_SEEDS)}")
    out(f"standard error of a coverage proportion : {error:.6f}")
    out("")
    out("The 4.00 row fires in the opposite direction: an interval four times too")
    out("wide covers the truth more often than its nominal level, which is as much a")
    out("defect as covering it less often, and a one-sided criterion would miss it.")
    out("A coverage that saturates at 1.000000 is only "
        f"{(1.0 - CONFIDENCE_LEVEL) / error:.4f} standard errors from")
    out("the nominal level, which is why this script uses 400 realisations and not")
    out("200: at 200 the same saturated coverage is 3.2444 away and could not reach")
    out(f"the declared {CONTROL_SIGMA:.1f}.")

    out.rule("G. the flatness verdict of baseline 3")
    out("The truth is that the reference set's profiles are flat and the two-channel")
    out("set's are not. Both directions are controlled against the same criterion.")
    out("")
    case_offset = np.array([0.06, 0.0, 0.0, 0.0, 0.0])
    rows = []
    for name, must_fire in (("rate_full", False), ("rate_torque_full", True)):
        this_set = TELEMETRY_SETS[name]
        observed, _ = measure(this_set, perturb(NOMINAL, case_offset), seed=777)
        outcome = profile_direction(
            this_set, observed, np.eye(5)[0], "log J", 0.0178, points=5,
            half_width=0.01,
        )
        fires = outcome.max_delta_chi2 >= FLAT_TOLERANCE
        rows.append([
            name, f"{outcome.max_delta_chi2:.6e}",
            "curved" if fires else "FLAT", "yes" if fires else "no",
        ])
        out.control(
            f"the log J profile on {name} is curved",
            fires,
            must_fire,
            f"max delta chi2 {outcome.max_delta_chi2:.6e} against a flatness "
            f"tolerance of {FLAT_TOLERANCE:.1e}",
        )
    out("")
    out(format_table(["set", "max delta chi2", "shape", "fires"], rows))

    out.rule("H. the learned classifier's accuracy")
    out("The truth is the case labels. The control shuffles the TRAINING labels with")
    out("a declared seed, which destroys the mapping the classifier is supposed to")
    out("learn while leaving everything else -- features, hyperparameters, split --")
    out("untouched. Accuracy must then fall to the five-class chance level.")
    out("")
    train_cases = [
        analyse_case(telemetry, case, report, twin)
        for case in __import__(
            "twindiverge.synthetic", fromlist=["generate_cases"]
        ).generate_cases(TRAIN_SEEDS, "single")
    ]
    test_cases = [
        analyse_case(telemetry, case, report, twin)
        for case in __import__(
            "twindiverge.synthetic", fromlist=["generate_cases"]
        ).generate_cases(TEST_SEEDS, "single")
    ]
    features = np.vstack([item.features for item in train_cases])
    labels = np.array([item.case.label for item in train_cases])
    test_features = np.vstack([item.features for item in test_cases])
    test_labels = np.array([item.case.label for item in test_cases])
    chance = 1.0 / 5.0
    chance_error = float(np.sqrt(chance * (1.0 - chance) / len(test_labels)))
    rows = []
    for label, used, must_fire in (
        ("the real labels", labels, False),
        ("labels shuffled, seed 424242",
         np.random.default_rng(424242).permutation(labels), True),
    ):
        model = ResidualClassifier("parameter").fit(features, used)
        answers = np.array([
            prediction.label for prediction in model.predict(test_features)
        ])
        accuracy = float(np.mean(answers == test_labels))
        fires = accuracy < chance + CONTROL_SIGMA * chance_error
        rows.append([label, f"{accuracy:.6f}", "yes" if fires else "no"])
        out.control(
            f"learned accuracy with {label}",
            fires,
            must_fire,
            f"accuracy {accuracy:.6f} against the chance level {chance:.6f} plus "
            f"{CONTROL_SIGMA:.1f} standard errors "
            f"({chance + CONTROL_SIGMA * chance_error:.6f})",
        )
    out("")
    out(format_table(["training labels", "test accuracy", "fires"], rows))

    out.rule("I. the equivalence family")
    out("The truth is that the family differs along the NULL direction. The control")
    out("builds a partner along a direction that is NOT in the null space, which")
    out("must then produce different telemetry.")
    out("")
    base = np.array([0.08, 0.0, 0.0, 0.0, 0.0])
    reference_rate = simulate(perturb(NOMINAL, base), campaign).rate_rad_s
    rows = []
    for label, direction, must_fire in (
        ("the scale direction (correct)", SCALE_DIRECTION, False),
        ("the best identifiable direction", report.identifiable_basis[0], True),
        ("a single coordinate axis", np.eye(5)[2], True),
    ):
        partner = base + 0.08 * np.sqrt(5.0) * direction
        partner_rate = simulate(perturb(NOMINAL, partner), campaign).rate_rad_s
        relative = float(
            np.abs(reference_rate - partner_rate).max() / np.abs(reference_rate).max()
        )
        fires = relative > 1.0e-13
        rows.append([label, f"{relative:.6e}", "yes" if fires else "no"])
        out.control(
            f"a partner displaced along {label}",
            fires,
            must_fire,
            f"relative rate difference {relative:.6e} against a limit of 1e-13",
        )
    out("")
    out(format_table(["displacement direction", "relative difference", "fires"], rows))

    out.rule("J. the refusal itself")
    out("The truth is that a parameter with a null-space component has no finite")
    out("bound. The control raises the declared null-component tolerance until the")
    out("component -- which is exactly 1/sqrt(5) = 0.447214 -- falls below it, which")
    out("makes the refusal disappear and the naive finite bound reappear.")
    out("")
    rows = []
    component = float(report.null_component[0])
    for tolerance, must_fire in ((NULL_COMPONENT_TOL, False), (0.5, True), (1.0, True)):
        refuses = component > tolerance
        rows.append([
            f"{tolerance:.1e}", f"{component:.6f}",
            "REFUSED" if refuses else "a finite bound appears",
            "yes" if not refuses else "no",
        ])
        out.control(
            f"a null-component tolerance of {tolerance:.1e} loses the refusal",
            not refuses,
            must_fire,
            f"the null component is {component:.6f}, so a tolerance of "
            f"{tolerance:.1e} "
            + ("keeps" if refuses else "loses")
            + " the refusal; the declared tolerance is "
            f"{NULL_COMPONENT_TOL:.1e} and the parameter is {report.verdict[0]}",
        )
    out("")
    out(format_table(
        ["null-component tolerance", "measured component", "verdict", "fires"], rows
    ))
    out("")
    out(f"The declared tolerance is {NULL_COMPONENT_TOL:.1e} and the measured")
    out(f"components are all {component:.6f}, so the verdict has "
        f"{np.log10(component / NULL_COMPONENT_TOL):.4f} decades of margin: there is")
    out("no population of parameters between 'in the null space' and 'not in it' on")
    out("this plant, because the null space is exactly the diagonal.")

    out.rule("summary")
    out(f"controls declared : {out.controls_expected}")
    out(f"controls that fired : {out.controls_fired}")
    out("")
    out("A control that does not fire when it must is a failure of the control and")
    out("not a success of the product: it would mean the check it guards could not")
    out("catch the violation it exists to catch.")
    _ = stats

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
