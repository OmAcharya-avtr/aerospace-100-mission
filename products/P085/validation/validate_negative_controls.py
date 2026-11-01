"""Negative controls, every one anchored on the true value and verified to fire.

Why this script exists separately
---------------------------------
A check that passes is evidence only if it would have failed on a wrong answer.
Injecting a violation by perturbing an OUTPUT does not guarantee that: the
output already sits inside whatever band is being tested. Every control here is
built by corrupting something whose correct value is known independently -- the
design matrix, the term set, the declared noise, the geometry -- and the result
is compared against the TRUE generating coefficients from
:mod:`pointmodel.synthetic`.

Each control declares in advance whether it must FIRE or must NOT fire, and
both directions are asserted. A control that does not fire when it must is
reported as a failure of the control, because it means the check it guards
would not have caught the thing it exists to catch. A control that fires when
it must not is equally wrong and is reported the same way.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import (
    OBSERVATIONS,
    SEED,
    SIGMA_ARCSEC,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from pointmodel import identifiability, refraction, robust, synthetic, terms, wls  # noqa: E402
from pointmodel.report import format_table  # noqa: E402

#: A recovery is declared WRONG when a coefficient sits this many declared
#: sigmas from the truth. Anchored on the truth; declared before measuring.
WRONG_SIGMAS = 5.0


def _worst_error(design: np.ndarray, observed: np.ndarray, truth: np.ndarray) -> float:
    result = wls.fit(design, observed, SIGMA_ARCSEC)
    return float(np.max(np.abs(result.coefficients - truth)))


def main() -> int:  # noqa: C901 - a flat list of controls reads better than a nest
    out = Tee("validate_negative_controls_output.txt")
    out.rule("pointmodel -- negative controls, anchored on the true value")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared sigma        : {SIGMA_ARCSEC:g} arcsec per residual row")
    out(f"declared observations : {OBSERVATIONS}")
    out(f"declared geometry seed: {SEED}")
    out(
        f"a recovery is WRONG when a coefficient is more than {WRONG_SIGMAS:g} declared "
        f"sigmas ({WRONG_SIGMAS * SIGMA_ARCSEC:g} arcsec) from the truth"
    )
    out("")

    clean = synthetic.generate(
        "full_sky", OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
    )
    noiseless = synthetic.generate("full_sky", OBSERVATIONS, sigma_arcsec=0.0, seed=SEED)
    truth = clean.coefficients
    baseline = _worst_error(clean.design, clean.observed, truth)
    baseline_noiseless = _worst_error(noiseless.design, noiseless.observed, truth)

    out.rule("A. the uncorrupted baseline, which every control is measured against")
    out(f"worst |coefficient error|, with noise      : {baseline:.6f} arcsec")
    out(f"worst |coefficient error|, noiseless       : {baseline_noiseless:.6f} arcsec")
    out("")
    out.control(
        "the correct model on noiseless data is WRONG",
        baseline_noiseless > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=False,
        detail=f"{baseline_noiseless:.6f} arcsec",
    )
    out.control(
        "the correct model on noisy data is WRONG",
        baseline > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=False,
        detail=f"{baseline:.6f} arcsec",
    )

    out.rule("B. corrupting the design matrix")
    out("Each corruption is applied to the MODEL while the data keep coming from the")
    out("exact rotation composition with the true coefficients, so the truth is")
    out("untouched and the comparison is against it.")
    out("")
    rows = []

    # B1: sign flip on each column in turn.
    for index, name in enumerate(terms.TERM_NAMES):
        corrupted = noiseless.design.copy()
        corrupted[:, index] *= -1.0
        error = _worst_error(corrupted, noiseless.observed, truth)
        rows.append([f"sign flip on {name}", f"{error:.4f}"])
        out.control(
            f"sign flip on the {name} column",
            error > WRONG_SIGMAS * SIGMA_ARCSEC,
            must_fire=True,
            detail=f"worst |error| {error:.4f} arcsec against a baseline of "
            f"{baseline_noiseless:.4f}",
        )

    # B2: the refraction column replaced by a plausible wrong shape.
    wrong_shape = noiseless.design.copy()
    cosine = np.cos(noiseless.elevation_rad)
    wrong_shape[OBSERVATIONS:, terms.TERM_INDEX["RF"]] = cosine
    error = _worst_error(wrong_shape, noiseless.observed, truth)
    rows.append(["refraction shape replaced by cos(E)", f"{error:.4f}"])
    out.control(
        "the refraction column replaced by cos(E)",
        error > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=True,
        detail=f"worst |error| {error:.4f} arcsec. cos(E) is in the same family as the "
        "flexure column, which is exactly why a wrong refraction shape is absorbed "
        "by the flexure terms and corrupts them rather than itself",
    )

    # B3: the refraction column replaced by the ray trace's own shape, which is
    # a DIFFERENT calculation of the same physical quantity and must NOT fire.
    ray_shape = noiseless.design.copy()
    ray_values = np.atleast_1d(
        refraction.raytrace_refraction(
            np.rad2deg(noiseless.elevation_rad),
            refractivity_constant=refraction.K_OPTICAL_550NM,
        )
    )
    ray_reference = float(
        np.atleast_1d(
            refraction.raytrace_refraction(
                45.0, refractivity_constant=refraction.K_OPTICAL_550NM
            )
        )[0]
    )
    ray_shape[OBSERVATIONS:, terms.TERM_INDEX["RF"]] = ray_values / ray_reference
    error = _worst_error(ray_shape, noiseless.observed, truth)
    rows.append(["refraction shape from the ray trace", f"{error:.4f}"])
    out.control(
        "the refraction column replaced by the independent ray trace's own shape",
        error > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=False,
        detail=f"worst |error| {error:.4f} arcsec. The two shapes differ by up to 2.0 "
        "per cent and the fit absorbs that into the flexure terms without exceeding "
        "the declared criterion. This is the control that says the preceding one "
        "measures a wrong shape and not merely a changed one",
    )

    # B4: the two blocks swapped.
    swapped = np.concatenate(
        [noiseless.design[OBSERVATIONS:], noiseless.design[:OBSERVATIONS]], axis=0
    )
    error = _worst_error(swapped, noiseless.observed, truth)
    rows.append(["cross-elevation and elevation blocks swapped", f"{error:.4f}"])
    out.control(
        "the cross-elevation and elevation blocks swapped",
        error > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=True,
        detail=f"worst |error| {error:.4f} arcsec",
    )
    out("")
    out(format_table(["corruption", "worst |coefficient error| arcsec"], rows))

    out.rule("C. omitting a term that the data contain")
    out("The model is fitted with eight columns while the truth has nine. The")
    out("omitted term's signal has to go somewhere, and where it goes is the")
    out("measurement. There are two places it can go, and the control tests for")
    out("BOTH, because an omission that goes to only one of them would pass a")
    out("control that tested only the other:")
    out("")
    out("  (i)  into the remaining coefficients, as a bias -- measured as the largest")
    out("       |bias| / se over the eight remaining terms, with the standard errors")
    out("       taken from the DECLARED sigma so that the quantity asks whether the")
    out("       bias would be visible against the noise the user declared. The")
    out("       criterion follows each term's own conditioning through its se rather")
    out("       than being an absolute number of arcseconds.")
    out("  (ii) into the post-fit residual, as unexplained signal -- measured against")
    out("       the correct model's own post-fit residual on the same data.")
    out("")
    out("An omission is DETECTABLE when either exceeds its criterion. The absolute")
    out("bias in arcseconds is reported beside both, because it is what a reader of a")
    out("pointing report would see.")
    out("")
    reference = wls.fit(
        noiseless.design, noiseless.observed, SIGMA_ARCSEC, error_model="declared"
    )
    reference_rms = float(reference.rms_arcsec)
    reference_z = float(
        np.max(np.abs(reference.coefficients - truth) / reference.standard_errors)
    )
    out(f"correct model, noiseless: post-fit rms {reference_rms:.6f} arcsec, "
        f"worst |bias|/se {reference_z:.6f}")
    out("")
    rows = []
    for index, name in enumerate(terms.TERM_NAMES):
        kept = [position for position in range(terms.N_TERMS) if position != index]
        # Pad back to nine columns so that wls.fit accepts the matrix, and read
        # only the kept coefficients. A zero column is rejected by the fit, so
        # the omitted column is replaced by a dummy carrying a single unit entry
        # on one row and no structure at all.
        padded = np.zeros_like(noiseless.design)
        padded[:, kept] = noiseless.design[:, kept]
        padded[0, index] = 1.0
        result = wls.fit(
            padded, noiseless.observed, SIGMA_ARCSEC, error_model="declared"
        )
        absolute_bias = float(np.max(np.abs(result.coefficients[kept] - truth[kept])))
        standardised = float(
            np.max(
                np.abs(result.coefficients[kept] - truth[kept])
                / result.standard_errors[kept]
            )
        )
        residual_ratio = float(result.rms_arcsec) / reference_rms
        detectable = standardised > WRONG_SIGMAS or residual_ratio > 10.0
        rows.append(
            [
                name,
                f"{absolute_bias:.4f}",
                f"{standardised:.3f}",
                f"{result.rms_arcsec:.4f}",
                f"{residual_ratio:.1f}",
                "bias" if standardised > WRONG_SIGMAS else "residual",
            ]
        )
        out.control(
            f"omitting the {name} term from the model",
            detectable,
            must_fire=True,
            detail=f"|bias| {absolute_bias:.4f} arcsec, |bias|/se {standardised:.3f}, "
            f"post-fit rms {result.rms_arcsec:.4f} arcsec "
            f"({residual_ratio:.1f}x the correct model's)",
        )
    out("")
    out(
        format_table(
            [
                "omitted term",
                "worst |bias| arcsec",
                "worst |bias|/se",
                "post-fit rms arcsec",
                "rms vs correct model",
                "detected by",
            ],
            rows,
        )
    )
    out("")
    out("Read the AN and AW rows. Omitting either leaves a 3 to 5 arcsec post-fit")
    out("residual and barely biases the remaining coefficients at all -- |bias|/se of")
    out("3.36 and 1.01 against 152 for IA. That is because the azimuth-axis tilt")
    out("columns are nearly orthogonal to every other column on a full-sky geometry,")
    out("where their variance inflation factor is close to one. It is a useful fact")
    out("about this model: a missing tilt term shows up in the residual and not in")
    out("the other coefficients, while a missing index or refraction term corrupts")
    out("the coefficients and can leave the residual almost clean. A control that")
    out("tested only one of the two routes would have missed half of these.")

    out.rule("D. the geometry")
    out("The rank analysis must refuse where the geometry cannot separate the terms")
    out("and must refuse nothing where it can. Both directions are controls.")
    out("")
    rows = []
    for name in sorted(synthetic.GEOMETRIES):
        residuals = synthetic.generate(
            name, OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
        )
        result = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        report = identifiability.analyse_fit(residuals.design, result)
        refused = len(report.unidentifiable_terms) > 0
        must_fire = name in {"narrow_el", "single_az", "low_el"}
        rows.append(
            [
                name,
                f"{report.condition_number:.4e}",
                "must refuse" if must_fire else "must NOT refuse",
                "refused" if refused else "did not refuse",
            ]
        )
        out.control(
            f"the rank analysis on the {name} geometry",
            refused,
            must_fire=must_fire,
            detail=f"condition number {report.condition_number:.4e}, "
            f"refused {', '.join(report.unidentifiable_terms) or 'nothing'}",
        )
    out("")
    out(format_table(["geometry", "condition", "declared", "verdict"], rows))

    out.rule("E. perturbation along a null direction")
    out("Adding a large multiple of a refused direction to the coefficients must")
    out("leave the predicted residual field alone -- that is what 'not identifiable'")
    out("means -- while adding the same step along the best-determined direction must")
    out("move it. Both are controls on the same geometry.")
    out("")
    narrow = synthetic.generate(
        "narrow_el", OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
    )
    result = wls.fit(narrow.design, narrow.observed, SIGMA_ARCSEC)
    report = identifiability.analyse_fit(narrow.design, result)
    base_prediction = narrow.design @ result.coefficients
    step = 1000.0
    rows = []
    for direction in report.null_directions:
        moved = narrow.design @ (result.coefficients + step * direction.weights)
        shift = float(np.sqrt(np.mean((moved - base_prediction) ** 2)))
        rows.append([f"null direction {direction.index}", f"{shift:.6e}"])
        out.control(
            f"a {step:g} arcsec step along null direction {direction.index} "
            "changes the prediction",
            shift > SIGMA_ARCSEC,
            must_fire=False,
            detail=f"predicted sky rms shift {shift:.6e} arcsec against a declared "
            f"sigma of {SIGMA_ARCSEC:g}",
        )
    best = report.combinations[0]
    unit = best.unit_weights / float(np.linalg.norm(best.unit_weights))
    moved = narrow.design @ (result.coefficients + step * unit)
    shift = float(np.sqrt(np.mean((moved - base_prediction) ** 2)))
    rows.append(["best-determined direction 0", f"{shift:.6e}"])
    out.control(
        f"a {step:g} arcsec step along the best-determined direction changes "
        "the prediction",
        shift > SIGMA_ARCSEC,
        must_fire=True,
        detail=f"predicted sky rms shift {shift:.6e} arcsec",
    )
    out("")
    out(format_table(["direction", "predicted sky rms shift arcsec"], rows))

    out.rule("F. the declared noise and the fit quality")
    out("A declared sigma ten times too small must show up as a reduced chi-square")
    out("near one hundred, and ten times too large as near one hundredth. The fit")
    out("does not correct the declaration and does not hide it.")
    out("")
    rows = []
    for factor in (0.1, 1.0, 10.0):
        result = wls.fit(clean.design, clean.observed, SIGMA_ARCSEC * factor)
        rows.append(
            [
                f"{factor:g}",
                f"{SIGMA_ARCSEC * factor:g}",
                f"{result.reduced_chi_square:.6f}",
            ]
        )
    out(
        format_table(
            ["sigma factor", "declared sigma arcsec", "reduced chi-square"], rows
        )
    )
    out("")
    out.control(
        "a reduced chi-square far from one on a correctly declared sigma",
        abs(float(rows[1][2]) - 1.0) > 0.5,
        must_fire=False,
        detail=f"reduced chi-square {rows[1][2]}",
    )
    out.control(
        "a reduced chi-square far from one on a sigma ten times too small",
        abs(float(rows[0][2]) - 1.0) > 0.5,
        must_fire=True,
        detail=f"reduced chi-square {rows[0][2]}, expected near 100",
    )

    out.rule("G. contamination and the diagnostic")
    contaminated = synthetic.generate(
        "full_sky",
        OBSERVATIONS,
        sigma_arcsec=SIGMA_ARCSEC,
        outlier_fraction=0.06,
        outlier_arcsec=300.0,
        seed=SEED,
    )
    least = wls.fit(contaminated.design, contaminated.observed, SIGMA_ARCSEC)
    huber = robust.huber_fit(contaminated.design, contaminated.observed, SIGMA_ARCSEC)
    contaminated_error = float(np.max(np.abs(least.coefficients - truth)))
    clean_least = wls.fit(clean.design, clean.observed, SIGMA_ARCSEC)
    clean_huber = robust.huber_fit(clean.design, clean.observed, SIGMA_ARCSEC)
    dirty_delta = robust.disagreement(
        contaminated.design, least, huber, sigma_reference=SIGMA_ARCSEC
    )
    clean_delta = robust.disagreement(
        clean.design, clean_least, clean_huber, sigma_reference=SIGMA_ARCSEC
    )
    out(f"least squares worst |error|, contaminated : {contaminated_error:.4f} arcsec")
    out(f"least squares worst |error|, clean        : {baseline:.4f} arcsec")
    out(f"huber worst |error|, contaminated         : "
        f"{float(np.max(np.abs(huber.coefficients - truth))):.4f} arcsec")
    out(f"disagreement, contaminated                : "
        f"{dirty_delta.prediction_rms_arcsec:.4f} arcsec")
    out(f"disagreement, clean                       : "
        f"{clean_delta.prediction_rms_arcsec:.4f} arcsec")
    out("")
    out.control(
        "6 per cent acquisition failures make least squares WRONG",
        contaminated_error > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=True,
        detail=f"{contaminated_error:.4f} arcsec",
    )
    out.control(
        "6 per cent acquisition failures make Huber WRONG",
        float(np.max(np.abs(huber.coefficients - truth))) > WRONG_SIGMAS * SIGMA_ARCSEC,
        must_fire=False,
        detail=f"{float(np.max(np.abs(huber.coefficients - truth))):.4f} arcsec",
    )
    out.control(
        "the diagnostic on a contaminated run",
        dirty_delta.flagged,
        must_fire=True,
        detail=f"predicted sky rms {dirty_delta.prediction_rms_arcsec:.4f} arcsec",
    )
    out.control(
        "the diagnostic on a clean run",
        clean_delta.flagged,
        must_fire=False,
        detail=f"predicted sky rms {clean_delta.prediction_rms_arcsec:.4f} arcsec",
    )

    out.rule("H. input rejection")
    out("Out-of-range input must raise rather than return a wrong answer. Each of")
    out("these is a control that must fire.")
    out("")
    rejections = (
        (
            "elevation below the refraction model's validity range",
            lambda: terms.design_matrix(np.array([0.0]), np.array([0.01])),
        ),
        (
            "elevation above the refraction model's validity range",
            lambda: terms.design_matrix(np.array([0.0]), np.array([1.5700])),
        ),
        ("a non-positive sigma", lambda: wls.fit(clean.design, clean.observed, 0.0)),
        (
            "an unknown term name",
            lambda: terms.coefficients_from_mapping({"TILT": 1.0}),
        ),
        (
            "a confidence level of one",
            lambda: wls.fit(
                clean.design, clean.observed, SIGMA_ARCSEC, confidence_level=1.0
            ),
        ),
        (
            "an unknown geometry",
            lambda: synthetic.generate("equator", 100),
        ),
        (
            "a refraction altitude at the zenith",
            lambda: refraction.bennett_refraction(90.0),
        ),
        (
            "an odd row count for RANSAC",
            lambda: robust.ransac_fit(
                np.ones((21, terms.N_TERMS)), np.zeros(21), SIGMA_ARCSEC
            ),
        ),
        (
            "inconsistent identifiability thresholds",
            lambda: identifiability.analyse(
                clean.design,
                np.zeros(terms.N_TERMS),
                null_threshold=0.5,
                marginal_threshold=0.1,
            ),
        ),
    )
    for label, action in rejections:
        try:
            action()
            raised = False
            message = "no exception"
        except ValueError as error:
            raised = True
            message = str(error).splitlines()[0][:90]
        out.control(label, raised, must_fire=True, detail=message)

    out.rule("summary")
    out(
        f"{out.controls_expected} controls declared, {out.controls_fired} fired, "
        f"{out.controls_expected - out.controls_fired} did not."
    )
    out("Each one's declared direction is in its line above. A control that did not")
    out("fire when it must is a FAILED CHECK and is listed at the end.")
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
