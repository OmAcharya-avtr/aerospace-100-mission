"""The bound check: the design matrix against an exact rotation composition.

This is the check class the build guide calls out as distinct: not "two
implementations agree to a tolerance I chose", but "the answer lies inside a
bound derived from different mathematics". The bound is equation (15) of
:mod:`pointmodel.geometry`, the operator-norm Taylor remainder of a product of
matrix exponentials. It has no fitted constant, and its slack follows the
conditioning of the problem through ``s^2`` in (15): double the misalignments
and the admissible error quadruples.

Four parts:

1. The bound over four decades of misalignment amplitude, with the measured
   worst ratio of error to bound.
2. The bound's quadratic scaling, measured against the derived exponent.
3. Negative controls: every design-matrix column is corrupted in turn, in two
   ways, and each corruption must push the reconstruction outside the bound.
   Each control is asserted to FIRE.
4. The elevation dependence, because the sky-plane error grows toward the
   zenith while the vector-space bound does not: the comparison is made in
   vector space precisely so that the ``sec(E)`` amplification cannot be
   mistaken for a model error.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from pointmodel import geometry, terms  # noqa: E402
from pointmodel.report import format_table  # noqa: E402

AMPLITUDES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SAMPLES = 400


def _draw(seed: int, count: int = SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    elevation = np.radians(rng.uniform(3.0, 88.0, count))
    return azimuth, elevation


def bound_holds(out: Tee) -> list[float]:
    """Part 1: the bound over four decades."""
    out.rule("1. equation (16) over five decades of misalignment amplitude")
    out("For each amplitude, 400 random directions over azimuth and 3 to 88 degrees")
    out("elevation, with all nine coefficients drawn uniformly on [-A, A]. The error")
    out("is the Euclidean distance between the exact beam vector of equation (12) and")
    out("the first-order reconstruction from the design matrix; the bound is B(s) of")
    out("equation (15) with s the sum of the five small rotation magnitudes.")
    out("")
    rows = []
    ratios = []
    for amplitude in AMPLITUDES:
        azimuth, elevation = _draw(int(amplitude) + 7)
        rng = np.random.default_rng(int(amplitude) + 11)
        coefficients = rng.uniform(-amplitude, amplitude, terms.N_TERMS)
        error, bound = geometry.linearisation_error(azimuth, elevation, coefficients)
        ratio = float(np.max(error / bound))
        ratios.append(ratio)
        rows.append(
            [
                f"{amplitude:g}",
                f"{float(np.max(error)):.6e}",
                f"{float(np.max(bound)):.6e}",
                f"{ratio:.6f}",
                str(int(np.count_nonzero(error > bound))),
            ]
        )
    out(
        format_table(
            [
                "amplitude arcsec",
                "worst error (vector norm)",
                "bound B(s)",
                "worst error / bound",
                "violations of 400",
            ],
            rows,
        )
    )
    out("")
    out.check(
        "the bound is never violated at any amplitude",
        all(int(row[4]) == 0 for row in rows),
    )
    out.check(
        "the bound is not vacuous: the worst ratio exceeds 0.1 at every amplitude",
        all(ratio > 0.1 for ratio in ratios),
        f"worst ratios {', '.join(f'{ratio:.4f}' for ratio in ratios)}",
    )
    out.check(
        "the worst ratio is stable across five decades",
        max(ratios) / min(ratios) < 3.0,
        f"max/min = {max(ratios) / min(ratios):.4f}",
    )
    return ratios


def quadratic_scaling(out: Tee) -> None:
    """Part 2: the error's exponent against the derived value of two."""
    out.rule("2. the truncation error scales as the square of the misalignment")
    out("Measured by fitting a straight line to log(worst error) against")
    out("log(amplitude) over the five decades above, on one fixed set of directions")
    out("and one fixed coefficient direction scaled by the amplitude. The derived")
    out("exponent is exactly 2.")
    out("")
    azimuth, elevation = _draw(31)
    rng = np.random.default_rng(37)
    direction = rng.uniform(-1.0, 1.0, terms.N_TERMS)
    worst = []
    for amplitude in AMPLITUDES:
        error, _bound = geometry.linearisation_error(
            azimuth, elevation, amplitude * direction
        )
        worst.append(float(np.max(error)))
    slope, intercept = np.polyfit(np.log10(AMPLITUDES), np.log10(worst), 1)
    out(
        format_table(
            ["amplitude arcsec", "worst error (vector norm)"],
            [
                [f"{amplitude:g}", f"{value:.6e}"]
                for amplitude, value in zip(AMPLITUDES, worst, strict=True)
            ],
        )
    )
    out("")
    out(f"fitted exponent : {float(slope):.6f}   (derived value: 2)")
    out(f"fitted prefactor: 10^{float(intercept):.6f}")
    out("")
    out.check(
        "the fitted exponent is 2 to within one per cent",
        abs(float(slope) - 2.0) < 0.02,
        f"slope = {float(slope):.6f}",
    )


def _violation_counter(
    coefficients: np.ndarray, seed: int, count: int
):
    """Build a closure counting bound violations for a given design matrix."""
    azimuth, elevation = _draw(seed, count)
    bounds = np.array(
        [
            geometry.linearisation_bound(
                geometry.total_small_angle(coefficients, float(value))
            )
            for value in elevation
        ]
    )
    exact = np.stack(
        [
            geometry.beam_direction(float(a), float(e), coefficients)
            for a, e in zip(azimuth, elevation, strict=True)
        ]
    )
    nominal = np.stack(
        [
            geometry.sky_vector(float(a), float(e))
            for a, e in zip(azimuth, elevation, strict=True)
        ]
    )
    tangents = [
        geometry.tangent_basis(float(a), float(e))
        for a, e in zip(azimuth, elevation, strict=True)
    ]
    tangent_cross = np.stack([pair[0] for pair in tangents])
    tangent_elev = np.stack([pair[1] for pair in tangents])
    cross_block, elev_block = terms.design_matrix(azimuth, elevation)

    def count_violations(index: int | None = None, factor: float = 1.0) -> int:
        cross = cross_block.copy()
        elev = elev_block.copy()
        if index is not None:
            # Only one of the two blocks carries each term; scaling the column
            # in both leaves the identically-zero one unaffected.
            cross[:, index] *= factor
            elev[:, index] *= factor
        linear = (
            nominal
            + ((cross @ coefficients) * geometry.RAD_PER_ARCSEC)[:, None] * tangent_cross
            + ((elev @ coefficients) * geometry.RAD_PER_ARCSEC)[:, None] * tangent_elev
        )
        error = np.linalg.norm(exact - linear, axis=1)
        return int(np.count_nonzero(error > bounds))

    return count_violations, azimuth.size, float(np.max(bounds))


def negative_controls(out: Tee) -> None:
    """Part 3: corrupt each column in turn; every control must FIRE or not, as declared."""
    out.rule("3. negative controls: every column corrupted in turn")
    out("Each control is anchored on the exact rotation model -- the TRUTH -- and not")
    out("on any output of the fit. A corruption that did not push the reconstruction")
    out("outside the bound is reported as having failed to fire.")
    out("")
    out("The bound has a RESOLUTION, and it is stated rather than discovered by a")
    out("reader. B(s) admits an error of B(s) * 206265 arcsec, so a corruption")
    out("smaller than that is inside the bound and cannot be detected by it. With all")
    out("nine coefficients at 100 arcsec the total small angle s of equation (14) is")
    out("about 800 arcsec and the bound admits roughly 3 arcsec of sky error; with")
    out("all nine at 10 arcsec it admits roughly 0.03 arcsec. So the scale-error")
    out("controls are run at 10 arcsec, where the bound is tight, and a deliberately")
    out("undetectable corruption is run at 100 arcsec as a must-NOT-fire control.")
    out("")
    big_counter, big_count, big_bound = _violation_counter(
        np.full(terms.N_TERMS, 100.0), 97, 200
    )
    small_counter, small_count, small_bound = _violation_counter(
        np.full(terms.N_TERMS, 10.0), 97, 200
    )
    out(f"bound at 100 arcsec coefficients : {big_bound:.6e} "
        f"= {big_bound * geometry.ARCSEC_PER_RAD:.6f} arcsec of sky error")
    out(f"bound at 10 arcsec coefficients  : {small_bound:.6e} "
        f"= {small_bound * geometry.ARCSEC_PER_RAD:.6f} arcsec of sky error")
    out("")
    out.check(
        "the uncorrupted design matrix violates the bound nowhere, at either amplitude",
        big_counter() == 0 and small_counter() == 0,
        f"{big_counter()} and {small_counter()} of {big_count} observations",
    )
    rows = []
    specification = (
        ("sign flip, 100 arcsec", big_counter, big_count, -1.0, True),
        ("1 per cent scale, 10 arcsec", small_counter, small_count, 1.01, True),
        ("1 part per million scale, 100 arcsec", big_counter, big_count, 1.000001, False),
    )
    for label, counter, population, factor, must_fire in specification:
        for index, name in enumerate(terms.TERM_NAMES):
            count = counter(index, factor)
            fired = count > 0
            rows.append(
                [name, label, str(count), str(population), "FIRED" if fired else "-"]
            )
            out.control(
                f"{name}: {label}",
                fired,
                must_fire=must_fire,
                detail=f"{count} of {population} observations outside the bound",
            )
    out("")
    out(
        format_table(
            ["term", "corruption", "outside the bound", "observations", "verdict"], rows
        )
    )
    out("")
    out("The third block is the honest statement of what this bound cannot do. A one")
    out("part per million column error at 100 arcsec coefficients produces 1e-4")
    out("arcsec of sky error, which is thirty thousand times inside the bound, and no")
    out("bound of this form can see it. Detecting an error that small needs the")
    out("term-recovery check of validate_term_recovery.py, which is a different")
    out("instrument and is why both exist.")
    out("")
    out("Measured detection threshold: the smallest relative column error the bound")
    out("catches, found by bisection on the scale factor, at 10 arcsec coefficients.")
    out("")
    threshold_rows = []
    for index, name in enumerate(terms.TERM_NAMES):
        low, high = 1.0, 2.0
        for _ in range(40):
            middle = 0.5 * (low + high)
            if small_counter(index, middle) > 0:
                high = middle
            else:
                low = middle
        threshold_rows.append([name, f"{(high - 1.0) * 100.0:.6f}"])
    out(
        format_table(
            ["term", "smallest detected column error, per cent"], threshold_rows
        )
    )
    out("")
    out.check(
        "every column has a finite detection threshold below 1 per cent at "
        "10 arcsec coefficients",
        all(float(row[1]) < 1.0 for row in threshold_rows),
        f"worst = {max(float(row[1]) for row in threshold_rows):.6f} per cent",
    )


def elevation_dependence(out: Tee) -> None:
    """Part 4: why the comparison is made in vector space."""
    out.rule("4. the sky-plane error grows toward the zenith; the bound does not")
    out("The vector-space comparison of equation (16) is deliberate. In sky-plane")
    out("angles the cross-elevation error of a rank-deficient direction carries a")
    out("factor of sec(E), so a comparison in angles would have to carry an")
    out("elevation-dependent tolerance and a reader could not tell an amplification")
    out("from a model error. In vector space there is no such factor and the bound is")
    out("a single expression.")
    out("")
    coefficients = np.full(terms.N_TERMS, 200.0)
    rows = []
    for degrees in (5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 88.0):
        elevation = np.full(12, np.radians(degrees))
        azimuth = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
        error, bound = geometry.linearisation_error(azimuth, elevation, coefficients)
        cross_exact, elev_exact = geometry.exact_residuals(
            azimuth, elevation, coefficients
        )
        cross_block, elev_block = terms.design_matrix(azimuth, elevation)
        sky = np.maximum(
            np.abs(cross_exact - cross_block @ coefficients),
            np.abs(elev_exact - elev_block @ coefficients),
        )
        rows.append(
            [
                f"{degrees:g}",
                f"{float(np.max(error)):.4e}",
                f"{float(np.max(bound)):.4e}",
                f"{float(np.max(error / bound)):.4f}",
                f"{float(np.max(sky)):.4e}",
                f"{1.0 / np.cos(np.radians(degrees)):.4f}",
            ]
        )
    out(
        format_table(
            [
                "elevation deg",
                "worst vector error",
                "bound B(s)",
                "error / bound",
                "worst sky-plane error arcsec",
                "sec(E)",
            ],
            rows,
        )
    )
    out("")
    out.check(
        "the vector-space bound holds at every elevation in the declared range",
        all(float(row[3]) <= 1.0 for row in rows),
    )
    out.check(
        "the sky-plane error grows with sec(E) while the bound does not",
        float(rows[-1][4]) > 3.0 * float(rows[3][4]),
        f"88 deg: {rows[-1][4]} arcsec against 45 deg: {rows[3][4]} arcsec",
    )


def main() -> int:
    out = Tee("validate_linearisation_output.txt")
    out.rule("pointmodel -- the design matrix against an independently derived bound")
    for line in environment_lines():
        out(line)
    out("")
    out("The bound is equation (15) of pointmodel.geometry: the operator-norm Taylor")
    out("remainder of a product of matrix exponentials. Nothing in")
    out("pointmodel.terms uses it. It has no fitted constant and no slack term, and")
    out("its tolerance follows the conditioning of the problem through s^2.")
    bound_holds(out)
    quadratic_scaling(out)
    negative_controls(out)
    elevation_dependence(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
