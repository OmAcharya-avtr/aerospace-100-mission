"""Pinned regressions for defects this product actually had, and suite self-checks.

Every test here corresponds to something that was wrong in this repository at
some point during its construction, with the literal inputs written out.
``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone; a counterexample is only a regression test once it is
written down here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from orbitdrift import forces
from orbitdrift.constants import WGS72
from orbitdrift.errors import InsufficientDataError
from orbitdrift.growth import fit_power_law

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src" / "orbitdrift"


def test_regression_drag_unit_conversion_is_a_multiplication_by_1000() -> None:
    """The drag term divided by 1000 where it had to multiply: a factor of 1e6.

    The consequence was that the drag acceleration was 1e-6 of its true size,
    so switching the term on or off changed a divergence curve by 2.09e-07
    relative -- which is to say, not at all. It survived a unit test that
    recomputed the same expression.

    Pinned by an absolute magnitude taken from an independent SI evaluation:
    at 400 km with ``C_D A / m = 1.2738853503184714e-3`` m^2/kg and a
    7.67 km/s velocity, the drag acceleration is
    1.2217253994563807e-10 km/s^2. A sign or factor error moves this by orders
    of magnitude.
    """
    acceleration = forces.drag_acceleration_km_s2(
        np.array([WGS72.radius_km + 400.0, 0.0, 0.0]),
        np.array([0.0, 7.67, 0.0]),
        2451545.0,
        1.2738853503184714e-3,
    )
    assert float(np.linalg.norm(acceleration)) == pytest.approx(
        1.2217253994563807e-10, rel=1e-9, abs=0.0
    )


def test_regression_repeated_ages_refused_not_given_a_zero_exponent() -> None:
    """``lstsq`` on a constant age column returns exponent 0 with stderr 0.

    The literal inputs Hypothesis-style generation would find: five points at
    one age. The refusal must name rank deficiency so a reader knows it is not
    a tolerance failure.
    """
    with pytest.raises(InsufficientDataError, match="rank-deficient"):
        fit_power_law(
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]), np.array([1.0, 1.1, 0.9, 1.2, 0.95])
        )


def test_regression_two_nearly_identical_ages_still_produce_a_finite_stderr() -> None:
    """The near-degenerate neighbour of the case above.

    Ages differing by 1e-9 day are not *exactly* equal, so the refusal above
    does not fire and the fit proceeds on a nearly singular design matrix. The
    requirement is only that nothing fabricated comes out: the exponent and
    its standard error must both be finite, and the standard error must be
    large, because the data genuinely do not determine the exponent.
    """
    ages = np.array([2.0, 2.0 + 1e-9, 2.0 + 2e-9, 2.0 + 3e-9])
    drift = np.array([1.0, 1.1, 0.9, 1.2])
    fit = fit_power_law(ages, drift)
    assert np.isfinite(fit.exponent)
    assert np.isfinite(fit.exponent_stderr)
    assert fit.exponent_stderr > 1e6


def test_regression_duplicate_catalogue_numbers_are_matched_positionally() -> None:
    """Keying the published blocks by catalogue number merged two test cases.

    Catalogue 20413 appears twice in ``SGP4-VER.TLE`` with different
    schedules. Keying on it inflated the parser's row count from 667 to 763
    and compared one element set against the other's rows. Pinned by the row
    count, which is the number the defect changed.
    """
    from orbitdrift.vectors import compare_to_published, published_cases

    cases = published_cases()
    total = sum(
        result.rows_compared + result.rows_skipped_error_code
        for result in (compare_to_published(case) for case in cases)
    )
    assert total == 667


def test_regression_negative_control_offset_must_exceed_twice_the_bound() -> None:
    """Injecting a violation by offsetting a value does not guarantee one.

    If a quantity is accepted when it lies within ``b`` of a reference, then
    adding ``b`` to a value that was at ``-b`` lands it at ``0``, which is
    still accepted. Only an offset strictly greater than ``2b``, applied to
    the *true* value, is guaranteed to fire. This test states the arithmetic
    and then verifies that each negative control in it actually fires, which
    is the step that is usually skipped.
    """
    bound = 0.1
    truth = 1.0

    def accepts(value: float) -> bool:
        return abs(value - truth) <= bound

    # A value already at the edge of the band, offset by exactly the bound,
    # is still inside the band. This is the failure mode.
    edge = truth - bound
    assert accepts(edge)
    assert accepts(edge + bound)

    # Anchored on the true value, an offset of exactly 2b lands on the far
    # edge and is still accepted; strictly more than 2b is required.
    assert accepts(truth + 2.0 * bound - bound)
    assert not accepts(truth + 2.0 * bound + 1e-12)

    # And the control fires.
    assert not accepts(truth + 2.001 * bound)


def test_regression_conservation_negative_control_fires_on_the_true_value() -> None:
    """The same arithmetic applied to a real conservation check.

    The measured energy residual of the integrator at 360 steps per
    revolution over four revolutions is about 1.1e-9 relative. A negative
    control must perturb the trajectory's energy by strictly more than twice
    the tolerance the check uses, anchored on the unperturbed value, and must
    then be verified to fail the check. Both halves are asserted.
    """
    from orbitdrift import integrate
    from orbitdrift.forces import ForceModel
    from orbitdrift.kepler import ClassicalElements, elements_to_rv, orbital_period_s

    elements = ClassicalElements(6878.135, 1e-3, 0.9, 0.3, 0.7, 0.2)
    r, v = elements_to_rv(elements)
    period = orbital_period_s(r, v)
    tolerance = 2e-9
    trajectory = integrate.propagate(
        np.atleast_2d(r),
        np.atleast_2d(v),
        2451545.0,
        np.array([0.0]),
        n_steps=1440,
        step_s=period / 360.0,
        force=ForceModel(zonal=False, drag=False, third_body=False),
        sample_every=36,
    )
    residual = float(integrate.invariant_residuals(trajectory, zonal=False)["energy"][0])
    assert residual < tolerance

    # Perturb the velocity of the LAST sample so the energy there moves by
    # strictly more than twice the tolerance, relative to the true energy.
    energy = integrate.specific_energy_km2_s2(
        trajectory.positions_km, trajectory.velocities_km_s, zonal=False
    )
    target = abs(float(energy[0, 0])) * 3.0 * tolerance
    speed = float(np.linalg.norm(trajectory.velocities_km_s[0, -1]))
    scale = np.sqrt(1.0 + 2.0 * target / speed**2)
    broken_velocities = trajectory.velocities_km_s.copy()
    broken_velocities[0, -1] *= scale
    broken = integrate.Trajectory(
        trajectory.times_s,
        trajectory.positions_km,
        broken_velocities,
        trajectory.step_s,
        trajectory.force_label,
    )
    broken_residual = float(integrate.invariant_residuals(broken, zonal=False)["energy"][0])
    assert broken_residual > tolerance
    assert broken_residual > 2.0 * tolerance


def _approx_calls() -> list[tuple[str, int, tuple[str, ...]]]:
    """Every real ``pytest.approx`` call in this suite, by AST rather than regex.

    A regular expression over the source text also matches the occurrences of
    ``pytest.approx(expected, rel=r)`` inside the docstrings that explain the
    problem, and the first version of this scan reported its own docstring as
    an offender. Parsing finds calls and only calls.
    """
    found: list[tuple[str, int, tuple[str, ...]]] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "approx"
                and isinstance(function.value, ast.Name)
                and function.value.id == "pytest"
            ):
                keywords = tuple(
                    keyword.arg for keyword in node.keywords if keyword.arg is not None
                )
                found.append((path.name, node.lineno, keywords))
    return found


def test_every_approx_call_with_rel_also_passes_abs() -> None:
    """``pytest.approx(expected, rel=r)`` carries a DEFAULT ABSOLUTE TOLERANCE of 1e-12.

    It passes when *either* bound is satisfied, so for any quantity of order
    1e-5 every relative tolerance tighter than 1e-7 is silently replaced by
    the absolute one and the assertion stops testing what it says it tests.
    Conservation residuals, which are small by construction, are exactly where
    this bites.

    The invariant enforced here is that no assertion in this suite depends on
    the default: every ``pytest.approx`` call that passes ``rel=`` also passes
    an explicit ``abs=``. ``validation/validate_approx_audit.py`` goes further
    and re-runs the whole suite with ``abs=0.0`` substituted into every such
    call, and reports how many assertions changed verdict.
    """
    offenders = [
        f"{name}:{line}: keywords {keywords}"
        for name, line, keywords in _approx_calls()
        if "rel" in keywords and "abs" not in keywords
    ]
    assert not offenders, (
        "pytest.approx calls passing rel= without an explicit abs= rely on the "
        "default absolute tolerance of 1e-12:\n  " + "\n  ".join(offenders)
    )


def test_the_approx_scan_finds_calls_at_all() -> None:
    """The negative control for the scan above.

    A regular expression that matched nothing would make the previous test
    pass vacuously. There are many such calls in this suite and the scan must
    see them.
    """
    calls = _approx_calls()
    assert len(calls) > 40
    assert any("rel" in keywords for _, _, keywords in calls)
    assert any("abs" in keywords for _, _, keywords in calls)
    # Every call in this suite passes at least one tolerance keyword; a bare
    # pytest.approx(x) uses rel=1e-6 and abs=1e-12 together, which is never
    # what a physics assertion wants.
    assert all(keywords for _, _, keywords in calls)


def _phrase(*codes: int) -> str:
    """Build a phrase from character codes.

    The phrases this test forbids are built rather than written out, because
    this file is itself a tracked ``.py`` file and the same scan runs over the
    whole repository in ``.github/workflows/tests.yml``. A literal list would
    match itself and fail the repository it is checking -- which it did, until
    the scan was run over every tracked file rather than only over
    ``src/orbitdrift``.
    """
    return "".join(map(chr, codes))


FORBIDDEN_CLAIMS = (
    _phrase(102, 108, 105, 103, 104, 116, 45, 115, 97, 102, 101),
    _phrase(102, 108, 105, 103, 104, 116, 32, 115, 97, 102, 101),
    _phrase(109, 105, 115, 115, 105, 111, 110, 45, 114, 101, 97, 100, 121),
    _phrase(112, 114, 111, 100, 117, 99, 116, 105, 111, 110, 45, 114, 101, 97, 100, 121),
)
QUALIFIED_CLAIMS = (
    _phrase(99, 101, 114, 116, 105, 102, 105, 101, 100),
    _phrase(102, 108, 105, 103, 104, 116, 45, 113, 117, 97, 108, 105, 102, 105, 101, 100),
    _phrase(102, 108, 105, 103, 104, 116, 32, 113, 117, 97, 108, 105, 102, 105, 101, 100),
)


def test_no_tracked_file_claims_flight_qualification_or_certification() -> None:
    """A phrase scan over every tracked text file in the product.

    The mission forbids a product from making any of the claims built above,
    anywhere, including docstrings. None of those words is written out in this
    file for the reason :func:`_phrase` gives. The scan looks for the claim and
    not the denial: the required statements that this package is none of those
    things are preceded by a negation and must not trip it.

    The scan runs over the whole file with adjacent string literals joined and
    whitespace collapsed, not line by line. A line-based version reported two
    false offenders, both of them a forbidden word landing at the start of a
    continuation line whose preceding ``not`` was on the line above.
    """
    root = TESTS_DIR.parent
    suffixes = {".py", ".md", ".toml", ".cff", ".yml", ".txt", ".tle"}
    offenders = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        text = re.sub(r'"\s*\n\s*"', "", path.read_text(encoding="utf-8").lower())
        text = re.sub(r"\s+", " ", text)
        for phrase in FORBIDDEN_CLAIMS:
            if phrase in text:
                offenders.append(f"{path.relative_to(root)}: {phrase}")
        for phrase in QUALIFIED_CLAIMS:
            for match in re.finditer(re.escape(phrase), text):
                before = text[max(0, match.start() - 14) : match.start()]
                if "not " not in before:
                    offenders.append(
                        f"{path.relative_to(root)}: {phrase} (context {before!r})"
                    )
    assert not offenders, offenders


def test_the_claim_scan_would_catch_a_claim_if_one_were_made() -> None:
    """The negative control for the scan above.

    A scan that matched nothing would pass vacuously. This asserts that each
    forbidden phrase is found in a synthetic text that makes the claim, and
    that the same phrase preceded by a denial is not.
    """
    for phrase in FORBIDDEN_CLAIMS:
        assert phrase in f"this product is {phrase} today"
    for phrase in QUALIFIED_CLAIMS:
        claim = f"this product is {phrase}"
        denial = f"this product is not {phrase}"
        found_claim = [
            match
            for match in re.finditer(re.escape(phrase), claim)
            if "not " not in claim[max(0, match.start() - 14) : match.start()]
        ]
        found_denial = [
            match
            for match in re.finditer(re.escape(phrase), denial)
            if "not " not in denial[max(0, match.start() - 14) : match.start()]
        ]
        assert found_claim, phrase
        assert not found_denial, phrase


def test_no_source_file_contains_an_absolute_home_directory_path() -> None:
    """Release-gate check: an absolute container path in a tracked file fails the batch.

    The needles are built from character codes so that this test does not
    match itself.
    """
    needles = [
        "".join(map(chr, (47, 104, 111, 109, 101, 47))),
        "".join(map(chr, (47, 85, 115, 101, 114, 115, 47))),
        "".join(map(chr, (47, 116, 109, 112, 47))),
    ]
    root = TESTS_DIR.parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "build", "dist", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text()
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(root)}: {needle!r}")
    assert not offenders, offenders


def test_regression_near_equatorial_inclination_recovery_floor() -> None:
    """Hypothesis's counterexample to a flat inclination tolerance, pinned.

    Found at ``a = 6600`` km, ``e = 0``, ``i = 5.960464477539063e-08`` radian
    with every angle zero. The element round trip recovers
    ``6.143906154658886e-08`` radian, an absolute departure of
    ``1.83e-09``, against a flat ``abs=1e-9`` tolerance the hand-written cases
    had no reason to question -- none of them used an inclination below a
    degree.

    It is not a defect. Inclination comes back as ``arccos(h_z / |h|)`` and
    ``d(arccos x)/dx = -1/sin(i)``, so a few machine epsilons of relative error
    in the ratio become ``eps / sin(i)`` radian of absolute error in the angle.
    At this inclination that floor is about 3e-08 radian, half the inclination
    itself. The test in ``tests/test_kepler.py`` now derives its tolerance from
    that conditioning; this test pins the literal inputs so the behaviour
    cannot regress on a fresh clone where ``.hypothesis/`` is absent.
    """
    from orbitdrift.kepler import ClassicalElements, elements_to_rv, rv_to_elements

    inclination = 5.960464477539063e-08
    elements = ClassicalElements(6600.0, 0.0, inclination, 0.0, 0.0, 0.0)
    r, v = elements_to_rv(elements)
    recovered = rv_to_elements(r, v)

    departure = abs(recovered.inclination_rad - inclination)
    floor = 8.0 * float(np.finfo(float).eps) / float(np.sin(inclination))
    assert departure > 1e-9, "the counterexample no longer exceeds a flat 1e-9 tolerance"
    assert departure < floor, f"departure {departure:.6e} exceeds the derived floor {floor:.6e}"

    # The shape parameters are unaffected: only the angle is ill-conditioned.
    assert recovered.semi_major_axis_km == pytest.approx(6600.0, rel=1e-12, abs=0.0)
    assert recovered.eccentricity == pytest.approx(0.0, rel=0.0, abs=1e-12)

    # And at a well-conditioned inclination the same round trip closes tightly,
    # which is what makes the statement above about conditioning rather than
    # about the implementation.
    well_conditioned = rv_to_elements(
        *elements_to_rv(ClassicalElements(6600.0, 0.0, 0.9, 0.0, 0.0, 0.0))
    )
    assert well_conditioned.inclination_rad == pytest.approx(0.9, rel=0.0, abs=1e-12)


def test_no_required_deliverable_is_matched_by_a_gitignore_pattern() -> None:
    """The committed fixtures, screenshots and validation outputs must ship.

    A `.gitignore` pattern with no slash matches at **any depth**: a bare
    `junit.xml` in a product `.gitignore` once silently excluded a required
    committed fixture elsewhere in this mission, and a cold clone then failed
    every test while the build machine passed.

    This check does not shell out to `git check-ignore`, and that is deliberate
    for a reason the mission has recorded: `git check-ignore` exits **0 when a
    negation pattern matches**, so a check that read only its exit status would
    report a deliberately re-included file as ignored. The matching is done
    here instead, with a leading `!` treated as *not ignored*, which is the
    semantics rather than the exit status.
    """
    import fnmatch

    root = TESTS_DIR.parent
    patterns: list[tuple[str, bool]] = []
    for line in (root / ".gitignore").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        negated = stripped.startswith("!")
        patterns.append((stripped.lstrip("!"), negated))
    assert patterns, ".gitignore has no patterns; this check would be vacuous"

    required = [
        "fixtures/population.tle",
        "fixtures/divergence_dataset.npz",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CITATION.cff",
        "MODEL_CARD.md",
        "DATASET_CARD.md",
        "docs/REQUIREMENTS.md",
        "validation/VALIDATION.md",
        ".github/workflows/tests.yml",
    ]
    required += [
        str(path.relative_to(root)) for path in sorted(root.glob("screenshots/*.png"))
    ]
    required += [
        str(path.relative_to(root)) for path in sorted(root.glob("validation/*_output.txt"))
    ]
    for path in required:
        assert (root / path).exists(), f"{path} is missing from the working tree"

    def matches(pattern: str, candidate: str) -> bool:
        anchored = pattern.startswith("/")
        body = pattern.lstrip("/").rstrip("/")
        if anchored:
            return fnmatch.fnmatch(candidate, body) or candidate.startswith(body + "/")
        # No leading slash: the pattern matches at ANY depth.
        parts = candidate.split("/")
        return any(fnmatch.fnmatch(part, body) for part in parts) or fnmatch.fnmatch(
            candidate, body
        )

    offenders = []
    for candidate in required:
        ignored = False
        for pattern, negated in patterns:
            if matches(pattern, candidate):
                ignored = not negated
        if ignored:
            offenders.append(candidate)
    assert not offenders, f"required deliverables matched by .gitignore: {offenders}"


def test_the_gitignore_check_would_catch_an_unanchored_pattern() -> None:
    """The negative control for the check above.

    A bare `junit.xml` pattern must be shown to match a nested file, and the
    anchored `/junit.xml` this repository actually uses must be shown not to.
    Without this, the check above would pass whatever the matching function did.
    """
    import fnmatch

    def matches(pattern: str, candidate: str) -> bool:
        anchored = pattern.startswith("/")
        body = pattern.lstrip("/").rstrip("/")
        if anchored:
            return fnmatch.fnmatch(candidate, body) or candidate.startswith(body + "/")
        parts = candidate.split("/")
        return any(fnmatch.fnmatch(part, body) for part in parts) or fnmatch.fnmatch(
            candidate, body
        )

    assert matches("junit.xml", "fixtures/sample/junit.xml")
    assert not matches("/junit.xml", "fixtures/sample/junit.xml")
    assert matches("/junit.xml", "junit.xml")
    assert "/junit.xml" in (TESTS_DIR.parent / ".gitignore").read_text()
    assert "\njunit.xml" not in (TESTS_DIR.parent / ".gitignore").read_text()
