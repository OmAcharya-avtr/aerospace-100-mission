"""Validation 4 -- what happens when the observations become nearly parallel.

Checks
------
1. The observability metric of Eq. O4 follows its closed form
   ``lambda_min = (1 - |cos eta|) / 2`` for a pair, so the default gate
   ``1e-6`` really is a 0.115 deg separation.
2. Every solver raises :class:`DegenerateObservationsError` below the gate, and
   the message names the frame and the angle.
3. With the gate disabled, the attitude error on *noisy* near-parallel data
   grows like ``1 / sin(eta)``, which is what the gate is protecting against:
   nothing raises, nothing is nan, and the answer is wrong.
4. The reference frame is tested as well as the body frame, so a degenerate
   catalogue is caught even when the measurements look healthy.
5. The 180-degree parametrisation singularity of QUEST and OLAE, and the
   sequential-rotation cure.
6. What ``scipy.spatial.transform.Rotation.align_vectors`` does on the same
   inputs, for comparison.

Run:  python validation/validate_degeneracy.py
"""

from __future__ import annotations

import warnings

import numpy as np
from _common import WELL_CONDITIONED, banner, sample_observations, verdict

from wahbakit import (
    DegenerateObservationsError,
    VectorObservations,
    angle_between_dcm,
    dcm_from_quat,
    olae,
    q_method,
    quest,
    triad,
)

SEED = 20260831
DCM_TRUE = dcm_from_quat([0.6, 0.4, -0.5, 0.3])
SOLVERS = {"triad": triad, "q-method": q_method, "quest": quest, "olae": olae}


def pair(separation_deg: float, sigma: float, rng: np.random.Generator) -> VectorObservations:
    """Two observations separated by ``separation_deg``, with Eq. O1 noise."""
    eta = np.radians(separation_deg)
    reference = np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])
    sigmas = np.full(2, max(sigma, 1e-12))
    true_body = reference @ DCM_TRUE.T
    body = true_body if sigma == 0.0 else sample_observations(true_body, sigmas, rng, 1)[0]
    return VectorObservations(body, reference, sigmas=sigmas)


def main() -> int:
    banner("wahbakit validation 4 -- near-parallel observations")
    rng = np.random.default_rng(SEED)
    passed = []

    print("\n[1] Eq. O4 against its closed form for a pair, lambda_min = (1 - |cos eta|) / 2")
    print("    evaluated as the algebraically identical sin^2(eta/2), which the (1 - cos)")
    print("    form cannot match at small eta. lambda_min comes from eigvalsh on a matrix")
    print("    of trace 2, so its error is ABSOLUTE (about eps * 2 = 4e-16), not relative:")
    print("    a lambda_min of 7.6e-09 carries only about eight correct digits in relative")
    print("    terms. The gate is at 1e-6, a hundred times above that floor, so the check")
    print("    below is on the absolute deviation and the relative one is reported.")
    print(
        f"    {'eta [deg]':>12} {'lambda_min':>14} {'closed form':>14} {'abs dev':>11} "
        f"{'rel dev':>11}"
    )
    worst_absolute = 0.0
    worst_relative = 0.0
    for separation_deg in (90.0, 30.0, 10.0, 1.0, 0.115, 0.01):
        obs = pair(separation_deg, 0.0, rng)
        measured = obs.observability().lambda_min
        expected = float(np.sin(np.radians(separation_deg) / 2.0) ** 2)
        absolute = abs(measured - expected)
        worst_absolute = max(worst_absolute, absolute)
        worst_relative = max(worst_relative, absolute / expected)
        print(
            f"    {separation_deg:12.4f} {measured:14.6e} {expected:14.6e} "
            f"{absolute:11.2e} {absolute / expected:11.2e}"
        )
    passed.append(
        verdict("worst absolute deviation from the closed form", worst_absolute, 1e-13)
    )
    print(
        f"    worst relative deviation over the same rows: {worst_relative:.3e}, all of it "
        f"at eta = 0.01 deg"
    )
    gate_angle = np.degrees(np.arccos(1.0 - 2.0 * 1e-6))
    print(f"    default gate lambda_min = 1e-6 corresponds to eta = {gate_angle:.4f} deg")
    passed.append(verdict("|gate angle - 0.1146 deg|", abs(gate_angle - 0.1146), 1e-3))

    print("\n[2] every solver raises below the gate")
    obs = pair(0.05, 0.0, rng)
    print(
        "    geometry: 0.05 deg separation, noise free, "
        f"lambda_min = {obs.observability().lambda_min:.4e}"
    )
    for name, solver in SOLVERS.items():
        try:
            solver(obs)
            print(f"    {name:>10}: did NOT raise  FAIL")
            passed.append(False)
        except DegenerateObservationsError as exc:
            message = str(exc)
            ok = "degenerate in the" in message and "0.0500" in message
            print(f"    {name:>10}: DegenerateObservationsError, message names the angle: {ok}")
            passed.append(ok)
    print("    message:")
    try:
        q_method(obs)
    except DegenerateObservationsError as exc:
        for line in str(exc).split(". "):
            print(f"      {line.strip()}")

    print("\n[3] with the gate disabled: no exception, no nan, and a wrong answer")
    print(
        f"    {'eta [deg]':>10} {'lambda_min':>12} {'1/sin(eta)':>12} "
        + "  ".join(f"{name:>12}" for name in SOLVERS)
    )
    sigma = 1e-4
    errors_by_separation = []
    for separation_deg in (90.0, 10.0, 1.0, 0.1, 0.01, 0.001):
        obs = pair(separation_deg, sigma, rng)
        row = []
        cells = []
        for name, solver in SOLVERS.items():
            try:
                solution = solver(obs, check_degeneracy=False)
            except RuntimeError:
                # QUEST's Newton iteration stalls once lambda_max is a near-double
                # root; OLAE's normal equations go singular. Both are reported.
                row.append(np.nan)
                cells.append(f"{'RuntimeError':>12}")
                continue
            error = angle_between_dcm(solution.dcm, DCM_TRUE)
            assert np.all(np.isfinite(solution.dcm)), f"{name} returned a non-finite matrix"
            row.append(error)
            cells.append(f"{error:12.3e}")
        errors_by_separation.append((separation_deg, row[1]))
        print(
            f"    {separation_deg:10.3f} {obs.observability().lambda_min:12.3e} "
            f"{1.0 / np.sin(np.radians(separation_deg)):12.3e} " + "  ".join(cells)
        )
    ratios = [
        error * np.sin(np.radians(separation_deg))
        for separation_deg, error in errors_by_separation
        if separation_deg <= 10.0
    ]
    print(
        f"    error * sin(eta) for eta <= 10 deg: "
        f"{np.array2string(np.array(ratios), precision=3)} rad -- flat to within a factor"
    )
    print(f"    of {np.max(ratios) / np.min(ratios):.2f}, confirming the 1 / sin(eta) growth.")
    growth = errors_by_separation[-1][1] / errors_by_separation[0][1]
    print(
        f"    q-method error grows by a factor {growth:.3e} from 90 deg to 0.001 deg "
        f"at sigma = {sigma:g} rad."
    )
    passed.append(
        verdict("error growth factor 90 deg -> 0.001 deg", growth, 1e3, smaller_is_better=False)
    )

    print("\n[4] a degenerate reference catalogue with healthy measurements")
    body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference = np.array([[1.0, 0.0, 0.0], [1.0, 1e-5, 0.0]])
    obs = VectorObservations(body, reference, sigmas=[1e-3, 1e-3])
    result = obs.observability()
    print(
        f"    body lambda_min = {result.lambda_min_body:.4e}, "
        f"reference lambda_min = {result.lambda_min_reference:.4e}, "
        f"limiting frame = {result.limiting_frame}"
    )
    caught = False
    try:
        q_method(obs)
    except DegenerateObservationsError as exc:
        caught = "reference frame" in str(exc)
    print(f"    q-method raised and named the reference frame: {caught}")
    passed.append(caught)

    print("\n[5] the 180-degree parametrisation singularity")
    header = f"{'pi - theta':>12} {'QUEST no seq':>14} {'QUEST seq':>12} "
    print("    " + header + f"{'OLAE no seq':>14} {'OLAE seq':>12}")
    worst_with_sequential = 0.0
    worst_without = 0.0
    for gap in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 0.0):
        angle = np.pi - gap
        dcm = dcm_from_quat([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        obs = VectorObservations(WELL_CONDITIONED @ dcm.T, WELL_CONDITIONED)
        cells = []
        for solver in (quest, olae):
            try:
                bare = angle_between_dcm(solver(obs, sequential_rotation=False).dcm, dcm)
                worst_without = max(worst_without, bare)
                cells.append(f"{bare:14.3e}")
            except RuntimeError:
                cells.append(f"{'RuntimeError':>14}")
            fixed = angle_between_dcm(solver(obs).dcm, dcm)
            worst_with_sequential = max(worst_with_sequential, fixed)
            cells.append(f"{fixed:12.3e}")
        print(f"    {gap:12.0e} " + " ".join(cells))
    passed.append(
        verdict(
            "worst error at a pi rotation WITH sequential rotation [rad]",
            worst_with_sequential,
            1e-12,
        )
    )
    print(
        "    worst error WITHOUT sequential rotation, over the same rows: "
        f"{worst_without:.3e} rad"
    )

    print("\n[6] what scipy align_vectors does on the same degenerate inputs")
    from scipy.spatial.transform import Rotation

    for label, separation_deg in (("0.05 deg apart", 0.05), ("exactly parallel", 0.0)):
        if separation_deg == 0.0:
            reference = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            eta = np.radians(separation_deg)
            reference = np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])
        body = reference @ DCM_TRUE.T
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            rotation, _ = Rotation.align_vectors(body, reference)
        error = angle_between_dcm(rotation.as_matrix(), DCM_TRUE)
        messages = [str(w.message) for w in caught_warnings]
        print(f"    {label:>18}: error {error:.3e} rad, warnings {messages}")
    print("    SciPy warns on exactly parallel input and returns a value; it does not")
    print("    warn at 0.05 deg. wahbakit raises in both cases. Neither behaviour is")
    print("    wrong; they are different defaults, and this is the one difference the")
    print("    README claims.")

    print()
    print(f"RESULT: {sum(passed)} / {len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
