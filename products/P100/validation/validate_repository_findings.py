"""Pin the repository audit's finding set as literals, and re-verify it fast.

``validate_repository_audit.py`` is the MEASUREMENT and takes tens of minutes:
it runs every candidate script twice, each run in its own pristine copy. This
script is the PINNED CHECK and takes under a second. It re-reads the
machine-readable evidence that run wrote, asserts that the finding set is
exactly the literal set recorded here, and recomputes the headline from the
evidence rather than trusting the number the measurement printed.

Splitting them is deliberate. The build guide requires every quoted number to
be re-executed as the last step before hand-back, and a check that takes forty
minutes will not be re-run as often as one that takes a second. It also means
the pinned set fails for real if the corpus changes: a new finding, a lost
finding or a changed value all break this script, which is the point of pinning
a finding set as literals rather than as a count.

The literals below were filled in from the measured evidence after the audit
ran. They are not a prediction and they are not a tolerance.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from seedaudit.power import escaped_rate_upper_bound  # noqa: E402

#: The evidence file the measurement writes.
EVIDENCE = "validation/repository_findings.json"

#: PINNED. The seed-attributable finding set, pinned two ways. The per-script
#: counts localise a change to a file; the digest covers every one of the
#: 5683 items including its before and after values, so a
#: changed VALUE inside an unchanged script fails this script too. Pinning
#: 5683 tuples as source literals would be a megabyte of
#: unreadable file; a digest over a canonical serialisation is the same claim.
PINNED_SEED_ATTRIBUTABLE_BY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("products/P004/validation/validate_availability_model.py", 62),
    ("products/P005/validation/val_detector.py", 9),
    ("products/P005/validation/val_pointing.py", 25),
    ("products/P005/validation/val_psd.py", 10),
    ("products/P006/validation/mc_crosscheck.py", 4),
    ("products/P024/validation/momentum_monotonicity.py", 14),
    ("products/P025/validation/cusum_delay.py", 45),
    ("products/P035/validation/validate_changepoint.py", 34),
    ("products/P035/validation/validate_cusum_delay.py", 21),
    ("products/P035/validation/validate_far_design.py", 131),
    ("products/P035/validation/validate_matched_far.py", 581),
    ("products/P052/validation/validate_benchmark.py", 1350),
    ("products/P052/validation/validate_difficulty.py", 97),
    ("products/P052/validation/validate_semantics.py", 20),
    ("products/P052/validation/validate_simulator.py", 28),
    ("products/P052/validation/validate_surrogate.py", 110),
    ("products/P052/validation/worked_example.py", 6),
    ("products/P055/validation/validate_graph_identities.py", 3),
    ("products/P061/validation/validate_gap.py", 14),
    ("products/P061/validation/validate_predictor.py", 5),
    ("products/P061/validation/validate_robust.py", 57),
    ("products/P061/validation/validate_warmstart.py", 7),
    ("products/P061/validation/worked_example.py", 7),
    ("products/P062/validation/validate_coverage.py", 32),
    ("products/P062/validation/validate_recovery.py", 170),
    ("products/P062/validation/validate_residual.py", 7),
    ("products/P062/validation/worked_example.py", 8),
    ("products/P065/validation/validate_properties.py", 20),
    ("products/P066/validation/validate_identities.py", 9),
    ("products/P068/validation/validate_baselines.py", 54),
    ("products/P068/validation/validate_calibration.py", 153),
    ("products/P068/validation/validate_crossing.py", 5),
    ("products/P068/validation/validate_headtohead.py", 167),
    ("products/P068/validation/validate_physics.py", 36),
    ("products/P068/validation/worked_example.py", 8),
    ("products/P070/validation/validate_false_trip.py", 4),
    ("products/P070/validation/worked_example.py", 1),
    ("products/P071/validation/validate_geometry.py", 5),
    ("products/P071/validation/validate_nonlinear.py", 110),
    ("products/P071/validation/validate_screening.py", 261),
    ("products/P071/validation/worked_example.py", 6),
    ("products/P072/validation/validate_compress.py", 58),
    ("products/P072/validation/validate_stability.py", 103),
    ("products/P072/validation/validate_valuemodel.py", 109),
    ("products/P073/validation/validate_geometry.py", 32),
    ("products/P074/validation/validate_frames.py", 6),
    ("products/P074/validation/validate_growth_baseline.py", 2),
    ("products/P074/validation/validate_kepler_reference.py", 5),
    ("products/P075/validation/validate_bridge.py", 6),
    ("products/P075/validation/validate_commoncause.py", 12),
    ("products/P075/validation/validate_montecarlo.py", 91),
    ("products/P075/validation/validate_negative_controls.py", 9),
    ("products/P075/validation/worked_example.py", 8),
    ("products/P076/validation/validate_aberration.py", 13),
    ("products/P076/validation/validate_invariants.py", 5),
    ("products/P076/validation/validate_kepler.py", 8),
    ("products/P076/validation/validate_lighttime.py", 2),
    ("products/P077/validation/validate_analytic.py", 3),
    ("products/P077/validation/validate_predictors.py", 199),
    ("products/P077/validation/worked_example.py", 46),
    ("products/P079/validation/regenerate_fixture.py", 2),
    ("products/P079/validation/validate_allan.py", 112),
    ("products/P079/validation/validate_asymmetry_benchmark.py", 148),
    ("products/P079/validation/validate_deployment.py", 13),
    ("products/P079/validation/validate_dryrun.py", 4),
    ("products/P079/validation/validate_parity.py", 7),
    ("products/P079/validation/validate_reproducibility.py", 2),
    ("products/P079/validation/validate_skew_estimators.py", 53),
    ("products/P079/validation/validate_symmetric_bias.py", 7),
    ("products/P079/validation/validate_uncertainty_coverage.py", 90),
    ("products/P079/validation/worked_example.py", 61),
    ("products/P080/validation/validate_learned.py", 18),
    ("products/P080/validation/validate_negative_control.py", 4),
    ("products/P080/validation/validate_rate_distortion.py", 10),
    ("products/P080/validation/worked_example.py", 60),
    ("products/P081/validation/validate_negative_controls.py", 6),
    ("products/P081/validation/worked_example.py", 81),
    ("products/P082/validation/validate_climatology.py", 68),
    ("products/P082/validation/validate_copula.py", 28),
    ("products/P082/validation/validate_empirical.py", 97),
    ("products/P082/validation/validate_geometry.py", 3),
    ("products/P082/validation/validate_negative_controls.py", 14),
    ("products/P082/validation/validate_overstatement.py", 55),
    ("products/P082/validation/worked_example.py", 12),
    ("products/P083/validation/validate_certificate.py", 3),
    ("products/P083/validation/validate_clock.py", 26),
    ("products/P083/validation/validate_negative_controls.py", 2),
    ("products/P083/validation/validate_plant.py", 4),
    ("products/P083/validation/validate_resimulation.py", 3),
    ("products/P083/validation/worked_example.py", 67),
    ("products/P084/validation/validate_bounds.py", 44),
    ("products/P084/validation/validate_independent_bound.py", 8),
    ("products/P084/validation/validate_inversion.py", 24),
    ("products/P084/validation/validate_negative_controls.py", 8),
    ("products/P084/validation/worked_example.py", 45),
    ("products/P085/validation/validate_linearisation.py", 8),
    ("products/P085/validation/validate_robust.py", 11),
    ("products/P085/validation/worked_example.py", 18),
    ("products/P086/validation/validate_refusal_controls.py", 1),
    ("products/P086/validation/worked_example.py", 21),
    ("products/P089/validation/validate_controls.py", 6),
    ("products/P089/validation/worked_example.py", 14),
    ("products/P090/validation/worked_example.py", 2),
)

#: sha256 over "script\x1fline\x1findex\x1fbefore\x1fafter" per item, sorted.
PINNED_SEED_ATTRIBUTABLE_DIGEST = "bf1709ade155744d6ffc45accb9ab05b9cc336d478be58cf424d5c16d07dd8f4"

#: PINNED. Numbers that move with the seed held at its DECLARED value.
PINNED_DECLARED_SEED_BY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("products/P004/validation/validate_availability_model.py", 1),
    ("products/P005/validation/val_detector.py", 2),
    ("products/P025/validation/cusum_delay.py", 1),
    ("products/P035/validation/validate_changepoint.py", 1),
    ("products/P035/validation/validate_far_design.py", 1),
    ("products/P035/validation/validate_matched_far.py", 2),
    ("products/P052/validation/validate_benchmark.py", 3),
    ("products/P052/validation/validate_difficulty.py", 2),
    ("products/P052/validation/validate_surrogate.py", 8),
    ("products/P061/validation/validate_gap.py", 62),
    ("products/P061/validation/validate_predictor.py", 11),
    ("products/P061/validation/validate_robust.py", 2),
    ("products/P061/validation/validate_warmstart.py", 65),
    ("products/P061/validation/worked_example.py", 1),
    ("products/P062/validation/validate_coverage.py", 1),
    ("products/P062/validation/validate_profile.py", 1),
    ("products/P062/validation/validate_recovery.py", 1),
    ("products/P062/validation/validate_residual.py", 6),
    ("products/P062/validation/worked_example.py", 1),
    ("products/P068/validation/validate_headtohead.py", 8),
    ("products/P070/validation/validate_false_trip.py", 1),
    ("products/P071/validation/validate_screening.py", 8),
    ("products/P072/validation/validate_compress.py", 2),
    ("products/P072/validation/validate_stability.py", 1),
    ("products/P072/validation/validate_valuemodel.py", 2),
    ("products/P075/validation/validate_bridge.py", 1),
    ("products/P075/validation/validate_commoncause.py", 1),
    ("products/P075/validation/validate_montecarlo.py", 1),
    ("products/P075/validation/validate_negative_controls.py", 1),
    ("products/P075/validation/worked_example.py", 1),
    ("products/P076/validation/validate_kepler.py", 2),
    ("products/P077/validation/validate_predictors.py", 2),
    ("products/P079/validation/validate_asymmetry_benchmark.py", 5),
    ("products/P079/validation/validate_benchmark_table.py", 38),
    ("products/P080/validation/validate_learned.py", 2),
    ("products/P080/validation/validate_rate_distortion.py", 49),
    ("products/P080/validation/worked_example.py", 4),
    ("products/P082/validation/validate_climatology.py", 1),
    ("products/P082/validation/validate_copula.py", 3),
    ("products/P082/validation/validate_empirical.py", 3),
    ("products/P082/validation/validate_overstatement.py", 1),
    ("products/P082/validation/worked_example.py", 1),
    ("products/P083/validation/validate_certificate.py", 21),
    ("products/P083/validation/validate_resimulation.py", 11),
    ("products/P084/validation/validate_bounds.py", 1),
    ("products/P084/validation/validate_inversion.py", 2),
)

PINNED_DECLARED_SEED_DIGEST = "2b7dab6e955040206439eccd0fb8b3057eba227c10307c0eb89010196cf909a6"

#: PINNED. The instrument's own footprint: numbers that moved only because the
#: script prints the seed the audit rewrote. Excluded from the headline.
PINNED_INSTRUMENT_FOOTPRINT_DIGEST = (
    "6df78cc0102c38a614785061e3c1f51b974344f5f4597f5e3a1ee28be2c8ad35"
)
PINNED_INSTRUMENT_FOOTPRINT_COUNT = 45
PINNED_INSTRUMENT_FOOTPRINT_SCRIPTS = 33

#: PINNED. The headline counts.
PINNED_HEADLINE_SEED_ATTRIBUTABLE = 5683
PINNED_HEADLINE_DECLARED_SEED = 345
PINNED_SCRIPTS_WITH_SEED_MOVEMENT = 103
PINNED_SCRIPTS_WITH_DECLARED_SEED_MOVEMENT = 46
PINNED_DECLARED_SEED_DURATION_MARKED = 88

#: PINNED. Coverage, so that a silently shrinking corpus is a failure.
PINNED_CANDIDATES_FOUND = 229
PINNED_CANDIDATES_ATTEMPTED = 229
PINNED_STATUS_COUNTS: dict[str, int] = {
    "compared": 110,
    "no_output": 73,
    "run_failed": 37,
    "timeout": 9,
}
PINNED_CLAIM_TOKENS_TOTAL = 43321
PINNED_CLAIM_TOKENS_COMPARED = 21414

#: PINNED. The static screen over the same corpus.
PINNED_STATIC_FILES_PARSED = 3564
PINNED_STATIC_FILES_WITH_FINDINGS = 53
PINNED_STATIC_FINDINGS_TOTAL = 275
PINNED_STATIC_BY_SUBKIND: dict[str, int] = {
    "write_to_read_path": 246,
    "write_under_glob": 16,
    "write_under_read": 13,
}
PINNED_STATIC_PARSE_FAILURES = 0

#: PINNED. Candidates whose claim could not be evaluated, by status.
PINNED_UNEVALUABLE: dict[str, tuple[str, ...]] = {
    "no_output": (
        "products/P012/validation/v1_riccati_steady_state.py",
        "products/P012/validation/v2_nees_nis_consistency.py",
        "products/P012/validation/v3_ukf_vs_ekf.py",
        "products/P012/validation/v4_mekf_quaternion.py",
        "products/P012/validation/v5_sensor_and_truth_models.py",
        "products/P012/validation/v6_adaptive_q_benchmark.py",
        "products/P012/validation/v7_performance.py",
        "products/P013/validation/round_trip_recovery.py",
        "products/P013/validation/saturation_regime.py",
        "products/P015/validation/horizon_sensitivity.py",
        "products/P015/validation/policy_comparison_ci.py",
        "products/P029/validation/learned_vs_fixed_ci.py",
        "products/P029/validation/magnetic_controllability.py",
        "products/P031/validation/deployment_checks.py",
        "products/P031/validation/dryrun_nowrite.py",
        "products/P031/validation/latency_percentiles.py",
        "products/P031/validation/overrun_handcount.py",
        "products/P031/validation/parity_simdev.py",
        "products/P031/validation/performance_benchmark.py",
        "products/P031/validation/predictor_benchmark.py",
        "products/P031/validation/recovery_abort.py",
        "products/P031/validation/timing_uncertainty.py",
        "products/P032/validation/validate_benchmark.py",
        "products/P032/validation/validate_calibration.py",
        "products/P032/validation/validate_ilp.py",
        "products/P032/validation/validate_routing.py",
        "products/P032/validation/validate_uncertainty.py",
        "products/P037/validation/crosscheck_berbench.py",
        "products/P037/validation/validate_conv_fer.py",
        "products/P037/validation/validate_false_sync.py",
        "products/P037/validation/validate_state_machine.py",
        "products/P037/validation/validate_uncoded_fer.py",
        "products/P043/validation/validate_afterpulse.py",
        "products/P043/validation/validate_capacity.py",
        "products/P043/validation/validate_correction.py",
        "products/P043/validation/validate_hal_contract.py",
        "products/P043/validation/validate_ppm.py",
        "products/P043/validation/validate_simulator.py",
        "products/P043/validation/validate_webb_limits.py",
        "products/P043/validation/worked_example.py",
        "products/P045/validation/export_policy_table.py",
        "products/P045/validation/validate_burst_channel.py",
        "products/P045/validation/validate_burst_goodput.py",
        "products/P045/validation/validate_closed_form.py",
        "products/P045/validation/validate_crc.py",
        "products/P045/validation/validate_harq.py",
        "products/P045/validation/validate_policy.py",
        "products/P045/validation/validate_window.py",
        "products/P045/validation/worked_example.py",
        "products/P049/validation/regenerate_outage_model.py",
        "products/P049/validation/worked_example.py",
        "products/P051/validation/validate_accounting.py",
        "products/P051/validation/validate_bound_violation.py",
        "products/P051/validation/validate_guard_exactness.py",
        "products/P051/validation/validate_invariant_set.py",
        "products/P051/validation/validate_predictor.py",
        "products/P051/validation/validate_support_function.py",
        "products/P051/validation/worked_example.py",
        "products/P053/validation/validate_ambiguity.py",
        "products/P053/validation/validate_arl_curve.py",
        "products/P053/validation/validate_classifier.py",
        "products/P053/validation/validate_thresholds.py",
        "products/P053/validation/validate_twin.py",
        "products/P053/validation/worked_example.py",
        "products/P060/validation/validate_coverage.py",
        "products/P067/validation/dependency_range_check.py",
        "products/P067/validation/validate_policies.py",
        "products/P067/validation/validate_spiral_coverage.py",
        "products/P067/validation/worked_example.py",
        "products/P069/validation/validate_dependency_ends.py",
        "products/P069/validation/validate_occupancy.py",
        "products/P069/validation/validate_uncertainty.py",
        "products/P069/validation/worked_example.py",
    ),
    "run_failed": (
        "products/P011/validation/validate_fitting_error.py",
        "products/P011/validation/validate_predictor.py",
        "products/P011/validation/validate_rejection_tf.py",
        "products/P011/validation/validate_strehl_marechal.py",
        "products/P017/validation/covariance_health.py",
        "products/P017/validation/smoother_rms.py",
        "products/P017/validation/ukf_vs_kf_linear.py",
        "products/P029/validation/wheel_allocation.py",
        "products/P033/validation/validate_backend_cost_model.py",
        "products/P033/validation/validate_budget.py",
        "products/P033/validation/validate_memory.py",
        "products/P033/validation/validate_onnxruntime.py",
        "products/P033/validation/validate_performance.py",
        "products/P033/validation/validate_predictor.py",
        "products/P033/validation/validate_uncertainty.py",
        "products/P037/validation/validate_rs_coding_gain.py",
        "products/P039/validation/crosscheck_edgeinfer.py",
        "products/P039/validation/validate_analytic_exactness.py",
        "products/P039/validation/validate_interval_coverage.py",
        "products/P039/validation/validate_model_comparison.py",
        "products/P039/validation/validate_tail_convergence.py",
        "products/P040/validation/worked_example.py",
        "products/P041/validation/validate_channel_statistics.py",
        "products/P041/validation/validate_crossing_convergence.py",
        "products/P041/validation/validate_depth_sweep.py",
        "products/P041/validation/validate_rs_known_answers.py",
        "products/P041/validation/worked_example.py",
        "products/P048/validation/worked_example.py",
        "products/P057/validation/validate_baseline_vs_learned.py",
        "products/P057/validation/validate_breaking_point.py",
        "products/P057/validation/validate_conditional_coverage.py",
        "products/P057/validation/validate_coverage_audit.py",
        "products/P057/validation/validate_known_answers.py",
        "products/P057/validation/worked_example.py",
        "products/P059/validation/validate_support_algebra.py",
        "products/P063/validation/validate_calibration.py",
        "products/P063/validation/validate_conservatism.py",
    ),
    "timeout": (
        "products/P070/validation/validate_latency_vs_simulation.py",
        "products/P071/validation/validate_montecarlo.py",
        "products/P074/validation/validate_reproducibility.py",
        "products/P078/validation/validate_determinism.py",
        "products/P078/validation/validate_irreversible_gate.py",
        "products/P078/validation/validate_lint_missrate.py",
        "products/P078/validation/validate_negative_control.py",
        "products/P089/validation/validate_recovery_grid.py",
        "products/P089/validation/validate_recursive_agreement.py",
    ),
}


def digest(items: list[dict[str, object]]) -> str:
    """sha256 over a canonical serialisation of a finding list.

    The key order is fixed and the items are sorted, so the digest depends on
    the finding SET and not on the order the audit happened to emit it in.
    """
    canonical = "\n".join(
        "\x1f".join(str(item[key]) for key in ("script", "line", "index", "before", "after"))
        for item in sorted(
            items,
            key=lambda item: tuple(
                str(item[key]) for key in ("script", "line", "index", "before", "after")
            ),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    """Re-verify the pinned finding set against the committed evidence."""
    out = _bootstrap.Tee("validate_repository_findings_output.txt")
    out("Pinned repository findings -- P100 SeedAudit")
    out("")
    out("this script re-reads the evidence written by validate_repository_audit.py")
    out("and fails if the finding set has changed. It runs no scripts and takes")
    out("under a second, so it can be re-run as often as the build guide requires.")
    out("")
    path = ROOT / EVIDENCE
    if not path.is_file():
        out.unavailable(
            "pinned finding set",
            f"{EVIDENCE} is absent; run validate_repository_audit.py first",
        )
        return out.finish()
    evidence = json.loads(path.read_text(encoding="utf-8"))

    out.rule("1. THE HEADLINE, recomputed from the evidence")
    seed_items = evidence["seed_attributable"]
    declared_items = evidence["declared_seed_moved"]
    out(f"seed-attributable moved numbers, counted from the itemised list: {len(seed_items)}")
    out(f"headline recorded by the measurement: {evidence['headline_seed_attributable']}")
    out(f"declared-seed moved numbers, counted: {len(declared_items)}")
    out(f"declared-seed headline recorded: {evidence['headline_declared_seed']}")
    out.check(
        "the itemised list and the recorded headline agree, seed-attributable",
        len(seed_items) == evidence["headline_seed_attributable"],
        f"{len(seed_items)} against {evidence['headline_seed_attributable']}",
    )
    out.check(
        "the itemised list and the recorded headline agree, declared seed",
        len(declared_items) == evidence["headline_declared_seed"],
        f"{len(declared_items)} against {evidence['headline_declared_seed']}",
    )
    out.check(
        "the seed-attributable headline is exactly the pinned literal",
        evidence["headline_seed_attributable"] == PINNED_HEADLINE_SEED_ATTRIBUTABLE,
        f"{evidence['headline_seed_attributable']} against "
        f"{PINNED_HEADLINE_SEED_ATTRIBUTABLE}",
    )
    out.check(
        "the declared-seed headline is exactly the pinned literal",
        evidence["headline_declared_seed"] == PINNED_HEADLINE_DECLARED_SEED,
        f"{evidence['headline_declared_seed']} against {PINNED_HEADLINE_DECLARED_SEED}",
    )

    out.rule("2. THE FINDING SET, pinned two ways")
    out("the per-script counts localise a change to a file; the digest covers")
    out("every item including its before and after values, so a changed VALUE")
    out("inside an unchanged script fails this check too. Pinning thousands of")
    out("tuples as source literals would be a megabyte of unreadable file; a")
    out("digest over a canonical serialisation is the same claim.")
    out("")
    for label, items, by_script, pinned_digest in (
        (
            "seed-attributable",
            seed_items,
            PINNED_SEED_ATTRIBUTABLE_BY_SCRIPT,
            PINNED_SEED_ATTRIBUTABLE_DIGEST,
        ),
        (
            "declared-seed",
            declared_items,
            PINNED_DECLARED_SEED_BY_SCRIPT,
            PINNED_DECLARED_SEED_DIGEST,
        ),
    ):
        measured_digest = digest(items)
        counts = collections.Counter(item["script"] for item in items)
        measured_counts = tuple(sorted(counts.items()))
        expected_counts = tuple(sorted(by_script))
        out(f"{label}: {len(items)} items across {len(counts)} scripts")
        out(f"  pinned digest  : {pinned_digest}")
        out(f"  measured digest: {measured_digest}")
        missing = [pair for pair in expected_counts if pair not in measured_counts]
        extra = [pair for pair in measured_counts if pair not in expected_counts]
        for pair in missing:
            out(f"  MISSING (pinned but not measured): {pair}")
        for pair in extra:
            out(f"  EXTRA (measured but not pinned): {pair}")
        out.check(
            f"the {label} per-script counts are exactly the pinned literals",
            measured_counts == expected_counts,
            f"{len(missing)} missing, {len(extra)} extra",
        )
        out.check(
            f"the {label} digest is exactly the pinned literal",
            measured_digest == pinned_digest,
            f"{measured_digest[:16]} against {pinned_digest[:16]}",
        )

    out.rule("2b. THE INSTRUMENT'S OWN FOOTPRINT, pinned and excluded")
    out("a script that prints its own seed reports a changed number because the")
    out("audit changed the seed. Counting it as a finding would be the audit")
    out("reporting its own perturbation back to itself.")
    out("")
    footprint = evidence["instrument_footprint"]
    out(f"instrument-footprint numbers: {len(footprint)}")
    out(f"scripts that print a seed the audit rewrote: "
        f"{len(evidence['scripts_with_instrument_footprint'])}")
    out(f"  pinned digest  : {PINNED_INSTRUMENT_FOOTPRINT_DIGEST}")
    out(f"  measured digest: {digest(footprint)}")
    out.check(
        "the instrument-footprint count is exactly the pinned literal",
        evidence["instrument_footprint_count"] == PINNED_INSTRUMENT_FOOTPRINT_COUNT,
        f"{evidence['instrument_footprint_count']} against "
        f"{PINNED_INSTRUMENT_FOOTPRINT_COUNT}",
    )
    out.check(
        "the instrument-footprint digest is exactly the pinned literal",
        digest(footprint) == PINNED_INSTRUMENT_FOOTPRINT_DIGEST,
    )
    out.check(
        "the count of scripts printing a rewritten seed is exactly the pinned literal",
        len(evidence["scripts_with_instrument_footprint"])
        == PINNED_INSTRUMENT_FOOTPRINT_SCRIPTS,
        f"{len(evidence['scripts_with_instrument_footprint'])} against "
        f"{PINNED_INSTRUMENT_FOOTPRINT_SCRIPTS}",
    )
    out("")
    out(f"of {evidence['headline_declared_seed']} numbers that move at the declared "
        f"seed, {evidence['declared_seed_duration_marked']} sit on a line carrying a")
    out("wall-clock duration marker. The marker GROUPS findings and never removes")
    out("one, and it is a line-local keyword test: a table row whose unit lives in")
    out("a column header elsewhere carries no marker, so the unmarked count is an")
    out("UPPER bound on the non-timing findings.")
    out.check(
        "the duration-marked count is exactly the pinned literal",
        evidence["declared_seed_duration_marked"] == PINNED_DECLARED_SEED_DURATION_MARKED,
        f"{evidence['declared_seed_duration_marked']} against "
        f"{PINNED_DECLARED_SEED_DURATION_MARKED}",
    )

    out.rule("3. COVERAGE, pinned so a silently shrinking corpus is a failure")
    out(f"candidates found: {evidence['candidates_found']}")
    out(f"candidates attempted: {evidence['candidates_attempted']}")
    out(f"status counts: {evidence['status_counts']}")
    out(f"deadline reached: {evidence['deadline_reached']}")
    out(f"claim tokens across all candidates: {evidence['claim_tokens_total']}")
    out(f"claim tokens the perturbed comparison evaluated: {evidence['claim_tokens_compared']}")
    out(f"repeats per claim: {evidence['repeats_per_claim']}")
    out(f"escaped-rate upper bound recorded: {evidence['escaped_bound']!r}")
    out(
        "escaped-rate upper bound recomputed: "
        f"{escaped_rate_upper_bound(max(1, evidence['repeats_per_claim']))!r}"
    )
    out.check(
        "the candidate count is exactly the pinned literal",
        evidence["candidates_found"] == PINNED_CANDIDATES_FOUND,
        f"{evidence['candidates_found']} against {PINNED_CANDIDATES_FOUND}",
    )
    out.check(
        "every candidate found was attempted",
        evidence["candidates_attempted"] == PINNED_CANDIDATES_ATTEMPTED
        and not evidence["deadline_reached"],
        f"{evidence['candidates_attempted']} attempted, deadline reached "
        f"{evidence['deadline_reached']}",
    )
    out.check(
        "the evaluated and total claim-token counts are exactly the pinned literals",
        evidence["claim_tokens_total"] == PINNED_CLAIM_TOKENS_TOTAL
        and evidence["claim_tokens_compared"] == PINNED_CLAIM_TOKENS_COMPARED,
        f"{evidence['claim_tokens_compared']} of {evidence['claim_tokens_total']}",
    )
    out.check(
        "the scripts contributing findings are exactly the pinned counts",
        len(evidence["scripts_with_seed_movement"]) == PINNED_SCRIPTS_WITH_SEED_MOVEMENT
        and len(evidence["scripts_with_declared_seed_movement"])
        == PINNED_SCRIPTS_WITH_DECLARED_SEED_MOVEMENT,
        f"{len(evidence['scripts_with_seed_movement'])} and "
        f"{len(evidence['scripts_with_declared_seed_movement'])}",
    )
    out.check(
        "the status counts are exactly the pinned literals",
        evidence["status_counts"] == PINNED_STATUS_COUNTS,
        f"{evidence['status_counts']} against {PINNED_STATUS_COUNTS}",
    )
    out.check(
        "the recorded escaped-rate bound equals the one recomputed from the model",
        abs(
            evidence["escaped_bound"]
            - escaped_rate_upper_bound(max(1, evidence["repeats_per_claim"]))
        )
        <= 1e-15,
    )

    out.rule("4. CANDIDATES WHOSE CLAIM COULD NOT BE EVALUATED, pinned")
    for status in sorted(set(evidence["unevaluable"]) | set(PINNED_UNEVALUABLE)):
        measured = tuple(sorted(evidence["unevaluable"].get(status, ())))
        pinned = tuple(sorted(PINNED_UNEVALUABLE.get(status, ())))
        out(f"{status}: measured {len(measured)}, pinned {len(pinned)}")
        for item in measured:
            out(f"  {item}")
        out.check(
            f"the {status} set is exactly the pinned literal set",
            measured == pinned,
            f"measured {len(measured)}, pinned {len(pinned)}",
        )

    out.rule("5. THE STATIC SCREEN, pinned")
    out(f"python files parsed: {evidence['static_files_parsed']}")
    out(f"files with at least one finding: {evidence['static_files_with_findings']}")
    out(f"findings in total: {evidence['static_findings_total']}")
    out(f"by subkind: {evidence['static_by_subkind']}")
    out(f"files that would not parse: {evidence['static_parse_failures']}")
    out.check(
        "the static files-parsed and parse-failure counts are exactly the pinned literals",
        evidence["static_files_parsed"] == PINNED_STATIC_FILES_PARSED
        and evidence["static_parse_failures"] == PINNED_STATIC_PARSE_FAILURES,
        f"{evidence['static_files_parsed']} parsed, "
        f"{evidence['static_parse_failures']} failures",
    )
    out.check(
        "the count of files with a static finding is exactly the pinned literal",
        evidence["static_files_with_findings"] == PINNED_STATIC_FILES_WITH_FINDINGS,
        f"{evidence['static_files_with_findings']} against "
        f"{PINNED_STATIC_FILES_WITH_FINDINGS}",
    )
    out.check(
        "the total static finding count is exactly the pinned literal",
        evidence["static_findings_total"] == PINNED_STATIC_FINDINGS_TOTAL,
        f"{evidence['static_findings_total']} against {PINNED_STATIC_FINDINGS_TOTAL}",
    )
    out.check(
        "the static findings by subkind are exactly the pinned literals",
        evidence["static_by_subkind"] == PINNED_STATIC_BY_SUBKIND,
        f"{evidence['static_by_subkind']} against {PINNED_STATIC_BY_SUBKIND}",
    )

    out.rule("6. THE EXCLUSIONS ARE PART OF THE RESULT")
    for item in evidence["exclusions"]:
        out(f"  excluded: {item}")
    out.check(
        "both declared exclusions are recorded in the evidence",
        set(evidence["exclusions"]) == {"products/P094", "products/P100"},
        f"{evidence['exclusions']}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
