"""Verify the frozen campaign evidence, fast, against pinned literals.

The corpus-wide audit is a **campaign**: `data/repository_audit.py`, which
takes most of an hour. It is not in `validation/` and the release gate does not
run it. Its output is committed, frozen, at `repository_findings.json`, and
**this** script is the authoritative fast check over that evidence. It runs in
well under a second, executes nothing, and exits 0 in a cold clone with no
campaign run.

Why the two were separated
--------------------------
In the first version of this product the campaign lived in `validation/`. The
release gate runs every script in that directory **in sorted order**, so
`validate_repository_audit.py` ran immediately before
`validate_repository_findings.py` and **rewrote the evidence that script
pins** -- from a tree that had changed, under a machine load that was not the
one the pins were taken under. Fifteen of twenty-six checks failed, and the
cause was structural rather than a bad pin: the two scripts contradicted each
other by construction and the gate's own ordering guaranteed it.

A script that silently overwrites the file another script pins is exactly the
self-polluting-check class this product exists to detect, arriving in this
product's own validation directory. It is recorded in `VALIDATION.md` in those
terms.

What is pinned and what is not
------------------------------
**Pinned as exact literals:** the sha256 of the evidence file itself, and every
quantity that comes from reading and parsing files rather than from executing
anything -- the candidate count, the committed-token count and the whole static
screen. An independent re-execution on a different machine under a different
load reproduced every one of these exactly.

**Reported with its conditions and deliberately NOT pinned:** every quantity
that depends on how many audited scripts finished inside the per-run limit. On
that independent re-execution the timeout set **doubled**, from 9 candidates to
18, purely because both cores were busy. Those are properties of the repository
**and the machine**. Pinning them claimed something untrue, and the fix is to
stop claiming it rather than to widen a tolerance until it passes.

The consequence for the headline is stated wherever the headline appears: it is
a **lower bound** under the recorded conditions. A claim that timed out
contributes nothing, so a machine with more headroom finds at least as many
moved numbers and never fewer.
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

#: The committed, frozen evidence this script verifies.
EVIDENCE = "validation/repository_findings.json"

#: The campaign that produced it, for the reader who wants to re-run it.
CAMPAIGN = "data/repository_audit.py"

#: PINNED. The sha256 of the committed evidence file itself. This is the
#: integrity pin: if a single byte of the evidence changes, every claim below
#: is about a different measurement and this check says so first.
PINNED_EVIDENCE_SHA256 = (
    "cafca4b3a4b63249a07229ed98531f4c27ebde4e12be1b81298d799cff589ef3"
)

# ---------------------------------------------------------------------------
# LOAD-INVARIANT QUANTITIES, pinned as exact literals.
#
# These come from reading and parsing files. No audited script is executed to
# produce them, so they are properties of the repository alone and a re-run on
# any machine must reproduce them exactly. The coordinator's independent
# re-execution on a two-core machine running another build's work confirmed
# every one of them.
# ---------------------------------------------------------------------------

PINNED_CANDIDATES_FOUND = 229
PINNED_CLAIM_TOKENS_TOTAL = 43321
PINNED_STATIC_FILES_PARSED = 3564
PINNED_STATIC_FILES_WITH_FINDINGS = 53
PINNED_STATIC_FINDINGS_TOTAL = 275
PINNED_STATIC_BY_SUBKIND: dict[str, int] = {
    "write_to_read_path": 246,
    "write_under_glob": 16,
    "write_under_read": 13,
}
PINNED_STATIC_PARSE_FAILURES = 0

# ---------------------------------------------------------------------------
# THE FINDING SET AS FROZEN. These are pinned against the committed evidence,
# which the hash above fixes. They are NOT a claim that a fresh campaign on a
# different machine would reproduce them: a campaign's coverage depends on how
# many claims finish inside the per-run limit, so the headline is a LOWER BOUND
# under the conditions recorded in the evidence. See REPORTED_NOT_PINNED.
# ---------------------------------------------------------------------------

PINNED_SEED_ATTRIBUTABLE_BY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("products/P004/validation/validate_availability_model.py", 62),
    ("products/P005/validation/val_detector.py", 9),
    ("products/P005/validation/val_pointing.py", 25),
    ("products/P005/validation/val_psd.py", 10),
    ("products/P006/validation/mc_crosscheck.py", 4),
    ("products/P024/validation/momentum_monotonicity.py", 14),
    ("products/P025/validation/cusum_delay.py", 45),
    ("products/P035/validation/validate_changepoint.py", 34),
    ("products/P035/validation/validate_cusum_delay.py", 20),
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
    ("products/P083/validation/validate_clock.py", 26),
    ("products/P083/validation/validate_negative_controls.py", 2),
    ("products/P083/validation/validate_plant.py", 4),
    ("products/P083/validation/validate_resimulation.py", 1),
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

PINNED_SEED_ATTRIBUTABLE_DIGEST = (
    "a759ddc58837aef96c5a6faef07270682ffbae0e8fd2ac15c6431af062d981d7"
)

PINNED_DECLARED_SEED_BY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("products/P004/validation/validate_availability_model.py", 1),
    ("products/P005/validation/val_detector.py", 2),
    ("products/P025/validation/cusum_delay.py", 1),
    ("products/P035/validation/validate_changepoint.py", 1),
    ("products/P035/validation/validate_cusum_delay.py", 1),
    ("products/P035/validation/validate_far_design.py", 1),
    ("products/P035/validation/validate_matched_far.py", 2),
    ("products/P052/validation/validate_benchmark.py", 3),
    ("products/P052/validation/validate_difficulty.py", 2),
    ("products/P052/validation/validate_surrogate.py", 8),
    ("products/P061/validation/validate_gap.py", 62),
    ("products/P061/validation/validate_predictor.py", 11),
    ("products/P061/validation/validate_robust.py", 2),
    ("products/P061/validation/validate_warmstart.py", 64),
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

PINNED_DECLARED_SEED_DIGEST = (
    "bc3be75a38808ae28ef0d93746d8ec34e0878d772886bad8c9b314afd55b75a1"
)

PINNED_INSTRUMENT_FOOTPRINT_DIGEST = (
    "6df78cc0102c38a614785061e3c1f51b974344f5f4597f5e3a1ee28be2c8ad35"
)
PINNED_INSTRUMENT_FOOTPRINT_COUNT = 45
PINNED_INSTRUMENT_FOOTPRINT_SCRIPTS = 33

PINNED_HEADLINE_SEED_ATTRIBUTABLE = 5677
PINNED_HEADLINE_DECLARED_SEED = 345
PINNED_SCRIPTS_WITH_SEED_MOVEMENT = 102
PINNED_SCRIPTS_WITH_DECLARED_SEED_MOVEMENT = 47
PINNED_DECLARED_SEED_DURATION_MARKED = 89

# ---------------------------------------------------------------------------
# DE-PINNED, AND WHY. These quantities were pinned as exact sets in the first
# version of this product and the pin was WRONG -- not too tight, but a claim
# about the wrong thing.
#
# The coordinator re-executed the campaign on a two-core machine that was
# running another build's work on both cores. The static screen, which executes
# nothing, matched every pin exactly. Every quantity that depends on EXECUTING
# an audited script moved, and the sharpest case was the timeout set, which
# DOUBLED: 18 candidates exceeded the per-run limit where 9 had before.
#
# That is a finding about this product and it is the deepest one it produced. A
# reproducibility auditor whose own findings move with machine load is
# reporting load, not reproducibility. The honest fix is not a wider tolerance;
# it is to stop claiming these are properties of the repository. They are
# properties of the repository AND the machine, so they are REPORTED with their
# conditions and checked only for internal consistency.
# ---------------------------------------------------------------------------

REPORTED_NOT_PINNED: tuple[str, ...] = (
    "status_counts: compared / no_output / run_failed / timeout",
    "the timeout, no_output and run_failed candidate SETS",
    "claim_tokens_compared, which depends on which claims completed",
    "wall_clock_seconds",
)


def digest(items: list[dict[str, object]]) -> str:
    """sha256 over a canonical serialisation of a finding list.

    The key order is fixed and the items are sorted, so the digest depends on
    the finding SET and not on the order the campaign happened to emit it in.
    """
    keys = ("script", "line", "index", "before", "after")
    canonical = "\n".join(
        "\x1f".join(str(item[key]) for key in keys)
        for item in sorted(items, key=lambda item: tuple(str(item[key]) for key in keys))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    """Verify the frozen evidence against the pinned literals."""
    out = _bootstrap.Tee("validate_repository_findings_output.txt")
    out("Frozen campaign evidence, verified -- P100 SeedAudit")
    out("")
    out(f"evidence : {EVIDENCE}")
    out(f"campaign : {CAMPAIGN}  (NOT run by this script and NOT run by the gate)")
    out("")
    out("this script executes nothing and runs in under a second. The campaign")
    out("that produced the evidence takes most of an hour and lives in data/,")
    out("because a script in validation/ that rewrites the file another script")
    out("pins is the self-polluting-check class this product detects.")

    path = ROOT / EVIDENCE
    if not path.is_file():
        out.unavailable(
            "frozen campaign evidence",
            f"{EVIDENCE} is absent; run {CAMPAIGN} --refresh-committed to produce it",
        )
        return out.finish()

    raw = path.read_bytes()
    measured_sha = hashlib.sha256(raw).hexdigest()
    evidence = json.loads(raw.decode("utf-8"))

    out.rule("1. INTEGRITY -- is this the evidence the pins were taken from?")
    out(f"bytes          : {len(raw)}")
    out(f"pinned sha256  : {PINNED_EVIDENCE_SHA256}")
    out(f"measured sha256: {measured_sha}")
    out("")
    out("if this check fails, every claim below is about a different measurement")
    out("and nothing after it should be read as verified.")
    out.check(
        "the evidence file is byte-identical to the one the pins were taken from",
        measured_sha == PINNED_EVIDENCE_SHA256,
        f"{measured_sha[:16]} against {PINNED_EVIDENCE_SHA256[:16]}",
    )

    out.rule("2. LOAD-INVARIANT QUANTITIES, pinned as exact literals")
    out("these come from reading and parsing files. No audited script is executed")
    out("to produce them, so a re-run on any machine must reproduce them exactly.")
    out("An independent re-execution on a busier two-core machine confirmed every")
    out("one of them.")
    out("")
    for label, measured, pinned in (
        ("candidates found", evidence["candidates_found"], PINNED_CANDIDATES_FOUND),
        (
            "committed numeric tokens across all candidates",
            evidence["claim_tokens_total"],
            PINNED_CLAIM_TOKENS_TOTAL,
        ),
        (
            "static: python files parsed",
            evidence["static_files_parsed"],
            PINNED_STATIC_FILES_PARSED,
        ),
        (
            "static: files with at least one finding",
            evidence["static_files_with_findings"],
            PINNED_STATIC_FILES_WITH_FINDINGS,
        ),
        (
            "static: findings in total",
            evidence["static_findings_total"],
            PINNED_STATIC_FINDINGS_TOTAL,
        ),
        (
            "static: files that would not parse",
            evidence["static_parse_failures"],
            PINNED_STATIC_PARSE_FAILURES,
        ),
    ):
        out(f"{label}: measured {measured}, pinned {pinned}")
        out.check(f"{label} is exactly the pinned literal", measured == pinned,
                  f"{measured} against {pinned}")
    out("")
    out(f"static findings by subkind: {evidence['static_by_subkind']}")
    out.check(
        "the static findings by subkind are exactly the pinned literals",
        evidence["static_by_subkind"] == PINNED_STATIC_BY_SUBKIND,
        f"{evidence['static_by_subkind']} against {PINNED_STATIC_BY_SUBKIND}",
    )

    out.rule("3. THE FROZEN FINDING SET, pinned two ways")
    out("the per-script counts localise a change to a file; the digest covers")
    out("every item including its before and after values, so a changed VALUE")
    out("inside an unchanged script fails this check too.")
    out("")
    seed_items = evidence["seed_attributable"]
    declared_items = evidence["declared_seed_moved"]
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
        counted = collections.Counter(item["script"] for item in items)
        measured_counts = tuple(sorted(counted.items()))
        expected_counts = tuple(sorted(by_script))
        out(f"{label}: {len(items)} items across {len(counted)} scripts")
        out(f"  pinned digest  : {pinned_digest}")
        out(f"  measured digest: {measured_digest}")
        for pair in [p for p in expected_counts if p not in measured_counts]:
            out(f"  MISSING (pinned but not measured): {pair}")
        for pair in [p for p in measured_counts if p not in expected_counts]:
            out(f"  EXTRA (measured but not pinned): {pair}")
        out.check(
            f"the {label} per-script counts are exactly the pinned literals",
            measured_counts == expected_counts,
            f"{len(counted)} scripts",
        )
        out.check(
            f"the {label} digest is exactly the pinned literal",
            measured_digest == pinned_digest,
            f"{measured_digest[:16]} against {pinned_digest[:16]}",
        )
    out("")
    out.check(
        "the itemised lists and the recorded headlines agree",
        len(seed_items) == evidence["headline_seed_attributable"]
        and len(declared_items) == evidence["headline_declared_seed"],
        f"{len(seed_items)} and {len(declared_items)}",
    )
    out.check(
        "the headline counts are exactly the pinned literals",
        evidence["headline_seed_attributable"] == PINNED_HEADLINE_SEED_ATTRIBUTABLE
        and evidence["headline_declared_seed"] == PINNED_HEADLINE_DECLARED_SEED,
        f"{evidence['headline_seed_attributable']} and "
        f"{evidence['headline_declared_seed']}",
    )
    out.check(
        "the contributing script counts are exactly the pinned literals",
        len(evidence["scripts_with_seed_movement"]) == PINNED_SCRIPTS_WITH_SEED_MOVEMENT
        and len(evidence["scripts_with_declared_seed_movement"])
        == PINNED_SCRIPTS_WITH_DECLARED_SEED_MOVEMENT,
        f"{len(evidence['scripts_with_seed_movement'])} and "
        f"{len(evidence['scripts_with_declared_seed_movement'])}",
    )
    out.check(
        "the duration-marked count is exactly the pinned literal",
        evidence["declared_seed_duration_marked"] == PINNED_DECLARED_SEED_DURATION_MARKED,
        f"{evidence['declared_seed_duration_marked']} against "
        f"{PINNED_DECLARED_SEED_DURATION_MARKED}",
    )

    out.rule("3b. THE INSTRUMENT'S OWN FOOTPRINT, pinned and excluded")
    footprint = evidence["instrument_footprint"]
    out(f"instrument-footprint numbers: {len(footprint)}")
    out(f"scripts that print a seed the campaign rewrote: "
        f"{len(evidence['scripts_with_instrument_footprint'])}")
    out(f"  pinned digest  : {PINNED_INSTRUMENT_FOOTPRINT_DIGEST}")
    out(f"  measured digest: {digest(footprint)}")
    out.check(
        "the instrument-footprint count, scripts and digest are the pinned literals",
        evidence["instrument_footprint_count"] == PINNED_INSTRUMENT_FOOTPRINT_COUNT
        and len(evidence["scripts_with_instrument_footprint"])
        == PINNED_INSTRUMENT_FOOTPRINT_SCRIPTS
        and digest(footprint) == PINNED_INSTRUMENT_FOOTPRINT_DIGEST,
        f"{evidence['instrument_footprint_count']} numbers, "
        f"{len(evidence['scripts_with_instrument_footprint'])} scripts",
    )

    out.rule("4. LOAD-SENSITIVE QUANTITIES -- reported, deliberately NOT pinned")
    out("these depend on how many audited scripts finished inside the per-run")
    out("limit, which depends on the machine. An independent re-execution on a")
    out("busier two-core machine saw the timeout set DOUBLE, from 9 candidates to")
    out("18, with no change to the repository. Pinning them claimed something")
    out("untrue; they are reported with their conditions instead.")
    out("")
    out("de-pinned, and what is now unpinned:")
    for item in REPORTED_NOT_PINNED:
        out(f"  {item}")
    out("")
    counts = evidence["status_counts"]
    timeouts = counts.get("timeout", 0)
    required = ("per_run_timeout_seconds", "workers", "cpu_count", "wall_clock_seconds")
    recorded = all(key in evidence for key in required)
    out("conditions the campaign recorded:")
    if not recorded:
        out("  NOT RECORDED. This evidence predates the provenance fields, so the")
        out("  conditions it was taken under cannot be stated and the bounds below")
        out("  cannot be qualified. Re-run the campaign with --refresh-committed.")
        out.check(
            "the evidence records the conditions it was produced under",
            False,
            f"missing {[k for k in required if k not in evidence]}",
        )
    else:
        out(f"  per-run timeout, seconds : {evidence['per_run_timeout_seconds']!r}")
        out(f"  workers                  : {evidence['workers']}")
        out(f"  cpu count                : {evidence['cpu_count']}")
        out(f"  campaign wall clock, s   : {evidence['wall_clock_seconds']}")
    out("")
    out("under those conditions:")
    out(f"  candidates attempted     : {evidence['candidates_attempted']}")
    out(f"  compared                 : {counts.get('compared', 0)}")
    out(f"  no_output                : {counts.get('no_output', 0)}")
    out(f"  run_failed               : {counts.get('run_failed', 0)}")
    out(f"  timeout                  : {timeouts}")
    out(f"  claim tokens evaluated   : {evidence['claim_tokens_compared']}")
    out("")
    out(
        f"AT LEAST {timeouts} of {evidence['candidates_found']} candidate claims did not "
        f"complete within {evidence.get('per_run_timeout_seconds', 'an unrecorded')!r} s "
        f"at the load this "
        f"campaign ran under. On a busier machine the figure was 18. The count is a "
        f"LOWER BOUND on how many claims a reader should treat as unevaluated."
    )
    out("")
    out(
        f"THE HEADLINE OF {evidence['headline_seed_attributable']} IS LIKEWISE A LOWER "
        f"BOUND. It counts moved numbers among the {counts.get('compared', 0)} claims "
        "that completed. A claim that timed out contributes nothing, so a machine with "
        "more headroom finds at least as many and never fewer."
    )
    out("")
    out("what IS checked here is internal consistency, which is load-invariant:")
    total = sum(counts.values())
    out.check(
        "the status counts sum to the candidates attempted",
        total == evidence["candidates_attempted"],
        f"{total} against {evidence['candidates_attempted']}",
    )
    out.check(
        "every candidate found was attempted and the deadline was not reached",
        evidence["candidates_attempted"] == evidence["candidates_found"]
        and not evidence["deadline_reached"],
        f"{evidence['candidates_attempted']} of {evidence['candidates_found']}, "
        f"deadline reached {evidence['deadline_reached']}",
    )
    out.check(
        "the unevaluable sets have the sizes the status counts report",
        all(
            len(evidence["unevaluable"].get(status, [])) == counts.get(status, 0)
            for status in ("no_output", "run_failed", "timeout")
        ),
        f"{ {k: len(v) for k, v in evidence['unevaluable'].items()} }",
    )
    out.check(
        "the evaluated token count does not exceed the committed total",
        0 < evidence["claim_tokens_compared"] <= evidence["claim_tokens_total"],
        f"{evidence['claim_tokens_compared']} of {evidence['claim_tokens_total']}",
    )
    out.check(
        "the recorded escaped-rate bound equals the one recomputed from the model",
        abs(
            evidence["escaped_bound"]
            - escaped_rate_upper_bound(max(1, evidence["repeats_per_claim"]))
        )
        <= 1e-15,
        f"{evidence['escaped_bound']!r}",
    )

    out.rule("5. THE EXCLUSIONS ARE PART OF THE RESULT")
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
