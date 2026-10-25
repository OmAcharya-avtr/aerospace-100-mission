"""The covariance prohibition: every refusal fires, every legitimate call passes.

The prohibition is that nothing this package produces is a covariance and
nothing it produces may be used as the input covariance of a conjunction
assessment. P071 ConjProb of this mission is the product that computes
collision probability, and its covariance has to come from orbit determination
-- a filter with measurement residuals, or a published covariance in a
conjunction data message.

A guard that raised on everything would pass a test file that only checked
that it raises. Every refusal below is therefore paired with a request that
must be accepted, and both counts are reported.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import guard  # noqa: E402
from orbitdrift.errors import CovarianceRefused  # noqa: E402

REFUSED_PURPOSES = (
    "covariance",
    "need a covariance for screening",
    "conjunction assessment",
    "collision probability",
    "compute Pc",
    "CDM generation",
    "close approach report",
    "miss distance estimate",
    "manoeuvre screening",
    "maneuver screening",
    "encounter geometry",
    "COVARIANCE FOR CONJUNCTION",
    "estimate pc",
)

ACCEPTED_PURPOSES = (
    "ephemeris staleness check",
    "decide whether to re-task tracking",
    "pass prediction for a ground station",
    "is this TLE too old for my pointing budget",
    "deciding when to request a fresh element set",
    "spacecraft operations",
    "pc12345 is an identifier, not the word",
    "recovery of a lost object",
)

REFUSED_KEYS = (
    "covariance",
    "position_covariance",
    "cov",
    "cov_matrix",
    "sigma_matrix",
    "p_matrix",
    "state_covariance",
    "COVARIANCE",
)

ACCEPTED_KEYS = (
    "drift_km",
    "log_residual_std",
    "coverage",
    "recovery",
    "interval_score",
    "ages_days",
    "conservative_horizon_days",
)


def main() -> int:
    out = Tee("validate_guard_output.txt")
    out.rule("orbitdrift -- the covariance prohibition")
    for line in environment_lines():
        out(line)

    out.rule("the cross-reference every refusal carries")
    try:
        guard.refuse_covariance()
    except CovarianceRefused as error:
        message = str(error)
    for line in message.split(". "):
        if line.strip():
            out(f"  {line.strip()}.")
    out.check("the refusal names P071", "P071" in message, "")
    out.check("the refusal names ConjProb", "ConjProb" in message, "")
    out.check("the refusal names orbit determination", "orbit determination" in message, "")

    out.rule("purposes that must be refused")
    refused = 0
    for purpose in REFUSED_PURPOSES:
        try:
            guard.check_purpose(purpose)
            out(f"  [NOT REFUSED] {purpose!r}")
        except CovarianceRefused:
            refused += 1
            out(f"  refused: {purpose!r}")
    out.check(
        f"all {len(REFUSED_PURPOSES)} forbidden purposes are refused",
        refused == len(REFUSED_PURPOSES),
        f"{refused} of {len(REFUSED_PURPOSES)}",
    )

    out.rule("purposes that must be accepted (the negative control)")
    accepted = 0
    for purpose in ACCEPTED_PURPOSES:
        try:
            assert guard.check_purpose(purpose) == purpose
            accepted += 1
            out(f"  accepted: {purpose!r}")
        except CovarianceRefused:
            out(f"  [WRONGLY REFUSED] {purpose!r}")
    out.check(
        f"all {len(ACCEPTED_PURPOSES)} legitimate purposes are accepted",
        accepted == len(ACCEPTED_PURPOSES),
        f"{accepted} of {len(ACCEPTED_PURPOSES)}; without this the refusals above "
        "would be satisfied by a guard that raised on everything",
    )

    out.rule("serialised keys that must be refused")
    key_refused = 0
    for key in REFUSED_KEYS:
        try:
            guard.audit_mapping_for_covariance_keys({key: 1.0}, where="probe")
            out(f"  [NOT REFUSED] {key!r}")
        except CovarianceRefused:
            key_refused += 1
            out(f"  refused: {key!r}")
    out.check(
        f"all {len(REFUSED_KEYS)} covariance-shaped keys are refused",
        key_refused == len(REFUSED_KEYS),
        f"{key_refused} of {len(REFUSED_KEYS)}",
    )

    out.rule("serialised keys that must be accepted (the negative control)")
    try:
        guard.audit_mapping_for_covariance_keys(
            dict.fromkeys(ACCEPTED_KEYS, 1.0), where="probe"
        )
        out(f"  accepted all {len(ACCEPTED_KEYS)} ordinary keys: {', '.join(ACCEPTED_KEYS)}")
        key_accepted = True
    except CovarianceRefused as error:
        out(f"  [WRONGLY REFUSED] {error}")
        key_accepted = False
    out.check("ordinary keys pass the audit", key_accepted, "")

    out.rule("a nested key is found")
    try:
        guard.audit_mapping_for_covariance_keys(
            {"outer": {"inner": {"state_covariance": 1.0}}}, where="probe"
        )
        nested = ""
    except CovarianceRefused as error:
        nested = str(error)
    out(f"  {nested.split('.')[0]}...")
    out.check(
        "the audit walks nested mappings and names the path",
        "probe.outer.inner.state_covariance" in nested,
        "",
    )

    out.rule("the two methods that exist only to refuse")
    from sgp4.api import Satrec

    from orbitdrift import build_report, model_divergence
    from orbitdrift.features import features_from_satrec
    from orbitdrift.staleness import staleness_verdict

    satrec = Satrec.twoline2rv(
        "1 25544U 98067A   26015.50000000  .00016717  00000-0  10270-3 0  9003",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    )
    curve = model_divergence(
        satrec, span_days=0.4, steps_per_rev=90, report_every_hours=4.0, sample_every=3
    )
    report = build_report(satrec, curve)
    for name, call in (
        ("DriftReport.as_covariance", report.as_covariance),
        (
            "StalenessVerdict.as_covariance",
            staleness_verdict(
                _ConstantPredictor(), features_from_satrec(satrec), 5.0
            ).as_covariance,
        ),
    ):
        try:
            call()
            out(f"  [NOT REFUSED] {name}")
            out.check(f"{name} raises", False, "it returned")
        except CovarianceRefused as error:
            out(f"  refused: {name}")
            out.check(f"{name} raises and names P071", "P071" in str(error), "")

    out.rule("the serialised report is audited by the report itself")
    payload = report.to_dict()
    out(f"  keys at the top level         : {len(payload)}")
    out(f"  quantity statement            : {payload['quantity'][:68]}...")
    out.check(
        "the payload states that it is not a covariance",
        "not a covariance" in payload["quantity"],
        "",
    )

    def _walk(mapping, prefix=""):
        offenders = []
        for key, value in mapping.items():
            if guard.COVARIANCE_KEY_PATTERN.search(str(key)):
                offenders.append(f"{prefix}{key}")
            if isinstance(value, dict):
                offenders.extend(_walk(value, f"{prefix}{key}."))
        return offenders

    offenders = _walk(payload)
    out.check(
        "no covariance-shaped key appears at any depth of the payload",
        not offenders,
        f"{offenders}" if offenders else f"{len(payload)} top-level keys scanned",
    )
    out("")
    out("  And the audit is live rather than decorative: inserting such a key into")
    out("  the same payload must raise.")
    payload["position_covariance"] = [[1.0, 0.0], [0.0, 1.0]]
    try:
        guard.audit_mapping_for_covariance_keys(payload, where="DriftReport")
        fired = False
    except CovarianceRefused:
        fired = True
    out.check("inserting a covariance key into the payload raises", fired, "")

    out.rule("counts")
    out(f"  refusals that fired           : {refused + key_refused + 2}")
    out(f"  legitimate calls accepted     : {accepted + len(ACCEPTED_KEYS)}")
    out("")
    out("  Both numbers matter. A guard that refused everything would make the")
    out("  first number large and the second zero, and would be useless.")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


class _ConstantPredictor:
    """A trivial predictor, so the verdict can be built without fitting anything."""

    name = "constant"

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(x, dtype=float))[:, -1]

    def predict_interval_km(self, x: np.ndarray, coverage: float = 0.90):
        del coverage
        centre = self.predict_median_km(x)
        return 0.5 * centre, 2.0 * centre


if __name__ == "__main__":
    sys.exit(main())
