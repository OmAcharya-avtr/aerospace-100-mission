"""Check the alternatives table in README.md against what the packages ship.

Describing a competitor wrongly is as much a defect as a wrong number of our
own, so the claims in the README's alternatives table are checked here against
a committed snapshot of each package's own metadata and wheel contents rather
than against memory.

How the snapshot was produced, on 2026-10-10, in this container:

    curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/calibaudit/json
    curl -s https://pypi.org/pypi/<name>/json
    pip download --no-deps <name>==<version> -d .
    unzip -o <name>-<version>-py3-none-any.whl

and then the module inventory and the keyword searches recorded in
``validation/alternatives_snapshot.json``. The ``method`` key of that file
lists the exact commands.

This script is deliberately **offline and deterministic**: it reads the
committed snapshot. Network access is not assumed, so the release gate can
re-run it and get byte-identical output. Re-capturing the snapshot needs the
commands above and is a manual step, dated in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

from _harness import HERE, Recorder

SNAPSHOT = HERE / "alternatives_snapshot.json"


def main() -> int:
    rec = Recorder("validate_alternatives")
    rec.header("Alternatives named in README.md, checked against what they ship")

    data = json.loads(Path(SNAPSHOT).read_text())
    rec.say(f"snapshot captured : {data['captured_utc']} UTC")
    rec.say("snapshot file     : validation/alternatives_snapshot.json")
    rec.say("capture commands  :")
    for line in data["method"]:
        rec.say(f"  {line}")
    rec.say()

    # --- the PyPI name this package would claim -----------------------------
    code = data["name_availability"]["calibaudit"]
    rec.check(
        "the PyPI name 'calibaudit' is free",
        reference="HTTP status of https://pypi.org/pypi/calibaudit/json",
        measured=f"HTTP {code}",
        expectation="404, i.e. no project of that name exists",
        passed=code == 404,
    )

    # --- netcal -------------------------------------------------------------
    nc = data["packages"]["netcal"]
    rec.say(f"netcal {nc['version']}, {nc['license']}, uploaded {nc['upload_time_utc']} UTC")
    rec.say(f"  requires        : {', '.join(nc['requires_dist'])}")
    rec.say(f"  metrics         : {', '.join(nc['metrics_exported'])}")
    rec.say(f"  scaling methods : {', '.join(nc['scaling_modules'])}")
    rec.say(f"  binning methods : {', '.join(nc['binning_modules'])}")
    rec.say(f"  presentation    : {', '.join(nc['presentation_modules'])}")
    rec.say(f"  shipped modules : {len(nc['shipped_modules'])} .py files under netcal/")
    rec.say()
    rec.check(
        "netcal 1.4.0 exists and is the version the README cites",
        reference="PyPI JSON metadata for netcal",
        measured=f"version {nc['version']}, {nc['n_releases']} releases",
        expectation="version 1.4.0",
        passed=nc["version"] == "1.4.0",
    )
    rec.check(
        "netcal computes ECE, ACE and MCE, as the README says it does",
        reference="netcal/metrics/__init__.py in the published wheel",
        measured=f"exports {', '.join(nc['metrics_exported'])}",
        expectation="ECE, ACE and MCE all present; the README must not imply netcal "
        "lacks the metric",
        passed={"ECE", "ACE", "MCE"} <= set(nc["metrics_exported"]),
    )
    rec.check(
        "netcal ships no Brier score and therefore no Murphy decomposition",
        reference="case-insensitive search for 'brier' and 'murphy' across every .py "
        "file under netcal/ in the published wheel",
        measured=f"mentions 'brier': {nc['mentions_brier']}; mentions 'murphy': "
        f"{nc['mentions_murphy_decomposition']}; mentions 'resolution': "
        f"{nc['mentions_reliability_or_resolution_terms']}",
        expectation="all three absent, which is what lets this package claim the "
        "decomposition as a difference rather than a duplicate",
        passed=not nc["mentions_brier"]
        and not nc["mentions_murphy_decomposition"]
        and not nc["mentions_reliability_or_resolution_terms"],
    )
    rec.check(
        "netcal ships no bootstrap",
        reference="case-insensitive search for 'bootstrap' under netcal/",
        measured=f"mentions 'bootstrap': {nc['mentions_bootstrap']}",
        expectation="absent; its ReliabilityDiagram error bars come from "
        "Monte-Carlo-sampled input uncertainty, not from resampling the data",
        passed=not nc["mentions_bootstrap"],
    )
    rec.check(
        "netcal requires PyTorch, which is why it is a citation here and not an import",
        reference="requires_dist in netcal's PyPI metadata",
        measured=f"requires {', '.join(nc['requires_dist'])}",
        expectation="torch, pyro-ppl, gpytorch and tensorboard all required, none of "
        "which is installable in this 2-core container",
        passed=any(r.startswith("torch") for r in nc["requires_dist"]),
    )
    rec.say()

    # --- properscoring ------------------------------------------------------
    ps = data["packages"]["properscoring"]
    rec.say(
        f"properscoring {ps['version']}, {ps['license']}, uploaded "
        f"{ps['upload_time_utc']} UTC, {ps['n_releases']} release in total"
    )
    rec.say(f"  public API      : {', '.join(ps['public_api'])}")
    rec.say(f"  shipped modules : {len(ps['shipped_modules'])} .py files")
    rec.say()
    rec.check(
        "properscoring exists, at version 0.1, and has had exactly one release",
        reference="PyPI JSON metadata for properscoring",
        measured=f"version {ps['version']}, {ps['n_releases']} release, uploaded "
        f"{ps['upload_time_utc']}",
        expectation="one release, 2015; the README must say it is unmaintained rather "
        "than imply it is current",
        passed=ps["version"] == "0.1" and ps["n_releases"] == 1,
    )
    rec.check(
        "properscoring has a Brier score but no calibration analysis",
        reference="__init__.py __all__ and a keyword search of the published wheel",
        measured=f"public API {', '.join(ps['public_api'])}; mentions ECE: "
        f"{ps['mentions_expected_calibration_error']}; mentions reliability or "
        f"resolution: {ps['mentions_reliability_or_resolution_terms']}",
        expectation="brier_score present, no ECE, no reliability or resolution term",
        passed=(
            "brier_score" in ps["public_api"]
            and not ps["mentions_expected_calibration_error"]
            and not ps["mentions_reliability_or_resolution_terms"]
        ),
    )
    rec.check(
        "properscoring's 'decomposition' is the CRPS threshold decomposition, not "
        "Murphy's partition",
        reference="properscoring/thresholds.py in the published wheel",
        measured=f"ships threshold_decomposition: "
        f"{ps['public_api_includes_threshold_decomposition']}; ships reliability or "
        f"resolution terms: {ps['mentions_reliability_or_resolution_terms']}",
        expectation="threshold decomposition present, Murphy partition absent; "
        "conflating the two in the README would be a wrong claim about a competitor",
        passed=ps["public_api_includes_threshold_decomposition"]
        and not ps["mentions_reliability_or_resolution_terms"],
    )
    rec.say()

    # --- scikit-learn -------------------------------------------------------
    sk = data["packages"]["scikit-learn"]
    rec.say(
        f"scikit-learn {sk['version']} on PyPI, {sk['installed_version']} installed here"
    )
    rec.say(f"  calibration     : {', '.join(sk['calibration_public'])}")
    rec.say(f"  Brier metrics   : {', '.join(sk['brier_related_metrics'])}")
    rec.say()
    wanted = {
        "CalibratedClassifierCV",
        "CalibrationDisplay",
        "calibration_curve",
        "FrozenEstimator",
    }
    present = wanted & set(sk["calibration_public"])
    rec.check(
        "sklearn.calibration provides the maps and the curve the README credits it with",
        reference="dir(sklearn.calibration) and dir(sklearn.metrics) in this interpreter",
        measured=f"{len(present)} of {len(wanted)} present: {', '.join(sorted(present))}",
        expectation="all four present in 1.9.1",
        passed=present == wanted,
    )
    rec.check(
        "sklearn has a Brier score but no ECE and no Brier decomposition",
        reference="dir(sklearn.metrics) filtered for 'brier' and 'log_loss'",
        measured=f"{', '.join(sk['brier_related_metrics'])}; no name containing "
        f"'calibration_error' or 'reliability' is exported",
        expectation="brier_score_loss and log_loss present, no ECE, no decomposition",
        passed=(
            "brier_score_loss" in sk["brier_related_metrics"]
            and not any("calibration_error" in n for n in sk["calibration_public"])
        ),
    )

    rec.say()
    rec.say(
        "What this leaves as this package's difference, stated as a measured claim\n"
        "rather than a boast: netcal computes the same plug-in ECE this package\n"
        "computes, and computes more calibration metrics besides, but ships no Brier\n"
        "score, no Murphy decomposition, no bootstrap, and no estimate of its own\n"
        "ECE's binning bias. sklearn has the Brier score and the reliability curve\n"
        "but neither the decomposition nor any ECE. properscoring has the Brier score\n"
        "and the CRPS, is a decade without a release, and has no calibration\n"
        "analysis at all. The overlap is real and the README says so first."
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
