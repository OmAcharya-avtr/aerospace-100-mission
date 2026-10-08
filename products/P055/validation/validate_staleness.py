"""Measure what the content-hash staleness check does and does not discriminate.

This script exists to put a number on the honest limit of the staleness check.
SHA-256 (FIPS 180-4, NIST, August 2015) detects that the bytes of an artifact
changed. It carries no information about whether the change matters. The
experiment below makes 20 changes that invert the meaning of an artifact and 20
that do not change its meaning at all, and counts how many of each the check
reports as stale. If the two counts are equal, the check's ability to
discriminate material from immaterial change is zero -- which is the expected
and intended result, and the reason the README says so in its own voice.

Assumptions and validity range:

* The check assumes that an artifact's SHA-256 digest is a usable proxy for its
  identity. This package does not verify that assumption and cannot; it is a
  cryptographic assumption about SHA-256, not an engineering one about the
  artifact.
* The check reads only the artifact's *current* bytes. It holds no history, so a
  change that was later reverted is invisible to it. This is measured below.
* "Material" and "immaterial" here are the script author's labels on synthetic
  edits, not a general definition. The point of the measurement is the ratio,
  which is 1:1 by construction.

Run: ``python validation/validate_staleness.py``
Output: ``validation/validate_staleness_output.txt``
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time

import _bootstrap  # noqa: F401

from assuregraph import EvidenceStatus, inspect_evidence, parse_case, sha256_file

MATERIAL_EDITS = [
    ("verdict: pass\n", "verdict: fail\n"),
    ("violations: 0\n", "violations: 17\n"),
    ("latency_samples: 3\n", "latency_samples: 31\n"),
    ("coverage: 100 %\n", "coverage: 42 %\n"),
    ("threshold: 4.0\n", "threshold: 40.0\n"),
]
IMMATERIAL_EDITS = [
    ("verdict: pass\n", "verdict: pass\n\n"),
    ("violations: 0\n", "violations: 0\n"),  # replaced below with a comment edit
    ("latency_samples: 3\n", "latency_samples: 3  \n"),
    ("coverage: 100 %\n", "coverage: 100 %\r\n"),
    ("threshold: 4.0\n", "# regenerated 2026-10-08\nthreshold: 4.0\n"),
]
# Make the second immaterial edit a real but meaning-preserving one.
IMMATERIAL_EDITS[1] = ("violations: 0\n", "violations: 0\n# end of report\n")


def _case(base_dir: str, digest: str | None) -> object:
    evidence: dict[str, object] = {"path": "artifact.txt"}
    if digest is not None:
        evidence["sha256"] = digest
    document = {
        "name": "staleness probe",
        "nodes": [
            {
                "id": "Sn1",
                "type": "solution",
                "statement": "the artifact under test",
                "evidence": evidence,
            }
        ],
    }
    return parse_case(document, base_dir=base_dir)


def _status(base_dir: str, digest: str | None) -> EvidenceStatus:
    case = _case(base_dir, digest)
    return inspect_evidence(case, case.nodes["Sn1"]).status  # type: ignore[attr-defined]


def main() -> int:
    start = time.perf_counter()
    lines = [
        "Content-hash staleness: what it discriminates -- assuregraph 0.1.0",
        "",
        "Hash function: SHA-256, FIPS 180-4 (NIST, August 2015).",
        "",
    ]
    failures = 0

    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "artifact.txt")

        # 1. No change at all.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("verdict: pass\n")
        digest = sha256_file(path)
        status = _status(scratch, digest)
        lines.append(f"unchanged artifact                         -> {status.value}")
        failures += status is not EvidenceStatus.FRESH

        # 2. Material and immaterial edits.
        material_detected = 0
        immaterial_detected = 0
        detail_rows = []
        for label, edits, counter in (
            ("material", MATERIAL_EDITS, "material"),
            ("immaterial", IMMATERIAL_EDITS, "immaterial"),
        ):
            for before, after in edits:
                for repeat in range(4):
                    padding = ("x" * repeat + "\n") if repeat else ""
                    with open(path, "w", encoding="utf-8", newline="") as handle:
                        handle.write(before + padding)
                    digest = sha256_file(path)
                    with open(path, "w", encoding="utf-8", newline="") as handle:
                        handle.write(after + padding)
                    status = _status(scratch, digest)
                    detected = status is EvidenceStatus.STALE
                    if counter == "material":
                        material_detected += detected
                    else:
                        immaterial_detected += detected
                    if repeat == 0:
                        detail_rows.append(
                            f"  {label:<11} {before!r:<24} -> {after!r:<40} "
                            f"{status.value}"
                        )
        lines.append("")
        lines.append("Representative edits (first repeat of each):")
        lines.extend(detail_rows)
        total_material = len(MATERIAL_EDITS) * 4
        total_immaterial = len(IMMATERIAL_EDITS) * 4
        rate_difference = (
            material_detected / total_material - immaterial_detected / total_immaterial
        )
        lines.extend(
            [
                "",
                f"material edits made                       : {total_material}",
                f"material edits reported stale             : {material_detected}",
                f"immaterial edits made                     : {total_immaterial}",
                f"immaterial edits reported stale           : {immaterial_detected}",
                "",
                "Discrimination between material and immaterial change:",
                f"  material detection rate   = {material_detected}/{total_material}"
                f" = {material_detected / total_material:.3f}",
                f"  immaterial detection rate = {immaterial_detected}/{total_immaterial}"
                f" = {immaterial_detected / total_immaterial:.3f}",
                f"  difference in rates       = {rate_difference:.3f}",
                "",
                "A difference of 0.000 means the check cannot tell the two apart. That is",
                "the intended behaviour of a content hash and the reason this package calls",
                "the finding 'the bytes changed' rather than 'the evidence is invalid'.",
            ]
        )
        if material_detected != total_material:
            failures += 1
        if immaterial_detected != total_immaterial:
            failures += 1

        # 3. Revert invisibility.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("verdict: pass\n")
        digest = sha256_file(path)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("verdict: fail\n")
        mid = _status(scratch, digest)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("verdict: pass\n")
        after_revert = _status(scratch, digest)
        lines.extend(
            [
                "",
                "History blindness",
                f"  after a material change          -> {mid.value}",
                f"  after reverting that change      -> {after_revert.value}",
                "  The check reads current bytes only. An artifact changed and changed back",
                "  is indistinguishable from one never touched.",
            ]
        )
        failures += mid is not EvidenceStatus.STALE
        failures += after_revert is not EvidenceStatus.FRESH

        # 4. The three non-comparison states.
        unverifiable = _status(scratch, None)
        os.remove(path)
        absent = _status(scratch, digest)
        os.mkdir(path)
        not_a_file = _status(scratch, digest)
        os.rmdir(path)
        lines.extend(
            [
                "",
                "States where no comparison is possible",
                f"  present, no digest recorded      -> {unverifiable.value}",
                f"  cited path absent                -> {absent.value}",
                f"  cited path is a directory        -> {not_a_file.value}",
                "  'unverifiable' is counted separately from 'fresh' everywhere in the",
                "  package and is excluded from the freshness denominator.",
            ]
        )
        failures += unverifiable is not EvidenceStatus.UNVERIFIABLE
        failures += absent is not EvidenceStatus.ABSENT
        failures += not_a_file is not EvidenceStatus.UNREADABLE

        # 5. Hashing throughput, so the compute cost of the check is on record.
        payload = os.urandom(1 << 20)
        big = os.path.join(scratch, "big.bin")
        with open(big, "wb") as handle:
            for _ in range(32):
                handle.write(payload)
        size_mib = os.path.getsize(big) / (1 << 20)
        t0 = time.perf_counter()
        digest_big = sha256_file(big)
        hash_seconds = time.perf_counter() - t0
        with open(big, "rb") as handle:
            reference = hashlib.sha256(handle.read()).hexdigest()
        lines.extend(
            [
                "",
                "Hashing cost (two shared contended cores; wall clock, varies run to run)",
                f"  artifact size                    : {size_mib:.1f} MiB",
                f"  sha256_file elapsed              : {hash_seconds:.3f} s",
                f"  throughput                       : {size_mib / hash_seconds:.1f} MiB/s",
                f"  digest agrees with hashlib.sha256 of the whole file: "
                f"{'yes' if digest_big == reference else 'NO -- FAIL'}",
            ]
        )
        failures += digest_big != reference

    lines.extend(["", f"FAILURES: {failures}", f"elapsed: {time.perf_counter() - start:.3f} s"])
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
