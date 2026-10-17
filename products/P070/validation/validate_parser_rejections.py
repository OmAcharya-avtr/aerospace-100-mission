"""Every malformed-document path, counted, with the key path it named.

The test suite parametrises the same corpus. This script exists so that the
counts quoted in README.md and VALIDATION.md come from a committed run rather
than from a test summary line.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _support import Emitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_parse_malformed import CASES  # noqa: E402
from watchdogspec import SpecFormatError, load_spec, parse_spec  # noqa: E402


def main() -> int:
    """Replay the rejection corpus. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("watchdogspec validation: parser rejections")
    emit()
    emit(f"  schema rejection cases : {len(CASES)}")

    raised = 0
    named = 0
    accepted: list[str] = []
    wrong_path: list[str] = []
    for label, document, expected in CASES:
        try:
            parse_spec(document)
        except SpecFormatError as exc:
            raised += 1
            if expected in str(exc):
                named += 1
            else:
                wrong_path.append(f"{label}: expected {expected!r}, got {exc}")
        else:
            accepted.append(label)

    emit(f"  raised SpecFormatError : {raised}/{len(CASES)}")
    emit(f"  named the key path     : {named}/{len(CASES)}")
    emit(f"  silently accepted      : {len(accepted)}")
    for label in accepted:
        emit(f"    accepted: {label}")
    for line in wrong_path:
        emit(f"    wrong path: {line}")
    emit.check("every malformed document was rejected", raised == len(CASES))
    emit.check("every rejection named the offending key path", named == len(CASES))
    emit.check("no malformed document was accepted", not accepted)

    emit()
    emit("  input and output paths")
    io_cases = 0
    io_raised = 0
    # Named relatively so the product's own error text -- which quotes the path
    # it was given -- stays free of this machine's absolute paths in the
    # committed output. _support anchors the working directory on __file__.
    here = Path("validation")
    for label, target in (
        ("missing file", here / "does-not-exist.yaml"),
        ("directory", here),
    ):
        io_cases += 1
        try:
            load_spec(target)
        except SpecFormatError as exc:
            io_raised += 1
            emit(f"    {label:<14}: {exc}")
    bad_utf8 = here / "_invalid_utf8_probe.yaml"
    try:
        bad_utf8.write_bytes(b"\xff\xfe\x00spec_version: 1")
        io_cases += 1
        try:
            load_spec(bad_utf8)
        except SpecFormatError as exc:
            io_raised += 1
            emit(f"    {'invalid utf-8':<14}: {exc}")
    finally:
        bad_utf8.unlink(missing_ok=True)
    emit.check("every input/output path raised SpecFormatError", io_raised == io_cases,
               f"{io_raised}/{io_cases}")

    emit()
    emit("  the three shipped specifications all parse")
    for name in ("minimal_consistent.yaml", "planted_defects.yaml", "smallsat_fc.yaml"):
        spec = load_spec(Path("examples") / "specs" / name)
        emit(f"    {name:<26} tasks {len(spec.tasks):>2}  ladders {len(spec.ladders):>2}")
    emit.check("all three shipped specifications parse", True)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
