"""Check that the committed junit and collection fixtures are reproducible.

The fixtures under fixtures/sample_project/ are produced by a real pytest run
and then normalised by fixtures/regenerate_fixtures.py. This script re-runs
that generation into a scratch directory and compares the result, byte for
byte, with what is committed. A mismatch means either the generator is not
deterministic or the committed file was edited by hand; either is worth
knowing and is printed as FAIL.

Exits 0 regardless. It reports; it does not gate.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ROOT / "fixtures" / "sample_project"
FAILURES: list[str] = []


def load_generator():
    path = ROOT / "fixtures" / "regenerate_fixtures.py"
    spec = importlib.util.spec_from_file_location("regenerate_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<54s} got {got!r}   expected {expected!r}")


def main() -> int:
    generator = load_generator()
    print("1. regenerating the fixtures into a scratch directory")
    with tempfile.TemporaryDirectory() as scratch:
        produced = generator.generate(Path(scratch))
    for name, text in produced.items():
        committed = (SAMPLE / name).read_text(encoding="utf-8")
        print(f"  {name}: committed {len(committed)} bytes, regenerated {len(text)} bytes")
        check(f"{name} is byte-identical", text == committed, True)

    print()
    print("2. the committed fixtures contain no absolute path")
    prefixes = ("/" + "home" + "/", "/" + "Users" + "/", "/" + "root" + "/")
    for name in produced:
        text = (SAMPLE / name).read_text(encoding="utf-8")
        check(f"{name} has no absolute home or root path",
              not any(prefix in text for prefix in prefixes), True)

    print()
    print("3. what normalisation removes, and why")
    print("   time, timestamp and hostname attributes are not reproducible, and")
    print("   pytest writes the absolute path of the skipping line into the text")
    print("   of the <skipped> element. Element structure, attributes, outcomes")
    print("   and properties are exactly as pytest wrote them.")

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
