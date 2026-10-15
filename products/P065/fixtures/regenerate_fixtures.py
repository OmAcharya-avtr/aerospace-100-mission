"""Regenerate the committed cross-link fixtures.

``p024_bdot/junit.xml`` is a **real** pytest run of P024 DetumbleSim's own test
suite, normalised for reproducibility: every ``time`` attribute is set to
``0.000``, the suite ``timestamp`` to a fixed instant and ``hostname`` to
``fixture``, because those three move between runs and a fixture that changes
on every run is not a fixture.

The run is performed on a **copy** of P024, made into a scratch directory, so
that nothing in another product's directory is written to. Reproducing it:

    cp -r <portfolio>/products/P024/{src,tests,conftest.py,pyproject.toml,data} /tmp/p024_copy
    cd /tmp/p024_copy && python -m pytest tests/ -q --junitxml=/tmp/p024_raw.xml
    python fixtures/regenerate_fixtures.py /tmp/p024_raw.xml

The measured run on 2026-10-15 in this container: 305 tests, 0 failures,
0 errors, 0 skipped, 18.36 s wall clock on 2 cores.

``p024_bdot/docs/REQUIREMENTS.md`` is **not** regenerated: it is written by
hand for the worked example and is not a document P024 ships. See its own
header, and the Limitations section of README.md.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FIXED_TIMESTAMP = "2026-10-15T00:00:00+00:00"
OUT = Path(__file__).resolve().parent / "p024_bdot" / "junit.xml"


def normalise(raw_path: Path, out_path: Path = OUT) -> int:
    """Normalise a junit XML report and write it. Returns the test count."""
    tree = ET.parse(raw_path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = 0
    for suite in suites:
        suite.set("time", "0.000")
        suite.set("timestamp", FIXED_TIMESTAMP)
        suite.set("hostname", "fixture")
        for case in suite.iter("testcase"):
            case.set("time", "0.000")
            total += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    out_path.write_text(out_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return total


def main(argv: list[str]) -> int:
    """Entry point: ``python fixtures/regenerate_fixtures.py <raw junit.xml>``."""
    if len(argv) != 2:
        sys.stderr.write(__doc__ or "")
        sys.stderr.write("\nusage: regenerate_fixtures.py <raw junit.xml>\n")
        return 2
    count = normalise(Path(argv[1]))
    sys.stdout.write(f"wrote {OUT} with {count} test case(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
