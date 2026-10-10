"""Regenerate the committed junit XML and collection-report fixtures.

The fixtures under ``fixtures/sample_project/`` are produced by a real pytest
run, not written by hand, so that the shapes this package parses are the
shapes pytest actually emits.  Two things in a raw pytest report are not
reproducible and both are normalised here:

* wall-clock values -- ``time``, ``timestamp``, ``hostname``;
* the absolute path pytest writes into the text of a ``<skipped>`` element,
  which would otherwise commit this container's directory layout.

Nothing else is altered: element structure, attributes, outcomes and
properties are exactly what pytest wrote.  Normalisation makes regeneration
byte-deterministic, which ``validation/validate_fixtures.py`` checks.

Run from anywhere:  ``python fixtures/regenerate_fixtures.py``
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "sample_project"
PRODUCT_ROOT = HERE.parent

FIXED_TIMESTAMP = "2026-10-10T00:00:00+00:00"
FIXED_HOSTNAME = "fixture"


def normalise(xml_text: str, project_dir: Path) -> str:
    """Remove the non-reproducible parts of a pytest junit report."""
    text = xml_text.replace(str(project_dir) + "/", "")
    text = text.replace(str(project_dir), ".")
    text = re.sub(r'\btime="[0-9.]+"', 'time="0.000"', text)
    text = re.sub(r'\btimestamp="[^"]*"', f'timestamp="{FIXED_TIMESTAMP}"', text)
    text = re.sub(r'\bhostname="[^"]*"', f'hostname="{FIXED_HOSTNAME}"', text)
    if not text.endswith("\n"):
        text += "\n"
    return text


def generate(target_dir: Path) -> dict[str, str]:
    """Run pytest on the sample project and return the normalised artifacts."""
    target_dir.mkdir(parents=True, exist_ok=True)
    xml_path = target_dir / "junit.xml"
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider",
            "-c", "pytest.ini", "-o", "junit_family=xunit1", f"--junitxml={xml_path}",
        ],
        cwd=PROJECT, capture_output=True, text=True, timeout=300, check=False,
    )
    xml_text = normalise(xml_path.read_text(encoding="utf-8"), PROJECT)
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-c", "pytest.ini"],
        cwd=PROJECT, capture_output=True, text=True, timeout=300, check=False,
    )
    lines = [
        ln for ln in collect.stdout.splitlines()
        if "::" in ln and ln.strip().endswith(tuple("abcdefghijklmnopqrstuvwxyz0123456789_]"))
    ]
    collect_text = "\n".join(lines) + "\n"
    return {"junit.xml": xml_text, "collect_only.txt": collect_text}


def main() -> int:
    artifacts = generate(PROJECT)
    for name, text in artifacts.items():
        path = PROJECT / name
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(PRODUCT_ROOT)}  ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
