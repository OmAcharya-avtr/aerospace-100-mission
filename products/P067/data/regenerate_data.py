"""Regenerate the committed scenario file. Deterministic, fixed seed.

    python data/regenerate_data.py

Writes ``data/scenarios.json``. Running it twice produces byte-identical
output; the test suite asserts that in ``tests/test_scenarios.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acqsearch.scenarios import generate_scenarios, save_scenarios, split_scenarios  # noqa: E402

N_SCENARIOS = 300
SEED = 20261016
TRAIN_FRACTION = 0.65
OUTPUT = ROOT / "data" / "scenarios.json"


def main() -> int:
    """Write the scenario file and print what was written."""
    scenarios = generate_scenarios(N_SCENARIOS, SEED)
    save_scenarios(scenarios, OUTPUT)
    train, test = split_scenarios(scenarios, TRAIN_FRACTION)
    size = OUTPUT.stat().st_size
    print(f"wrote {len(scenarios)} scenarios, seed {SEED}, to {OUTPUT.relative_to(ROOT)}")
    print(f"train {len(train)}  held out {len(test)}  file size {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
