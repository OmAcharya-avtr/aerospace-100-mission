"""Shared path setup for the validation scripts. Not part of the package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "fixtures" / "sample_project"
HISTORY_JSON = ROOT / "fixtures" / "history" / "history.json"
FAULTS_JSON = ROOT / "fixtures" / "faults" / "faults.json"
FAULT_MATRIX_JSON = ROOT / "fixtures" / "faults" / "fault_matrix.json"
TRAIN_FRACTION = 0.70

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
