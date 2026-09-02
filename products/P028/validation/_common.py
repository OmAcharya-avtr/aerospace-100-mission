"""Shared helpers for the skymatch validation scripts. Seeded and deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: One seed for every script, so a rerun reproduces the committed output.
SEED = 20260902

#: The reference camera used throughout validation: a 12 deg square field on a
#: 1024 x 1024 detector, 42.19 arcsec/pixel. Inside the 10-25 deg range typical
#: of small-satellite star trackers.
CAMERA_FOV_DEG = 12.0
CAMERA_PIXELS = 1024


def banner(title: str) -> None:
    """Print a section banner."""
    print("=" * 86)
    print(title)
    print("=" * 86)


def verdict(name: str, value: float, tolerance: float, mode: str = "<=") -> bool:
    """Print ``name value <op> tolerance PASS/FAIL`` and return whether it passed."""
    if mode == "<=":
        ok = bool(value <= tolerance)
        op = "<="
    elif mode == ">=":
        ok = bool(value >= tolerance)
        op = ">="
    else:
        raise ValueError(f"mode must be '<=' or '>=', got {mode!r}")
    print(f"{name:<62s} {value:11.4e} {op} {tolerance:9.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def report(name: str, value: float, unit: str = "") -> None:
    """Print a measured quantity that is reported rather than gated."""
    suffix = f" {unit}" if unit else ""
    print(f"{name:<62s} {value:11.4e}{suffix}")


def rate_row(label: str, correct: int, false: int, none: int, trials: int) -> str:
    """One formatted outcome row with the Wilson interval on the false-ID rate."""
    from skymatch.benchmark import wilson_interval  # noqa: PLC0415

    lo, hi = wilson_interval(false, trials)
    return (
        f"{label:<26s} {correct / trials:8.4f} {false / trials:9.4f} "
        f"{none / trials:8.4f}   [{lo:.4f}, {hi:.4f}]  {false}/{trials}"
    )


RATE_HEADER = (
    f"{'method':<26s} {'ident':>8s} {'false ID':>9s} {'none':>8s}   "
    f"{'95% CI on false ID':^20s}  count"
)


def finish(passed: list[bool]) -> int:
    """Print the pass count and return the process exit code."""
    print()
    print(f"RESULT: {sum(passed)} / {len(passed)} checks passed")
    return 0 if all(passed) else 1


def unit_sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    """``(n, 3)`` directions uniform on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]
