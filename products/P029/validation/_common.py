"""Shared bootstrap for the validation scripts: put ``src/`` on the path and provide a
small PASS/FAIL recorder. No numerical content lives here."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


class Checks:
    """Accumulates PASS/FAIL lines and prints a summary."""

    def __init__(self) -> None:
        self.n_pass = 0
        self.n_fail = 0

    def check(self, label: str, value: float, target: float, tol: float, kind: str = "rel") -> bool:
        """Record one numeric comparison; ``kind`` is ``'rel'`` or ``'abs'``."""
        if kind == "rel":
            denom = abs(target) if target != 0.0 else 1.0
            err = abs(value - target) / denom
        else:
            err = abs(value - target)
        ok = err <= tol
        self.n_pass += ok
        self.n_fail += not ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {label:<62} value={value: .10e} "
            f"target={target: .10e} {kind}_err={err:.3e} tol={tol:.1e}"
        )
        return ok

    def assert_true(self, label: str, ok: bool, note: str = "") -> bool:
        """Record one boolean check."""
        self.n_pass += bool(ok)
        self.n_fail += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<62} {note}")
        return bool(ok)

    def summary(self, title: str) -> None:
        """Print the pass/fail tally."""
        print(f"\n{title}: {self.n_pass} passed, {self.n_fail} failed")
