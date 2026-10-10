"""The worked example quoted in README.md. Run it to regenerate that output.

Everything printed below is also written to
``validation/outputs/worked_example_output.txt``, so the README's output block
is checkable against a committed file rather than trusted. The body of the
script, from the generator down, is exactly what the README quotes.
"""

from __future__ import annotations

import atexit
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))


class _Tee(io.TextIOBase):
    """Write to the real stdout and to the committed output file at once."""

    def __init__(self, stream: io.TextIOBase, path: Path) -> None:
        self._stream = stream
        self._buffer: list[str] = []
        self._path = path

    def write(self, text: str) -> int:
        self._buffer.append(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def save(self) -> None:
        self._path.parent.mkdir(exist_ok=True)
        self._path.write_text("".join(self._buffer))


_tee = _Tee(sys.stdout, _HERE / "outputs" / "worked_example_output.txt")
sys.stdout = _tee
atexit.register(_tee.save)

import numpy as np  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from calibaudit import (  # noqa: E402
    PlattScaling,
    binned_decomposition,
    bootstrap_reliability,
    debiased_ece,
    expected_calibration_error,
    log_score,
    murphy_decomposition,
    recalibration_audit,
)

# A forecaster that is overconfident by a factor of 1.4 in the logit, and that
# quantises its output to steps of 0.05. The quantisation means the exact
# Murphy decomposition applies with no binning at all.
rng = np.random.default_rng(2026)
truth = rng.beta(2.0, 2.0, size=1200)                   # latent P(event)
outcome = (rng.random(1200) < truth).astype(float)      # the event, 0 or 1
sharp = expit(logit(truth) * 1.4)                       # overconfident forecaster
forecast = np.round(np.clip(sharp, 0.025, 0.975) * 20.0) / 20.0   # 0.05 steps

exact = murphy_decomposition(forecast, outcome)
print(exact.report())

binned = binned_decomposition(forecast, outcome, n_bins=10, strategy="equal_width")
print(f"\nthree-term residual, exact grouping : {exact.three_term_residual:+.3e}")
print(f"three-term residual, 10 equal bins  : {binned.three_term_residual:+.3e}")

print(f"\nraw ECE, 10 bins : {expected_calibration_error(forecast, outcome, n_bins=10):.6f}")
print(debiased_ece(forecast, outcome, n_bins=10, strategy="equal_width",
                   n_replicates=500, seed=1).report())

curve = bootstrap_reliability(forecast, outcome, n_bins=10, n_bootstrap=1000,
                              level=0.9, seed=2)
print(f"\nbins flagged miscalibrated at 90 % pointwise: "
      f"{int(np.sum(curve.diagonal_excluded()))} of {curve.n_occupied}")

audit = recalibration_audit(forecast, outcome, test_fraction=0.5, n_bins=10,
                            strategy="equal_width", n_bootstrap=1000, seed=3)
print(f"\nheld-out audit, {audit.n_train} train / {audit.n_test} test")
print(audit.table())

platt = PlattScaling(n_bootstrap=200, seed=4).fit(forecast, outcome)
point, lo, hi = platt.predict_with_interval([0.1, 0.5, 0.9], level=0.9)
print(f"\nfitted Platt map: a = {platt.a_:.6f}, b = {platt.b_:.6f}  "
      f"(exact inverse would be a = {1 / 1.4:.6f}, b = 0)")
for f, p, a, b in zip([0.1, 0.5, 0.9], point, lo, hi, strict=True):
    print(f"  f = {f:.1f} -> {p:.4f}  90 % interval on the map [{a:.4f}, {b:.4f}]")

# What a forecast of exactly 0 or 1 does. Keep this in mind before handing a
# clipped model output to any of this.
hard = np.where(sharp > 0.5, 1.0, 0.0)
hard_platt = PlattScaling().fit(hard, outcome)
print("\nsame data, forecasts hard-thresholded to exactly 0 or 1:")
print(f"  log score of the raw forecast : {log_score(hard, outcome):.6f}  "
      f"(clipped at 1e-15, so this is finite only by convention)")
print(f"  Platt fit on it               : a = {hard_platt.a_:.6f}, "
      f"b = {hard_platt.b_:.6f}")
print("  the slope collapses because logit(0) is clipped to -27.6 and dominates "
      "the fit")
