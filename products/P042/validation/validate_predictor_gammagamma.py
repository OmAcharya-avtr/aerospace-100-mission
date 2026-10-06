"""Validation V5: the same V4 protocol on the gamma-gamma channel.

The lognormal channel of :mod:`acmpilot.channel` is an AR(1) process in
log-amplitude **by construction**, so its optimal predictor is linear and a
learned model has nothing structural to find. The gamma-gamma channel is built
by pushing two independent Gauss-Markov drivers through a Gaussian copula, so
``snr_dB`` is a *nonlinear* function of two latent AR(1) states: the conditional
mean is no longer linear in the last report and the conditional variance is no
longer constant. If a learned predictor is ever going to beat the analytic
linear one in this package, this is where it happens.

This script reuses ``validation/validate_predictor.py`` so the protocol is
literally the same code: disjoint train, tune and test seeds, every baseline
tuned on the tuning seeds, every reported number from the test seeds.

Runtime: about 55 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from validate_predictor import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("gamma-gamma", (2.0, 5.0, 10.0, 20.0), "gamma-gamma"))
