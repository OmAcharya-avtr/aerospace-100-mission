"""The worked example quoted in README.md, run verbatim. Exits 0."""

from __future__ import annotations

import numpy as np

from conformalband import (
    GaussianResidualInterval,
    PhysicsRegressor,
    SplitConformal,
    WeightedConformal,
    effective_sample_size,
    make_audit_split,
    split_conformal_coverage_bound,
)

ALPHA = 0.1


def main() -> None:
    data = make_audit_split(seed=57001, n_calibration=500, n_test=1000, severity=2.0)
    shift = data.shift

    model = PhysicsRegressor().fit(data.fit.features, data.fit.energy)   # the baseline, first
    pred_cal = model.predict(data.calibration.features)
    pred_test = model.predict(data.test.features)

    parametric = GaussianResidualInterval(ALPHA, n_parameters=PhysicsRegressor.n_parameters)
    parametric.fit(data.calibration.energy, pred_cal)
    conformal = SplitConformal(ALPHA).calibrate(data.calibration.energy, pred_cal)

    w_cal = shift.likelihood_ratio(data.calibration.mass, data.calibration.headwind)
    w_test = shift.likelihood_ratio(data.test.mass, data.test.headwind)
    weighted = WeightedConformal(ALPHA).calibrate(data.calibration.energy, pred_cal, w_cal)

    print(shift.describe())
    bound = split_conformal_coverage_bound(conformal.n_calibration, ALPHA)
    print(f"finite-sample bound: rank {bound.rank}, exact {bound.exact:.12f}")
    print(f"sigma_hat {parametric.sigma:.6f} Wh, conformal quantile {conformal.quantile:.6f} Wh")
    print(f"weight sum {weighted.weight_sum:.4f}, ESS {effective_sample_size(w_cal):.2f} of 500")
    header = f"{'method':12s} {'coverage':>9s} {'width_Wh':>9s} {'infinite':>9s}"
    print(header)
    print("-" * len(header))
    for name, interval in (
        ("parametric", parametric.interval(pred_test)),
        ("split", conformal.interval(pred_test)),
        ("weighted", weighted.interval(pred_test, w_test)),
    ):
        finite = np.isfinite(interval.width)
        print(
            f"{name:12s} {interval.covers(data.test.energy).mean():9.5f} "
            f"{np.mean(interval.width[finite]):9.5f} {np.mean(~finite):9.5f}"
        )


if __name__ == "__main__":
    main()
