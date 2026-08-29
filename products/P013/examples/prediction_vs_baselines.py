"""Example 2: learned model predictions with intervals vs both classical
closed-form baselines, across a sweep from weak to saturated turbulence.

Trains the default seeded model (a few seconds on 2 cores), synthesises one
noisy measurement per point on a truth sweep, and plots the learned model's
median + 90% prediction interval against the scintillometer-only and
DIMM-only closed-form inversions and the ground truth.

    python examples/prediction_vs_baselines.py

Writes ``screenshots/prediction_vs_baselines.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope.model import (  # noqa: E402
    DimmOnlyBaseline,
    ScintillometerWeakBaseline,
    train_default_model,
)
from turbscope.scintillometer import rytov_variance  # noqa: E402
from turbscope.synthetic import (  # noqa: E402
    SCINT_WAVELENGTH_M,
    WAVE_TYPE,
    Scenario,
    cn2_from_target_rytov,
    synthesize_measurement,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "prediction_vs_baselines.png"


def main() -> int:
    model, _ = train_default_model()
    scint_base = ScintillometerWeakBaseline()
    dimm_base = DimmOnlyBaseline()

    length = 800.0
    targets = np.geomspace(1e-3, 30.0, 26)
    rng = np.random.default_rng(2026)

    truth, med, lo, hi, scint_pred, dimm_pred = [], [], [], [], [], []
    for target in targets:
        cn2_true = cn2_from_target_rytov(float(target), length)
        sc = Scenario(cn2_path=cn2_true, path_length_m=length, rytov_variance_true=float(target))
        m = synthesize_measurement(sc, rng)
        pred = model.predict(m.sigma_i2_scint, m.var_long_dimm, m.var_trans_dimm, m.path_length_m)
        x = np.array(
            [[
                np.log10(m.sigma_i2_scint), np.log10(m.var_long_dimm),
                np.log10(m.var_trans_dimm), np.log10(m.path_length_m),
            ]]
        )
        truth.append(cn2_true)
        med.append(pred.cn2_path)
        lo.append(pred.cn2_lower)
        hi.append(pred.cn2_upper)
        scint_pred.append(10.0 ** scint_base.predict_log10_cn2(x)[0])
        dimm_pred.append(10.0 ** dimm_base.predict_log10_cn2(x)[0])

    rytov_true = np.array(
        [float(rytov_variance(t, length, SCINT_WAVELENGTH_M, WAVE_TYPE)) for t in truth]
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(rytov_true, truth, color="k", lw=2.0, label="true Cn2_path")
    ax.fill_between(
        rytov_true, lo, hi, color="tab:blue", alpha=0.20, label="TurbScope 90% interval"
    )
    ax.plot(rytov_true, med, color="tab:blue", lw=1.8, label="TurbScope learned median")
    ax.plot(
        rytov_true, scint_pred, color="tab:red", lw=1.4, ls="--",
        label="scintillometer weak baseline (mandated)",
    )
    ax.plot(rytov_true, dimm_pred, color="tab:green", lw=1.4, ls=":", label="DIMM-only baseline")
    ax.axvspan(1.0, 3.0, color="tab:orange", alpha=0.10, label="approx. saturation transition")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true Rytov variance $\sigma_R^2$ (weak $\to$ saturated)")
    ax.set_ylabel(r"$C_n^2$ [m$^{-2/3}$]")
    ax.set_title(
        "Learned model vs closed-form single-sensor baselines, one noisy draw per point\n"
        "Synthetic data -- see DATASET_CARD.md; not certified for operational flight use",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    rel_err_learned = np.abs(np.asarray(med) - np.asarray(truth)) / np.asarray(truth)
    rel_err_scint = np.abs(np.asarray(scint_pred) - np.asarray(truth)) / np.asarray(truth)
    sys.stdout.write(
        f"wrote {OUT}\n"
        f"median |rel err| learned model  : {np.median(rel_err_learned):.2%}\n"
        f"median |rel err| scint baseline : {np.median(rel_err_scint):.2%}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
