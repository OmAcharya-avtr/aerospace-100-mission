"""Validation 5: the learned LLR corrector against every analytic alternative.

The comparison, and what each method is allowed to know
-------------------------------------------------------
======================= ==================================================
method                  information used
======================= ==================================================
known_csi               the **true** ``h``. Unreachable upper bound.
csi_aware_exact         ``(y, h_hat)`` plus the error-model parameters.
                        Bayes-optimal given the observables, analytic.
csi_aware_maxlog        the same, with the posterior average replaced by a
                        max. The realistic baseline named in the spec.
plugin                  ``(y, h_hat)`` treated as if ``h_hat = h``.
plugin_scaled           plug-in times ``alpha``, ``alpha`` tuned on the
                        tuning split.
plugin_clipped          plug-in saturated at ``L_max``, tuned likewise.
plugin_scaled_clipped   both, tuned jointly.
learned                 ``(y, h_hat)`` plus **pilot bits**. It is never told
                        the bias, the jitter or the fading parameters.
======================= ==================================================

Splits are disjoint by construction (``softdecode.datasets``): the forest is
fitted on the train split, every free parameter including the forest's own
output clip level is chosen on the tune split, and every number reported below
comes from the report split, which neither of the other two touched.

Honest expectation, stated before the numbers
---------------------------------------------
``csi_aware_exact`` is the posterior-mean likelihood ratio given ``(y,
h_hat)``; no function of ``(y, h_hat)`` can beat it under the stated error
model. The learned corrector is therefore competing for second place, and the
interesting questions are (a) how close it gets without being told the error
model and (b) whether the two one-parameter non-learned fixes get there first.

Runtime: about 150 s on one shared core.
"""

from __future__ import annotations

import numpy as np

from softdecode.channel import LognormalFading
from softdecode.corrector import (
    FEATURE_NAMES,
    LlrCorrector,
    build_features,
    gmi_of,
    tune_clip,
    tune_scale_and_clip,
)
from softdecode.csi import MultiplicativeCsiError, StaleCsiError
from softdecode.datasets import TRAIN_EBN0_DB, make_split, make_training_set
from softdecode.ldpc import make_regular_ldpc
from softdecode.llr import clip_llr
from softdecode.metrics import llr_error
from softdecode.simulate import decode_ber, demap_ook

TRAIN_BLOCKS = 500
TUNE_BLOCKS = 600
REPORT_BLOCKS = 1500
REPORT_EBN0_DB = 8.0


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def evaluate(code, fading, error, label: str) -> dict[str, tuple[float, float, float]]:
    banner(f"Case: {label}")
    train = make_training_set(
        code, fading, error, blocks_per_point=TRAIN_BLOCKS, ebn0_db=TRAIN_EBN0_DB
    )
    print(f"  train split: {train.n_samples} channel-bit samples over "
          f"Eb/N0 {train.ebn0_db} dB, {TRAIN_BLOCKS} blocks per point, seed {train.seed}")
    corrector = LlrCorrector()
    corrector.fit(train.features, train.bits)
    print(f"  forest: {corrector.model.n_estimators} trees, max_depth "
          f"{corrector.model.max_depth}, min_samples_leaf "
          f"{corrector.model.min_samples_leaf}, seed {corrector.seed}")
    importances = dict(zip(FEATURE_NAMES, corrector.model.feature_importances_, strict=True))
    for name, value in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"    feature {name:<20} importance {value:.6f}")

    tune = make_split(code, REPORT_EBN0_DB, fading, error, TUNE_BLOCKS, "tune")
    tuned = tune_scale_and_clip(code, tune, fading, error)
    for key, value in tuned.items():
        print(f"  tuned on the tune split: {key:<22} {value}")
    tune_features = build_features(tune, demap_ook(tune, "plugin", fading, error))
    learned_clip = tune_clip(code, tune, corrector, tune_features)
    print(f"  tuned on the tune split: {'learned output clip':<22} {learned_clip}")

    report = make_split(code, REPORT_EBN0_DB, fading, error, REPORT_BLOCKS, "report")
    plugin = demap_ook(report, "plugin", fading, error)
    truth = demap_ook(report, "known_csi", fading, error)
    aware = demap_ook(report, "csi_aware_exact", fading, error)
    features = build_features(report, plugin)
    learned, uncertainty = corrector.predict(features, with_uncertainty=True)
    learned = learned.reshape(report.codeword.shape)

    candidates = {
        "known_csi (upper bound)": truth,
        "csi_aware_exact": aware,
        "csi_aware_maxlog": demap_ook(report, "csi_aware_maxlog", fading, error),
        "plugin": plugin,
        "plugin_scaled": plugin * tuned["plugin_scaled"].scale,
        "plugin_clipped": clip_llr(plugin, tuned["plugin_clipped"].clip),
        "plugin_scaled_clipped": clip_llr(
            plugin * tuned["plugin_scaled_clipped"].scale,
            tuned["plugin_scaled_clipped"].clip,
        ),
        "learned": learned,
    }
    print(f"\n  report split: {REPORT_BLOCKS} blocks "
          f"({REPORT_BLOCKS * code.dimension} information bits) at "
          f"Eb/N0 {REPORT_EBN0_DB:.2f} dB")
    print(f"  {'method':<24} {'decoded BER':>14} {'binom se':>11} {'BER / best':>11} "
          f"{'GMI':>9} {'rmse vs truth':>14}")
    results: dict[str, tuple[float, float, float]] = {}
    best = min(decode_ber(code, report, v).rate for v in candidates.values())
    for name, llr in candidates.items():
        ber = decode_ber(code, report, llr)
        gmi = gmi_of(llr, report)
        rmse = llr_error(llr, truth).rmse
        results[name] = (ber.rate, gmi, rmse)
        ratio = ber.rate / best if best > 0 else float("inf")
        print(f"  {name:<24} {ber.rate:14.6e} {ber.standard_error:11.3e} {ratio:11.3f} "
              f"{gmi:+9.5f} {rmse:14.5f}")

    banner_small("  Uncertainty output")
    print("    The forest reports the standard deviation of its per-tree LLRs,")
    print("    saturated at the same clip level as the mean, so the reference it is")
    print("    compared against is clipped the same way; comparing an output capped")
    print("    at L_max against an uncapped analytic LLR measures the cap, not the")
    print("    model's uncertainty.")
    reference = clip_llr(aware, corrector.clip).ravel()
    residual = np.abs(learned.ravel() - reference)
    errors = (learned.ravel() < 0.0).astype(np.int8) != report.codeword.ravel()
    order = np.argsort(uncertainty)
    quintiles = np.array_split(order, 5)
    relative = uncertainty / (1.0 + np.abs(learned.ravel()))
    print(f"    Pearson correlation of sigma with |learned - clipped reference|: "
          f"{np.corrcoef(uncertainty, residual)[0, 1]:.6f}")
    print(f"    Pearson correlation of sigma with the hard-decision error indicator: "
          f"{np.corrcoef(uncertainty, errors.astype(float))[0, 1]:.6f}")
    print(f"    Pearson correlation of sigma/(1+|L|) with that indicator:          "
          f"{np.corrcoef(relative, errors.astype(float))[0, 1]:.6f}")
    magnitude = np.abs(learned.ravel())
    print(f"    Pearson correlation of |L| with that indicator:                    "
          f"{np.corrcoef(magnitude, errors.astype(float))[0, 1]:.6f}")
    rel_quintiles = np.array_split(np.argsort(relative), 5)
    mag_quintiles = np.array_split(np.argsort(magnitude), 5)
    print(f"    {'quintile':>9} {'mean sigma':>12} {'mean |error|':>13} "
          f"{'err by sigma':>14} {'err by sigma/(1+|L|)':>22} {'err by |L|':>12}")
    by_sigma, by_rel, by_mag = [], [], []
    for i, (idx, ridx, midx) in enumerate(
        zip(quintiles, rel_quintiles, mag_quintiles, strict=True)
    ):
        by_sigma.append(float(errors[idx].mean()))
        by_rel.append(float(errors[ridx].mean()))
        by_mag.append(float(errors[midx].mean()))
        print(f"    {i + 1:9d} {uncertainty[idx].mean():12.6f} "
              f"{residual[idx].mean():13.6f} {by_sigma[-1]:14.6f} "
              f"{by_rel[-1]:22.6f} {by_mag[-1]:12.6f}")
    print(f"    sigma range {uncertainty.min():.6f} .. {uncertainty.max():.6f}, "
          f"mean {uncertainty.mean():.6f}")

    def monotone(values: list[float]) -> str:
        if all(b >= a for a, b in zip(values, values[1:], strict=False)):
            return "monotonically increasing"
        if all(b <= a for a, b in zip(values, values[1:], strict=False)):
            return "monotonically decreasing"
        return "NOT monotone"

    print("    Reading, from the three right-hand columns:")
    print(f"      error rate against sigma              {monotone(by_sigma)}")
    print(f"      error rate against sigma/(1+|L|)      {monotone(by_rel)}")
    print(f"      error rate against |L|                {monotone(by_mag)}")
    print("    The dispersion measures how much the trees disagree, and it grows")
    print("    with the magnitude of the LLR it is attached to, so as a standalone")
    print("    risk score it points the wrong way. Normalising by 1 + |L| removes")
    print("    the scale but does not produce a monotone risk ordering either. The")
    print("    quantity that does order risk here is |L| itself. All three are")
    print("    published as measured; none is a calibrated interval, the dispersion")
    print("    is a model-disagreement diagnostic, and the README says so.")
    return results


def banner_small(text: str) -> None:
    print()
    print(text)
    print("  " + "-" * (len(text) - 2))


def verdicts(results: dict[str, tuple[float, float, float]], label: str) -> None:
    banner(f"Verdict: {label}")
    learned = results["learned"][0]
    for name in (
        "known_csi (upper bound)",
        "csi_aware_exact",
        "csi_aware_maxlog",
        "plugin",
        "plugin_scaled",
        "plugin_clipped",
        "plugin_scaled_clipped",
    ):
        other = results[name][0]
        if other == 0.0 and learned == 0.0:
            verdict = "both zero errors"
        elif other == 0.0:
            verdict = "the other method made no errors"
        else:
            ratio = learned / other
            verdict = (
                f"learned is {1 / ratio:.3f}x better" if ratio < 1
                else f"learned is {ratio:.3f}x worse"
            )
        print(f"  learned {learned:.6e} vs {name:<24} {other:.6e}   {verdict}")


def main() -> int:
    print("softdecode validation 5: learned LLR corrector against the analytic methods")
    print("raw output committed as validation/corrector_output.txt")
    code = make_regular_ldpc()
    fading = LognormalFading(0.3)
    print(f"LDPC n={code.length} k={code.dimension} rate={code.rate:.6f}; "
          f"lognormal sigma_I2={fading.scintillation_index}")
    print("PyTorch is unavailable in the build container, so the learned model is a")
    print("scikit-learn random forest, not a neural network.")

    over = MultiplicativeCsiError(2.0, 1.0)
    results_over = evaluate(
        code, fading, over, "over-estimating CSI, bias +2.00 dB, jitter 1.00 dB"
    )
    verdicts(results_over, "over-estimating CSI")

    under = MultiplicativeCsiError(-2.0, 1.0)
    results_under = evaluate(
        code, fading, under, "under-estimating CSI, bias -2.00 dB, jitter 1.00 dB"
    )
    verdicts(results_under, "under-estimating CSI")

    stale = StaleCsiError(0.8)
    results_stale = evaluate(code, fading, stale, "stale CSI, log-amplitude correlation 0.80")
    verdicts(results_stale, "stale CSI")

    banner("Summary table, decoded BER on the report split")
    names = list(results_over)
    print(f"  {'method':<24} {'bias +2 dB':>14} {'bias -2 dB':>14} {'stale rho=0.8':>14}")
    for name in names:
        print(f"  {name:<24} {results_over[name][0]:14.6e} "
              f"{results_under[name][0]:14.6e} {results_stale[name][0]:14.6e}")
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
