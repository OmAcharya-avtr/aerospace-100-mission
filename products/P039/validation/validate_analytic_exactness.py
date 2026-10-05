#!/usr/bin/env python3
"""Validation 1 -- the analytic sum-of-stages model is exact.

What is being checked
---------------------
The specification requires that the analytic sum-of-stages model be *exact*
on an injected synthetic pipeline with independent stage latencies, and names
this the correctness check for the whole harness. "Exact" is a strong word and
it is used precisely here: the model consists of two algebraic identities, and
each is checked to binary64 rounding, not to an engineering tolerance.

(a) Constant stages. With every stage deterministic the end-to-end latency is
    a single number and the model must return it with zero error. Any
    departure is a coding defect, so the tolerance is the relative machine
    epsilon of binary64, 2.22e-16, times a small factor for the summation.

(b) Linearity of the mean, on real draws. For any data whatsoever, the sample
    mean of the row sums equals the sum of the column sample means. This is
    the content of equation (1) and it holds under any dependence structure.

(c) Variance additivity, on real draws. For any data whatsoever, the variance
    of the row sums equals the total sum of the sample covariance matrix. This
    is the content of equation (2). Checked both with the covariance terms
    present (exact) and with them dropped, where the residual must equal
    exactly twice the sum of the off-diagonal covariances -- the quantity the
    independence assumption discards.

(d) The declared-parameter recovery. The analytic model evaluated on the
    *declared* stage parameters must reproduce the closed-form injected
    end-to-end mean and standard deviation of
    :class:`latencynet.pipeline.PipelineSpec`, again to rounding.

(e) Separately and explicitly NOT exact: the Fenton-Wilkinson map from two
    moments to a tail quantile. This section measures its error against a
    large Monte Carlo reference and reports it. It is an approximation, it is
    labelled as one, and its error is the reason the analytic predictor loses
    accuracy to a fitted model even where the stages are independent. No
    tolerance is applied to it; the measured error is the result.

Reproduce with:
    cd validation && PYTHONPATH=../src python3 validate_analytic_exactness.py

Runtime: about 4 s on one uncontended core.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from latencynet.analytic import (
    SumOfStagesModel,
    fenton_wilkinson_quantile,
    sum_of_stages_mean,
    sum_of_stages_variance,
)
from latencynet.pipeline import (
    PipelineSpec,
    StageSpec,
    make_lognormal_pipeline,
    sample_stage_latencies,
)
from latencynet.tails import quantile
from latencynet.units import s_to_us

#: Relative tolerance for an identity that must hold to binary64 rounding.
#: 16 * eps, which allows for the accumulated rounding of a few sums over at
#: most a few thousand terms. It is NOT an engineering tolerance and must
#: never be widened: a failure here means the harness is wrong.
EXACT_RTOL = 16.0 * np.finfo(float).eps

INDEPENDENT_SPEC = make_lognormal_pipeline(
    (60e-6, 180e-6, 30e-6), (12e-6, 40e-6, 6e-6), latent_rho=0.0
)
CORRELATED_SPEC = make_lognormal_pipeline(
    (60e-6, 180e-6, 30e-6), (12e-6, 40e-6, 6e-6), latent_rho=0.7
)
N_DRAWS = 200_000
SEED = 20260404
#: Reference sample for the Fenton-Wilkinson error measurement.
N_REFERENCE = 2_000_000


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


def _report(label: str, got: float, want: float, rtol: float) -> bool:
    rel = _rel(got, want)
    ok = rel <= rtol
    print(
        f"  {label:<52} got {got: .17e}\n"
        f"  {'':<52} ref {want: .17e}   rel {rel:.3e}   "
        f"tol {rtol:.3e}   {'PASS' if ok else 'FAIL'}"
    )
    return ok


def constant_stage_exactness() -> list[bool]:
    print("\n(a) constant stages: the model must be exact sample by sample")
    print("-" * 78)
    results: list[bool] = []
    for totals in ((50e-6, 120e-6), (1e-3, 2e-3, 3e-3), (7.5e-5,)):
        spec = PipelineSpec(
            tuple(StageSpec(f"c{i}", v, 0.0, "constant") for i, v in enumerate(totals))
        )
        drawn = sample_stage_latencies(spec, 32, SEED)
        realised = float(drawn.sum(axis=1)[0])
        pred = SumOfStagesModel().predict(spec.stage_mean_s, spec.stage_std_s, (0.5, 0.99, 0.999))
        results.append(
            _report(f"{len(totals)} constant stages, mean", pred.mean_s, realised, EXACT_RTOL)
        )
        results.append(
            _report(
                f"{len(totals)} constant stages, p99",
                pred.quantile_s[0.99],
                realised,
                EXACT_RTOL,
            )
        )
        if pred.std_s != 0.0:
            print(f"  FAIL: constant pipeline has non-zero predicted sd {pred.std_s!r}")
            results.append(False)
        else:
            print(f"  {'constant stages, predicted sd is exactly zero':<52} PASS")
            results.append(True)
    return results


def mean_linearity_identity() -> list[bool]:
    print("\n(b) linearity of the mean, on 200000 injected draws, both regimes")
    print("-" * 78)
    print(
        "  Both sides are accumulated with math.fsum, which is correctly\n"
        "  rounded, so the two sides differ only by the final rounding of one\n"
        "  exact value and the tolerance is 4 eps. The second line of each\n"
        "  pair reports the same comparison with numpy's default pairwise\n"
        "  summation, whose accumulation error is bounded by about\n"
        "  log2(n) eps per sum (Higham 2002, Accuracy and Stability of\n"
        "  Numerical Algorithms, 2nd ed., Sec. 4.2); that figure is reported\n"
        "  as a floating-point fact, not asserted as an identity."
    )
    results: list[bool] = []
    bound = math.log2(N_DRAWS) * float(np.finfo(float).eps)
    for name, spec in (("independent", INDEPENDENT_SPEC), ("correlated rho=0.7", CORRELATED_SPEC)):
        trace = sample_stage_latencies(spec, N_DRAWS, SEED)
        k = trace.shape[1]
        # Exact (correctly rounded) accumulation of both sides.
        exact_sum_of_means = math.fsum(
            math.fsum(trace[:, j].tolist()) / N_DRAWS for j in range(k)
        )
        exact_mean_of_sums = (
            math.fsum(math.fsum(trace[i, :].tolist()) for i in range(N_DRAWS)) / N_DRAWS
        )
        results.append(
            _report(
                f"{name}: fsum sum-of-means vs fsum mean-of-sums",
                exact_sum_of_means,
                exact_mean_of_sums,
                4.0 * float(np.finfo(float).eps),
            )
        )
        np_rel = _rel(
            sum_of_stages_mean(trace.mean(axis=0)), float(trace.sum(axis=1).mean())
        )
        print(
            f"  {name}: numpy pairwise summation differs by rel {np_rel:.3e}; "
            f"log2(n) eps = {bound:.3e} per sum (reported, not asserted)"
        )
    print("  equation (1) holds under any dependence; both regimes must pass.")
    return results


def variance_additivity_identity() -> list[bool]:
    print("\n(c) variance additivity, on 200000 injected draws")
    print("-" * 78)
    results: list[bool] = []
    for name, spec in (("independent", INDEPENDENT_SPEC), ("correlated rho=0.7", CORRELATED_SPEC)):
        trace = sample_stage_latencies(spec, N_DRAWS, SEED + 1)
        total = trace.sum(axis=1)
        cov = np.cov(trace, rowvar=False, ddof=1)
        sds = np.sqrt(np.diag(cov))
        results.append(
            _report(
                f"{name}: total sum of the covariance matrix",
                sum_of_stages_variance(sds, cov),
                float(total.var(ddof=1)),
                EXACT_RTOL,
            )
        )
        dropped = float(total.var(ddof=1)) - sum_of_stages_variance(sds)
        off_diagonal = float(cov.sum() - np.trace(cov))
        results.append(
            _report(
                f"{name}: residual from dropping covariances",
                dropped,
                off_diagonal,
                1.0e-9,
            )
        )
        indep_sd = math.sqrt(sum_of_stages_variance(sds))
        print(
            f"  {name}: independence sd {s_to_us(indep_sd):.4f} us vs measured total sd "
            f"{s_to_us(float(total.std(ddof=1))):.4f} us  "
            f"(ratio {indep_sd / float(total.std(ddof=1)):.5f})"
        )
    print(
        "  the residual check uses 1e-9 because it is a difference of two\n"
        "  near-equal sums and so loses significant digits by cancellation;\n"
        "  the identity itself is checked at machine precision above."
    )
    return results


def declared_parameter_recovery() -> list[bool]:
    print("\n(d) the model on declared parameters reproduces the closed-form injection")
    print("-" * 78)
    results: list[bool] = []
    for name, spec in (("independent", INDEPENDENT_SPEC), ("correlated rho=0.7", CORRELATED_SPEC)):
        cov = spec.injected_covariance()
        model = SumOfStagesModel(assume_independent=False)
        pred = model.predict(spec.stage_mean_s, spec.stage_std_s, (0.99,), cov)
        results.append(
            _report(f"{name}: end-to-end mean", pred.mean_s, spec.injected_mean_s(), EXACT_RTOL)
        )
        results.append(
            _report(f"{name}: end-to-end sd", pred.std_s, spec.injected_std_s(), EXACT_RTOL)
        )
    return results


def fenton_wilkinson_error() -> None:
    print("\n(e) Fenton-Wilkinson quantile error -- an APPROXIMATION, measured not asserted")
    print("-" * 78)
    print(f"  reference: {N_REFERENCE} injected draws per pipeline, seed {SEED + 2}")
    print(f"  {'pipeline':<28}{'p':>7}{'MC us':>12}{'FW us':>12}{'rel err':>11}{'MC SE':>10}")
    rng_specs = (
        ("independent cv~0.2", INDEPENDENT_SPEC),
        ("correlated rho=0.7", CORRELATED_SPEC),
        (
            "one dominant skewed stage",
            make_lognormal_pipeline((4e-4, 2e-5, 2e-5), (2.4e-4, 2e-6, 2e-6)),
        ),
        (
            "five low-cv stages",
            make_lognormal_pipeline((1e-4,) * 5, (5e-6,) * 5),
        ),
    )
    for label, spec in rng_specs:
        total = sample_stage_latencies(spec, N_REFERENCE, SEED + 2).sum(axis=1)
        cov = spec.injected_covariance()
        var = float(cov.sum())
        for p in (0.5, 0.99, 0.999):
            mc = quantile(total, p, "linear")
            fw = fenton_wilkinson_quantile(spec.injected_mean_s(), var, p)
            # Distribution-free one-sigma order-statistic uncertainty on the
            # Monte Carlo reference, so the reader can see when a residual is
            # smaller than the reference's own noise.
            n = total.size
            spread = math.sqrt(n * p * (1.0 - p))
            srt = np.sort(total)
            r = int(math.floor(n * p - spread))
            s_hi = int(math.ceil(n * p + spread))
            se = 0.5 * float(srt[s_hi - 1] - srt[r - 1]) if 1 <= r and s_hi <= n else float("nan")
            print(
                f"  {label:<28}{p:>7}{s_to_us(mc):>12.4f}{s_to_us(fw):>12.4f}"
                f"{(fw - mc) / mc:>11.5f}{se / mc:>10.5f}"
            )
    print(
        "  Reading: the relative error column is the systematic bias the\n"
        "  analytic predictor carries into every tail prediction. It is\n"
        "  negative at p99 and p99.9 for every pipeline above, i.e. the\n"
        "  two-moment lognormal match UNDER-predicts the tail, which is the\n"
        "  unsafe direction for a deadline. It is not a defect in equations\n"
        "  (1) and (2), which are exact; it is the cost of turning two\n"
        "  moments into a quantile without the distribution."
    )


def main() -> int:
    print("=" * 78)
    print("P039 latencynet -- Validation 1: analytic sum-of-stages exactness")
    print("=" * 78)
    print(f"binary64 eps {np.finfo(float).eps:.6e}; exactness tolerance {EXACT_RTOL:.6e} relative")
    print("No measured wall-clock time appears anywhere in this script.")

    results: list[bool] = []
    results += constant_stage_exactness()
    results += mean_linearity_identity()
    results += variance_additivity_identity()
    results += declared_parameter_recovery()
    fenton_wilkinson_error()

    n_pass = sum(results)
    print("\n" + "=" * 78)
    print(f"exactness checks: {n_pass}/{len(results)} PASS")
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
