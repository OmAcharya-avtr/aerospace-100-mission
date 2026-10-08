"""Validate the claim that this monitor cannot attribute a residual to a cause.

Two worlds, one shared noise realisation:

World A, asset fault
    The asset's sensor develops a bias ``b`` at sample ``onset``; the twin's
    declared measurement offset stays 0.
World B, twin invalidation
    The asset is unchanged and healthy; the twin's declared measurement offset
    is revised to ``-b`` at sample ``onset``.

Both give ``e_k = C (x_k - x_hat_k) + b + v_k``, so the residual streams are
algebraically identical and the difference measured below is floating-point
rounding from the different evaluation order.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate import (  # noqa: E402
    DetectorSpec,
    calibrate_threshold,
    in_control_streams,
    max_absolute_difference,
    paired_streams,
)


def main() -> int:
    print("=== paired-world simulation ===")
    pair = paired_streams(offset=0.02, onset=400, n_runs=40, n_samples=1200, seed=53400)
    diff = max_absolute_difference(pair)
    print(f"measurement offset b   {pair.offset:g} rad "
          f"(2 x the declared measurement noise sd of 0.01 rad)")
    print(f"onset                  sample {pair.onset}")
    print(f"batch                  {pair.asset_fault.shape[0]} runs x "
          f"{pair.asset_fault.shape[1]} samples")
    print(f"residual mean before   {pair.asset_fault[:, : pair.onset].mean():+.6f}")
    print(f"residual mean after    {pair.asset_fault[:, pair.onset + 100 :].mean():+.6f}")
    print(f"residual sd before     {pair.asset_fault[:, : pair.onset].std():.6f}")
    print()
    print(f"max |world A - world B| {diff:.4e}")
    print(f"typical |residual|      {np.abs(pair.asset_fault).mean():.4f}")
    print(f"relative difference     {diff / np.abs(pair.asset_fault).mean():.4e}")
    print(f"double-precision eps    {np.finfo(float).eps:.4e}")
    print(f"difference in ulps      about {diff / np.finfo(float).eps:.0f}")
    ok = diff < 1e-10
    print(f"CHECK identity          tolerance 1e-10   {'PASS' if ok else 'FAIL'}")
    print("NOTE  the difference is NOT zero and this package does not claim it is.")
    print("NOTE  World A adds b to the measurement and World B subtracts it from the")
    print("NOTE  prediction, so the two sums round differently. The claim is that the")
    print("NOTE  two streams are the same function of the same quantities, which the")
    print("NOTE  derivation establishes and this number corroborates.")

    print()
    print("=== every detector returns the same verdict in both worlds ===")
    bank = in_control_streams(n_runs=100, n_samples=2000, seed=53001)
    print(f"  {'detector':<32}{'max |stat_A - stat_B|':>22}{'alarm index A':>15}"
          f"{'alarm index B':>15}")
    results = [ok]
    for name in ("cusum", "ewma", "glr", "varcusum"):
        spec = DetectorSpec(name)
        threshold = calibrate_threshold(spec, bank, 1000.0).threshold
        sa = spec.statistic(pair.asset_fault)
        sb = spec.statistic(pair.twin_invalid)
        sdiff = float(np.max(np.abs(sa - sb)))
        from twininvalidate import first_alarm

        ia = first_alarm(sa, threshold)
        ib = first_alarm(sb, threshold)
        same = bool(np.array_equal(ia, ib))
        results.append(same)
        print(
            f"  {spec.label():<32}{sdiff:>22.3e}"
            f"{float(ia[ia >= 0].mean()) if (ia >= 0).any() else float('nan'):>15.1f}"
            f"{float(ib[ib >= 0].mean()) if (ib >= 0).any() else float('nan'):>15.1f}"
            f"   identical alarm indices: {same}"
        )

    print()
    print("CONCLUSION  an alarm from this package means the twin and the asset no")
    print("CONCLUSION  longer agree. It does not mean the asset is faulty and it does")
    print("CONCLUSION  not mean the twin is wrong. The two worlds above are physically")
    print("CONCLUSION  different -- one needs a sensor replaced, the other needs a")
    print("CONCLUSION  configuration rolled back -- and no function of the residual")
    print("CONCLUSION  can separate them, because the residual is identical. Deciding")
    print("CONCLUSION  which it is needs information this monitor is not given: a")
    print("CONCLUSION  second independently instrumented measurement path, a known")
    print("CONCLUSION  reference manoeuvre, or a maintenance record.")
    print()
    print(f"SUMMARY  {sum(results)}/{len(results)} checks PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
