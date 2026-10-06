"""Validation V3: the three non-learned policies against feedback delay.

Checks:

V3.1  Invariant: the clairvoyant upper bound's goodput is >= every causal
      policy's goodput on the *same* sample path, at every delay. A violation
      would be a defect in the accounting, not an interesting result.
V3.2  Invariant: the clairvoyant bound is independent of the feedback delay,
      because it never reads the feedback. Any variation is a bug.
V3.3  Goodput, outage and the two kinds of mis-selection against feedback delay,
      for fixed margin and hysteresis, with the standard error over seeds so a
      reader can see whether two policies are separated at all.
V3.4  The same against channel correlation time at fixed delay: the thing that
      matters is tau / tau_c, and this shows it.
V3.5  The chatter-against-outage trade of the hysteresis dead band: switch rate
      and outage against dead-band width.
V3.6  Margin sweep for the fixed-margin policy: the best margin is a function of
      delay, which is why a single number cannot be published as "the" margin.

Runtime: about 100 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from acmpilot.channel import ChannelConfig  # noqa: E402
from acmpilot.modcod import ModcodTable, measure_thresholds  # noqa: E402
from acmpilot.policy import (  # noqa: E402
    ClairvoyantUpperBound,
    FixedMargin,
    ThresholdHysteresis,
    baseline_policies,
)
from acmpilot.simulate import run_episode, sweep_delay  # noqa: E402

TABLE_JSON = HERE / "modcod_thresholds.json"
N_SLOTS = 20_000
SEEDS = (1, 2, 3, 4, 5)
TAUS_MS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)


def _load_table() -> ModcodTable:
    if TABLE_JSON.exists():
        print("  MODCOD table loaded from validation/modcod_thresholds.json")
        return ModcodTable.load_json(TABLE_JSON)
    print("  MODCOD table measured in-process (run validate_modcod_thresholds.py first)")
    table, _ = measure_thresholds()
    return table


def main() -> int:
    print("V3 POLICY VALIDATION -- acmpilot")
    print("script: validation/validate_policies.py")
    table = _load_table()
    config = ChannelConfig(slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0)
    print(
        f"  channel: lognormal, slot = {config.slot_s * 1e3:g} ms, "
        f"tau_c = {config.tau_c_s * 1e3:g} ms, sigma_I^2 = {config.sigma_i2:g}, "
        f"SNR at mean irradiance = {config.mean_snr_db:g} dB"
    )
    print(f"  {N_SLOTS} slots per episode, {len(SEEDS)} seeds, paired across policies")
    print()

    policies = baseline_policies(margin_db=3.0, up_margin_db=2.0, down_margin_db=0.5)

    # ---- V3.1 / V3.2 invariants ---------------------------------------------
    print("V3.1 and V3.2 invariants: clairvoyant dominates, and does not see the delay")
    print(
        f"{'tau_ms':>7} {'clair_G':>9} {'fixed_G':>9} {'hyst_G':>9} "
        f"{'clair>=fixed':>13} {'clair>=hyst':>12}"
    )
    clair_values = []
    all_ok = True
    for tau_ms in TAUS_MS:
        _, results = run_episode(
            table, config, policies, n_slots=N_SLOTS, seed=1, tau_s=tau_ms * 1e-3
        )
        by_causal = {r.causal: r for r in results}
        clair = by_causal[False].accounting.goodput_bit_per_symbol
        causal = [r for r in results if r.causal]
        fixed_g = causal[0].accounting.goodput_bit_per_symbol
        hyst_g = causal[1].accounting.goodput_bit_per_symbol
        ok_f, ok_h = clair >= fixed_g, clair >= hyst_g
        all_ok &= ok_f and ok_h
        clair_values.append(clair)
        print(
            f"{tau_ms:7.1f} {clair:9.4f} {fixed_g:9.4f} {hyst_g:9.4f} "
            f"{'PASS' if ok_f else 'FAIL':>13} {'PASS' if ok_h else 'FAIL':>12}"
        )
    spread = float(np.ptp(np.array(clair_values)))
    print(f"  V3.1 all dominance checks: {'PASS' if all_ok else 'FAIL'}")
    print(
        f"  V3.2 clairvoyant goodput spread across delays: {spread:.3e} bit/symbol "
        f"({'PASS, exactly invariant' if spread == 0.0 else 'FAIL, should be 0'})"
    )
    print()

    # ---- V3.3 delay sweep ----------------------------------------------------
    print("V3.3 goodput, outage and the two mis-selection kinds against feedback delay")
    rows = sweep_delay(
        table, config, policies, tau_list_s=[t * 1e-3 for t in TAUS_MS],
        n_slots=N_SLOTS, seeds=SEEDS,
    )
    print(
        f"{'tau_ms':>7} {'policy':>34} {'goodput':>8} {'sem':>8} {'eff':>6} "
        f"{'outage':>8} {'unavoid':>8} {'aggr_avoid':>11} {'conserv':>8} "
        f"{'wasted':>8} {'lost':>8} {'switch':>7}"
    )
    for row in rows:
        print(
            f"{row['tau_s'] * 1e3:7.1f} {row['policy'][:34]:>34} "
            f"{row['goodput_bit_per_symbol']:8.4f} {row['goodput_sem']:8.5f} "
            f"{row['efficiency']:6.4f} {row['outage_fraction']:8.5f} "
            f"{row['unavoidable_outage_fraction']:8.5f} "
            f"{row['avoidable_outage_fraction']:11.5f} "
            f"{row['conservative_fraction']:8.4f} "
            f"{row['wasted_bit_per_symbol']:8.4f} {row['lost_bit_per_symbol']:8.4f} "
            f"{row['switch_rate_per_slot']:7.4f}"
        )
    print("  aggressive mis-selection equals outage when the ladder is monotone;")
    print("  'unavoid' is the part no selection rule could have prevented.")
    print()

    # ---- V3.4 correlation-time sweep ----------------------------------------
    print("V3.4 the ratio tau / tau_c is what matters: fixed tau = 10 ms, tau_c swept")
    print(
        f"{'tau_c_ms':>9} {'tau/tau_c':>10} {'policy':>34} {'goodput':>8} "
        f"{'eff':>6} {'outage':>8} {'conserv':>8}"
    )
    for tau_c_ms in (2.5, 5.0, 10.0, 20.0, 40.0):
        cfg = ChannelConfig(
            slot_s=1e-3, tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0
        )
        sub = sweep_delay(
            table, cfg, policies, tau_list_s=[10e-3], n_slots=N_SLOTS, seeds=SEEDS
        )
        for row in sub:
            print(
                f"{tau_c_ms:9.1f} {10.0 / tau_c_ms:10.3f} {row['policy'][:34]:>34} "
                f"{row['goodput_bit_per_symbol']:8.4f} {row['efficiency']:6.4f} "
                f"{row['outage_fraction']:8.5f} {row['conservative_fraction']:8.4f}"
            )
    print()

    # ---- V3.5 dead-band trade ----------------------------------------------
    print("V3.5 hysteresis dead band: the chatter-against-outage trade, tau = 10 ms")
    print(
        f"{'up_dB':>6} {'down_dB':>8} {'dead_dB':>8} {'goodput':>8} {'eff':>6} "
        f"{'outage':>8} {'conserv':>8} {'switch':>7}"
    )
    for up_db, down_db in (
        (0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5), (4.0, 0.5), (6.0, 0.5),
        (4.0, 2.0), (6.0, 4.0),
    ):
        pol = (ThresholdHysteresis(up_margin_db=up_db, down_margin_db=down_db),)
        sub = sweep_delay(
            table, config, pol, tau_list_s=[10e-3], n_slots=N_SLOTS, seeds=SEEDS
        )[0]
        print(
            f"{up_db:6.1f} {down_db:8.1f} {up_db - down_db:8.1f} "
            f"{sub['goodput_bit_per_symbol']:8.4f} {sub['efficiency']:6.4f} "
            f"{sub['outage_fraction']:8.5f} {sub['conservative_fraction']:8.4f} "
            f"{sub['switch_rate_per_slot']:7.4f}"
        )
    print()

    # ---- V3.6 margin sweep --------------------------------------------------
    print("V3.6 best fixed margin is a function of the delay, so no single margin wins")
    margins = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    print(f"{'tau_ms':>7} " + " ".join(f"{m:7.1f}" for m in margins) + "   best_dB")
    for tau_ms in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        goodputs = []
        for margin in margins:
            sub = sweep_delay(
                table, config, (FixedMargin(margin_db=margin),),
                tau_list_s=[tau_ms * 1e-3], n_slots=N_SLOTS, seeds=SEEDS,
            )[0]
            goodputs.append(sub["goodput_bit_per_symbol"])
        best = margins[int(np.argmax(goodputs))]
        print(
            f"{tau_ms:7.1f} " + " ".join(f"{g:7.4f}" for g in goodputs)
            + f"   {best:6.1f}"
        )
    print("  At zero delay the best margin is 0 dB, as it must be. The optimum moves")
    print("  off zero as soon as the delay is non-zero and then settles; the measured")
    print("  best-margin column above is the evidence, and the shipped 3 dB default is")
    print("  NOT the optimum on this channel. Every comparison against a learned")
    print("  predictor therefore uses the margin tuned on held-out seeds, not 3 dB.")
    print()

    # ---- headline numbers ----------------------------------------------------
    print("V3 headline, tau = 10 ms, tau_c = 10 ms, averaged over 5 seeds")
    sub = sweep_delay(
        table, config, (*policies, ClairvoyantUpperBound()), tau_list_s=[10e-3],
        n_slots=N_SLOTS, seeds=SEEDS,
    )
    seen = set()
    for row in sub:
        if row["policy"] in seen:
            continue
        seen.add(row["policy"])
        print(
            f"  {row['policy'][:52]:52s} goodput={row['goodput_bit_per_symbol']:.4f}"
            f" +/-{row['goodput_sem']:.4f} efficiency={row['efficiency']:.4f}"
        )
    print()
    print("V3 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
