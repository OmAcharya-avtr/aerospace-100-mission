"""Validation V2: the MODCOD ladder and its measured thresholds.

Checks:

V2.1  Monte Carlo uncoded BER against the closed forms of
      ``src/acmpilot/modulation.py``. Exact comparison for BPSK and QPSK
      (equations (2)-(3)); approximate for 8PSK and 16QAM (equations (4)-(5)),
      where the discrepancy is reported and is *expected* to be large at low
      SNR because those forms are high-SNR approximations.
V2.2  The Reed-Solomon MDS properties ``d_min = n-k+1`` and ``t = floor((n-k)/2)``
      for every shipped code.
V2.3  The post-decoding combinatorics of ``acmpilot.coding`` equations (4)-(5)
      against a **direct Monte Carlo through the real modulator**: BPSK, where
      the i.i.d.-bit assumption holds exactly, and 16QAM with and without a bit
      interleaver, which measures what the ideal-interleaving assumption is
      worth.
V2.4  The threshold table itself, with its Monte Carlo uncertainty, written to
      ``validation/modcod_thresholds.json`` for the other scripts and examples
      to load.
V2.5  Ladder sanity: thresholds strictly increase with spectral efficiency, and
      no MODCOD is dominated (higher threshold *and* lower efficiency than
      another).

Runtime: about 75 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from acmpilot.coding import SHIPPED_CODES, ReedSolomonCode  # noqa: E402
from acmpilot.modcod import measure_thresholds  # noqa: E402
from acmpilot.modulation import (  # noqa: E402
    CONSTELLATION_NAMES,
    measure_ber,
    simulate_bit_errors,
    theoretical_ber,
)

TABLE_JSON = "validation/modcod_thresholds.json"


def _check_ber_vs_theory() -> None:
    print("V2.1 Monte Carlo uncoded BER vs closed form")
    print("  BPSK and QPSK closed forms are EXACT; 8PSK and 16QAM are high-SNR")
    print("  approximations and are expected to deviate at low SNR.")
    print(
        f"{'mod':>6} {'SNR_dB':>7} {'ber_mc':>11} {'ber_form':>11} {'n_bits':>9} "
        f"{'n_err':>7} {'z_score':>8} {'exact?':>7}"
    )
    rng = np.random.default_rng(991)
    exact = {"BPSK", "QPSK"}
    worst_exact_z = 0.0
    for name in CONSTELLATION_NAMES:
        for snr in (2.0, 6.0, 10.0, 14.0, 18.0):
            ber, n_err, n_bits = measure_ber(
                name, snr, rng=rng, target_errors=4000, max_symbols=600_000
            )
            form = float(theoretical_ber(name, snr))
            se = float(np.sqrt(max(form * (1.0 - form), 1e-300) / n_bits))
            z = (ber - form) / se if se > 0 else float("nan")
            if name in exact and n_err > 50:
                worst_exact_z = max(worst_exact_z, abs(z))
            print(
                f"{name:>6} {snr:7.1f} {ber:11.5e} {form:11.5e} {n_bits:9d} "
                f"{n_err:7d} {z:8.2f} {'yes' if name in exact else 'approx':>7}"
            )
    print(
        f"  worst |z| over the EXACT closed forms with > 50 observed errors: "
        f"{worst_exact_z:.2f} (3.0 would be a 3-sigma disagreement)"
    )
    print()


def _check_mds() -> None:
    print("V2.2 Reed-Solomon MDS properties, equation (1) of acmpilot.coding")
    print(f"{'code':>14} {'rate':>8} {'d_min':>6} {'n-k+1':>6} {'t':>4} {'(n-k)//2':>9} {'ok':>4}")
    for code in SHIPPED_CODES:
        ok = code.d_min == code.n - code.k + 1 and code.t == (code.n - code.k) // 2
        print(
            f"{code.label:>14} {code.rate:8.5f} {code.d_min:6d} "
            f"{code.n - code.k + 1:6d} {code.t:4d} {(code.n - code.k) // 2:9d} "
            f"{'PASS' if ok else 'FAIL':>4}"
        )
    print()


def _direct_codeword_mc(
    modulation: str,
    snr_db: float,
    code: ReedSolomonCode,
    n_codewords: int,
    *,
    interleave: bool,
    rng: np.random.Generator,
) -> dict[str, float]:
    """FER and output symbol error rate measured through the real modem."""
    m = code.bits_per_symbol
    bits_per_word = code.n * m
    from acmpilot.modulation import constellation

    bps = constellation(modulation).bits_per_symbol
    total_bits = n_codewords * bits_per_word
    n_symbols = int(np.ceil(total_bits / bps))
    errors = simulate_bit_errors(modulation, snr_db, n_symbols, rng=rng)[:total_bits]
    if interleave:
        errors = rng.permutation(errors)
    grid = errors.reshape(n_codewords, code.n, m)
    sym_err = grid.any(axis=2)
    counts = sym_err.sum(axis=1)
    failed = counts > code.t
    return {
        "p_b_meas": float(errors.mean()),
        "fer_meas": float(failed.mean()),
        "ser_out_meas": float((counts * failed).sum() / (n_codewords * code.n)),
        "n_codewords": float(n_codewords),
        "n_failed": float(failed.sum()),
    }


def _check_block_combinatorics() -> None:
    print("V2.3 post-decoding combinatorics vs direct Monte Carlo through the modem")
    print("  BPSK: one bit per modulation symbol, so the i.i.d.-bit assumption of")
    print("  acmpilot.coding equations (2)-(3) holds exactly and this is a clean")
    print("  test of the combinatorics.")
    rng = np.random.default_rng(2024)
    code = ReedSolomonCode(255, 223)
    header = (
        f"{'case':>26} {'SNR_dB':>7} {'p_b':>10} {'n_fail':>7} {'FER_mc':>9} "
        f"{'FER_form':>9} {'ratio':>7} {'SERo_mc':>10} {'SERo_form':>10} {'ratio':>7}"
    )
    print(header)
    for snr in (4.5, 4.8, 5.1):
        meas = _direct_codeword_mc(
            "BPSK", snr, code, 10_000, interleave=False, rng=rng
        )
        pb = meas["p_b_meas"]
        fer = float(code.frame_error_rate(pb))
        sero = float(code.output_symbol_error_rate(pb))
        print(
            f"{'BPSK, no interleaver':>26} {snr:7.2f} {pb:10.5f} "
            f"{meas['n_failed']:7.0f} "
            f"{meas['fer_meas']:9.5f} {fer:9.5f} {meas['fer_meas'] / fer:7.3f} "
            f"{meas['ser_out_meas']:10.3e} {sero:10.3e} "
            f"{meas['ser_out_meas'] / sero:7.3f}"
        )
    print()
    print("  16QAM: four bits per modulation symbol, so without a bit interleaver")
    print("  the bit errors inside one code symbol are correlated, more code symbols")
    print("  are hit than the i.i.d. model expects, and the model is therefore")
    print("  OPTIMISTIC -- the measured error rate is HIGHER than the formula. With a")
    print("  bit interleaver, which is what the model assumes, the agreement returns.")
    print("  The ratio column measures the cost of omitting the interleaver.")
    print(header)
    for snr in (13.5, 14.0, 14.5):
        for interleave in (False, True):
            meas = _direct_codeword_mc(
                "16QAM", snr, code, 10_000, interleave=interleave, rng=rng
            )
            pb = meas["p_b_meas"]
            fer = float(code.frame_error_rate(pb))
            sero = float(code.output_symbol_error_rate(pb))
            label = "16QAM, interleaved" if interleave else "16QAM, no interleaver"
            print(
                f"{label:>26} {snr:7.2f} {pb:10.5f} {meas['n_failed']:7.0f} "
                f"{meas['fer_meas']:9.5f} {fer:9.5f} {meas['fer_meas'] / fer:7.3f} "
                f"{meas['ser_out_meas']:10.3e} {sero:10.3e} "
                f"{meas['ser_out_meas'] / max(sero, 1e-300):7.3f}"
            )
    print()


def main() -> int:
    print("V2 MODCOD THRESHOLD VALIDATION -- acmpilot")
    print("script: validation/validate_modcod_thresholds.py")
    print()
    _check_ber_vs_theory()
    _check_mds()
    _check_block_combinatorics()

    print("V2.4 measured MODCOD threshold table")
    table, curves = measure_thresholds(seed=20261006)
    print(f"  target post-decoding BER: {table.target_ber:g}")
    print("  thresholds are measured by this script; no standard's table is used.")
    print(
        f"{'idx':>3} {'MODCOD':>22} {'mod_bits':>9} {'code_rate':>10} "
        f"{'eta_bit_per_sym':>16} {'thr_dB':>8} {'sigma_dB':>9}"
    )
    for i, (mode, thr, sig) in enumerate(
        zip(table.modcods, table.thresholds_db, table.threshold_sigma_db, strict=True)
    ):
        print(
            f"{i:3d} {mode.name:>22} {mode.bits_per_symbol:9d} "
            f"{mode.code.rate:10.5f} {mode.spectral_efficiency:16.4f} "
            f"{thr:8.3f} {sig:9.3f}"
        )
    print()
    print("  Monte Carlo support per constellation:")
    print(f"{'mod':>6} {'grid_lo_dB':>11} {'grid_hi_dB':>11} {'points':>7} {'min_ber':>11}")
    for name in sorted(curves):
        cur = curves[name]
        print(
            f"{name:>6} {cur['snr_db'][0]:11.2f} {cur['snr_db'][-1]:11.2f} "
            f"{cur['snr_db'].size:7d} {cur['ber'][-1]:11.3e}"
        )
    table.save_json(HERE / "modcod_thresholds.json")
    print(f"  written: {TABLE_JSON}")
    print()

    print("V2.5 ladder sanity")
    eta = table.spectral_efficiencies
    thr = table.thresholds_db
    print(f"  thresholds strictly increasing: {table.is_monotone}")
    print(f"  efficiencies strictly increasing: {bool(np.all(np.diff(eta) > 0))}")
    dominated = [
        (i, j)
        for i in range(table.n_modes)
        for j in range(table.n_modes)
        if i != j and thr[i] >= thr[j] and eta[i] <= eta[j]
    ]
    print(f"  dominated MODCODs (higher threshold and lower efficiency): {dominated}")
    print(f"  threshold span: {thr[0]:.3f} dB to {thr[-1]:.3f} dB "
          f"({thr[-1] - thr[0]:.3f} dB)")
    print(f"  efficiency span: {eta[0]:.3f} to {eta[-1]:.3f} bit/symbol")
    print(f"  worst threshold uncertainty: {np.nanmax(table.threshold_sigma_db):.3f} dB")
    print()
    print("V2 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
