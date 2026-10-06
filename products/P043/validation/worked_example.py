"""The worked example reproduced in the README, with its printed output.

A photon-starved 64-PPM downlink: received power to slot counts, slot counts to
symbol error probability and available rate, then the same link read out through
a detector with dead time and afterpulsing and corrected back to the incident
rate by the closed forms and by the learned model.

Every number the README quotes from this example comes from this script. Run it
from this directory; it writes nothing.

Runtime: about 5 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.capacity import (  # noqa: E402
    hard_decision_capacity,
    minimum_photons_per_bit,
    photons_per_bit,
    soft_decision_achievable_rate,
)
from photoncount.correction import (  # noqa: E402
    RateCorrector,
    composed_baseline,
    matched_baseline,
)
from photoncount.dataset import make_features  # noqa: E402
from photoncount.deadtime import paralyzable_maximum  # noqa: E402
from photoncount.hal import AcquisitionRequest, BackendMode, SimulatedBackend  # noqa: E402
from photoncount.ops import RunRecord, run_preflight  # noqa: E402
from photoncount.poisson import counting_snr, mean_counts_from_power  # noqa: E402
from photoncount.ppm import (  # noqa: E402
    PPMConfig,
    bit_llrs,
    sample_counts,
    slot_metric_scale,
    symbol_error_probability,
)
from photoncount.simulate import DetectorSpec  # noqa: E402

MODEL_FILE = "rate_corrector.joblib"


def main() -> int:
    print("worked_example.py --- 64-PPM photon-starved downlink")
    print("=" * 78)

    # --- 1. link budget to slot counts ------------------------------------
    power_w = 3.0e-11
    wavelength_m = 1550e-9
    slot_s = 2.0e-8
    efficiency = 0.65
    n_s = mean_counts_from_power(power_w, wavelength_m, slot_s, efficiency)
    background_w = 2.0e-14
    n_b = mean_counts_from_power(background_w, wavelength_m, slot_s, efficiency)
    n_d = 2.0e-8 * 200.0  # 200 counts/s of dark current over one slot
    print("\n1. Link budget to slot counts")
    print(f"   received signal power   {power_w:.3e} W at {wavelength_m * 1e9:.0f} nm")
    print(f"   slot length             {slot_s * 1e9:.1f} ns")
    print(f"   detection efficiency    {efficiency:.2f}")
    print(f"   signal counts per pulsed slot   n_s = {n_s:.6f}")
    print(f"   background counts per slot      n_b = {n_b:.6f}")
    print(f"   dark counts per slot            n_d = {n_d:.6f}")
    print(f"   counting SNR of one slot        {counting_snr(n_s, n_b + n_d):.4f}")

    # --- 2. slot statistics ------------------------------------------------
    cfg = PPMConfig(64, n_s, n_b, n_d)
    exact = symbol_error_probability(cfg)
    print("\n2. 64-PPM slot statistics")
    print(f"   non-signalled slot mean n_0 = {cfg.noise_counts:.6f}")
    print(f"   symbol error probability    {exact['error']:.6f} "
          f"(truncation K = {int(exact['truncation'])}, tail {exact['tail_mass']:.1e})")
    print(f"   soft metric scale ln(1 + n_s/n_0) = {slot_metric_scale(cfg):.6f} nats/count")
    rng = np.random.default_rng(20261006)
    counts = sample_counts(cfg, [17], rng)
    print(f"   one simulated symbol, slot 17 signalled, non-zero slots: "
          f"{[(int(i), int(counts[0, i])) for i in np.nonzero(counts[0])[0]]}")
    print(f"   bit LLRs (nats, positive favours 0): "
          f"{np.round(bit_llrs(counts, cfg)[0], 3).tolist()}")

    # --- 3. available rate -------------------------------------------------
    hard = hard_decision_capacity(cfg)
    soft = soft_decision_achievable_rate(cfg, 60_000, rng)
    print("\n3. Available rate")
    print(f"   hard-decision capacity   {hard['bits_per_symbol']:.5f} bits/symbol "
          f"({hard['bits_per_slot']:.6f} bits/slot)")
    print(f"   soft-decision rate       {soft['bits_per_symbol']:.5f} "
          f"+- {soft['standard_error']:.5f} bits/symbol")
    print(f"   photons/bit, hard        {photons_per_bit(cfg, hard['bits_per_symbol']):.5f}")
    print(f"   photons/bit, soft        {photons_per_bit(cfg, soft['bits_per_symbol']):.5f}")
    print(f"   64-PPM limit 1/log2(64)  {minimum_photons_per_bit(64):.5f}")
    print(f"   symbol rate at this slot length  {1.0 / (64 * slot_s) / 1e3:.3f} ksymbol/s")
    print(f"   information rate, soft           "
          f"{soft['bits_per_symbol'] / (64 * slot_s) / 1e6:.4f} Mbit/s")

    # --- 4. the detector is not an ideal counter ---------------------------
    tau = 2.4e-8
    spec = DetectorSpec(
        dead_time_s=tau,
        model="paralyzable",
        afterpulse_probability=0.045,
        afterpulse_mean_delay_s=1.1e-7,
    )
    incident_hz = 6.0e6
    n_max, m_max = paralyzable_maximum(tau)
    print("\n4. The detector")
    print(f"   dead time               {tau * 1e9:.1f} ns, paralyzable")
    print(f"   afterpulse probability  {spec.afterpulse_probability:.3f}, "
          f"mean delay {spec.afterpulse_mean_delay_s * 1e9:.1f} ns")
    print(f"   incident rate           {incident_hz:.3e} counts/s "
          f"(n tau = {incident_hz * tau:.4f})")
    print(f"   paralyzable peak at     n = {n_max:.4e} counts/s, "
          f"m_max = {m_max:.4e} counts/s")

    backend = SimulatedBackend(spec, incident_hz, np.random.default_rng(4242))
    request = AcquisitionRequest(window_s=1e-3, n_windows=200, label="worked_example")
    report = run_preflight(backend, request, incident_hz, directory=".")
    print(f"   preflight passed: {report.passed}; "
          f"warnings: {[c.name for c in report.warnings]}")
    with backend:
        acq = backend.acquire(request, BackendMode.SIMULATION)
    record = RunRecord.from_acquisition(acq, request, backend, report, seed=4242)
    print(f"   observed rate           {acq.observed_rate_hz:.4e} counts/s")
    print(f"   Fano factor             {acq.fano_factor:.4f} "
          "(below 1: dead time dominates the clustering)")
    print(f"   is_measurement          {acq.is_measurement} "
          "(simulated counts are never a measurement)")
    print(f"   record would be written as  {record.label}.json")

    # --- 5. correcting back ------------------------------------------------
    m = np.array([acq.observed_rate_hz])
    taus = np.array([tau])
    par = np.array([1.0])
    p_ap = np.array([spec.afterpulse_probability])
    matched = float(matched_baseline(m, taus, par)[0])
    composed = float(composed_baseline(m, taus, par, p_ap)[0])
    print("\n5. Correcting the observed rate back to the incident rate")
    print(f"   truth                           {incident_hz:.4e} counts/s")
    print(f"   paralyzable closed form (D5)     {matched:.4e}  "
          f"({(matched / incident_hz - 1) * 100:+.2f} %)")
    print(f"   composed closed form (A2 + D5)   {composed:.4e}  "
          f"({(composed / incident_hz - 1) * 100:+.2f} %)")
    model_path = Path(__file__).resolve().parent / MODEL_FILE
    if model_path.exists():
        corrector = RateCorrector.load(model_path)
        feats = make_features(
            m,
            taus,
            np.array([acq.fano_factor]),
            p_ap,
            np.array([spec.afterpulse_mean_delay_s / tau]),
            par,
        )
        out = corrector.predict_rate(feats, taus)
        lo, mid, hi = (float(out[k][0]) for k in ("lower", "median", "upper"))
        print(f"   learned correction               {mid:.4e}  "
              f"({(mid / incident_hz - 1) * 100:+.2f} %)")
        print(f"   learned 5-95 interval            [{lo:.4e}, {hi:.4e}]  "
              f"contains the truth: {lo <= incident_hz <= hi}")
    else:
        print(f"   learned correction               {MODEL_FILE} not present; "
              "run validate_correction.py first")
    print("\n   One row is one draw, not a result. This row sits at n tau = 0.14 with")
    print("   p = 0.045; which estimator happens to be closest here is counting noise.")
    print("   The result is the regime table in VALIDATION.md, which says the closed")
    print("   forms win where afterpulsing is absent, the learned model wins where it")
    print("   is present, and nothing works past n tau = 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
