"""Level-2 validation of the Kim/Kruse baseline implementations.

Checks the implemented formulas against independently hand-computed reference values
(alpha = (10/ln 10) * (3.912/V) * (lambda/550)^(-q)) at reference visibilities, and
verifies the published qualitative behaviours:
- Kim q(V) branch values (SPIE 4214, 2001),
- wavelength independence of Kim attenuation for V <= 0.5 km,
- Kim == Kruse for V > 6 km,
- Kim-vs-Kruse disagreement at low visibility.

Writes the evidence table to validate_baselines_output.txt next to this script.
"""

import math
from pathlib import Path

from fogcast import kim_attenuation_db_km, kim_q, kruse_attenuation_db_km, kruse_q

DB = 10.0 / math.log(10.0)
K = 3.912


def ref_alpha(v: float, lam: float, q: float) -> float:
    """Independent recomputation of the attenuation formula (dB/km)."""
    return DB * (K / v) * (lam / 550.0) ** (-q)


def main() -> None:
    lines = []
    lines.append("FogCast baseline validation — computed in-session")
    lines.append("Formula: alpha_dB/km = (10/ln10)*(3.912/V)*(lambda/550)^(-q)")
    lines.append("References: Kruse et al. 1962; Kim, McArthur, Korevaar, SPIE 4214, 2001")
    lines.append("")

    # 1. Kim q(V) branches
    lines.append("1. Kim q(V) branches (expected from SPIE 4214 piecewise definition):")
    q_cases = [
        (0.3, 0.0, "V<=0.5: q=0"),
        (0.75, 0.25, "0.5<V<=1: q=V-0.5"),
        (3.0, 0.82, "1<V<=6: q=0.16V+0.34"),
        (10.0, 1.3, "6<V<=50: q=1.3"),
        (60.0, 1.6, "V>50: q=1.6"),
    ]
    ok_all = True
    for v, expected, label in q_cases:
        got = kim_q(v)
        ok = abs(got - expected) < 1e-12
        ok_all &= ok
        lines.append(f"   V={v:6.2f} km  q={got:.4f}  expected {expected:.4f}  [{label}]"
                     f"  {'PASS' if ok else 'FAIL'}")

    # 2. Attenuation known answers at reference visibilities
    lines.append("")
    lines.append("2. Attenuation at reference visibilities (implementation vs hand formula):")
    lines.append(f"   {'V (km)':>7} {'lam (nm)':>9} {'model':>6} {'computed':>10}"
                 f" {'reference':>10} {'rel err':>9}")
    cases = [
        (0.3, 1550.0, "kim", kim_attenuation_db_km, kim_q),
        (0.3, 1550.0, "kruse", kruse_attenuation_db_km, kruse_q),
        (0.75, 850.0, "kim", kim_attenuation_db_km, kim_q),
        (3.0, 1310.0, "kim", kim_attenuation_db_km, kim_q),
        (3.0, 1310.0, "kruse", kruse_attenuation_db_km, kruse_q),
        (10.0, 1550.0, "kim", kim_attenuation_db_km, kim_q),
        (60.0, 850.0, "kim", kim_attenuation_db_km, kim_q),
    ]
    for v, lam, name, fn, qfn in cases:
        got = fn(v, lam)
        ref = ref_alpha(v, lam, qfn(v))
        rel = abs(got - ref) / ref
        ok = rel < 1e-12
        ok_all &= ok
        lines.append(f"   {v:7.2f} {lam:9.0f} {name:>6} {got:10.4f} {ref:10.4f} {rel:9.2e}"
                     f"  {'PASS' if ok else 'FAIL'}")

    # 3. Published qualitative behaviours
    lines.append("")
    lines.append("3. Published behaviours:")
    a850 = kim_attenuation_db_km(0.3, 850.0)
    a1550 = kim_attenuation_db_km(0.3, 1550.0)
    ok = abs(a850 - a1550) < 1e-12
    ok_all &= ok
    lines.append(f"   Kim wavelength-independence at V=0.3 km: 850nm={a850:.4f},"
                 f" 1550nm={a1550:.4f} dB/km  {'PASS' if ok else 'FAIL'}")
    kim20 = kim_attenuation_db_km(20.0, 1310.0)
    kruse20 = kruse_attenuation_db_km(20.0, 1310.0)
    ok = abs(kim20 - kruse20) < 1e-12
    ok_all &= ok
    lines.append(f"   Kim==Kruse for V=20 km, 1310 nm: {kim20:.4f} vs {kruse20:.4f} dB/km"
                 f"  {'PASS' if ok else 'FAIL'}")
    kim_lo = kim_attenuation_db_km(0.3, 1550.0)
    kruse_lo = kruse_attenuation_db_km(0.3, 1550.0)
    ok = kim_lo > 1.4 * kruse_lo
    ok_all &= ok
    lines.append(f"   Low-visibility disagreement V=0.3 km, 1550 nm: Kim={kim_lo:.2f},"
                 f" Kruse={kruse_lo:.2f} dB/km (Kim/Kruse={kim_lo / kruse_lo:.2f})"
                 f"  {'PASS' if ok else 'FAIL'}")

    lines.append("")
    lines.append(f"OVERALL: {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'}")

    out = Path(__file__).resolve().parent / "validate_baselines_output.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
