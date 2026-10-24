"""Deliberate bound violations the certificate must catch and fail on.

A certificate that always says HOLDS certifies nothing. This script injects
violations of eight distinct kinds into reconstructions the codec itself
produced, hands each to the same :func:`boundcompress.certificate.certify` the
codec's own output goes through, and requires every one to come back with
``bound_holds=False`` and the right sample index.

The sharpest control is the last one: a reconstruction exactly at the bound
must pass and the same reconstruction moved by one unit in the last place must
fail. That is the assertion that the comparison carries no hidden tolerance,
which is the property the whole product rests on.

A note on how a negative control gets silently weakened
-------------------------------------------------------

The first version of this script built three of its controls by *offsetting
the reconstruction* by 1.5 times the bound, and two of them were not caught.
That is a defect in the control and not in the certificate: a reconstruction
is already somewhere inside the bound, so adding 1.5 b to it leaves an error
of at least 0.5 b and at most 2.5 b -- which need not be outside the bound at
all. Only an offset strictly greater than 2 b guarantees a violation. The
measured consequence was that the index-0 and final-sample controls reported
zero violations, and the 50-sample run control caught 26 of its 50 samples.

Every control here is now anchored on the **true** sample value rather than on
the reconstruction, so the injected error is exactly the stated multiple of
the bound and the violation is guaranteed by construction. The episode is
recorded rather than quietly fixed because a negative control that cannot
fail is indistinguishable from one that is not there.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.certificate import certify  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096


def main() -> int:
    out = Tee("validate_negative_control_output.txt")
    out.rule("boundcompress -- deliberate bound violations, which must be caught")
    for line in environment_lines():
        out(line)

    signal = dataset.generate("battery_voltage", samples=SAMPLES, seed=0)
    values = signal.values
    bound = 1e-3
    result = encode(values, "abs", bound, predictor="linear")
    out("")
    out(f"signal                 : {signal.name} ({signal.samples} samples, {signal.unit})")
    out(f"declared bound         : abs {bound:.6e}")
    out(f"clean certificate      : {'HOLDS' if result.certificate.bound_holds else 'FAILED'}")
    out(f"clean max abs error    : {result.certificate.max_absolute_error:.6e}")
    out(f"clean margin           : {result.certificate.margin:.6e}")
    out.check("the clean encode holds its bound", result.certificate.bound_holds)

    rows: list[list[str]] = []

    def control(label: str, corrupted: np.ndarray, expect_index: int | None,
                mode: str = "abs", declared: float = bound) -> None:
        certificate = certify(values, corrupted, mode, declared)
        caught = certificate.bound_holds is False
        located = expect_index is None or expect_index in certificate.violation_indices
        rows.append(
            [
                label,
                str(certificate.violations),
                str(list(certificate.violation_indices[:4])),
                f"{certificate.max_absolute_error:.6e}",
                f"{certificate.margin:.6e}",
                "CAUGHT" if caught else "MISSED",
                "yes" if located else "NO",
            ]
        )
        out.check(f"negative control caught: {label}", caught and located)

    # Every injection below is anchored on the TRUE value, so the realised
    # error is exactly the stated multiple of the bound. See the module
    # docstring for why offsetting the reconstruction instead is not a control.

    # 1. One sample at 2.5 bounds of error.
    corrupted = result.reconstruction.copy()
    corrupted[1234] = values[1234] + 2.5 * bound
    control("single sample, error = 2.5 x bound at index 1234", corrupted, 1234)

    # 2. One sample in the other direction.
    corrupted = result.reconstruction.copy()
    corrupted[77] = values[77] - 4.0 * bound
    control("single sample, error = -4.0 x bound at index 77", corrupted, 77)

    # 3. The first sample, which a loop that starts at one would miss.
    corrupted = result.reconstruction.copy()
    corrupted[0] = values[0] + 1.5 * bound
    control("first sample, error = 1.5 x bound at index 0", corrupted, 0)

    # 4. The last sample, which a loop that stops early would miss.
    corrupted = result.reconstruction.copy()
    corrupted[-1] = values[-1] + 1.5 * bound
    control("last sample, error = 1.5 x bound", corrupted, SAMPLES - 1)

    # 5. A whole run of samples.
    corrupted = result.reconstruction.copy()
    corrupted[2000:2050] = values[2000:2050] + 1.01 * bound
    control("run of 50 samples, error = 1.01 x bound from index 2000", corrupted, 2000)

    # 6. A sample set to nan. ``nan <= bound`` is false, so this must be caught.
    corrupted = result.reconstruction.copy()
    corrupted[500] = float("nan")
    control("nan reconstruction at index 500", corrupted, 500)

    # 7. A sample set to inf.
    corrupted = result.reconstruction.copy()
    corrupted[501] = float("inf")
    control("inf reconstruction at index 501", corrupted, 501)

    # 8. A violation smaller than one part in a million of the bound, so a
    #    tolerance of 1e-6 relative would hide it.
    corrupted = result.reconstruction.copy()
    worst = int(result.certificate.argmax_index)
    direction = 1.0 if corrupted[worst] >= values[worst] else -1.0
    corrupted[worst] = values[worst] + direction * bound * (1.0 + 1e-9)
    control("violation of 1 part in 1e9 of the bound", corrupted, worst)

    out("")
    out(
        format_table(
            ["injected violation", "violations", "first indices", "max_abs_error",
             "margin", "verdict", "located"],
            rows,
        )
    )

    out.rule("the comparison carries no tolerance: at the bound passes, one ulp past fails")
    probe = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    limit = 0.25
    at_bound = probe + limit
    one_ulp_past = at_bound.copy()
    one_ulp_past[2] = np.nextafter(one_ulp_past[2], np.inf)
    first = certify(probe, at_bound, "abs", limit)
    second = certify(probe, one_ulp_past, "abs", limit)
    out(f"error exactly at the bound   : holds={first.bound_holds} margin={first.margin:.6e}")
    out(
        f"error one ulp past the bound : holds={second.bound_holds} "
        f"margin={second.margin:.6e} indices={list(second.violation_indices)}"
    )
    out(f"the ulp step itself          : {np.nextafter(1.25, 2.0) - 1.25:.6e}")
    out.check("a reconstruction exactly at the bound is admissible", first.bound_holds)
    out.check(
        "a reconstruction one ulp past the bound is not",
        second.bound_holds is False and second.violation_indices == (2,),
    )

    out.rule("pointwise relative mode")
    positive = dataset.generate("panel_temperature", samples=SAMPLES, seed=1).values
    relative_bound = 1e-5
    clean = encode(positive, "rel", relative_bound, predictor="linear")
    corrupted = clean.reconstruction.copy()
    corrupted[900] = positive[900] * (1.0 + 1.5 * relative_bound)
    certificate = certify(positive, corrupted, "rel", relative_bound)
    out(f"clean rel certificate  : {'HOLDS' if clean.certificate.bound_holds else 'FAILED'}")
    out(f"clean max rel error    : {clean.certificate.max_relative_error:.6e}")
    out(f"injected rel error     : {certificate.max_relative_error:.6e}")
    out(f"violating indices      : {list(certificate.violation_indices)}")
    out.check("the clean rel encode holds", clean.certificate.bound_holds)
    out.check(
        "a 1.5x pointwise relative violation is caught at the right index",
        certificate.bound_holds is False and certificate.violation_indices == (900,),
    )

    out.rule("totals")
    out(f"negative controls injected : {len(rows) + 3}")
    out(f"negative controls caught   : {len(rows) + 3 - len(out.failures)}")
    out("")
    out("Every control above is anchored on the true sample value. An earlier")
    out("version offset the reconstruction by 1.5 x the bound instead, and two")
    out("controls were then not caught and a third caught 26 of its 50 samples,")
    out("because an offset below 2 x the bound need not leave the bound. The")
    out("certificate was correct in every one of those cases; the control was")
    out("not. See the module docstring of validation/validate_negative_control.py.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
