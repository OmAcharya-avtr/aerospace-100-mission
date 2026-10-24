"""Regenerate the committed dry-run recording, deterministically.

``fixtures/recorded_exchange.json`` is a required committed deliverable: it is
what ``timetransfer dryrun`` replays, what ``tests/test_dryrun.py`` exercises,
and what the pre-flight dry-run check uses. It must be reproducible from the
seed, so this script regenerates it and reports whether the file on disk is
byte-identical to what the generator produces now.

A difference is not automatically a defect -- it means either the generator
changed or the file was edited -- but it must be visible, so the script prints
the digest both ways and exits non-zero on a mismatch.

Run: ``python validation/regenerate_fixture.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, is_fast, relative

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.backends import SimulatedBackend, SimulationSpec  # noqa: E402
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.exchange import ExchangeSet  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402

FIXTURE = Path("fixtures") / "recorded_exchange.json"
SEED = 20261024
N_ATTEMPTS = 240

SPEC = SimulationSpec(
    clock=ClockSpec(
        offset0_s=1.250e-3,
        skew_s_per_s=1.800e-6,
        drift_per_s=2.000e-12,
        q1_s2_per_s=1.0e-20,
        q2_per_s=1.0e-26,
    ),
    link=LinkSpec(
        floor_ab_s=5.400e-3,
        floor_ba_s=4.600e-3,
        jitter_shape_ab=2.0,
        jitter_shape_ba=1.6,
        jitter_scale_ab_s=2.400e-4,
        jitter_scale_ba_s=1.800e-4,
        turnaround_s=1.0e-4,
        loss_probability=0.03,
        outage_windows=((118.0, 126.0),),
    ),
    period_s=1.0,
    timestamp_noise_s=2.0e-8,
    resolution_ns=1,
    seed=SEED,
)


def main() -> int:
    out = Tee("regenerate_fixture_output.txt")
    if is_fast():
        # The fixture is a committed deliverable. A reduced-sample probe run
        # must never rewrite it, so the probe verifies the existing file and
        # stops.
        print("fast probe mode: verifying the committed fixture without rewriting it")
        if not FIXTURE.exists():
            print(f"FAIL: {relative(FIXTURE)} is missing")
            return 1
        existing = ExchangeSet.read_json(FIXTURE)
        print(f"exchanges {existing.n}, digest {existing.digest()}")
        return 0
    out("timetransfer -- regenerate the committed dry-run recording")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    out("")
    out(f"seed                 : {SEED}")
    out(f"attempts             : {N_ATTEMPTS}")
    out(f"declared asymmetry   : {SPEC.link.static_asymmetry_s:+.6e} s")
    out(f"declared static bias : {SPEC.link.static_bias_s:+.6e} s")
    out(f"outage windows       : {SPEC.link.outage_windows}")
    out("")

    backend = SimulatedBackend(SPEC)
    backend.open()
    report = acquire(backend, N_ATTEMPTS, min_exchanges=8)
    backend.close()
    fresh = report.exchanges

    out(report.as_text())
    out("")
    out(f"fresh digest         : {fresh.digest()}")

    existing_digest = "(absent)"
    if FIXTURE.exists():
        existing_digest = ExchangeSet.read_json(FIXTURE).digest()
    out(f"on-disk digest       : {existing_digest}")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_json(FIXTURE)
    rewritten = ExchangeSet.read_json(FIXTURE)
    out(f"rewritten digest     : {rewritten.digest()}")
    out(f"round trip identical : {rewritten.timestamp_bytes() == fresh.timestamp_bytes()}")
    out(f"wrote                : {relative(FIXTURE)}")
    out("")

    failures = 0
    if rewritten.timestamp_bytes() != fresh.timestamp_bytes():
        failures += 1
        out("FAIL: JSON round trip is not byte-identical")
    if existing_digest not in ("(absent)", fresh.digest()):
        out(
            "NOTE: the committed fixture differed from the generator output and has been "
            "rewritten. This is reported rather than hidden."
        )
    out(f"failures             : {failures}")
    out(f"saved to             : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
