"""The README's worked example, executed, with its output committed.

One run, worked through in the order the product insists on: the clock first,
then the certificate, then re-simulation on the residual only, then the
verdict. The last block is the paragraph a programme would actually write into
a test report, printed verbatim from the numbers above it.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    ALLOWANCE_PPM,
    STEPS,
    TOLERANCE,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from hilverdict.certificate import certify_run  # noqa: E402
from hilverdict.clock import estimate_clock  # noqa: E402
from hilverdict.plant import DEFAULT_METRIC  # noqa: E402
from hilverdict.report import (  # noqa: E402
    render_certificates,
    render_clock,
    render_resimulations,
    render_run_certificate,
    render_verdict,
)
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.synthetic import generate_run  # noqa: E402
from hilverdict.timeline import overrun_counts, utilisation  # noqa: E402
from hilverdict.verdict import run_verdict  # noqa: E402

SEED = 902
PROFILE = "loaded"
AMPLITUDE_DEG = 8.0


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("hilverdict -- one run, worked through")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared run: seed {SEED}, profile {PROFILE}, "
        f"{AMPLITUDE_DEG:g} deg manoeuvre, {STEPS} iterations at 10 ms")
    out(f"declared tolerance: {TOLERANCE:g} in a metric of "
        f"{math.degrees(DEFAULT_METRIC.alpha_tolerance_rad):.4f} deg incidence and "
        f"{math.degrees(DEFAULT_METRIC.rate_tolerance_rad_s):.4f} deg/s pitch rate")
    log = generate_run(
        seed=SEED, profile=PROFILE, amplitude_deg=AMPLITUDE_DEG, steps=STEPS
    )

    out.rule("step 0: what the rig recorded")
    counts = overrun_counts(log)
    out(f"  iterations                 : {log.steps}")
    out(f"  mean utilisation           : {utilisation(log):.6f}")
    out(f"  overruns, direct count     : {counts.direct}")
    out(f"  overruns, cascade count    : {counts.cascade}")
    out(f"  cascade / direct           : {counts.ratio:.6f}")
    out(f"  longest overrun            : "
        f"{float(np.max(np.asarray(log.durations_s))) * 1e3:.4f} ms "
        f"against a {log.model.sampling.deadline_s * 1e3:g} ms deadline")
    out(f"  peak reference incidence   : {math.degrees(log.peak_alpha_rad):.4f} deg")
    out(f"  saturated iterations       : {log.saturated_steps}")
    out("")
    out("A timing report that stops here has not answered the question. Twenty-two")
    out("overruns is not a verdict.")

    out.rule("step 1: the clock, before any duration is believed")
    out(render_clock(
        estimate_clock(log.host_time_s, log.device_time_s, ALLOWANCE_PPM)
    ))

    out.rule("step 2: the analytic certificate, equation (13)")
    certificate = certify_run(log, TOLERANCE)
    out(render_run_certificate(certificate))
    out("")
    out(render_certificates(certificate.certificates, limit=12))

    out.rule("step 3: re-simulation, on the residual only")
    undecided = tuple(item.event.index for item in certificate.undecided)
    out(f"the certificate left {len(undecided)} of "
        f"{len(certificate.certificates)} misses undecided: {list(undecided)}")
    out("")
    results = resimulate_run(log, TOLERANCE, subset=undecided)
    out(render_resimulations(results))
    out("")
    paid = sum(result.plant_calls for result in results)
    everything = sum(
        result.plant_calls for result in resimulate_run(log, TOLERANCE)
    )
    out(f"  plant evaluations paid for          : {paid}")
    out(f"  plant evaluations if every miss had been re-simulated : {everything}")
    out(f"  saved by the certificate            : "
        f"{1.0 - paid / everything:.6f}")

    out.rule("step 4: the verdict")
    verdict = run_verdict(log, TOLERANCE)
    out(render_verdict(verdict))

    out.rule("the paragraph a programme would write down")
    consequential = [
        result for result in resimulate_run(log, TOLERANCE)
        if result.consequential
    ]
    worst = max(consequential, key=lambda result: result.peak_divergence)
    out("")
    out(f"  Run {log.label} recorded {counts.direct} deadline overruns in")
    out(f"  {log.steps} iterations at a mean utilisation of "
        f"{utilisation(log):.4f}. The host-to-device")
    out(f"  clock was within {ALLOWANCE_PPM:g} ppm with no reset, so the recorded")
    out("  durations are believable.")
    out("")
    out(f"  {len(certificate.dismissed)} of those {counts.direct} overruns are")
    out("  dismissed in closed form: for the declared plant, controller and")
    out("  linearisation, no overrun of that duration holding that input step can")
    out("  move the trajectory past the declared tolerance. That is a proof about")
    out("  the declared model, not an estimate, and it is why only")
    out(f"  {len(undecided)} overruns needed re-simulating.")
    out("")
    out(f"  Of those {len(undecided)}, {len(consequential)} measurably changed the")
    out("  trajectory. The worst, iteration")
    out(f"  {worst.event.index} with a {worst.event.overrun_s * 1e3:.4f} ms overrun,")
    out(f"  moved it by {worst.peak_divergence:.4f} times the declared tolerance,")
    out(f"  peaking {worst.peak_index - worst.event.index} iterations later.")
    out("")
    out(f"  The run is therefore {verdict.verdict}: its result is not timing-faithful")
    out("  at the declared tolerance, and the iterations to look at are")
    out(f"  {list(verdict.consequential)}.")
    out("")
    out("  This is research-grade software. The statement above is evidence about a")
    out("  declared model, not a certification, and nothing here is flight-qualified")
    out("  or approved for operational aerospace use.")
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
