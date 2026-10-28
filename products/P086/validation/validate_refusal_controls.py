"""Negative and positive controls. Every one anchored on a truth, every one verified to fire.

Why the anchoring matters
-------------------------
A control that injects a violation by offsetting an *output* does not guarantee
a violation: the output already sits inside the band, so only an offset
strictly greater than twice the bound is guaranteed to leave it. An earlier
product of this mission shipped two of eleven controls silently too weak, and
the certificate was right while the control was wrong, which reads as evidence
and is the worse of the two failures.

So every control below is anchored on a quantity that is true independently of
anything this package computed -- an analytic bound from equation (12), a
declared noise sigma, a ground-truth anomaly class from the generator, or a
constructed counter arithmetic -- and every control is asserted to have
actually fired. Four controls are the other direction and must **not** fire;
both directions are asserted.

The fifteen controls are listed in the output with the quantity each is
anchored on.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from _bootstrap import Tee, bootstrap, default_tolerances, environment_lines

ROOT = bootstrap()

from telemgap import classify, dataset, policy  # noqa: E402
from telemgap.record import ChannelSpec, RecordSchema, TelemetryRecord  # noqa: E402
from telemgap.report import format_table  # noqa: E402
from telemgap.screens import screen_counters  # noqa: E402


def single_channel_record(
    values: np.ndarray,
    timestamps: np.ndarray,
    counters: np.ndarray,
    channel: ChannelSpec,
    modulus: int = 256,
) -> TelemetryRecord:
    """A one-channel record built from engineering values, round-tripped through words."""
    schema = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=modulus,
        channels=(channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )
    raw = np.mod(
        np.rint((values - channel.offset) / channel.scale).astype(np.int64),
        np.int64(1 << channel.word_bits),
    ).astype(np.uint64)
    decoded = raw.astype(np.int64)
    if channel.signed:
        span = 1 << channel.word_bits
        decoded = np.where(decoded >= span // 2, decoded - span, decoded)
    return TelemetryRecord(
        schema=schema,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        counters=np.asarray(counters, dtype=np.int64),
        values=(decoded.astype(np.float64) * channel.scale + channel.offset).reshape(-1, 1),
        raw_words=raw.reshape(-1, 1),
        validity=np.ones(values.size, dtype=bool),
        provenance={"time_source": "constructed control"},
    )


FLAT_CHANNEL = ChannelSpec(
    name="control",
    unit="u",
    scale=1.0e-4,
    offset=0.0,
    low=0.0,
    high=6.0,
    max_slew=0.05,
    max_curvature=0.02,
    noise_sigma=0.001,
)


def gap_record(slots: int, channel: ChannelSpec = FLAT_CHANNEL) -> TelemetryRecord:
    """Forty frames of a flat signal with one gap of ``slots`` cadence intervals."""
    before = np.arange(20, dtype=np.float64)
    after = before[-1] + slots + 1.0 + np.arange(20, dtype=np.float64)
    timestamps = np.concatenate((before, after))
    counters = np.concatenate(
        (np.arange(20, dtype=np.int64), 20 + slots + np.arange(20, dtype=np.int64))
    )
    values = np.full(timestamps.size, 3.0, dtype=np.float64)
    return single_channel_record(values, timestamps, counters, channel, modulus=4096)


def main() -> int:
    out = Tee("validate_refusal_controls_output.txt")
    out.rule("telemgap -- fifteen controls, each anchored on a truth and verified to fire")
    for line in environment_lines():
        out(line)
    out("")
    out("A control marked MUST FIRE is a negative control: the policy or the")
    out("classifier is required to refuse or to withhold. A control marked MUST NOT")
    out("FIRE is the other direction and is asserted too.")

    rows: list[list[str]] = []

    def record_row(name: str, anchored_on: str, must_fire: bool, fired: bool, detail: str) -> None:
        rows.append(
            [
                name,
                anchored_on,
                "MUST FIRE" if must_fire else "MUST NOT FIRE",
                "fired" if fired else "did not fire",
                "ok" if fired == must_fire else "CONTROL FAILED",
                detail,
            ]
        )
        out.check(f"control {name}", fired == must_fire, detail)

    tolerances = default_tolerances()

    # 1. The long outage.
    out.rule("controls 1 to 4: the policy's gates, anchored on analytic quantities")
    generated = dataset.long_outage_record(samples=300, outage_frames=21_600)
    report = classify.classify(generated.record)
    plan = policy.plan(generated.record, report, tolerances)
    record_row(
        "1 long outage refused",
        "a constructed outage of 21,600 cadence intervals",
        True,
        len(plan.permissions) == 0 and len(plan.refusals) == 4,
        f"{len(plan.refusals)} refusals, {len(plan.permissions)} permissions",
    )

    # 2 and 3. Tolerance against the analytic bound of equation (12).
    slots = 2
    duration = (slots + 1) * 1.0
    analytic = policy.linear_error_bound(FLAT_CHANNEL.max_curvature, duration)
    coverage = policy.DEFAULT_COVERAGE_K * FLAT_CHANNEL.noise_sigma
    anchor = analytic + coverage
    record = gap_record(slots)
    report = classify.classify(record)
    tight = policy.plan(record, report, {"control": 0.5 * anchor})
    loose = policy.plan(record, report, {"control": 2.0 * anchor})
    record_row(
        "2 tolerance below the analytic prediction refused",
        f"equation (12) curvature {FLAT_CHANNEL.max_curvature:g} over h = {duration:g} s "
        f"gives {analytic:.6g}; tolerance set to half of {anchor:.6g}",
        True,
        all(decision.refused for decision in tight.decisions),
        f"{len(tight.refusals)} of {len(tight.decisions)} refused",
    )
    record_row(
        "3 tolerance above the analytic prediction permitted",
        f"the same anchor {anchor:.6g}, tolerance set to twice it",
        False,
        any(decision.refused for decision in loose.decisions),
        f"{len(loose.refusals)} of {len(loose.decisions)} refused, "
        f"method {loose.decisions[0].method}",
    )

    # 4 and 5. The Nyquist gate.
    over = gap_record(int(policy.DEFAULT_MAX_GAP_SLOTS) + 1)
    over_report = classify.classify(over)
    over_plan = policy.plan(over, over_report, {"control": 1e6})
    under = gap_record(int(policy.DEFAULT_MAX_GAP_SLOTS) - 2)
    under_report = classify.classify(under)
    under_plan = policy.plan(under, under_report, {"control": 1e6})
    record_row(
        "4 gap beyond max_gap_slots refused",
        f"a constructed gap of {int(policy.DEFAULT_MAX_GAP_SLOTS) + 1} cadence intervals "
        f"against the declared limit of {policy.DEFAULT_MAX_GAP_SLOTS:g}",
        True,
        all("nyquist_slots" in decision.failed_gates for decision in over_plan.decisions),
        f"failed gates {over_plan.decisions[0].failed_gates}",
    )
    record_row(
        "5 gap inside max_gap_slots permitted",
        f"a constructed gap of {int(policy.DEFAULT_MAX_GAP_SLOTS) - 2} cadence intervals "
        "with an effectively unlimited tolerance",
        False,
        any(decision.refused for decision in under_plan.decisions),
        f"{len(under_plan.refusals)} of {len(under_plan.decisions)} refused",
    )

    # 6. The tolerance-above-noise gate.
    noisy_channel = dataclasses.replace(FLAT_CHANNEL, noise_sigma=0.5)
    noisy = gap_record(1, noisy_channel)
    noisy_report = classify.classify(noisy)
    noisy_plan = policy.plan(noisy, noisy_report, {"control": 0.5})
    record_row(
        "6 tolerance below k sigma refused",
        f"declared noise sigma {noisy_channel.noise_sigma:g} and k = "
        f"{policy.DEFAULT_COVERAGE_K:g}, so k sigma = "
        f"{policy.DEFAULT_COVERAGE_K * noisy_channel.noise_sigma:g}; tolerance 0.5",
        True,
        all(
            "tolerance_above_noise" in decision.failed_gates
            for decision in noisy_plan.decisions
        ),
        f"failed gates {noisy_plan.decisions[0].failed_gates}",
    )

    out.rule("controls 7 to 10: the endpoint, measurability and resolvability gates")
    # 7 and 8. The endpoint gate, anchored on an injected corruption of known class.
    clean = gap_record(1)
    clean_report = classify.classify(clean)
    clean_plan = policy.plan(clean, clean_report, {"control": 1e6})
    damaged = gap_record(1)
    endpoint = int(clean_report.findings.gaps[0].before_index)
    damaged.validity[endpoint] = False
    damaged_report = classify.classify(damaged)
    damaged_plan = policy.plan(damaged, damaged_report, {"control": 1e6})
    record_row(
        "7 gap bounded by a frame whose error control failed is refused",
        f"the frame error-control flag of frame {endpoint} set to failed, which is a "
        "declared input and not an output of this package",
        True,
        all(
            "endpoints_clean" in decision.failed_gates for decision in damaged_plan.decisions
        ),
        f"failed gates {damaged_plan.decisions[0].failed_gates}",
    )
    record_row(
        "8 the same gap with clean endpoints is permitted",
        "the identical record with the error-control flag left passing",
        False,
        any(decision.refused for decision in clean_plan.decisions),
        f"{len(clean_plan.refusals)} of {len(clean_plan.decisions)} refused",
    )

    # 9. Unmeasurable curvature on a pure-noise channel.
    rng = np.random.default_rng(4242)
    noise_channel = dataclasses.replace(
        FLAT_CHANNEL, max_curvature=0.0, noise_sigma=0.01, low=2.0, high=4.0
    )
    timestamps = np.concatenate(
        (np.arange(200, dtype=np.float64), 202.0 + np.arange(200, dtype=np.float64))
    )
    counters = np.concatenate(
        (np.arange(200, dtype=np.int64), 202 + np.arange(200, dtype=np.int64))
    )
    values = 3.0 + rng.normal(0.0, 0.01, size=timestamps.size)
    noise_record = single_channel_record(
        values, timestamps, counters, noise_channel, modulus=4096
    )
    noise_report = classify.classify(noise_record)
    noise_plan = policy.plan(noise_record, noise_report, {"control": 1.0})
    estimate = noise_plan.curvatures["control"]
    record_row(
        "9 a curvature estimate below its own noise floor refuses the linear method",
        f"a constructed pure-noise channel of sigma {noise_channel.noise_sigma:g} with no "
        "declared curvature; the noise floor is derived from the estimator's own "
        "coefficients",
        True,
        any(
            "curvature_measurable" in decision.failed_gates
            for decision in noise_plan.decisions
        ),
        f"estimate {estimate.value:.6g} against floor {estimate.noise_floor:.6g}, "
        f"measurable {estimate.measurable}",
    )

    # 10. Resolvability on a near-Nyquist channel.
    fast_channel = dataclasses.replace(
        FLAT_CHANNEL, max_curvature=0.0, noise_sigma=1e-5, low=0.0, high=6.0
    )
    times = np.concatenate(
        (np.arange(300, dtype=np.float64), 302.0 + np.arange(300, dtype=np.float64))
    )
    fast_counters = np.concatenate(
        (np.arange(300, dtype=np.int64), 302 + np.arange(300, dtype=np.int64))
    )
    fast_values = 3.0 + 0.5 * np.sin(2.0 * np.pi * 0.35 * times)
    fast_record = single_channel_record(
        fast_values, times, fast_counters, fast_channel, modulus=4096
    )
    fast_report = classify.classify(fast_record)
    fast_plan = policy.plan(fast_record, fast_report, {"control": 1.0})
    fast_estimate = fast_plan.curvatures["control"]
    record_row(
        "10 a channel with content above the estimator bandwidth refuses the linear method",
        "a constructed sinusoid at 0.35 of the sampling frequency, whose analytic "
        f"curvature is {0.5 * (2.0 * np.pi * 0.35) ** 2:.6g} per s^2",
        True,
        any(
            "curvature_resolvable" in decision.failed_gates
            for decision in fast_plan.decisions
        ),
        f"estimate {fast_estimate.value:.6g} against analytic "
        f"{0.5 * (2.0 * np.pi * 0.35) ** 2:.6g}, high-frequency fraction "
        f"{fast_estimate.high_frequency_fraction:.6f}",
    )

    out.rule("controls 11 to 15: the classification and the determinability limits")
    sample = dataset.test_record(samples=4000)
    full = classify.classify(sample.record)
    no_raw = classify.classify(sample.record.without_raw_words())
    no_validity = classify.classify(sample.record.without_validity())
    record_row(
        "11 withholding the raw words makes the misparsed class unreachable",
        "the ground-truth misparse count of the generated record, "
        f"{int((sample.value_truth == 2).sum())} value-level misparses",
        True,
        all(
            anomaly.verdict is not classify.AnomalyClass.MISPARSED
            for anomaly in no_raw.anomalies
            if anomaly.channel >= 0
        ),
        f"misparsed verdicts at the value level: "
        f"{sum(1 for a in full.anomalies if a.channel >= 0 and a.verdict.value == 'misparsed')}"
        f" -> "
        f"{sum(1 for a in no_raw.anomalies if a.channel >= 0 and a.verdict.value == 'misparsed')}",
    )
    record_row(
        "12 withholding the validity flag makes the corrupt class unreachable",
        "the ground-truth corruption count of the generated record, "
        f"{int((sample.value_truth == 1).sum())} value-level corruptions",
        True,
        all(
            anomaly.verdict is not classify.AnomalyClass.CORRUPT
            for anomaly in no_validity.anomalies
        ),
        f"corrupt verdicts: {full.counts['corrupt']} -> {no_validity.counts['corrupt']}",
    )

    # 13. A clean record must produce nothing.
    pristine = single_channel_record(
        np.full(60, 3.0),
        np.arange(60, dtype=np.float64),
        np.arange(60, dtype=np.int64),
        FLAT_CHANNEL,
        modulus=4096,
    )
    pristine_report = classify.classify(pristine)
    record_row(
        "13 a constructed clean record produces no anomaly",
        "a flat in-range signal on a nominal timeline, constructed with nothing injected",
        False,
        len(pristine_report.anomalies) > 0,
        f"{len(pristine_report.anomalies)} anomalies, counts {pristine_report.counts}",
    )

    # 14. A forward gap beyond the reorder horizon reads as a reorder.
    modulus = 256
    horizon = modulus // 2
    long_gap = modulus - horizon + 4
    beyond = single_channel_record(
        np.full(2, 3.0),
        np.array([0.0, float(long_gap + 1)]),
        np.array([0, long_gap + 1], dtype=np.int64),
        FLAT_CHANNEL,
        modulus=modulus,
    )
    finding = screen_counters(beyond)[0]
    limit = classify.classify(beyond).determinability["counter_horizon"]
    record_row(
        "14 a forward gap beyond the reorder horizon is read as a reorder",
        f"constructed counter arithmetic: modulus {modulus}, horizon {horizon}, a true "
        f"forward gap of {long_gap} frames",
        True,
        finding.kind == "reorder",
        f"counter delta {finding.delta} read as {finding.kind!r}; the counter_horizon "
        f"limit states {limit.quantity:g} s and is marked "
        f"{'determinable' if limit.determinable else 'NOT determinable'}",
    )

    # 15. Positive control: each injected misparse kind is classified misparsed.
    kinds: dict[str, tuple[int, int]] = {}
    value_level = {
        (anomaly.index, anomaly.channel): anomaly.verdict.value
        for anomaly in full.anomalies
        if anomaly.channel >= 0
    }
    for kind in ("signedness", "byteswap", "scale_decade_down"):
        positions = np.argwhere(sample.misparse_kind == kind)
        hit = sum(
            1
            for index, column in positions
            if value_level.get((int(index), int(column))) == "misparsed"
        )
        kinds[kind] = (hit, int(positions.shape[0]))
    record_row(
        "15 every injected misparse kind is detected as misparsed",
        "the generator's ground-truth misparse kind, per frame and channel",
        True,
        all(hit > 0 for hit, _ in kinds.values()),
        ", ".join(f"{kind}: {hit}/{total}" for kind, (hit, total) in kinds.items()),
    )

    out.rule("control summary")
    out(
        format_table(
            ["control", "anchored on", "direction", "outcome", "verdict", "detail"],
            rows,
        )
    )
    out("")
    fired = sum(1 for row in rows if row[3] == "fired")
    must_fire = sum(1 for row in rows if row[2] == "MUST FIRE")
    out(f"controls: {len(rows)}; must fire: {must_fire}; fired: {fired}")
    out.check(
        "every control that must fire did, and every control that must not did not",
        all(row[4] == "ok" for row in rows),
        f"{sum(1 for row in rows if row[4] == 'ok')} of {len(rows)}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
