"""Property-based tests. Hypothesis searches where hand-written cases do not.

On P071 of this mission Hypothesis found six equation defects that the
hand-written tests did not. The properties below are algebraic identities and
simulator invariants that must hold for every admissible input, not for the
inputs a person thought of. Every counterexample Hypothesis has found in this
package is pinned as a named regression test with literal inputs in
``tests/test_regressions.py``, because ``.hypothesis/`` is not committed and a
falsifying example found by one run is otherwise lost on a fresh clone.

Floating-point comparisons here carry a tolerance on both sides, and no test
asserts equality of a boolean produced by a strict threshold comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from relfiledelivery.analytic import (
    analytic_delivery,
    analytic_delivery_time_quantile,
    expected_rounds,
    expected_total_transmissions,
    extended_expected_delivery_time,
    rounds_cdf,
    rounds_quantile,
    transmissions_quantile,
)
from relfiledelivery.link import (
    ContactSchedule,
    LinkConditions,
    OutageModel,
    OutageTrace,
    uniform_windows,
)
from relfiledelivery.modes import ALL_MODES, TransferMode
from relfiledelivery.payload import FileTransferSpec, digest, reassemble, source_payload
from relfiledelivery.simulate import accounting_discrepancy, simulate_transfer

SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

counts = st.integers(min_value=1, max_value=400)
losses = st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False)
levels = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
payloads = st.sampled_from([64, 100, 256, 512, 1024])
headers = st.integers(min_value=0, max_value=32)
rates = st.floats(min_value=100.0, max_value=5.0e6, allow_nan=False, allow_infinity=False)
light_times = st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=2**31 - 1)


@st.composite
def specs(draw: st.DrawFn) -> FileTransferSpec:
    payload = draw(payloads)
    count = draw(st.integers(min_value=1, max_value=60))
    tail = draw(st.integers(min_value=1, max_value=payload))
    return FileTransferSpec((count - 1) * payload + tail, payload, draw(headers))


@st.composite
def clean_links(draw: st.DrawFn) -> LinkConditions:
    forward = draw(rates)
    asymmetry = draw(st.floats(min_value=1.0, max_value=300.0, allow_nan=False))
    return LinkConditions(
        forward_rate_bps=forward,
        return_rate_bps=forward / asymmetry,
        one_way_light_time_s=draw(light_times),
        pdu_loss_probability=draw(
            st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False)
        ),
    )


class TestRoundsAlgebra:
    @SETTINGS
    @given(counts, losses)
    def test_expected_rounds_is_bracketed_by_the_single_unit_and_the_total(
        self, n: int, p: float
    ) -> None:
        """``1/(1-p) <= E[max G_i] <= sum E[G_i]``: a maximum is at least any
        one term and at most the sum of non-negative terms."""
        single = 1.0 / (1.0 - p)
        total = expected_total_transmissions(n, p)
        value = expected_rounds(n, p)
        assert value >= single - 1e-9
        assert value <= total + 1e-9

    @SETTINGS
    @given(counts, losses, losses)
    def test_expected_rounds_is_monotone_in_the_loss_probability(
        self, n: int, p_low: float, p_high: float
    ) -> None:
        lo, hi = sorted((p_low, p_high))
        assert expected_rounds(n, lo) <= expected_rounds(n, hi) + 1e-9

    @SETTINGS
    @given(st.integers(min_value=1, max_value=200), losses)
    def test_expected_rounds_is_monotone_in_the_unit_count(
        self, n: int, p: float
    ) -> None:
        assert expected_rounds(n, p) <= expected_rounds(n + 1, p) + 1e-9

    @SETTINGS
    @given(counts, losses, st.integers(min_value=0, max_value=60))
    def test_rounds_cdf_is_a_cdf(self, n: int, p: float, k: int) -> None:
        here = rounds_cdf(n, p, k)
        there = rounds_cdf(n, p, k + 1)
        assert 0.0 <= here <= 1.0
        assert here <= there + 1e-12

    @SETTINGS
    @given(counts, losses, levels)
    def test_rounds_quantile_satisfies_its_own_definition(
        self, n: int, p: float, q: float
    ) -> None:
        k = rounds_quantile(n, p, q)
        assert k >= 1
        assert rounds_cdf(n, p, k) >= q - 1e-12
        if k > 1:
            assert rounds_cdf(n, p, k - 1) <= q + 1e-12


class TestTransmissionAlgebra:
    @SETTINGS
    @given(counts, losses, levels)
    def test_transmission_quantile_is_at_least_the_unit_count(
        self, n: int, p: float, q: float
    ) -> None:
        assert transmissions_quantile(n, p, q) >= n - 1e-9

    @SETTINGS
    @given(counts, losses, levels, levels)
    def test_transmission_quantile_is_monotone_in_the_level(
        self, n: int, p: float, q_low: float, q_high: float
    ) -> None:
        lo, hi = sorted((q_low, q_high))
        assert transmissions_quantile(n, p, lo) <= transmissions_quantile(n, p, hi) + 1e-9


class TestClosedForm:
    @SETTINGS
    @given(specs(), clean_links())
    def test_expected_delivery_time_exceeds_its_own_lower_bound(
        self, spec: FileTransferSpec, link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(spec, link, TransferMode.DEFERRED_NAK)
        assert prediction.expected_delivery_time_s >= (
            prediction.delivery_time_lower_bound_s - 1e-9
        )

    @SETTINGS
    @given(specs(), clean_links())
    def test_expected_transmissions_is_mode_independent(
        self, spec: FileTransferSpec, link: LinkConditions
    ) -> None:
        deferred = analytic_delivery(spec, link, TransferMode.DEFERRED_NAK)
        immediate = analytic_delivery(spec, link, TransferMode.IMMEDIATE_NAK)
        assert deferred.expected_transmissions == pytest.approx(
            immediate.expected_transmissions, rel=1e-12
        )

    @SETTINGS
    @given(specs(), clean_links())
    def test_overhead_ratio_is_at_least_the_lossless_floor(
        self, spec: FileTransferSpec, link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(spec, link, TransferMode.DEFERRED_NAK)
        assert prediction.expected_overhead_ratio >= spec.minimum_overhead_ratio - 1e-12

    @SETTINGS
    @given(specs(), clean_links(), levels, levels)
    def test_semi_analytic_quantile_is_monotone_in_the_level(
        self, spec: FileTransferSpec, link: LinkConditions, q_low: float, q_high: float
    ) -> None:
        lo, hi = sorted((q_low, q_high))
        assert analytic_delivery_time_quantile(
            spec, link, lo
        ) <= analytic_delivery_time_quantile(spec, link, hi) + 1e-9

    @SETTINGS
    @given(specs(), clean_links())
    def test_unacknowledged_completion_probability_is_a_probability(
        self, spec: FileTransferSpec, link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(spec, link, TransferMode.UNACKNOWLEDGED)
        assert 0.0 <= prediction.completion_probability <= 1.0

    @SETTINGS
    @given(
        specs(),
        clean_links(),
        st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
        st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    )
    def test_the_extension_never_shortens_the_closed_form(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mean_up: float,
        mean_down: float,
    ) -> None:
        """Dividing by an availability of at most 1 cannot make a transfer faster."""
        from dataclasses import replace

        bursty = replace(link, outage=OutageModel(mean_up_s=mean_up, mean_down_s=mean_down))
        plain = analytic_delivery(
            spec, link, TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        extended = extended_expected_delivery_time(spec, bursty, TransferMode.DEFERRED_NAK)
        assert extended >= plain - 1e-9


class TestSpecAlgebra:
    @SETTINGS
    @given(specs())
    def test_slices_partition_the_file(self, spec: FileTransferSpec) -> None:
        slices = spec.slices()
        assert len(slices) == spec.pdu_count
        assert slices[0][0] == 0
        assert slices[-1][1] == spec.file_bytes
        assert sum(stop - start for start, stop in slices) == spec.file_bytes
        for (_, stop), (start, _) in zip(slices, slices[1:], strict=False):
            assert stop == start

    @SETTINGS
    @given(specs())
    def test_wire_bytes_sum_to_the_declared_total(self, spec: FileTransferSpec) -> None:
        assert sum(
            spec.wire_bytes_of(index) for index in range(spec.pdu_count)
        ) == spec.total_wire_bytes

    @SETTINGS
    @given(specs(), seeds)
    def test_payload_round_trips_through_reassembly(
        self, spec: FileTransferSpec, seed: int
    ) -> None:
        source = source_payload(spec, seed)
        received = {
            index: source[start:stop] for index, (start, stop) in enumerate(spec.slices())
        }
        assert digest(reassemble(spec, received)) == digest(source)


class TestLinkAlgebra:
    @SETTINGS
    @given(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    )
    def test_duty_cycle_is_a_probability(self, mean_up: float, mean_down: float) -> None:
        duty = OutageModel(mean_up_s=mean_up, mean_down_s=mean_down).duty_cycle
        assert 0.0 < duty <= 1.0

    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=12),
        st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
        st.floats(min_value=1.0, max_value=20.0, allow_nan=False),
    )
    def test_contact_duty_cycle_is_a_probability(
        self, count: int, duration: float, stretch: float
    ) -> None:
        windows = uniform_windows(count, duration, duration * stretch)
        link = LinkConditions(1000.0, 100.0, windows=windows)
        assert 0.0 < link.contact_duty_cycle <= 1.0
        assert 0.0 < link.availability <= 1.0

    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=8),
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    )
    def test_placed_transmissions_lie_inside_a_declared_window(
        self,
        count: int,
        duration: float,
        stretch: float,
        after: float,
        request: float,
    ) -> None:
        windows = uniform_windows(count, duration, duration * stretch)
        schedule = ContactSchedule(windows)
        placed = schedule.place(after, request)
        if placed is None:
            return
        start, end, index = placed
        window = windows[index]
        assert start >= window.start_s - 1e-9
        assert end <= window.end_s + 1e-9
        assert start >= after - 1e-9
        assert end - start == pytest.approx(request, abs=1e-9)

    @SETTINGS
    @given(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
        seeds,
        st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
    )
    def test_outage_time_never_exceeds_the_span(
        self, mean_up: float, mean_down: float, seed: int, span: float
    ) -> None:
        trace = OutageTrace(
            OutageModel(mean_up_s=mean_up, mean_down_s=mean_down),
            np.random.default_rng(seed),
        )
        down = trace.outage_seconds_in(0.0, span)
        assert -1e-9 <= down <= span + 1e-9


class TestSimulatorInvariants:
    @SETTINGS
    @given(specs(), clean_links(), st.sampled_from(ALL_MODES), seeds)
    def test_accounting_closes_on_every_run(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        result = simulate_transfer(spec, link, mode, seed)
        assert accounting_discrepancy(result) == 0
        assert result.forward_transmissions >= result.received_count

    @SETTINGS
    @given(specs(), clean_links(), st.sampled_from(ALL_MODES), seeds)
    def test_a_delivered_run_reproduces_the_source_digest(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        result = simulate_transfer(spec, link, mode, seed)
        if result.delivered:
            assert result.delivered_digest == result.source_digest
            assert result.overhead_ratio >= spec.minimum_overhead_ratio - 1e-12
        else:
            assert result.delivered_digest is None

    @SETTINGS
    @given(specs(), clean_links(), st.sampled_from(ALL_MODES), seeds)
    def test_idle_time_is_never_negative(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        """Every forward transmission precedes the completing arrival, so the
        delivery time cannot be shorter than the forward occupancy."""
        result = simulate_transfer(spec, link, mode, seed)
        if result.delivered:
            assert result.idle_s >= -1e-9

    @SETTINGS
    @given(specs(), clean_links(), st.sampled_from(ALL_MODES), seeds)
    def test_identical_seeds_give_identical_results(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        assert simulate_transfer(spec, link, mode, seed) == simulate_transfer(
            spec, link, mode, seed
        )

    @SETTINGS
    @given(specs(), clean_links(), seeds)
    def test_unacknowledged_sends_each_unit_exactly_once(
        self, spec: FileTransferSpec, link: LinkConditions, seed: int
    ) -> None:
        result = simulate_transfer(spec, link, TransferMode.UNACKNOWLEDGED, seed)
        assert result.forward_transmissions == spec.pdu_count
        assert result.retransmissions == 0
        assert set(result.attempts) <= {1}

    @SETTINGS
    @given(specs(), clean_links(), st.sampled_from(ALL_MODES), seeds)
    def test_a_lossless_link_delivers_in_one_pass(
        self,
        spec: FileTransferSpec,
        link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        from dataclasses import replace

        lossless = replace(link, pdu_loss_probability=0.0)
        result = simulate_transfer(spec, lossless, mode, seed)
        assert result.delivered
        assert result.forward_transmissions == spec.pdu_count
        reference = (
            8.0 * spec.total_wire_bytes / lossless.forward_rate_bps
            + lossless.one_way_light_time_s
        )
        assert result.delivery_time_s == pytest.approx(reference, rel=1e-9)
