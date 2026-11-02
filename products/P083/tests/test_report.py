"""Rendering: the table primitive and every renderer, including the empty cases."""

from __future__ import annotations

import numpy as np
import pytest

from hilverdict.certificate import certify_run
from hilverdict.clock import estimate_clock
from hilverdict.report import (
    format_table,
    render_array,
    render_certificates,
    render_clock,
    render_resimulations,
    render_run_certificate,
    render_scores,
    render_verdict,
)
from hilverdict.resimulate import resimulate_run
from hilverdict.scoring import score_decision
from hilverdict.synthetic import generate_run
from hilverdict.verdict import run_verdict


def test_format_table_has_a_header_and_a_rule():
    text = format_table(["a", "b"], [[1, 2]])
    lines = text.splitlines()
    assert lines[0].split() == ["a", "b"]
    assert set(lines[1].replace(" ", "")) == {"-"}


def test_format_table_sizes_columns_to_the_widest_cell():
    text = format_table(["a"], [["xxxxx"]])
    assert all(len(line) >= 5 for line in text.splitlines())


def test_format_table_with_no_rows_is_just_the_header():
    assert len(format_table(["a", "b"], []).splitlines()) == 2


def test_format_table_rejects_a_short_row():
    with pytest.raises(ValueError, match="has 1 cells"):
        format_table(["a", "b"], [[1]])


def test_format_table_rejects_a_long_row():
    with pytest.raises(ValueError, match="has 3 cells"):
        format_table(["a", "b"], [[1, 2, 3]])


def test_format_table_stringifies_every_cell():
    text = format_table(["a"], [[None]])
    assert "None" in text


def test_render_certificates_names_every_verdict(long_run):
    certificate = certify_run(long_run, 1.0)
    text = render_certificates(certificate.certificates)
    assert "DISMISSED" in text
    assert "proof" in text


def test_render_certificates_honours_the_limit(long_run):
    certificate = certify_run(long_run, 1.0)
    text = render_certificates(certificate.certificates, limit=3)
    assert "showing 3 of" in text


def test_render_certificates_with_no_misses_still_renders():
    text = render_certificates([])
    assert "iter" in text


def test_render_run_certificate_shows_the_threshold(long_run):
    text = render_run_certificate(certify_run(long_run, 1.0))
    assert "certified overrun" in text
    assert "dismissal rate" in text


def test_render_run_certificate_puts_refusals_first():
    log = generate_run(seed=3, profile="loaded", amplitude_deg=14.0, steps=100)
    text = render_run_certificate(certify_run(log, 1.0))
    assert text.startswith("REFUSED")


def test_render_resimulations_names_the_cost(loaded_run):
    text = render_resimulations(resimulate_run(loaded_run, 1.0))
    assert "plant calls" in text
    assert "peak divergence" in text


def test_render_resimulations_with_nothing_renders_the_header():
    assert "iter" in render_resimulations([])


def test_render_clock_puts_the_verdict_first(loaded_run):
    text = render_clock(
        estimate_clock(loaded_run.host_time_s, loaded_run.device_time_s, 100.0)
    )
    assert text.startswith("CLOCK VERDICT:")
    assert "theil-sen" in text


def test_render_scores_marks_the_proof_column():
    truth = np.array([True, False, True])
    scores = [
        score_decision("proved", np.ones(3, dtype=bool), truth, sound=True),
        score_decision("guessed", np.array([True, False, False]), truth),
    ]
    text = render_scores(scores)
    assert "yes" in text
    assert "NO" in text
    assert "recall" in text


def test_render_scores_with_no_methods_renders_the_header():
    assert "method" in render_scores([])


def test_render_verdict_names_the_verdict(long_run):
    text = render_verdict(run_verdict(long_run, 1.0))
    assert text.startswith("RUN VERDICT:")
    assert "cascade / direct overruns" in text


def test_render_verdict_states_the_safety_wording(long_run):
    text = render_verdict(run_verdict(long_run, 1.0))
    assert "not a certification" in text


def test_render_verdict_for_an_undecidable_run_lists_the_reason():
    from hilverdict.synthetic import ClockSpec

    spec = ClockSpec(drift_ppm=0.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(0.5,), name="t")
    log = generate_run(seed=3, profile="loaded", amplitude_deg=6.0, steps=100,
                       clock=spec)
    text = render_verdict(run_verdict(log, 1.0))
    assert "UNDECIDABLE" in text
    assert "reset" in text


def test_render_array_prints_every_value():
    text = render_array("x", np.array([1.0, 2.5]))
    assert text.startswith("x: ")
    assert "2.5000" in text


def test_render_array_spells_out_non_finite_values():
    text = render_array("x", np.array([np.nan, np.inf, -np.inf]))
    assert "nan" in text
    assert "inf" in text
    assert "-inf" in text


def test_renderers_use_scientific_notation_for_tiny_numbers():
    text = render_array("x", np.array([1e-9]))
    assert "e-" in text
