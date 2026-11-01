"""Unit, known-answer and input-validation tests for the sample-count inversions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from transfergap.bounds import empirical_bernstein_slack
from transfergap.plan import plan_existence, plan_for_slack


def test_plan_for_slack_returns_both_mean_bounds(small_field):
    plans = plan_for_slack(small_field, 2.0, 0.95)
    assert [plan.method for plan in plans] == ["hoeffding", "empirical_bernstein"]
    assert [plan.basis for plan in plans] == ["exact", "plug-in"]


def test_hoeffding_plan_known_answer(small_field):
    """Hand calculation at R = 40 kN.m, eps = 2 kN.m, 1-delta = 0.95.

    R^2 ln(1/delta) / (2 eps^2) = 1600 * 2.995732273553991 / 8
                                = 4793.1716376863855 / 8 ... no:
      1600 * 2.995732273553991 = 4793.171637686386
      / (2 * 4)                = 599.1464547107982
    ceil                        = 600
    additional at n = 15        = 585
    """
    plan = plan_for_slack(small_field, 2.0, 0.95)[0]
    assert plan.required_n == 600
    assert plan.current_n == 15
    assert plan.additional_n == 585
    assert plan.achieved_slack is not None
    assert plan.achieved_slack <= 2.0


def test_hoeffding_plan_achieved_slack_matches_the_forward_expression(small_field):
    plan = plan_for_slack(small_field, 2.0, 0.95)[0]
    expected = 40.0 * math.sqrt(math.log(20.0) / (2.0 * plan.required_n))
    assert plan.achieved_slack == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_bernstein_plan_achieved_slack_matches_the_forward_expression(small_field):
    plan = plan_for_slack(small_field, 2.0, 0.95)[1]
    variance = float(np.var(small_field.gaps, ddof=1))
    expected = empirical_bernstein_slack(plan.required_n, variance, 40.0, 0.95)
    assert plan.achieved_slack == pytest.approx(expected, rel=1e-13, abs=0.0)
    assert plan.achieved_slack <= 2.0


def test_bernstein_plan_is_labelled_a_prediction(small_field):
    plan = plan_for_slack(small_field, 2.0, 0.95)[1]
    assert plan.basis == "plug-in"
    assert "PREDICTION" in plan.note


def test_bernstein_plan_needs_fewer_samples_than_hoeffding_at_this_target(small_field):
    """The variance penalty pays off once the target slack is small.

    At a 2 kN.m target with R = 40 kN.m the Hoeffding count is 600 and the
    empirical Bernstein count is smaller, because the 1/(n-1) term has become
    negligible by then. That ordering reverses at a loose target, which is why
    both are always reported.
    """
    hoeffding, bernstein = plan_for_slack(small_field, 2.0, 0.95)
    assert bernstein.required_n < hoeffding.required_n


def test_plan_existence_known_answers(small_field):
    """Wilks needs 59 by equation (7); split conformal needs 19 by equation (12)."""
    wilks, conformal = plan_existence(small_field, 0.95, 0.95)
    assert wilks.required_n == 59
    assert wilks.additional_n == 44
    assert conformal.required_n == 19
    assert conformal.additional_n == 4
    assert wilks.basis == conformal.basis == "existence"
    assert wilks.target_slack is None


def test_plan_existence_marks_conformal_as_the_weaker_statement(small_field):
    _, conformal = plan_existence(small_field, 0.95, 0.95)
    assert "MARGINAL" in conformal.note


def test_plan_additional_is_zero_when_the_sample_already_suffices(field_sample):
    _, conformal = plan_existence(field_sample, 0.95, 0.95)
    assert conformal.required_n == 19
    assert conformal.additional_n == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_plan_for_slack_rejects_a_bad_target(small_field, bad):
    with pytest.raises(ValueError, match="target_slack"):
        plan_for_slack(small_field, bad, 0.95)


def test_plan_for_slack_rejects_a_bad_confidence(small_field):
    with pytest.raises(ValueError, match="strictly inside"):
        plan_for_slack(small_field, 2.0, 1.0)
