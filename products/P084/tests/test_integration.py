"""The integration test: the whole pipeline, baseline first, model benchmarked after.

This is one end-to-end run in the order the product is meant to be used:

1. the field sample arrives;
2. exchangeability is TESTED before anything that depends on it is reported;
3. the distribution-free baseline is computed and is the reference;
4. the resampling bounds are computed and compared to it;
5. the learned model is fitted and benchmarked against both on the same
   held-out pool;
6. the sample-count inversion says how many more flights each method needs.

The coverage numbers asserted here are from ONE draw and are deliberately
loose. The repeated-draw measurement is in ``validation/`` and is what the
product is scored on.
"""

from __future__ import annotations

import numpy as np

from transfergap.bounds import (
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    wilks_tolerance_upper,
)
from transfergap.exchangeability import permutation_test
from transfergap.learned import ConformalisedDiscrepancyModel, DiscrepancyModel
from transfergap.plan import plan_existence, plan_for_slack
from transfergap.report import render_bounds, render_exchangeability, render_plan
from transfergap.resample import bootstrap_mean_upper, split_conformal_upper


def test_end_to_end_pipeline_on_a_sixty_point_campaign(field_sample, pool, truth):
    # 2. the assumption is tested BEFORE the bounds that rest on it are used.
    tests = [
        permutation_test(field_sample.gaps, name, permutations=299, seed=1)
        for name in ("trend", "half_split")
    ]
    assert render_exchangeability(tests)
    assert all(not test.rejected for test in tests), (
        "the exchangeable campaign was flagged; the bootstrap and conformal rows "
        "below would then carry no guarantee"
    )

    # 3. baseline 1, the distribution-free reference.
    hoeffding = hoeffding_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, field_sample.unit
    )
    bernstein = empirical_bernstein_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, field_sample.unit
    )
    wilks = wilks_tolerance_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, 0.95, field_sample.unit
    )
    normal = normal_tolerance_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, 0.95, field_sample.unit
    )
    assert hoeffding.upper > truth.mean
    assert bernstein.upper > truth.mean

    # 4. baseline 2, tighter and conditional on the assumption just tested.
    bootstrap = bootstrap_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, resamples=2000, seed=1,
        unit=field_sample.unit,
    )
    conformal = split_conformal_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, field_sample.unit
    )
    assert bootstrap.slack < hoeffding.slack, (
        "the bootstrap is supposed to be the tighter answer; if it is not, the "
        "comparison in the README is wrong"
    )

    # the full table renders with both targets side by side and marked apart.
    text = render_bounds(
        [hoeffding, bernstein, bootstrap, wilks, normal, conformal], field_sample.unit
    )
    assert "not comparable" in text

    # 5. the learned model, benchmarked on the SAME held-out pool.
    model = DiscrepancyModel(upper_level=0.95, lower_level=0.05, seed=7)
    model.fit(field_sample.features, field_sample.gaps)
    prediction = model.predict_with_uncertainty(pool.features)
    learned_coverage = float(np.mean(pool.gaps <= prediction.upper))

    calibrated = ConformalisedDiscrepancyModel(confidence=0.95, seed=7)
    calibrated.fit(field_sample.features, field_sample.gaps)
    calibrated_coverage = float(np.mean(pool.gaps <= calibrated.predict_upper(pool.features)))

    wilks_coverage = float(np.mean(pool.gaps <= wilks.upper))
    conformal_coverage = float(np.mean(pool.gaps <= conformal.upper))

    # The expected result, stated in the specification and measured here: at a
    # field sample size of tens the raw learned quantile under-covers its
    # declared 0.95 and the distribution-free limits do not. One draw is not
    # evidence; validation/validate_learned_model_output.txt is. What this
    # assertion pins is the DIRECTION, with a margin wide enough that it is
    # not asserting a seed.
    assert learned_coverage < 0.95
    assert wilks_coverage > 0.95
    assert calibrated_coverage > conformal_coverage - 0.2

    # 6. the inversion: how many more flights, and on what basis.
    plans = plan_for_slack(field_sample, 2.0, 0.95) + plan_existence(field_sample, 0.95, 0.95)
    assert [plan.basis for plan in plans] == ["exact", "plug-in", "existence", "existence"]
    assert plans[0].additional_n > 0
    assert render_plan(plans)


def test_end_to_end_on_a_non_exchangeable_campaign_flags_the_assumption(swept_sample):
    """The same pipeline on a swept card: the test fires and the report says so.

    This is the integration-level negative control. The bounds still COMPUTE
    on a non-exchangeable sample -- nothing raises -- and the only thing that
    stops a reader trusting the conformal row is the exchangeability verdict
    printed above it.
    """
    tests = [
        permutation_test(swept_sample.gaps, name, permutations=299, seed=2)
        for name in ("trend", "half_split")
    ]
    assert any(test.rejected for test in tests)
    text = render_exchangeability(tests)
    assert "REJECTED" in text
    assert "no guarantee" in text

    conformal = split_conformal_upper(
        swept_sample.gaps, swept_sample.bound_range, 0.95, swept_sample.unit
    )
    assert conformal.upper > 0.0
