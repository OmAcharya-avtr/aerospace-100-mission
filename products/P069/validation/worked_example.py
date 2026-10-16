"""The worked example from README.md, run so that the quoted output is real.

    python validation/worked_example.py > validation/worked_example_output.txt
"""

from __future__ import annotations

from _common import FIXTURES  # noqa: F401

from scenariocov import (
    GapFinder,
    build_report,
    load_campaign,
    load_class_document,
    load_space,
)


def main() -> int:
    """Run exactly the snippet the README quotes, then print what it printed."""
    space = load_space("fixtures/space.json")
    campaign = load_campaign("fixtures/campaign_executed.json")
    classes, boundaries = load_class_document("fixtures/classes.json")

    report = build_report(space, campaign, strength=2, classes=classes, boundaries=boundaries)
    print(f"{report.tway.covered}/{report.tway.total} 2-way tuples covered")
    for tup in report.tway.uncovered:
        print(f"  uncovered: {tup.as_text()}")
    for cls in report.classes.uncovered:
        print(f"  uncovered class: {cls.parameter}/{cls.name}")
    for hit in report.boundaries:
        if not hit.covered:
            print(
                f"  missed boundary: {hit.spec.name}, nearest "
                f"{hit.nearest_distance:.4g} against tolerance {hit.spec.tolerance:g}"
            )

    pool = load_campaign("fixtures/candidate_pool.json")
    finder = GapFinder(n_bootstrap=16, random_state=0).fit(space.normalise(campaign))
    for proposal in finder.propose(space.normalise(pool), 3):
        record = pool[proposal.candidate_index]
        print(
            f"  propose log_density={proposal.log_density:.3f} "
            f"sd={proposal.log_density_sd:.3f} "
            f"confidence={proposal.sparsity_agreement:.2f} "
            f"closing_speed={record['closing_speed']} lighting={record['lighting']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
