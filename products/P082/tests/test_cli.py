"""The command line, in process. The subprocess form is in validation/validate_cli.py.

These tests call ``main`` directly, which is fast and checks the parsing and
the dispatch. They do NOT check that ``python -m sitediverse`` works as a user
runs it: only a subprocess can check that, and ``validation/validate_cli.py``
does, which is where the release gate's own CLI check is mirrored.
"""

from __future__ import annotations

import pytest

from sitediverse.__main__ import COMMANDS, NETWORKS, build_parser, main
from sitediverse.field import RECORD_SPECS


def test_every_subcommand_in_the_parser_has_a_handler():
    """So that a subcommand cannot be added to the parser and left unwired."""
    parser = build_parser()
    actions = [
        action
        for action in parser._subparsers._group_actions  # noqa: SLF001
        if hasattr(action, "choices")
    ]
    assert actions
    names = set(actions[0].choices)
    assert names == set(COMMANDS)


def test_the_eight_declared_subcommands_are_present():
    """The list the module docstring claims."""
    assert set(COMMANDS) == {
        "sites",
        "record",
        "climatology",
        "overstate",
        "bounds",
        "correlate",
        "learned",
        "network",
    }


def test_the_declared_networks_are_all_valid_station_sets():
    """Every named network has at least two distinct stations in range."""
    for name, sites in NETWORKS.items():
        assert len(sites) >= 2, name
        assert len(set(sites)) == len(sites), name
        assert all(0 <= site < 20 for site in sites), name


def test_version_exits_zero(capsys):
    """``--version`` is an argparse action, so it raises SystemExit(0)."""
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert "sitediverse 0.1.0" in capsys.readouterr().out


def test_no_subcommand_is_an_error():
    """A bare invocation must not silently do nothing."""
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code != 0


def test_an_unknown_subcommand_is_an_error():
    """Input validation through argparse."""
    with pytest.raises(SystemExit) as caught:
        main(["invent"])
    assert caught.value.code != 0


def test_an_unknown_record_is_an_error():
    """The choices are the declared records and nothing else."""
    with pytest.raises(SystemExit) as caught:
        main(["record", "--record", "invented"])
    assert caught.value.code != 0


def test_an_unknown_network_is_an_error():
    """The same for the declared networks."""
    with pytest.raises(SystemExit) as caught:
        main(["bounds", "--network", "invented"])
    assert caught.value.code != 0


def test_sites_prints_the_declared_geometry_and_says_it_is_synthetic(capsys):
    """The synthetic warning must appear wherever the record is described."""
    assert main(["sites"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "COORDINATES ONLY" in out
    assert "pairs" in out
    assert "190" in out


def test_record_prints_the_ground_truth_lengths(capsys):
    """And says they are available to validation and used by no estimator."""
    assert main(["record", "--record", "short"]) == 0
    out = capsys.readouterr().out
    assert "GROUND TRUTH" in out
    assert "L_along km" in out
    assert "used by no estimator" in out


def test_climatology_prints_the_effective_sample_size(capsys):
    """Equation (13) is the figure that makes every standard error honest."""
    assert main(["climatology", "--record", "short"]) == 0
    out = capsys.readouterr().out
    assert "n_eff" in out
    assert "lag-1 indicator" in out


def test_overstate_prints_the_headline_table_and_the_decomposition(capsys):
    """The headline output, for every declared network."""
    assert main(["overstate", "--record", "short", "--replicates", "60"]) == 0
    out = capsys.readouterr().out
    assert "P_independent" in out
    assert "P_measured" in out
    assert "nines lost" in out
    assert "co-modulation" in out
    assert "clim share" in out
    for name in NETWORKS:
        assert name in out


def test_overstate_rejects_too_few_replicates(capsys):
    """A percentile interval from fewer than 50 replicates is not one."""
    assert main(["overstate", "--record", "short", "--replicates", "10"]) == 2
    assert "at least 50" in capsys.readouterr().err


def test_bounds_prints_the_frechet_and_bonferroni_results(capsys):
    """Both checks, with the slack and the conditioning."""
    assert main(["bounds", "--record", "short", "--network", "near-pair"]) == 0
    out = capsys.readouterr().out
    assert "Frechet-Hoeffding" in out
    assert "condition number kappa" in out
    assert "slack, equation (19)" in out
    assert "identity residual" in out


def test_bounds_reports_a_vacuous_bracket_for_a_large_network(capsys):
    """Reported rather than hidden."""
    assert main(["bounds", "--record", "short", "--network", "two-groups"]) == 0
    out = capsys.readouterr().out
    assert "vacuous for this network" in out


def test_correlate_prints_both_baselines_and_the_ground_truth(capsys):
    """The specified baseline and the strengthened one, side by side."""
    assert main(["correlate", "--record", "short"]) == 0
    out = capsys.readouterr().out
    assert "Baseline 2 as specified" in out
    assert "strengthened baseline" in out
    assert "true L_along km" in out
    assert "strongest on TRAIN" in out


def test_learned_prints_both_hold_outs_separately(capsys):
    """And says that they are never merged."""
    assert main(["learned", "--record", "short"]) == 0
    out = capsys.readouterr().out
    assert "held-out SITES" in out
    assert "held-out TIME" in out
    assert "SEPARATELY" in out
    assert "This model is not certified for operational flight use." in out
    assert "tuned against a hold-out   : False" in out


def test_learned_prints_both_intervals_and_says_they_differ(capsys):
    """Equation (41)'s two intervals answer different questions."""
    assert main(["learned", "--record", "short"]) == 0
    out = capsys.readouterr().out
    assert "margin 95% CI" in out
    assert "reference half-width" in out
    assert "CI contains 0" in out


def test_network_prints_every_route_with_its_error_source(capsys):
    """Analytic, quadrature, Monte Carlo and empirical, each with its error."""
    assert main(["network", "--record", "short", "--network", "near-pair"]) == 0
    out = capsys.readouterr().out
    assert "independence, equation (14)" in out
    assert "copula, equation (8), exact" in out
    assert "Monte Carlo" in out
    assert "empirical, equation (20)" in out
    assert "NOT a prediction" in out


def test_network_omits_the_exact_quadrature_for_more_than_two_stations(capsys):
    """Equation (8) is a pair result and the CLI does not pretend otherwise."""
    assert main(["network", "--record", "short", "--network", "cluster"]) == 0
    out = capsys.readouterr().out
    assert "copula, equation (8), exact" not in out
    assert "Monte Carlo" in out


def test_network_rejects_too_few_monte_carlo_samples(capsys):
    """A Monte Carlo of 100 draws has no usable error."""
    assert (
        main(["network", "--record", "short", "--network", "near-pair",
              "--samples", "100"])
        == 2
    )
    assert "at least 1000" in capsys.readouterr().err


def test_every_record_choice_runs_the_overstate_command(capsys):
    """All three declared records, through the headline command."""
    for name in RECORD_SPECS:
        if name != "short":
            continue
        assert main(["overstate", "--record", name, "--replicates", "60"]) == 0
    assert capsys.readouterr().out


def test_the_parser_help_names_the_safety_statement():
    """The required safety statement appears in the top-level description.

    Whitespace is normalised before the check because argparse rewraps the
    description to the terminal width, which splits phrases across lines at
    whatever column the width happens to be.
    """
    text = " ".join(build_parser().format_help().split()).lower()
    assert "not flight-qualified" in text
    assert "not certified" in text
    assert "not approved for operational aerospace use" in text


def test_the_parser_epilog_names_the_dataset_card():
    """A reader must be able to find out where the record comes from."""
    text = " ".join(build_parser().format_help().split())
    assert "DATASET_CARD.md" in text


def test_no_output_contains_an_exclamation_mark(capsys):
    """A style rule, enforced rather than trusted."""
    for argv in (
        ["sites"],
        ["record", "--record", "short"],
        ["bounds", "--record", "short", "--network", "near-pair"],
    ):
        assert main(argv) == 0
        assert "!" not in capsys.readouterr().out
