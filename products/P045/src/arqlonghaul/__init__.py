"""arqlonghaul: ARQ and hybrid-ARQ goodput where the round-trip time dominates.

Research-grade.  Not flight-qualified, not certified, not approved for
operational aerospace use.

Module map, in the order the package is meant to be read:

    link        link geometry, the bandwidth-delay product, N = 1 + RTT/T_f
    closedform  the classical ARQ throughput expressions and their validity
    protocols   slotted state-machine simulators for the same three protocols
    window      window sizing against the bandwidth-delay product
    channel     independent and Gilbert-Elliott frame-error channels
    crc         in-package CRC frame check, verified against two references
    harq        type-I chase combining and type-II incremental redundancy
    datasets    the learned-policy environment and the three disjoint seed sets
    policy      three baselines, then the learned redundancy policy
"""

from __future__ import annotations

__version__ = "0.1.0"

from .channel import (
    FrameChannel,
    GilbertElliottChannel,
    IndependentFrameChannel,
    ber_to_fer,
    bpsk_ber,
    fer_to_ber,
)
from .closedform import (
    expected_transmissions,
    gbn_throughput,
    slots_per_cycle,
    sr_throughput,
    sw_throughput,
    throughput,
)
from .crc import CATALOGUE, CRC16_CCITT_FALSE, CRC32, CrcSpec, append_fcs, check_fcs, crc
from .datasets import (
    DEFAULT_ACTIONS,
    SEED_SPLIT,
    FadeHarqEnv,
    SeedSplit,
    long_burst_env,
    short_burst_env,
)
from .harq import (
    HarqConfig,
    HarqResult,
    block_fer,
    crossover_rtt,
    harq_goodput,
    harq_goodput_exact,
    harq_simulate,
    optimal_first_rate,
    schedule_metrics,
    singleton_t,
)
from .link import PRESETS, LinkParams, preset
from .policy import (
    FEATURE_NAMES,
    AnalyticFixedPolicy,
    EscalatingPolicy,
    LearnedRedundancyPolicy,
    TunedFixedPolicy,
    analytic_fixed,
    collect_transitions,
    evaluate,
    run_episode,
    tune_escalating,
    tune_fixed,
)
from .protocols import (
    PROTOCOL_NAMES,
    SimResult,
    simulate,
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)
from .window import WindowSizing, marginal_gain, size_window, window_sweep

__all__ = [
    "__version__",
    # link
    "LinkParams",
    "PRESETS",
    "preset",
    # closed forms
    "slots_per_cycle",
    "sw_throughput",
    "gbn_throughput",
    "sr_throughput",
    "throughput",
    "expected_transmissions",
    # simulators
    "SimResult",
    "simulate",
    "simulate_stop_and_wait",
    "simulate_go_back_n",
    "simulate_selective_repeat",
    "PROTOCOL_NAMES",
    # window
    "WindowSizing",
    "size_window",
    "window_sweep",
    "marginal_gain",
    # channel
    "FrameChannel",
    "IndependentFrameChannel",
    "GilbertElliottChannel",
    "bpsk_ber",
    "ber_to_fer",
    "fer_to_ber",
    # crc
    "CrcSpec",
    "CRC32",
    "CRC16_CCITT_FALSE",
    "CATALOGUE",
    "crc",
    "append_fcs",
    "check_fcs",
    # harq
    "HarqConfig",
    "HarqResult",
    "block_fer",
    "singleton_t",
    "harq_goodput",
    "harq_goodput_exact",
    "harq_simulate",
    "schedule_metrics",
    "optimal_first_rate",
    "crossover_rtt",
    # datasets
    "FadeHarqEnv",
    "SeedSplit",
    "SEED_SPLIT",
    "DEFAULT_ACTIONS",
    "long_burst_env",
    "short_burst_env",
    # policy
    "FEATURE_NAMES",
    "AnalyticFixedPolicy",
    "TunedFixedPolicy",
    "EscalatingPolicy",
    "LearnedRedundancyPolicy",
    "analytic_fixed",
    "tune_fixed",
    "tune_escalating",
    "collect_transitions",
    "run_episode",
    "evaluate",
]
