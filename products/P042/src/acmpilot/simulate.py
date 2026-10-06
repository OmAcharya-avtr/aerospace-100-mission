"""Episode runner: channel, feedback delay, policy, accounting.

One episode is a seeded SNR sample path from :mod:`acmpilot.channel`, a feedback
delay in whole slots, and a selection sequence from a policy, scored by
:mod:`acmpilot.accounting`. The feedback channel is **ideal except for its
age**: the report is the exact SNR of slot ``n - d``, not a quantised or noisy
estimate of it. That isolates the one effect this product is about. Report
quantisation and measurement noise would each add their own penalty, and leaving
them out means every number here is optimistic about a real terminal; this is
recorded in the README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accounting import Accounting, account
from .channel import ChannelConfig, snr_db_path
from .modcod import ModcodTable
from .policy import Policy


def delayed_observation(snr_db: np.ndarray, delay_slots: int) -> np.ndarray:
    """The channel-state report available to the transmitter at each slot.

    ``observation[n] = snr_db[n - delay_slots]``. The first ``delay_slots``
    entries are held at ``snr_db[0]`` because no earlier report exists; callers
    exclude those slots from the accounting through ``warmup``.

    Parameters
    ----------
    snr_db
        True SNR per slot, dB.
    delay_slots
        Round-trip feedback delay in whole slots, >= 0.
    """
    x = np.asarray(snr_db, dtype=float)
    if delay_slots < 0:
        raise ValueError(f"delay_slots must be >= 0, got {delay_slots}")
    if delay_slots == 0:
        return x.copy()
    if delay_slots >= x.size:
        raise ValueError(
            f"delay_slots ({delay_slots}) must be < n_slots ({x.size})"
        )
    out = np.empty_like(x)
    out[:delay_slots] = x[0]
    out[delay_slots:] = x[: x.size - delay_slots]
    return out


@dataclass(frozen=True)
class EpisodeResult:
    """One policy on one sample path.

    Attributes
    ----------
    policy_name
        As reported by the policy.
    causal
        False only for the clairvoyant upper bound.
    delay_slots
        Feedback delay used, slots.
    chosen
        Selected MODCOD index per slot.
    accounting
        Scored summary.
    """

    policy_name: str
    causal: bool
    delay_slots: int
    chosen: np.ndarray
    accounting: Accounting


def run_policy(
    table: ModcodTable,
    snr_db: np.ndarray,
    policy: Policy,
    *,
    delay_slots: int,
    warmup: int | None = None,
) -> EpisodeResult:
    """Run one policy on one SNR path and score it.

    Parameters
    ----------
    table
        MODCOD table with measured thresholds.
    snr_db
        True SNR per slot, dB.
    policy
        Any object satisfying :class:`acmpilot.policy.Policy`.
    delay_slots
        Feedback delay in whole slots, >= 0.
    warmup
        Slots excluded from accounting. Defaults to ``max(delay_slots, 50)``.
    """
    snr = np.asarray(snr_db, dtype=float)
    observation = delayed_observation(snr, delay_slots)
    chosen = np.asarray(policy.select(table, observation, snr), dtype=int)
    if chosen.shape != snr.shape:
        raise ValueError(
            f"policy {policy.name!r} returned shape {chosen.shape}, expected {snr.shape}"
        )
    pad = max(delay_slots, 50) if warmup is None else warmup
    return EpisodeResult(
        policy_name=policy.name,
        causal=bool(policy.causal),
        delay_slots=delay_slots,
        chosen=chosen,
        accounting=account(table, chosen, snr, warmup=pad),
    )


def run_episode(
    table: ModcodTable,
    config: ChannelConfig,
    policies: list[Policy] | tuple[Policy, ...],
    *,
    n_slots: int,
    seed: int,
    tau_s: float,
) -> tuple[np.ndarray, list[EpisodeResult]]:
    """Generate one sample path and run every policy on it.

    All policies see the identical path, so differences between them are not
    Monte Carlo noise between paths.

    Returns
    -------
    (snr_db, results)
    """
    snr = snr_db_path(config, n_slots, seed)
    d = config.delay_slots(tau_s)
    return snr, [run_policy(table, snr, p, delay_slots=d) for p in policies]


def sweep_delay(
    table: ModcodTable,
    config: ChannelConfig,
    policies: list[Policy] | tuple[Policy, ...],
    *,
    tau_list_s: list[float] | tuple[float, ...],
    n_slots: int = 20_000,
    seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
) -> list[dict[str, float | str | bool]]:
    """Score every policy at every feedback delay, averaged over seeds.

    Returns
    -------
    list of dict
        One row per ``(tau, policy)``, carrying ``tau_s``, ``delay_slots``,
        ``policy``, ``causal``, ``n_seeds``, every :class:`Accounting` field
        averaged over seeds, and ``goodput_sem`` --- the standard error of the
        mean goodput across seeds, so a reader can see whether two policies are
        actually separated.
    """
    rows: list[dict[str, float | str | bool]] = []
    for tau in tau_list_s:
        per_policy: dict[str, list[dict[str, float]]] = {}
        causal_flag: dict[str, bool] = {}
        for seed in seeds:
            _, results = run_episode(
                table, config, policies, n_slots=n_slots, seed=seed, tau_s=tau
            )
            for res in results:
                per_policy.setdefault(res.policy_name, []).append(
                    res.accounting.as_dict()
                )
                causal_flag[res.policy_name] = res.causal
        for name, dicts in per_policy.items():
            row: dict[str, float | str | bool] = {
                "tau_s": float(tau),
                "delay_slots": float(config.delay_slots(tau)),
                "policy": name,
                "causal": causal_flag[name],
                "n_seeds": float(len(dicts)),
            }
            for key in dicts[0]:
                row[key] = float(np.mean([d[key] for d in dicts]))
            goodputs = np.array([d["goodput_bit_per_symbol"] for d in dicts])
            row["goodput_sem"] = float(
                goodputs.std(ddof=1) / np.sqrt(goodputs.size) if goodputs.size > 1 else 0.0
            )
            rows.append(row)
    return rows
