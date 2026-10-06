"""The MODCOD ladder and its **measured** switching thresholds.

A MODCOD is a (constellation, code) pair. Its two numbers are

    spectral efficiency  eta_k = bits_per_symbol(k) * rate(k)   bit/symbol
    required SNR         thr_k = min{ SNR : BER_out(SNR) <= target }   dB

``thr_k`` is the only quantity in this package that a reader might expect to
find in a standard's table, and it is deliberately **not** taken from one. It is
produced by :func:`measure_thresholds`, which

1. measures the uncoded channel bit error rate of the constellation by Monte
   Carlo through the actual modulator and detector
   (:func:`acmpilot.modulation.measure_ber`), on an SNR grid refined until the
   error rate falls below what any shipped code needs;
2. maps each measured ``p_b`` through the exact bounded-distance accounting of
   :mod:`acmpilot.coding`;
3. interpolates the crossing of the target post-decoding BER in SNR;
4. repeats (2)-(3) at ``p_b`` plus and minus one binomial standard error to
   obtain a Monte Carlo uncertainty on the threshold itself.

A threshold without an uncertainty would be a false precision: the whole
adaptive-rate problem is about operating close to a boundary, and a boundary
known to +/-0.3 dB is a different engineering problem from one known to
+/-0.01 dB. The uncertainty is reported with the table.

Thresholds are a property of *this* simulator at the stated target BER, under
the ideal-interleaving assumption of :mod:`acmpilot.coding`. They are not DVB-S2
thresholds and must not be read as such. DVB-S2 (ETSI EN 302 307) is the
canonical real-world ACM MODCOD table and is named in the README as the
reference a reader should consult for standardised numbers; its codes are LDPC
plus BCH, not Reed-Solomon, and its thresholds are therefore several dB better
than the ones below at comparable spectral efficiency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coding import SHIPPED_CODES, ReedSolomonCode
from .modulation import constellation, measure_ber

#: Default post-decoding BER that defines a MODCOD threshold, dimensionless.
DEFAULT_TARGET_BER: float = 1e-6


@dataclass(frozen=True)
class Modcod:
    """One modulation-and-coding mode.

    Attributes
    ----------
    name
        Short label, e.g. ``"QPSK RS(255,223)"``.
    modulation
        A name in :data:`acmpilot.modulation.CONSTELLATION_NAMES`.
    code
        The :class:`acmpilot.coding.ReedSolomonCode` applied.
    """

    name: str
    modulation: str
    code: ReedSolomonCode

    @property
    def bits_per_symbol(self) -> int:
        """Channel bits per modulation symbol, dimensionless."""
        return constellation(self.modulation).bits_per_symbol

    @property
    def spectral_efficiency(self) -> float:
        """Information bits per channel symbol, ``bits_per_symbol * rate``."""
        return self.bits_per_symbol * self.code.rate


def shipped_modcods() -> tuple[Modcod, ...]:
    """The eight shipped MODCODs, in increasing spectral efficiency.

    Built from four coherent constellations and four RS code rates. The choice
    is a ladder with roughly even spectral-efficiency spacing; it is not a
    standard's ladder and no standard is implied.
    """
    by_rate = {c.k: c for c in SHIPPED_CODES}
    spec = [
        ("BPSK", 127),
        ("BPSK", 191),
        ("QPSK", 191),
        ("QPSK", 223),
        ("8PSK", 191),
        ("8PSK", 223),
        ("16QAM", 223),
        ("16QAM", 239),
    ]
    modcods = [
        Modcod(name=f"{mod} {by_rate[k].label}", modulation=mod, code=by_rate[k])
        for mod, k in spec
    ]
    return tuple(sorted(modcods, key=lambda m: m.spectral_efficiency))


@dataclass(frozen=True)
class ModcodTable:
    """A MODCOD ladder with its measured thresholds.

    Attributes
    ----------
    modcods
        Ladder in increasing spectral efficiency.
    thresholds_db
        Required SNR per MODCOD, dB, strictly increasing if the ladder is sane.
    threshold_sigma_db
        Monte Carlo 1-sigma uncertainty on each threshold, dB.
    target_ber
        Post-decoding BER the thresholds were measured at.
    provenance
        Free-form record of how the thresholds were produced.
    """

    modcods: tuple[Modcod, ...]
    thresholds_db: np.ndarray
    threshold_sigma_db: np.ndarray
    target_ber: float
    provenance: str = ""

    def __post_init__(self) -> None:
        if len(self.modcods) != len(self.thresholds_db):
            raise ValueError(
                f"{len(self.modcods)} modcods but {len(self.thresholds_db)} thresholds"
            )
        if len(self.modcods) == 0:
            raise ValueError("a MODCOD table needs at least one mode")

    @property
    def n_modes(self) -> int:
        """Number of MODCODs in the ladder."""
        return len(self.modcods)

    @property
    def spectral_efficiencies(self) -> np.ndarray:
        """Spectral efficiency per MODCOD, bit/symbol."""
        return np.array([m.spectral_efficiency for m in self.modcods], dtype=float)

    @property
    def is_monotone(self) -> bool:
        """True if thresholds strictly increase with spectral efficiency."""
        return bool(np.all(np.diff(self.thresholds_db) > 0))

    def best_supported(self, snr_db: np.ndarray | float) -> np.ndarray:
        """Highest MODCOD index whose threshold is met, ``-1`` if none.

        Parameters
        ----------
        snr_db
            True SNR per symbol, dB.

        Returns
        -------
        ndarray of int
            Index into :attr:`modcods`, or ``-1`` meaning the channel supports
            no shipped mode at all (an unavoidable outage, not a policy error).
        """
        snr = np.atleast_1d(np.asarray(snr_db, dtype=float))
        supported = snr[:, None] >= self.thresholds_db[None, :]
        count = supported.sum(axis=1)
        idx = count - 1
        out = idx.astype(int)
        return out if np.ndim(snr_db) else out.reshape(())

    def to_dict(self) -> dict:
        """JSON-serialisable form. Contains no filesystem paths."""
        return {
            "target_ber": self.target_ber,
            "provenance": self.provenance,
            "modcods": [
                {
                    "name": m.name,
                    "modulation": m.modulation,
                    "code_n": m.code.n,
                    "code_k": m.code.k,
                    "code_bits_per_symbol": m.code.bits_per_symbol,
                    "code_rate": m.code.rate,
                    "code_t": m.code.t,
                    "bits_per_symbol": m.bits_per_symbol,
                    "spectral_efficiency_bit_per_symbol": m.spectral_efficiency,
                    "threshold_db": float(thr),
                    "threshold_sigma_db": float(sig),
                }
                for m, thr, sig in zip(
                    self.modcods, self.thresholds_db, self.threshold_sigma_db, strict=True
                )
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ModcodTable:
        """Inverse of :meth:`to_dict`."""
        modcods = []
        thr, sig = [], []
        for row in payload["modcods"]:
            code = ReedSolomonCode(
                row["code_n"], row["code_k"], row["code_bits_per_symbol"]
            )
            modcods.append(
                Modcod(name=row["name"], modulation=row["modulation"], code=code)
            )
            thr.append(row["threshold_db"])
            sig.append(row["threshold_sigma_db"])
        return cls(
            modcods=tuple(modcods),
            thresholds_db=np.array(thr, dtype=float),
            threshold_sigma_db=np.array(sig, dtype=float),
            target_ber=float(payload["target_ber"]),
            provenance=str(payload.get("provenance", "")),
        )

    def save_json(self, path: str | Path) -> None:
        """Write :meth:`to_dict` as JSON to ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load_json(cls, path: str | Path) -> ModcodTable:
        """Read a table written by :meth:`save_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def measure_ber_curve(
    modulation: str,
    *,
    rng: np.random.Generator,
    snr_start_db: float = -4.0,
    snr_step_db: float = 0.25,
    stop_below_ber: float = 3e-4,
    snr_max_db: float = 34.0,
    target_errors: int = 2000,
    max_symbols: int = 400_000,
) -> dict[str, np.ndarray]:
    """Monte Carlo uncoded BER curve, extended until BER < ``stop_below_ber``.

    Returns
    -------
    dict of ndarray
        ``snr_db``, ``ber``, ``n_bits``, ``n_errors``, ``ber_se`` (binomial
        standard error). All arrays the same length.
    """
    if snr_step_db <= 0:
        raise ValueError(f"snr_step_db must be > 0, got {snr_step_db}")
    snr_db: list[float] = []
    ber: list[float] = []
    bits: list[int] = []
    errs: list[int] = []
    snr = float(snr_start_db)
    while snr <= snr_max_db:
        value, n_err, n_bit = measure_ber(
            modulation,
            snr,
            rng=rng,
            target_errors=target_errors,
            max_symbols=max_symbols,
        )
        snr_db.append(snr)
        ber.append(value)
        bits.append(n_bit)
        errs.append(n_err)
        if value < stop_below_ber:
            break
        snr += snr_step_db
    ber_arr = np.array(ber, dtype=float)
    bits_arr = np.array(bits, dtype=float)
    return {
        "snr_db": np.array(snr_db, dtype=float),
        "ber": ber_arr,
        "n_bits": bits_arr,
        "n_errors": np.array(errs, dtype=float),
        "ber_se": np.sqrt(np.maximum(ber_arr * (1.0 - ber_arr), 0.0) / bits_arr),
    }


def _crossing_db(snr_db: np.ndarray, metric: np.ndarray, target: float) -> float:
    """Lowest SNR at which ``metric`` first drops to ``target``, log-linear interp.

    ``metric`` is assumed non-increasing in SNR. Returns ``nan`` if the target
    is not bracketed by the grid.
    """
    snr = np.asarray(snr_db, dtype=float)
    val = np.asarray(metric, dtype=float)
    tiny = 1e-300
    log_v = np.log10(np.maximum(val, tiny))
    log_t = np.log10(target)
    below = log_v <= log_t
    if not np.any(below):
        return float("nan")
    first = int(np.argmax(below))
    if first == 0:
        return float("nan")
    x0, x1 = snr[first - 1], snr[first]
    y0, y1 = log_v[first - 1], log_v[first]
    if y0 == y1:
        return float(x1)
    return float(x0 + (log_t - y0) * (x1 - x0) / (y1 - y0))


def measure_thresholds(
    modcods: tuple[Modcod, ...] | None = None,
    *,
    target_ber: float = DEFAULT_TARGET_BER,
    seed: int = 20261006,
    curves: dict[str, dict[str, np.ndarray]] | None = None,
    **curve_kwargs: float,
) -> tuple[ModcodTable, dict[str, dict[str, np.ndarray]]]:
    """Measure the threshold of every MODCOD and return the table and the curves.

    Parameters
    ----------
    modcods
        Ladder to measure. Defaults to :func:`shipped_modcods`.
    target_ber
        Post-decoding BER defining a threshold, must be in (0, 1).
    seed
        Seed for the Monte Carlo. The table is reproducible from it.
    curves
        Pre-measured BER curves keyed by modulation name, to avoid repeating the
        Monte Carlo. If given, ``seed`` is unused.
    **curve_kwargs
        Forwarded to :func:`measure_ber_curve`.

    Returns
    -------
    (table, curves)
        ``table`` is a :class:`ModcodTable`; ``curves`` maps modulation name to
        the measured BER curve, so a caller can plot or re-use it.
    """
    if not 0.0 < target_ber < 1.0:
        raise ValueError(f"target_ber must be in (0, 1), got {target_ber}")
    ladder = shipped_modcods() if modcods is None else tuple(modcods)
    if curves is None:
        rng = np.random.default_rng(seed)
        needed = sorted({m.modulation for m in ladder})
        curves = {
            name: measure_ber_curve(name, rng=rng, **curve_kwargs) for name in needed
        }
    thresholds, sigmas = [], []
    for mode in ladder:
        curve = curves[mode.modulation]
        p_b = curve["ber"]
        se = curve["ber_se"]
        centre = _crossing_db(
            curve["snr_db"], mode.code.output_bit_error_rate(p_b), target_ber
        )
        lo = _crossing_db(
            curve["snr_db"],
            mode.code.output_bit_error_rate(np.clip(p_b - se, 0.0, 1.0)),
            target_ber,
        )
        hi = _crossing_db(
            curve["snr_db"],
            mode.code.output_bit_error_rate(np.clip(p_b + se, 0.0, 1.0)),
            target_ber,
        )
        thresholds.append(centre)
        sigmas.append(abs(hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else np.nan)
    table = ModcodTable(
        modcods=ladder,
        thresholds_db=np.array(thresholds, dtype=float),
        threshold_sigma_db=np.array(sigmas, dtype=float),
        target_ber=target_ber,
        provenance=(
            f"Monte Carlo uncoded BER through acmpilot.modulation.measure_ber, "
            f"seed={seed}, mapped through acmpilot.coding bounded-distance "
            f"accounting; threshold = log-linear crossing of post-decoding "
            f"BER = {target_ber:g}; sigma from +/-1 binomial SE on the measured "
            f"channel BER."
        ),
    )
    return table, curves
