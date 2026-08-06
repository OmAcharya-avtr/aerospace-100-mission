"""Analytic BER expressions for OOK, BPSK and M-ary PPM over AWGN and
lognormal fading.

Conventions
-----------
* ``snr_db`` is the electrical signal-to-noise ratio per bit, gamma = Eb/N0,
  in dB. Negative dB values are physically meaningful and accepted.
* All BER values are dimensionless probabilities in [0, 1].
* Fading model: mean-normalised lognormal irradiance I (E[I] = 1) multiplies
  the received *electrical* signal amplitude, i.e. the instantaneous SNR per
  bit is gamma * I^2. This is the standard intensity-modulation model used
  in FSO analyses (Zhu & Kahn 2002; Popoola & Ghassemlooy 2009).

Expressions (gamma = Eb/N0, linear; Q = Gaussian tail)
------------------------------------------------------
BPSK (coherent, AWGN):
    Pb = Q(sqrt(2 gamma))
    [Proakis & Salehi 2008, "Digital Communications" 5th ed., Eq. (4.3-13)]

OOK (unipolar 2-ASK, matched filter, equiprobable bits):
    ON amplitude A with average energy Eb = A^2/2 => A = 2 sqrt(gamma) sigma_n.
    Threshold tau = t*A, t in (0,1):
        Pb(t) = 1/2 [ Q(2 t sqrt(gamma)) + Q(2 (1-t) sqrt(gamma)) ]
    Optimal (midpoint) threshold t = 1/2 under signal-independent AWGN:
        Pb = Q(sqrt(gamma))
    i.e. OOK is exactly 3 dB worse than BPSK.
    [Proakis & Salehi 2008, Sec. 4.3 (binary ASK); FSO OOK context:
    Zhu & Kahn 2002, Eq. (12)]

M-ary PPM, modelled as M-ary orthogonal signalling with coherent detection:
    EXACT symbol error probability (Es = k Eb, k = log2 M):
        Ps = 1 - Integral phi(y - sqrt(2 Es/N0)) [Phi(y)]^(M-1) dy
    evaluated here by Gauss-Hermite quadrature in a numerically stable
    expm1/log form.
    [Proakis & Salehi 2008, Eq. (4.4-17); equivalently Eq. (4.2-98) region]
    UNION BOUND (upper bound, tight at high SNR):
        Ps <= (M-1) Q(sqrt(Es/N0))
    [Proakis & Salehi 2008, Eq. (4.4-22) region — pairwise distance
    d^2 = 2 Es for orthogonal signals]
    Bit error probability for orthogonal signalling:
        Pb = M / (2 (M-1)) * Ps
    [Proakis & Salehi 2008, Eq. (4.4-18)]

Lognormal fading average (weak turbulence, sigma_I^2 < ~1):
    Pb = E_I[ Pb_cond(I) ], evaluated with N-point Gauss-Hermite quadrature
    over the lognormal pdf (Andrews & Phillips 2005 Ch. 8-9 for the model;
    Zhu & Kahn 2002 Sec. III for the BER-averaging technique).
    For OOK a *fixed* threshold (set for the mean irradiance, no channel
    state information) is distinguished from the *adaptive/optimal*
    threshold tau = I*A/2 (perfect CSI): the fixed threshold exhibits the
    well-known irreducible BER floor.

Caveat: real direct-detection PPM/OOK receivers may be shot-noise limited;
this package assumes additive signal-independent Gaussian noise (thermal /
background limited regime).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._math import gauss_hermite, qfunc
from .channels import lognormal_irradiance_nodes, validate_channel
from .results import AnalyticResult

__all__ = ["analytic_ber", "MODULATIONS"]

MODULATIONS = ("ook", "bpsk", "ppm")

_REFS = {
    "bpsk": "Proakis & Salehi 2008, Eq. (4.3-13): Pb = Q(sqrt(2 Eb/N0))",
    "ook": "Proakis & Salehi 2008 Sec. 4.3; Zhu & Kahn 2002 Eq. (12): Pb = Q(sqrt(Eb/N0))",
    "ppm-exact": (
        "Proakis & Salehi 2008, Eqs. (4.4-17)/(4.4-18): exact M-ary orthogonal "
        "signalling SER/BER (EXACT expression)"
    ),
    "ppm-union": (
        "Proakis & Salehi 2008, union bound for orthogonal signalling: "
        "Ps <= (M-1) Q(sqrt(Es/N0)) (UPPER BOUND, not exact)"
    ),
    "lognormal": (
        "Andrews & Phillips 2005 Ch. 8-9 (lognormal model); Zhu & Kahn 2002 "
        "Sec. III (Gauss-Hermite BER averaging)"
    ),
}


def _validate_common(mod: str, snr_db: Any) -> np.ndarray:
    if mod not in MODULATIONS:
        raise ValueError(f"unknown modulation {mod!r}; expected one of {MODULATIONS}")
    snr = np.atleast_1d(np.asarray(snr_db, dtype=float))
    if snr.ndim != 1:
        raise ValueError("snr_db must be a scalar or 1-D array of Eb/N0 values in dB")
    if not np.all(np.isfinite(snr)):
        raise ValueError("snr_db must be finite (negative dB is allowed; NaN/inf is not)")
    return snr


def _validate_m(m: int) -> int:
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool):
        raise TypeError(f"M must be an integer power of two >= 2, got {m!r}")
    m = int(m)
    if m < 2 or (m & (m - 1)) != 0:
        raise ValueError(f"M must be a power of two >= 2 (2, 4, 8, ...), got {m}")
    if m > 4096:
        raise ValueError(f"M > 4096 not supported (got {m})")
    return m


def _validate_threshold(threshold: Any) -> Any:
    if threshold == "optimal":
        return "optimal"
    try:
        t = float(threshold)
    except (TypeError, ValueError):
        raise ValueError(
            f"threshold must be 'optimal' or a fixed fraction in (0, 1), got {threshold!r}"
        ) from None
    if not (0.0 < t < 1.0) or not math.isfinite(t):
        raise ValueError(
            f"threshold must be 'optimal' or a fixed fraction in (0, 1) of the ON "
            f"amplitude, got {threshold!r}"
        )
    return t


def _ber_bpsk_cond(gamma: np.ndarray, irr: np.ndarray | float) -> np.ndarray:
    """Conditional BPSK BER at irradiance gain I: Q(I sqrt(2 gamma))."""
    return qfunc(np.sqrt(2.0 * gamma) * irr)


def _ber_ook_cond(
    gamma: np.ndarray, irr: np.ndarray | float, threshold: Any
) -> np.ndarray:
    """Conditional OOK BER at irradiance gain I.

    Adaptive/optimal threshold (tau = I*A/2, perfect CSI): Q(I sqrt(gamma)).
    Fixed threshold tau = t*A (set for mean irradiance I=1, no CSI):
        Pb(I) = 1/2 [ Q(2 t sqrt(gamma)) + Q(2 (I - t) sqrt(gamma)) ].
    Derivation: sigma_n = 1, A = 2 sqrt(gamma); P(err|0) = Q(t A) and
    P(err|1) = Q(I A - t A).
    """
    rt = np.sqrt(gamma)
    if threshold == "optimal":
        return qfunc(irr * rt)
    t = threshold
    return 0.5 * (qfunc(2.0 * t * rt) + qfunc(2.0 * (irr - t) * rt))


def _ppm_ser_exact(gamma_s: np.ndarray, m: int, n_nodes: int = 64) -> np.ndarray:
    """Exact SER of M-ary orthogonal signalling, coherent detection, AWGN.

    Ps = 1 - Integral phi(y - a) Phi(y)^(M-1) dy with a = sqrt(2 Es/N0)
    (Proakis & Salehi 2008, Eq. (4.4-17)). Substituting y = sqrt(2) x + a
    gives the Gauss-Hermite form

        Ps = (1/sqrt(pi)) sum_i w_i [1 - Phi(sqrt(2) x_i + a)^(M-1)]

    where the bracket is computed as -expm1((M-1) ln Phi(u)) for numerical
    stability at high SNR (avoids catastrophic cancellation in 1 - (1-eps)).
    """
    from scipy.special import log_ndtr

    x, w = gauss_hermite(n_nodes)
    a = np.sqrt(2.0 * gamma_s)  # (n_snr,)
    u = math.sqrt(2.0) * x[None, :] + a[:, None]  # (n_snr, n_nodes)
    per_node = -np.expm1((m - 1) * log_ndtr(u))
    return per_node @ w / math.sqrt(math.pi)


def _ppm_ser_union(gamma_s: np.ndarray, m: int) -> np.ndarray:
    """Union bound on the SER of M-ary orthogonal signalling (AWGN).

    Ps <= (M-1) Q(sqrt(Es/N0)); pairwise distance d^2 = 2 Es for orthogonal
    signals (Proakis & Salehi 2008, Sec. 4.4). Clipped to <= 1 since the
    bound is vacuous above 1.
    """
    return np.minimum(1.0, (m - 1) * qfunc(np.sqrt(gamma_s)))


def _ppm_ber_cond(
    gamma: np.ndarray, irr: np.ndarray | float, m: int, method: str, n_nodes: int
) -> np.ndarray:
    """Conditional M-PPM BER at irradiance gain I (instantaneous Es scaled by I^2)."""
    k = int(math.log2(m))
    gamma_s = k * gamma * np.square(irr)
    ser = _ppm_ser_exact(gamma_s, m, n_nodes) if method == "exact" else _ppm_ser_union(gamma_s, m)
    return ser * (m / (2.0 * (m - 1.0)))  # Proakis & Salehi 2008, Eq. (4.4-18)


def analytic_ber(
    mod: str,
    snr_db: float | np.ndarray,
    *,
    channel: str = "awgn",
    sigma_i2: float | None = None,
    M: int = 4,
    threshold: str | float = "optimal",
    ppm_method: str = "exact",
    n_gh_nodes: int = 64,
) -> AnalyticResult:
    """Analytic bit error ratio for a modulation / channel pair.

    Parameters
    ----------
    mod : {"ook", "bpsk", "ppm"}
        Modulation format.
    snr_db : float or 1-D array
        Electrical SNR per bit Eb/N0 in dB (dimensionless ratio in dB).
        Negative values are valid; NaN/inf raise ValueError.
    channel : {"awgn", "lognormal"}
        Channel model. "lognormal" = weak-turbulence FSO fading with
        mean-normalised irradiance (see module docstring).
    sigma_i2 : float, optional
        Scintillation index sigma_I^2 (> 0, dimensionless). Required for
        (and only valid with) channel="lognormal". Values > 1 exceed the
        weak-fluctuation validity range and emit a UserWarning.
    M : int
        PPM alphabet size, power of two >= 2 (only used for mod="ppm").
    threshold : "optimal" or float in (0, 1)
        OOK decision threshold. "optimal" = midpoint of the instantaneous
        levels (tau = I*A/2; requires CSI under fading). A float t is a
        FIXED threshold tau = t*A referenced to the mean-irradiance ON
        amplitude A (no CSI; exhibits a BER floor under fading). Only used
        for mod="ook".
    ppm_method : {"exact", "union"}
        "exact" = exact orthogonal-signalling expression (Gauss-Hermite
        evaluated); "union" = (M-1) Q(sqrt(Es/N0)) upper BOUND.
    n_gh_nodes : int
        Gauss-Hermite node count for the exact-PPM and fading integrals.

    Returns
    -------
    AnalyticResult
        With ``ber`` an ndarray matching ``snr_db``'s (atleast-1d) shape.

    References are recorded in the result's ``reference`` field; see module
    docstring for the equations, sources, units and validity ranges.
    """
    snr = _validate_common(mod, snr_db)
    validate_channel(channel, sigma_i2)
    gamma = np.power(10.0, snr / 10.0)

    params: dict[str, Any] = {}
    if mod == "ppm":
        m = _validate_m(M)
        if ppm_method not in ("exact", "union"):
            raise ValueError(f"ppm_method must be 'exact' or 'union', got {ppm_method!r}")
        params["M"] = m
        params["ppm_method"] = ppm_method
    if mod == "ook":
        threshold = _validate_threshold(threshold)
        params["threshold"] = threshold

    def cond(irr: np.ndarray | float) -> np.ndarray:
        if mod == "bpsk":
            return _ber_bpsk_cond(gamma, irr)
        if mod == "ook":
            return _ber_ook_cond(gamma, irr, threshold)
        return _ppm_ber_cond(gamma, irr, m, ppm_method, n_gh_nodes)

    base_ref = _REFS[f"ppm-{ppm_method}" if mod == "ppm" else mod]
    if channel == "awgn":
        ber = cond(1.0)
        method = params.get("ppm_method", "closed-form")
        ref = base_ref
    else:
        params["sigma_i2"] = float(sigma_i2)  # type: ignore[arg-type]
        # The fixed-threshold OOK integrand becomes step-like at high SNR and
        # needs more Gauss-Hermite nodes than the smooth adaptive case (see
        # validation/lognormal_gh_vs_quad.py): use >= 256 nodes there.
        eff_nodes = n_gh_nodes
        if mod == "ook" and threshold != "optimal":
            eff_nodes = max(n_gh_nodes, 256)
        irr_nodes, wts = lognormal_irradiance_nodes(float(sigma_i2), eff_nodes)
        # E_I[Pb(I)]: stack conditional BER over irradiance nodes -> weighted sum
        stack = np.stack([cond(float(i)) for i in irr_nodes], axis=-1)  # (n_snr, n_nodes)
        ber = stack @ wts
        method = f"{params.get('ppm_method', 'closed-form')} + GH({eff_nodes}) fading average"
        ref = f"{base_ref}; fading: {_REFS['lognormal']}"

    ber = np.clip(ber, 0.0, 1.0)
    return AnalyticResult(
        mod=mod, channel=channel, snr_db=snr, ber=ber, method=str(method),
        reference=ref, params=params,
    )
