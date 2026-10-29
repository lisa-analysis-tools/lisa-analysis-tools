"""Load golden outputs and compare new results against them.

    import _use_worktree  # noqa: F401
    from compare import golden_inputs, assert_close_to_golden

    inp = golden_inputs("gal1_spl0_sym", xp)           # identical inputs used for the golden
    ll = sm.compute_log_like(**inp["log_like_kwargs"], convolve_window=True)
    S = sm.compute_sensitivity_matrix(**inp["sens_mat_kwargs"], convolve_window=True)
    assert_close_to_golden("gal1_spl0_sym", "cuda12x", ll, S, rtol=1e-12)

Golden files: ``golden/<config>_<backend>.npz`` written by ``make_golden.py`` from the
unmodified HEAD implementation (see the ``prov_*`` keys for sha / file hashes).
"""

from __future__ import annotations

import json
import os

import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

_LLK_KEYS = (
    "data_in_all", "data_index_all", "Soms_in_all", "Sa_in_all", "Amp_in_all", "alpha_in_all",
    "f_1_in_all", "kn_in_all", "f_2_in_all", "knots_position_all", "knots_amplitude_all",
)
_SMK_KEYS = (
    "Soms_d_in", "Sa_a_in", "Amp", "alpha", "f_1", "kn", "f_2",
    "knots_position_all", "knots_amplitude_all",
)


def golden_path(config: str, backend: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{config}_{backend}.npz")


def load_golden(config: str, backend: str) -> dict:
    with np.load(golden_path(config, backend), allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    g["config"] = json.loads(str(g["config_json"]))
    return g


def _none_if_empty(g, key):
    return None if bool(g.get(f"{key}__is_none", False)) else g[key]


def golden_inputs(config: str, xp=np, backend: str = "cpu") -> dict:
    """The exact inputs the golden was computed with, as ``xp`` arrays.

    Returns ``{"log_like_kwargs": ..., "sens_mat_kwargs": ...}`` ready to splat into
    ``compute_log_like`` / ``compute_sensitivity_matrix`` (add ``convolve_window``).
    Inputs are identical for the cpu and cuda12x goldens.
    """
    g = load_golden(config, backend)

    def conv(a):
        return None if a is None else xp.asarray(a)

    llk = {k: conv(_none_if_empty(g, f"in_{k}")) for k in _LLK_KEYS}
    smk = {k: conv(_none_if_empty(g, f"sm_{k}")) for k in _SMK_KEYS}
    return dict(log_like_kwargs=llk, sens_mat_kwargs=smk)


def _host(x):
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def reldiff(new, ref, floor_rel: float = 1e-300) -> tuple[float, tuple]:
    """Max per-element ``|new - ref| / max(|ref|, floor)``; ``floor = eps * max|ref|``.

    The floor keeps bins where the reference is ~0 (masked dips, f = 0) from blowing up a
    naive ratio. Non-finite mismatches (NaN vs number) give ``inf``. Returns
    ``(max_reldiff, argmax_index)``.
    """
    new, ref = _host(new), _host(ref)
    if new.shape != ref.shape:
        raise AssertionError(f"shape mismatch: new {new.shape} vs golden {ref.shape}")
    both_nan = np.isnan(new) & np.isnan(ref)
    same_inf = np.isinf(new) & np.isinf(ref) & (new == ref)
    ok = both_nan | same_inf
    aref = np.abs(np.where(ok, 0.0, ref))
    floor = max(np.finfo(np.float64).eps * float(np.nanmax(aref, initial=0.0)), floor_rel)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(new - ref) / np.maximum(aref, floor)
    r = np.where(ok, 0.0, r)
    r = np.where(np.isnan(r), np.inf, r)
    if r.size == 0:
        return 0.0, ()
    i = np.unravel_index(int(np.argmax(r)), r.shape)
    return float(r[i]), tuple(int(j) for j in i)


def band_reldiff(new, ref) -> tuple[float, tuple]:
    """Max ``|new - ref| / max_f |ref|`` per matrix entry (normalised along the last,
    frequency, axis). This is the natural error scale of an FFT convolution: round-off is
    ~eps * max|C| on every bin, so elementwise reldiffs on bins where |C| is ~1e-7 of the
    band maximum (low f, TDI dips) are not meaningful."""
    new, ref = _host(new), _host(ref)
    if new.shape != ref.shape:
        raise AssertionError(f"shape mismatch: new {new.shape} vs golden {ref.shape}")
    scale = np.nanmax(np.abs(ref), axis=-1, keepdims=True)
    r = np.abs(new - ref) / np.where(scale > 0, scale, 1.0)
    r = np.where(np.isnan(r), np.inf, r)
    i = np.unravel_index(int(np.argmax(r)), r.shape)
    return float(r[i]), tuple(int(j) for j in i)


def assert_close_to_golden(
    config: str,
    backend: str,
    loglike,
    sens_mat=None,
    rtol: float = 1e-12,
    which: str = "ll_conv",
    sens_metric: str = "band",
    ll_atol: float | None = None,
    verbose: bool = True,
) -> dict:
    """Compare ``loglike`` (and optionally ``sens_mat``) with the golden; raise on failure.

    * ``loglike`` vs golden ``which`` (``"ll_conv"`` = convolve_window=True, default, or
      ``"ll_noconv"``): passes if max elementwise reldiff <= ``rtol`` OR, when ``ll_atol`` is
      given, max ``|delta ll|`` <= ``ll_atol``.
    * ``sens_mat`` vs golden ``compute_sensitivity_matrix(..., convolve_window=True)`` (first
      2 walkers): ``sens_metric="band"`` (default) uses :func:`band_reldiff`, ``"elementwise"``
      uses :func:`reldiff`; must be <= ``rtol``.

    NOTE (measured on the HEAD goldens): the legacy convolved log-likelihood is only
    reproducible to ~1e-8 relative (|delta ll| ~ 0.01-0.6 on ll ~ 4e6) between CPU and GPU, and
    to ~1e-9 between two GPU backend instances with the galaxy on (the GPU galactic-grid
    init is not bitwise deterministic). ``sens_mat`` agrees to ~1e-15 band-normalised.
    Choose tolerances accordingly (see README in the report).

    Returns ``{"ll": (max_reldiff, idx), "ll_abs": max|delta|, "sens_mat": (r, idx)}``.
    """
    g = load_golden(config, backend)
    ref_ll = g[which]
    out = {"ll": reldiff(loglike, ref_ll), "ll_abs": float(np.max(np.abs(_host(loglike) - ref_ll)))}
    if sens_mat is not None:
        f = band_reldiff if sens_metric == "band" else reldiff
        out["sens_mat"] = f(sens_mat, g["sens_mat_conv"])
        out["sens_mat_elementwise"] = reldiff(sens_mat, g["sens_mat_conv"])
    if verbose:
        msg = (
            f"ll: max reldiff {out['ll'][0]:.3e} at {out['ll'][1]}, max |dll| {out['ll_abs']:.3e}"
        )
        if sens_mat is not None:
            msg += (
                f"; sens_mat ({sens_metric}): {out['sens_mat'][0]:.3e} at {out['sens_mat'][1]}"
                f" [elementwise {out['sens_mat_elementwise'][0]:.3e}]"
            )
        print(f"[{config}/{backend}] {msg}")
    ll_ok = out["ll"][0] <= rtol or (ll_atol is not None and out["ll_abs"] <= ll_atol)
    bad = []
    if not ll_ok:
        bad.append(f"ll reldiff {out['ll'][0]:.3e} (|dll| {out['ll_abs']:.3e})")
    if sens_mat is not None and not out["sens_mat"][0] <= rtol:
        bad.append(f"sens_mat {sens_metric} reldiff {out['sens_mat'][0]:.3e}")
    if bad:
        raise AssertionError(f"[{config}/{backend}] exceeds rtol={rtol:g}: " + "; ".join(bad))
    return out
