"""Compiled window-convolution kernels (PSD.cu) vs the legacy compiled kernels.

    cd scripts/window_conv
    [WCONV_NATIVE=1] CUDA_VISIBLE_DEVICES=7 .venv/bin/python test_native.py [--backends cpu cuda12x]

(WCONV_NATIVE=1: the isolated build_native.sh build; otherwise the installed binary,
which must include the PSD.cu window-convolution methods.)

Per config x backend (``fixtures.CONFIGS``, golden inputs):

1. ``psd_likelihood_basis_wrap`` (``fused_log_like``) vs the legacy
   ``psd_likelihood_wrap`` fed the same covariance, assembled on the host side from the
   bases (external-matrix path): instrument only, + galaxy rows (galaxy configs),
   + random multiplicative spline weights (contiguous and a strided view), + both.
   Relative |d ll| / |ll|.
2. ``get_galactic_row_wrap`` (``_gal_row``) vs the galaxy-only
   ``get_noise_covariance_wrap`` XX entry (``_gal_eval``; expected bitwise) and vs the
   galaxy-only ``_compute_matrix_elements(convolve_window=False)``.
3. int64 fix: legacy ``compute_log_like`` (``window_convolution_mode="legacy"``,
   convolve_window True / False) vs the HEAD goldens: bitwise on CPU and on GPU without
   galaxy (the GPU galactic-grid init is not bitwise reproducible across instances).

``--overflow``: synthetic int32-overflow test, ``data_index * 3 * T > 2^31`` without a
34 GB buffer: the data array handed to the binding is a VIEW whose base address is
``data_index * 3 * T`` elements BEFORE one real block (never dereferenced except at the
real block; GPU: cupy ``UnownedMemory``, CPU: ctypes ``from_address``). The result with
``data_index = DI`` must be bitwise the one with the real block at ``data_index = 0``.
With the pre-fix int32 offset the kernel would read ``base + (int32 wrap)`` -> unmapped
memory. ``--overflow-head`` runs the same GPU check against the shared-venv (HEAD) binary
in this process (expect an illegal-address error or a wrong value) -- run it WITHOUT
WCONV_NATIVE and in a separate process.
"""

from __future__ import annotations

import argparse
import ctypes
import os

import numpy as np

import _use_worktree  # noqa: F401
from compare import band_reldiff, golden_inputs, load_golden
from fixtures import CONFIGS, asnumpy, make_backend

import lisatools.convolution_helpers as sw

LL_REL_TOL = 1e-12
FAIL = []
WORST = {}


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def rel(a, b):
    a, b = asnumpy(a), asnumpy(b)
    return float(np.max(np.abs(a - b) / np.abs(b))), float(np.max(np.abs(a - b)))


def _coeffs(sm, llk):
    n = sm.n_noise_par
    xp = sm.xp
    Soms = xp.asarray(llk["Soms_in_all"], dtype=xp.float64)
    Sa = xp.asarray(llk["Sa_in_all"], dtype=xp.float64)
    if n > 1:
        Soms, Sa = Soms.reshape(-1, n), Sa.reshape(-1, n)
    return sw.basis_coeffs(sm, Soms, Sa)


def _assembled_entries(sm, co, gal_conv=None, gal_ratios=None, spline_weights=None):
    """The 6 covariance entries ``(P, total_terms)`` that ``fused_log_like`` builds per bin."""
    xp = sm.xp
    bases = sw.get_instrument_bases(sm)
    P, nb = co.shape
    nt = sm.num_times
    if spline_weights is None:
        out = [co.astype(B.dtype) @ B for B in bases]
    else:
        wb = spline_weights[:, xp.arange(nb) // sm.n_noise_par, :] * co[:, :, None]
        out = [xp.einsum("pbf,btf->ptf", wb.astype(B.dtype), B.reshape(nb, nt, -1)).reshape(P, -1)
               for B in bases]
    if gal_conv is not None:
        g = gal_conv.reshape(P, nt, -1)
        for c, k in zip(out, sw._FUSED_GAL_K):
            c.reshape(P, nt, -1)[..., : g.shape[-1]] += gal_ratios[None, :, k, None] * g
    return out


def _legacy_external_ll(sm, llk, entries):
    """Legacy ``compute_log_like`` (``psd_likelihood_wrap``) on given covariance entries."""
    mode = sm.window_convolution_mode
    sm.window_convolution_mode = "legacy"
    sm._log_like_matrix_elements = lambda *a, **k: entries
    try:
        return sm.compute_log_like(**llk, convolve_window=True)
    finally:
        del sm._log_like_matrix_elements
        sm.window_convolution_mode = mode


def fused_vs_legacy_kernel(config, sm, llk):
    xp = sm.xp
    co = _coeffs(sm, llk)
    P = co.shape[0]
    gal = None
    if CONFIGS[config]["galaxy"]:
        gal = sw.galaxy_convolved_rows(
            sm, *[llk[k] for k in ("Amp_in_all", "alpha_in_all", "f_1_in_all", "kn_in_all", "f_2_in_all")]
        )
    rng = np.random.default_rng(11)
    w = xp.asarray(10.0 ** (0.1 * rng.standard_normal((P, 2, sm.num_freqs))))
    w_t = xp.asarray(np.ascontiguousarray(np.swapaxes(asnumpy(w), 0, 1)))  # (2, P, nf)
    w_view = xp.swapaxes(w_t, 0, 1)  # (P, 2, nf) non-contiguous view, same values as w
    cases = [("instrument", {})]
    if gal is not None:
        cases.append(("+galaxy", dict(gal_conv=gal[0], gal_ratios=gal[1])))
    cases.append(("+spline w", dict(spline_weights=w)))
    cases.append(("+spline w (strided view)", dict(spline_weights=w_view)))
    if gal is not None:
        cases.append(("+galaxy +spline w", dict(gal_conv=gal[0], gal_ratios=gal[1], spline_weights=w_view)))
    for name, kw in cases:
        a = sw.fused_log_like(sm, llk["data_in_all"], llk["data_index_all"], co, **kw)
        b = _legacy_external_ll(sm, llk, _assembled_entries(sm, co, **kw))
        r, d = rel(a, b)
        key = (sm.xp.__name__, name)
        WORST[key] = max(WORST.get(key, 0.0), r)
        check(r <= LL_REL_TOL, f"fused {name:26s} vs legacy kernel rel {r:.2e} |d| {d:.2e} (ll ~ {float(asnumpy(b)[0]):.4g})")
    a = asnumpy(sw.fused_log_like(sm, llk["data_in_all"], llk["data_index_all"], co, spline_weights=w))
    b = asnumpy(sw.fused_log_like(sm, llk["data_in_all"], llk["data_index_all"], co, spline_weights=w_view))
    check(np.array_equal(a, b), "strided spline view bitwise == contiguous")


def galaxy_rows(config, sm, llk):
    xp = sm.xp
    pl = [xp.ascontiguousarray(xp.asarray(llk[k], dtype=xp.float64))
          for k in ("Amp_in_all", "alpha_in_all", "f_1_in_all", "kn_in_all", "f_2_in_all")]
    n = int(sm.num_freqs)
    mine = asnumpy(sw._gal_row(sm, n, pl))
    kern = asnumpy(sw._gal_eval(sm, xp.asarray(sm.f_arr), pl, sm.time_indices[:1])[0][:, 0, :])
    fin = np.isfinite(kern)
    check(np.array_equal(fin, np.isfinite(mine)), f"galaxy row finite pattern == kernel ({int((~fin).sum())} non-finite, f[0]={float(sm.f_arr[0]):.3g})")
    check(np.array_equal(np.where(fin, mine, 0), np.where(fin, kern, 0)),
          "galaxy row compiled vs get_noise_covariance XX (galaxy-only): bitwise")
    z = xp.zeros((len(pl[0]), sm.n_noise_par)) if sm.n_noise_par > 1 else xp.zeros(len(pl[0]))
    kp = ka = None
    if sm.use_splines:
        f = asnumpy(sm.f_arr)
        f = f[f > 0]
        kp = xp.asarray(np.broadcast_to(np.linspace(np.log10(f[0]) - .1, np.log10(f[-1]) + .1, 8), (2, len(pl[0]), 8)).copy())
        ka = xp.zeros_like(kp)
    c00 = sm._compute_matrix_elements(sm.f_arr, z, z, *pl, kp, ka, convolve_window=False)[0]
    c00 = asnumpy(c00).reshape(len(pl[0]), sm.num_times, -1)[:, 0, :]
    fin2 = np.isfinite(c00) & fin
    r, _ = band_reldiff(np.where(fin2, mine, 0), np.where(fin2, c00, 0))
    check(r <= 1e-15, f"galaxy row compiled vs galaxy-only _compute_matrix_elements band {r:.2e} "
          f"(non-finite there: {int((~np.isfinite(c00)).sum())})")


def legacy_goldens(config, backend):
    sm = make_backend(config, backend, window_convolution_mode="legacy")
    inp = golden_inputs(config, sm.xp)
    llk = inp["log_like_kwargs"]
    g = load_golden(config, backend)
    gal = CONFIGS[config]["galaxy"]
    strict = backend == "cpu" or not gal
    for conv, key in ((True, "ll_conv"), (False, "ll_noconv")):
        ll = asnumpy(sm.compute_log_like(**llk, convolve_window=conv))
        same = np.array_equal(ll, g[key])
        r = float(np.max(np.abs(ll - g[key]) / np.abs(g[key])))
        if strict:
            check(same, f"legacy {key} vs golden bitwise (rel {r:.1e})")
        else:
            print(f"  info legacy {key} vs golden rel {r:.1e} (galaxy on GPU: not bitwise across instances)")


def run(config, backend):
    print(f"[{config} / {backend}]")
    sm = make_backend(config, backend, window_convolution_mode="exact")
    inp = golden_inputs(config, sm.xp)
    llk = inp["log_like_kwargs"]
    fused_vs_legacy_kernel(config, sm, llk)
    if CONFIGS[config]["galaxy"]:
        galaxy_rows(config, sm, llk)
    legacy_goldens(config, backend)


# ------------------------------------------------------------------ overflow test


def _fake_view(xp, real, off_elems, n_elems):
    """View of ``n_elems`` complex128 starting ``off_elems`` elements BEFORE ``real``."""
    off = off_elems * 16
    if xp is np:
        buf = (ctypes.c_char * (n_elems * 16)).from_address(real.ctypes.data - off)
        return np.frombuffer(buf, dtype=np.complex128)
    import cupy as cp

    mem = cp.cuda.UnownedMemory(real.data.ptr - off, n_elems * 16, real, device_id=real.device.id)
    return cp.ndarray((n_elems,), dtype=cp.complex128, memptr=cp.cuda.MemoryPointer(mem, 0))


def overflow(backend, config="gal0_spl0_sym"):
    sm = make_backend(config, backend, window_convolution_mode="legacy")
    xp = sm.xp
    inp = golden_inputs(config, xp)
    llk = dict(inp["log_like_kwargs"])
    T = sm.num_times * sm.num_freqs
    DI = (2**31) // (3 * T) + 7  # data_index * 3 * T > 2^31
    assert DI * 3 * T > 2**31 - 1
    real = xp.ascontiguousarray(xp.asarray(llk["data_in_all"]).reshape(-1, 3 * T)[0])  # one block
    fake = _fake_view(xp, real, DI * 3 * T, 3 * T)  # base = real - DI blocks; block DI = real
    one = {k: (v[:1] if v is not None and k not in ("data_in_all", "data_index_all") else v) for k, v in llk.items()}
    if one["knots_position_all"] is not None:
        one["knots_position_all"] = llk["knots_position_all"][:, :1]
        one["knots_amplitude_all"] = llk["knots_amplitude_all"][:, :1]
    print(f"[overflow / {backend}] T={T}, data_index={DI}: data_index*3*T = {DI * 3 * T:.4g} > 2^31-1 = {2**31 - 1}")
    for conv in (False, True):
        ref = asnumpy(sm.compute_log_like(**{**one, "data_in_all": real, "data_index_all": xp.zeros(1, dtype=xp.int32)}, convolve_window=conv))
        got = asnumpy(sm.compute_log_like(**{**one, "data_in_all": fake, "data_index_all": xp.full(1, DI, dtype=xp.int32)}, convolve_window=conv))
        check(np.array_equal(ref, got), f"legacy psd_likelihood_wrap (convolve_window={conv}) data_index={DI}: {got[0]!r} vs {ref[0]!r}")
    sm.window_convolution_mode = "exact"
    co = _coeffs(sm, one)
    ref = asnumpy(sw.fused_log_like(sm, real, xp.zeros(1, dtype=xp.int32), co))
    got = asnumpy(sw.fused_log_like(sm, fake, xp.full(1, DI, dtype=xp.int32), co))
    check(np.array_equal(ref, got), f"psd_likelihood_basis_wrap data_index={DI}: {got[0]!r} vs {ref[0]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda12x"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--overflow", action="store_true", help="only the synthetic int32-overflow test")
    ap.add_argument("--overflow-head", action="store_true",
                    help="overflow test on the HEAD (shared-venv) binary; run without WCONV_NATIVE")
    a = ap.parse_args()
    if a.overflow_head:
        assert not _use_worktree.NATIVE, "--overflow-head must run without WCONV_NATIVE"
        overflow("cuda12x")
    elif a.overflow:
        for b in a.backends:
            overflow(b)
    else:
        for b in a.backends:
            for c in a.configs:
                run(c, b)
        print("\nworst fused-vs-legacy-kernel relative |d ll| per (backend, case):")
        for k, v in sorted(WORST.items()):
            print(f"  {k[0]:6s} {k[1]:28s} {v:.2e}")
    print(f"\n{len(FAIL)} failures")
    for f in FAIL:
        print("  " + f)
    raise SystemExit(1 if FAIL else 0)
