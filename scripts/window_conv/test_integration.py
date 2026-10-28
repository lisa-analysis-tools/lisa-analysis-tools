"""Coordinator: XYZSensitivityBackend with window_convolution_mode="exact" vs "legacy".

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python test_integration.py [--backends cpu cuda12x]

Per config x backend, on ONE backend instance (the mode is switched in place, so the
GPU galactic grid is shared):
* compute_sensitivity_matrix(convolve_window=True): band reldiff exact vs legacy;
* compute_log_like(convolve_window=True): |delta ll| exact vs legacy, judged against the
  legacy round-off spread (golden cpu vs golden cuda12x);
* exact vs golden; psd_batch_size chunking; walker copies (__call__ / deepcopy) share
  the _wconv_* caches; splines fall back to legacy bitwise.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

import _use_worktree  # noqa: F401
from compare import band_reldiff, golden_inputs, load_golden
from fixtures import CONFIGS, asnumpy, make_backend

SENS_BAND = 1e-14
FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def legacy_spread(config):
    g_cpu, g_gpu = load_golden(config, "cpu"), load_golden(config, "cuda12x")
    return float(np.max(np.abs(g_cpu["ll_conv"] - g_gpu["ll_conv"])))


def run(config, backend):
    import lisatools  # noqa: F401

    print(f"[{config} / {backend}]")
    sm = make_backend(config, backend, window_convolution_mode="legacy")
    xp = sm.xp
    inp = golden_inputs(config, xp)
    llk, smk = inp["log_like_kwargs"], inp["sens_mat_kwargs"]
    spl = CONFIGS[config]["use_splines"]

    S_leg = sm.compute_sensitivity_matrix(**smk, convolve_window=True)
    ll_leg = asnumpy(sm.compute_log_like(**llk, convolve_window=True))

    sm.window_convolution_mode = "exact"
    S_ex = sm.compute_sensitivity_matrix(**smk, convolve_window=True)
    ll_ex = asnumpy(sm.compute_log_like(**llk, convolve_window=True))

    if spl:
        check(np.array_equal(ll_ex, ll_leg), "splines: exact mode falls back to legacy (bitwise)")
        return

    r, _ = band_reldiff(S_ex, S_leg)
    check(r <= SENS_BAND, f"sens_mat exact vs legacy band reldiff {r:.2e}")
    spread = legacy_spread(config)
    d = float(np.max(np.abs(ll_ex - ll_leg)))
    check(d <= max(2 * spread, 1e-6), f"ll exact vs legacy |d| {d:.3g} (legacy cpu-gpu spread {spread:.3g}; ll ~ {ll_leg[0]:.4g})")
    dg = float(np.max(np.abs(ll_ex - load_golden(config, backend)["ll_conv"])))
    check(dg <= max(2 * spread, 1e-6), f"ll exact vs golden |d| {dg:.3g}")

    # chunking in exact mode
    sm.psd_batch_size = 2
    ll_ch = asnumpy(sm.compute_log_like(**llk, convolve_window=True))
    sm.psd_batch_size = None
    dc = float(np.max(np.abs(ll_ch - ll_ex)))
    check(dc <= 1e-6 * max(1.0, abs(ll_ex[0]) * 1e-6), f"psd_batch_size=2 vs unchunked |d| {dc:.3g}")

    # walker copies share the caches (no rebuild, no copy)
    caches = {k: id(v) for k, v in vars(sm).items() if k.startswith("_wconv_")}
    check(len(caches) > 0, f"caches present: {sorted(caches)}")
    dc_ = copy.deepcopy(sm)
    check(all(id(getattr(dc_, k)) == i for k, i in caches.items()), "deepcopy shares _wconv_* caches")
    n = sm.n_noise_par
    psd_params = xp.asarray([15e-12] * n + [3e-15] * n)
    gal = xp.asarray([10 ** -43.7, 1.8, 2e-3, 3e-3, 8e-4]) if CONFIGS[config]["galaxy"] else None
    w = sm("w0", psd_params, gal)
    check(all(id(getattr(w, k)) == i for k, i in caches.items()), "__call__ copy shares _wconv_* caches")
    sm.window_convolution_mode = "legacy"
    w_leg = sm("w0", psd_params, gal)
    r, _ = band_reldiff(w.sens_mat, w_leg.sens_mat)
    check(r <= SENS_BAND, f"__call__ sens_mat (smoothed) exact vs legacy band reldiff {r:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda12x"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    for b in a.backends:
        for c in a.configs:
            run(c, b)
    print(f"\n{len(FAIL)} failures")
    for f in FAIL:
        print("  " + f)
    raise SystemExit(1 if FAIL else 0)
