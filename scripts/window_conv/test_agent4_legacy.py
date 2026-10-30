"""Agent 4: legacy convolution-path hardening + walker chunking vs the HEAD goldens.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python test_agent4_legacy.py [--backends cpu cuda12x]

Checks per config x backend:
* golden: cpu bitwise (ll_conv, ll_noconv, sens_mat_conv); cuda12x within the agent-1
  tolerances (ll rtol ``LL_RTOL_GPU``, sens_mat band-normalised ``SENS_BAND_GPU``);
* psd_batch_size in {1, 2, 3, None}: bitwise identical to unchunked (golden data_index
  map with repeats -> gather path; identity map -> zero-copy view path);
* fft_batch_size in {1, 3, None, "auto"}: identical on cpu, spread reported on gpu;
* ``_convolve_matrix_components`` leaves its inputs untouched by default.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import _use_worktree  # noqa: F401
import lisatools
from compare import band_reldiff, golden_inputs, load_golden, reldiff
from fixtures import CONFIGS, asnumpy, make_backend

# GPU vs golden: the galactic grid init is not bitwise reproducible across instances,
# which the legacy convolved ll amplifies to ~1e-9..3e-8 (measured, also with HEAD code);
# the strict GPU check is new-vs-HEAD on the SAME instance (bitwise), see head_class().
LL_RTOL_GPU = 5e-8
SENS_BAND_GPU = 1e-14
FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


_HEAD = []


# the unmodified implementation the goldens were made from (make_golden.py provenance)
REFERENCE_COMMIT = "673152b2"


def head_class():
    """The reference XYZSensitivityBackend (``git show REFERENCE_COMMIT:src/lisatools/sensitivity.py``)."""
    if not _HEAD:
        import importlib.util, os, subprocess, tempfile

        src = subprocess.run(
            ["git", "-C", _use_worktree.WORKTREE, "show", f"{REFERENCE_COMMIT}:src/lisatools/sensitivity.py"],
            capture_output=True, check=True,
        ).stdout
        path = os.path.join(tempfile.mkdtemp(), "sensitivity_head.py")
        with open(path, "wb") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("lisatools.sensitivity_head", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _HEAD.append(mod.XYZSensitivityBackend)
    return _HEAD[0]


def ll(sm, llk, conv):
    out = sm.compute_log_like(**llk, convolve_window=conv)
    return asnumpy(out)


def run_config(name, be):
    print(f"[{name}/{be}]")
    sm = make_backend(name, be)
    xp = sm.xp
    g = load_golden(name, be)
    inp = golden_inputs(name, xp, be)
    llk, smk = inp["log_like_kwargs"], inp["sens_mat_kwargs"]

    llc, llnc = ll(sm, llk, True), ll(sm, llk, False)
    S = asnumpy(sm.compute_sensitivity_matrix(**smk, convolve_window=True))
    if be == "cpu":
        check(np.array_equal(llc, g["ll_conv"]), f"ll_conv bitwise vs golden (reldiff {reldiff(llc, g['ll_conv'])[0]:.2e})")
        check(np.array_equal(llnc, g["ll_noconv"]), "ll_noconv bitwise vs golden")
        check(np.array_equal(S, g["sens_mat_conv"], equal_nan=True),
              f"sens_mat bitwise vs golden (band {band_reldiff(S, g['sens_mat_conv'])[0]:.2e})")
    else:
        r = reldiff(llc, g["ll_conv"])[0]
        check(r <= LL_RTOL_GPU, f"ll_conv vs golden reldiff {r:.2e} (|dll| {np.max(np.abs(llc - g['ll_conv'])):.2e})")
        r = reldiff(llnc, g["ll_noconv"])[0]
        check(r <= LL_RTOL_GPU, f"ll_noconv vs golden reldiff {r:.2e}")
        r = band_reldiff(S, g["sens_mat_conv"])[0]
        check(r <= SENS_BAND_GPU, f"sens_mat vs golden band reldiff {r:.2e}")
        # HEAD code on the same instance (same galactic grid): must be bitwise
        new_cls, sm.__class__ = sm.__class__, head_class()
        try:
            llc_old = ll(sm, llk, True)
            S_old = asnumpy(sm.compute_sensitivity_matrix(**smk, convolve_window=True))
        finally:
            sm.__class__ = new_cls
        check(np.array_equal(llc, llc_old) and np.array_equal(S, S_old, equal_nan=True),
              "ll_conv + sens_mat bitwise == HEAD code on the same instance")

    # walker chunking, same instance: bitwise
    ident = dict(llk, data_index_all=xp.arange(len(llk["data_index_all"]), dtype=xp.int32))
    for tag, kw in (("golden map", llk), ("identity map", ident)):
        sm.psd_batch_size = None
        ref_c, ref_n = ll(sm, kw, True), ll(sm, kw, False)
        for bs in (1, 2, 3):
            sm.psd_batch_size = bs
            a, b = ll(sm, kw, True), ll(sm, kw, False)
            check(np.array_equal(a, ref_c) and np.array_equal(b, ref_n),
                  f"psd_batch_size={bs} == None ({tag}) conv/noconv")
    sm.psd_batch_size = None

    # fft batch size, same instance
    sm._fft_batch_size = None
    ref_ll, ref_S = ll(sm, llk, True), asnumpy(sm.compute_sensitivity_matrix(**smk, convolve_window=True))
    spread = []
    for fb in (1, 3, "auto", 512):
        sm._fft_batch_size = fb
        a = ll(sm, llk, True)
        Sb = asnumpy(sm.compute_sensitivity_matrix(**smk, convolve_window=True))
        rl, rs = reldiff(a, ref_ll)[0], band_reldiff(Sb, ref_S)[0]
        spread.append((fb, rl, rs))
        if be == "cpu":
            check(np.array_equal(a, ref_ll) and np.array_equal(Sb, ref_S, equal_nan=True),
                  f"fft_batch_size={fb} bitwise == None")
        else:
            check(rl <= 1e-12 and rs <= 1e-15,
                  f"fft_batch_size={fb} vs None: ll reldiff {rl:.2e}, sens band {rs:.2e}")
    sm._fft_batch_size = 512

    # non-mutating convolution by default
    raw = sm._compute_matrix_elements(
        sm.f_arr, *[llk[k] for k in ("Soms_in_all", "Sa_in_all", "Amp_in_all", "alpha_in_all",
                                     "f_1_in_all", "kn_in_all", "f_2_in_all",
                                     "knots_position_all", "knots_amplitude_all")],
        convolve_window=False,
    )
    before = [asnumpy(c).copy() for c in raw]
    sm._convolve_matrix_components(*raw, num_psds=raw[0].shape[0])
    check(all(np.array_equal(asnumpy(c), b, equal_nan=True) for c, b in zip(raw, before)),
          "_convolve_matrix_components leaves inputs untouched")
    return spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda12x"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    for be in a.backends:
        if be != "cpu" and not lisatools.has_backend(be):
            print(f"{be} unavailable, skipped")
            continue
        for name in a.configs:
            run_config(name, be)
    print(f"\n{len(FAIL)} failure(s)")
    for f in FAIL:
        print("  " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
