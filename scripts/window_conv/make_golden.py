"""Write golden outputs of the UNMODIFIED ``XYZSensitivityBackend`` window-convolution path.

For every config x backend (``fixtures.CONFIGS`` x {cpu, cuda12x}) this stores in
``golden/<config>_<backend>.npz``:

* inputs (``in_*``: ``compute_log_like`` kwargs for 5 walkers; ``sm_*``:
  ``compute_sensitivity_matrix`` kwargs for the first 2) -- identical across backends;
* ``ll_conv``  = ``compute_log_like(..., convolve_window=True)``  (num_psds = 5);
* ``ll_noconv`` = ``compute_log_like(..., convolve_window=False)`` (reference);
* ``sens_mat_conv`` = ``compute_sensitivity_matrix(..., convolve_window=True)`` (2 PSDs),
  shape ``(2, 3, 3, nf)``;
* diagnostics: ``f0_guard_fires`` (any non-finite unconvolved entry at the first active
  bin -> ``_convolve_matrix_components`` copies column 1 into column 0), ``n_nonfinite_raw``,
  ``f_arr``, captured warnings, and ``ll_conv_repeat_bitwise`` (same call twice in-process);
* provenance: ``config_json`` (the full fixture config -> ``fixtures.make_backend`` rebuilds
  identically), ``prov_head``, ``prov_sha256_sensitivity``, ``prov_lisatools_file``,
  ``prov_backend``.

Refuses to write if ``src/lisatools/sensitivity.py`` or ``cutils/`` differ from HEAD (the
golden must be the current implementation).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python make_golden.py               # write all
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python make_golden.py --verify      # recompute + compare
    ... --configs gal1_spl0_sym --backends cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

import _use_worktree  # noqa: F401
import lisatools
from compare import GOLDEN_DIR, golden_inputs, golden_path, load_golden, reldiff
from fixtures import BACKENDS, CONFIGS, asnumpy, make_backend, make_inputs, sens_mat_kwargs, to_xp

WT = _use_worktree.WORKTREE
SENS = os.path.join(WT, "src", "lisatools", "sensitivity.py")
NUM_PSDS = 5
N_SENS = 2


def provenance() -> dict:
    head = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", WT, "diff", "HEAD", "--stat", "--", "src/lisatools/sensitivity.py", "src/lisatools/cutils"],
        capture_output=True, text=True,
    ).stdout.strip()
    with open(SENS, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return dict(head=head, dirty=dirty, sha256_sensitivity=sha, lisatools_file=lisatools.__file__)


def _flatten_inputs(inp: dict) -> dict:
    out = {}
    for prefix, d in (("in_", inp["log_like_kwargs"]), ("sm_", sens_mat_kwargs(inp, N_SENS))):
        for k, v in d.items():
            if v is None:
                out[f"{prefix}{k}"] = np.zeros(0)
                out[f"{prefix}{k}__is_none"] = np.array(True)
            else:
                out[f"{prefix}{k}"] = np.asarray(v)
    return out


def _sync(xp):
    if xp.__name__ == "cupy":
        xp.cuda.Device().synchronize()


def compute_outputs(sm, inp_xp: dict) -> dict:
    """Run the three golden computations (+ diagnostics) on backend ``sm``."""
    xp = sm.xp
    llk, smk = inp_xp["log_like_kwargs"], inp_xp["sens_mat_kwargs"]
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        ll_conv = sm.compute_log_like(**llk, convolve_window=True)
        _sync(xp)
        t_conv = time.perf_counter() - t0
        ll_conv2 = sm.compute_log_like(**llk, convolve_window=True)
        t0 = time.perf_counter()
        ll_noconv = sm.compute_log_like(**llk, convolve_window=False)
        _sync(xp)
        t_noconv = time.perf_counter() - t0
        S = sm.compute_sensitivity_matrix(**smk, convolve_window=True)
        raw = sm._compute_matrix_elements(
            sm.f_arr, llk["Soms_in_all"], llk["Sa_in_all"], llk["Amp_in_all"], llk["alpha_in_all"],
            llk["f_1_in_all"], llk["kn_in_all"], llk["f_2_in_all"],
            llk["knots_position_all"], llk["knots_amplitude_all"], convolve_window=False,
        )
    raw = [asnumpy(c).reshape(NUM_PSDS, -1) for c in raw]
    out = dict(
        ll_conv=asnumpy(ll_conv),
        ll_noconv=asnumpy(ll_noconv),
        sens_mat_conv=asnumpy(S),
        ll_conv_repeat_bitwise=np.array(bool(np.array_equal(asnumpy(ll_conv), asnumpy(ll_conv2)))),
        f0_guard_fires=np.array(not all(np.isfinite(c[:, 0]).all() for c in raw)),
        n_nonfinite_raw=np.array([int((~np.isfinite(c)).sum()) for c in raw]),
        f_arr=asnumpy(sm.f_arr),
        warnings=np.array([f"{w.category.__name__}: {w.message}" for w in wrec] or [""]),
        t_conv=np.array(t_conv),
        t_noconv=np.array(t_noconv),
    )
    return out


def run(configs, backends, verify: bool, seed0: int = 0) -> int:
    prov = provenance()
    print(f"lisatools: {prov['lisatools_file']}\nHEAD {prov['head']}  sha256(sensitivity.py) {prov['sha256_sensitivity'][:16]}")
    if prov["dirty"] and not verify:
        print(f"REFUSING to write golden: sensitivity/cutils differ from HEAD:\n{prov['dirty']}")
        return 2
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    status = 0
    for ic, name in enumerate(configs):
        cfg = CONFIGS[name]
        flat_inputs = None
        for be in backends:
            if be != "cpu" and not lisatools.has_backend(be):
                print(f"[{name}/{be}] backend unavailable, skipped")
                continue
            t0 = time.perf_counter()
            sm = make_backend(cfg, be)
            t_init = time.perf_counter() - t0
            if verify:
                inp_xp = golden_inputs(name, sm.xp, be)
            else:
                if flat_inputs is None:
                    inp = make_inputs(sm, NUM_PSDS, seed=seed0 + ic)
                    flat_inputs = _flatten_inputs(inp)
                inp_xp = dict(
                    log_like_kwargs=to_xp(inp["log_like_kwargs"], sm.xp),
                    sens_mat_kwargs=to_xp(sens_mat_kwargs(inp, N_SENS), sm.xp),
                )
            out = compute_outputs(sm, inp_xp)
            tag = (
                f"[{name}/{be}] nf={sm.num_freqs} n_alias={sm.n_alias} init={t_init:.1f}s "
                f"conv={float(out['t_conv']):.3f}s noconv={float(out['t_noconv']):.3f}s "
                f"guard={bool(out['f0_guard_fires'])} nonfinite_raw={out['n_nonfinite_raw'].tolist()} "
                f"repeat_bitwise={bool(out['ll_conv_repeat_bitwise'])}"
            )
            if any(out["warnings"]):
                tag += f" warnings={sorted(set(out['warnings'].tolist()))}"
            fin = all(np.isfinite(out[k]).all() for k in ("ll_conv", "ll_noconv", "sens_mat_conv"))
            if not fin:
                tag += " NONFINITE-OUTPUT"
            print(tag)
            if verify:
                g = load_golden(name, be)
                for k in ("ll_conv", "ll_noconv", "sens_mat_conv"):
                    r, idx = reldiff(out[k], g[k])
                    bit = np.array_equal(out[k], g[k], equal_nan=True)
                    print(f"    verify {k}: bitwise={bit} max reldiff={r:.3e} at {idx}")
                    status |= 0 if bit else 1
                continue
            np.savez(
                golden_path(name, be),
                **flat_inputs,
                **out,
                config_json=np.array(json.dumps(cfg, sort_keys=True)),
                prov_head=np.array(prov["head"]),
                prov_sha256_sensitivity=np.array(prov["sha256_sensitivity"]),
                prov_lisatools_file=np.array(prov["lisatools_file"]),
                prov_backend=np.array(be),
                num_psds=np.array(NUM_PSDS),
                seed=np.array(seed0 + ic),
            )
            del sm
    # CPU vs GPU agreement from the files on disk
    if set(BACKENDS) <= set(backends):
        print("\nCPU vs cuda12x (golden files):")
        for name in configs:
            if not (os.path.exists(golden_path(name, "cpu")) and os.path.exists(golden_path(name, "cuda12x"))):
                continue
            gc, gg = load_golden(name, "cpu"), load_golden(name, "cuda12x")
            same_in = all(
                np.array_equal(gc[k], gg[k]) for k in gc if k.startswith(("in_", "sm_"))
            )
            parts = [f"{k} {reldiff(gg[k], gc[k])[0]:.2e}" for k in ("ll_conv", "ll_noconv", "sens_mat_conv")]
            print(f"  {name:24s} inputs_identical={same_in}  " + "  ".join(parts))
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--backends", nargs="*", default=list(BACKENDS))
    ap.add_argument("--verify", action="store_true", help="recompute and compare against the files, don't write")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return run(a.configs, a.backends, a.verify, a.seed)


if __name__ == "__main__":
    sys.exit(main())
