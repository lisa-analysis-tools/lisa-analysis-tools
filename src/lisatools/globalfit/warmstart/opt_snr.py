"""Optimal-SNR gate for warm-start clusters (ruling 2026-09-10, limit 8).

Annotates every fitted component with its OPTIMAL SNR under the run's own
fitted sensitivity — psd + galfor of the BEST-logL cold walker at the LAST
valid stored sample — and drops components below ``--limit`` (default 8,
matching GB_OPT_SNR_LIMIT_SEARCH): sub-threshold clusters are
noise-fitting fodder that should not seed warm births.

Nothing waveform- or noise-side is re-derived: the sensitivity grids, the
FD waveform sizing and the SNR sum are IMPORTED from
``scripts/diagnostics/build_truth.py`` (the frozen truth-set route), and
the sampled-basis -> physical transform is the run's own stock container
(the referee's pattern). The only new choices here are the walker
selector (best-logL, per the ruling, vs build_truth's cold median) and the
gate itself.

Pipeline slot: fit -> **snr gate** -> referee -> apply. The output keeps
the exact fitter writer schema (plus an ``opt_snr`` column and gate
provenance in meta), so the referee, the apply step and
``WarmStartComponents.from_npz`` consume it unchanged.

Laptop rules: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1; peak RSS ~1 GB (orbit + waveform batches).

Usage:
  python warmstart_opt_snr.py --fit <components.npz> --store <h5>
      --out <gated.npz> [--limit 8] [--batch 20000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py
import numpy as np

def _build_truth_path() -> Path:
    """scripts/diagnostics/build_truth.py, resolved from the repo checkout
    (this module moved into the installed package 2026-09-14, so
    ``__file__``-relative no longer reaches scripts/)."""
    import lisatools

    repo = Path(lisatools.__file__).resolve().parents[2]
    for base in (repo, Path.cwd()):
        cand = base / "scripts" / "diagnostics" / "build_truth.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "scripts/diagnostics/build_truth.py not found -- the opt-snr gate "
        "needs the LAT repo checkout, not just the installed package."
    )


_BT = None  # resolved lazily by _build_truth()


def _build_truth():
    spec = importlib.util.spec_from_file_location(
        "build_truth", _build_truth_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_logl_noise(store: str) -> dict:
    """psd + galfor of the best-logL cold walker at the last VALID row.

    Valid = log_like written AND the gb coord slab nonzero (the
    keep-window/torn-tail guards of the fitter, applied to the row this
    gate reads its noise from). Raises on all-zero params — the
    build_truth unwritten-row trap surfaces loudly here."""
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        llv = g["log_like"][:, 0, 0, :]
        written = np.flatnonzero(np.any(llv != 0.0, axis=-1))
        chain_gb = g["chain"]["gb"]
        row = None
        for it in written[::-1][:8]:
            if chain_gb[it, 0, 0][:8].any():
                row = int(it)
                break
        if row is None:
            raise RuntimeError(f"{store}: no valid coord row found")
        w = int(np.argmax(llv[row]))
        psd = np.asarray(g["chain"]["psd"][row, 0, 0, w, 0, :], float)
        gal = np.asarray(g["chain"]["galfor"][row, 0, 0, w, 0, :], float)
    if not (np.all(np.isfinite(psd)) and np.any(psd != 0.0)):
        raise RuntimeError(
            f"{store}: psd params at row {row} walker {w} are zero/NaN "
            f"({psd}) — unwritten row?")
    return dict(psd_params=psd.tolist(), galfor_params=gal.tolist(),
                iteration=row, walker=w, log_like=float(llv[row, w]))


def boxed_means(means: np.ndarray, meta: dict) -> np.ndarray:
    """Clip bounded columns to their meta box before waveform transforms.

    Truncated-MLE fits (2026-09-11) legitimately place a rail-piled
    component's mean at/beyond the box edge; the physical transform
    (arccos etc.) needs in-box values. No bounded_cols meta = identity."""
    bc = dict(meta.get("bounded_cols", {}) or {})
    if not bc:
        return means
    out = np.array(means, copy=True)
    for c, (blo, bhi) in bc.items():
        c = int(c)
        out[:, c] = np.clip(out[:, c], float(blo), float(bhi))
    return out


def component_opt_snr(means: np.ndarray, store: str, noise: dict,
                      batch: int = 20000) -> np.ndarray:
    """Optimal SNR of each component mean under the fitted sensitivity."""
    bt = _build_truth()
    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.transforms import (
        make_gb_transform_container,
    )
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
        GB_MOJITO_T_REF,
    )

    tobs = bt.store_tobs(store)
    nw = bt.nw_for(tobs)
    df = 1.0 / tobs
    sa, se = bt.sens_grids(np.asarray(noise["psd_params"]),
                           np.asarray(noise["galfor_params"]), df)
    orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))
    tc = make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True)
    phys = np.asarray(tc.both_transforms(np.atleast_2d(means)), float)
    return bt.opt_snr(phys, sa, se, gbw, df, tobs, nw, batch=batch)


def gate(fit_npz: str, out: str, limit: float, snr: np.ndarray,
         noise: dict) -> str:
    """Write the gated npz: opt_snr column + comps below ``limit`` dropped."""
    with np.load(fit_npz, allow_pickle=False) as d:
        arrs = {k: np.array(d[k]) for k in d.files if k != "meta"}
        meta = json.loads(str(d["meta"]))
    n = len(arrs["means"])
    snr = np.asarray(snr, float)
    if snr.shape != (n,):
        raise ValueError(f"snr has shape {snr.shape}, fit has {n} comps")
    keep = snr >= float(limit)
    per_comp = {k for k, v in arrs.items()
                if v.ndim >= 1 and len(v) == n and k != "f0_window_edges"}
    out_arrs = {k: (v[keep] if k in per_comp else v)
                for k, v in arrs.items()}
    out_arrs["opt_snr"] = snr[keep]
    meta.update(
        opt_snr_limit=float(limit),
        opt_snr_dropped=int((~keep).sum()),
        opt_snr_noise=noise,
    )
    np.savez_compressed(out, **out_arrs, meta=json.dumps(meta))
    print(f"gated: {n} -> {int(keep.sum())} comps at opt SNR >= {limit:g} "
          f"(dropped {int((~keep).sum())}; dropped-p sum "
          f"{arrs['p'][~keep].sum():.1f} of {arrs['p'].sum():.1f}) | "
          f"noise: it {noise['iteration']} walker {noise['walker']} "
          f"logL {noise['log_like']:.1f} | wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True, help="fitter components npz")
    ap.add_argument("--store", required=True, help="run store h5 (noise "
                    "params + Tobs come from here)")
    ap.add_argument("--out", required=True, help="gated output npz")
    ap.add_argument("--limit", type=float, default=8.0,
                    help="optimal-SNR gate (default 8 = "
                         "GB_OPT_SNR_LIMIT_SEARCH); <= 0 annotates only")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args(argv)
    noise = best_logl_noise(args.store)
    print(f"noise: it {noise['iteration']} walker {noise['walker']} "
          f"logL {noise['log_like']:.1f} psd={noise['psd_params']} "
          f"galfor={np.round(noise['galfor_params'], 4)}")
    with np.load(args.fit, allow_pickle=False) as d:
        means = np.array(d["means"])
        fit_meta = json.loads(str(d["meta"]))
    snr = component_opt_snr(boxed_means(means, fit_meta), args.store,
                            noise, batch=args.batch)
    gate(args.fit, args.out, args.limit, snr, noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
