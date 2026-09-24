"""Prototype + efficiency report for the PRESET VGB band grid.

The VGB branch samples ~55 KNOWN fixed-frequency sources; its band grid is
pure scheduling bookkeeping (no RJ surface, f0 fixed per leaf), so the grid
can be anything that respects the same-unit support-separation rule. The
builder (:func:`lisatools.globalfit.stock.erebor.vgb.vgb_preset_band_edges`,
wired as ``VGB_BAND_EDGES_MODE=preset``) anchors windows on the catalogue
sources -- clusters share a window, don't-care filler bands sit between
windows, and every window lands at an ODD band index so stride-2 unit
scheduling puts ALL proposal work into one unit pass.

This script:

1. reads the mojito VGB catalogue (frequencies + amplitudes),
2. reconstructs the two measured baseline geometries -- (a) the production
   per-layer uniform VGB grid (the VGB slice of the GB-style grid;
   45 bands at 3 months, stride 2) and (b) the v6 narrow-band grid
   (GB_SUBBAND_DIVISOR=8 -> 360 VGB bands, stride 9),
3. builds preset grids over a width-cap sweep using the installed
   ``get_N`` for source supports,
4. VALIDATES every grid with the installed
   ``check_band_support_separation`` (the same gate the get_n builder and
   the move ctor use),
5. prints the efficiency table (n_bands / units / cells / predicted
   fill_slots / slab extent / predicted s-per-propose) against the two
   measured anchors:

       v5 vgb_pe : 8.5 s/propose, cells=4224, fill_slots=4800,  2 units
       v6 vgb_pe : 34  s/propose, cells=8064, fill_slots=55920, 9 units

Model conventions (documented, calibrated on the two anchors):

* ``cells``      = occupied interior bands x ntemps x nwalkers -- matches
  the move's ``tm.count("cells")`` (VGB is fixed-dimensional: an occupied
  band is occupied at every temp/walker). The reconstruction reproduces
  BOTH anchors exactly (22 x 192 and 42 x 192), which pins
  ntemps*nwalkers = 192 and validates the grid reconstruction.
* proposal fills = cells (each scheduled cell's slab is filled once).
* tempering fills: the swap grid covers ALL interior bands;
  with GB_TEMPER_SKIP_EMPTY=1 (default since d3f7ba01) only rows of
  occupied bands are filled -> occ x nwalkers x ntemps; without it,
  (n_bands - 2) x nwalkers x ntemps.
* wall-time model: t = alpha * cells * (slab/slab_ref) + beta * n_bands,
  with (alpha, beta) solved exactly from the two anchors (slab equal for
  both baselines, so the slab ratio only affects preset predictions; it
  encodes that per-cell likelihood/fill traffic scales with the shared
  slab extent ``band_slab_Nf``, which follows the WIDEST band). A second
  calibration replacing ``n_bands`` with tempering-grid cells is printed
  as a sensitivity check -- attribution between the two is degenerate at
  two anchors, but the preset grid improves both terms, so the
  prediction is robust.

Run (laptop budget, single process)::

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 python scripts/diagnostics/vgb_preset_grid.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

DEF_CAT = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/"
    "vgb_cat_mojito_lite_processed.hdf5"
)

# Measured anchors (3-mo production, [GB_TIMING vgb_pe]).
ANCHORS = {
    "v5(uniform, stride2)": dict(t=8.5, cells=4224, fill_slots=4800, units=2),
    "v6(div8, stride9)": dict(t=34.0, cells=8064, fill_slots=55920, units=9),
}


def load_catalogue(path):
    import h5py

    with h5py.File(path, "r") as f:
        B = f["Binaries"]
        f0 = np.asarray(B["GW22FrequencySSBFrame"][:], dtype=float).ravel()
        amp = np.asarray(B["Amplitude"][:], dtype=float).ravel()
    order = np.argsort(f0)
    return f0[order], amp[order]


def production_wdm(dt, tobs_target, wd_bounds):
    """(Nf, Nt, layer_df, Tobs) via the installed adjust_to_even_bins."""
    from lisatools.domains import WDMSettings

    Nf, Nt, _wd = WDMSettings.adjust_to_even_bins(
        t_min=wd_bounds[0], t_max=wd_bounds[1], dt=dt, Tobs=tobs_target
    )
    return int(Nf), int(Nt), 1.0 / (2 * Nf * dt), Nf * Nt * dt


def vgb_span(f0, layer_df, band_layers, min_freq, max_freq):
    """start/end_freq exactly as prepare_vgb_branch derives them (WDM)."""
    guard = 2.0 * band_layers * layer_df
    return (
        max(float(f0.min()) - guard, min_freq),
        min(float(f0.max()) + guard, max_freq),
    )


def uniform_edges(start, end, layer_df, div=1):
    """GBSetup.init_band_structure uniform mode (per-layer / div edges)."""
    k_lo = int(np.ceil(start / layer_df)) * div
    k_hi = int(np.floor(end / layer_df)) * div
    return np.asarray([k * layer_df / div for k in range(k_lo, k_hi + 1)])


def grid_metrics(edges, f0, stride, ntemps, nwalkers, Tobs, layer_df):
    """Efficiency metrics for one band grid (documented model above)."""
    from lisatools.globalfit.moves.gbbands import (
        SubBandBuffer,
        check_band_support_separation,
    )

    edges = np.asarray(edges, dtype=float)
    nb = len(edges) - 1
    band_of = np.searchsorted(edges, f0, side="right") - 1
    interior = (band_of >= 1) & (band_of <= nb - 2)
    occ_bands, occ_counts = np.unique(band_of[interior], return_counts=True)
    occ = len(occ_bands)
    tw = ntemps * nwalkers

    units_populated = len(np.unique(occ_bands % stride))
    rounds = sum(
        int(occ_counts[occ_bands % stride == u].max())
        for u in np.unique(occ_bands % stride)
    )
    cells = occ * tw
    temper_rows = max(nb - 2, 0) * nwalkers
    fills_noskip = cells + temper_rows * ntemps
    fills_skip = cells + occ * nwalkers * ntemps

    slab = SubBandBuffer.recommend_band_slab_layers(
        edges, layer_df, xp=np, Tobs=Tobs
    )
    sep = check_band_support_separation(
        edges, Tobs, stride, enforce=False,
        context="vgb_preset_grid report",
    )
    return dict(
        n_bands=nb, stride=stride, occ=occ, cells=cells,
        units_populated=units_populated, rounds=rounds,
        temper_cells=temper_rows * ntemps,
        fills_noskip=fills_noskip, fills_skip=fills_skip,
        slab=int(slab), sep_passes=bool(sep["passes"]),
        min_safe_stride=sep["min_safe_stride"],
        n_src_covered=int(interior.sum()), max_cluster=int(occ_counts.max()),
        storage_floats=nb * (2 * ntemps - 1 + 2 * ntemps + ntemps * nwalkers),
    )


def calibrate(m_v5, m_v6):
    """Solve the two 2x2 wall-time models exactly on the anchors."""
    t5, t6 = ANCHORS["v5(uniform, stride2)"]["t"], ANCHORS["v6(div8, stride9)"]["t"]
    out = {}
    # model A: t = a*cells + b*n_bands
    A = np.array([[m_v5["cells"], m_v5["n_bands"]],
                  [m_v6["cells"], m_v6["n_bands"]]], dtype=float)
    out["A(cells,bands)"] = np.linalg.solve(A, [t5, t6])
    # model B: t = a*cells + b*temper_cells
    B = np.array([[m_v5["cells"], m_v5["temper_cells"]],
                  [m_v6["cells"], m_v6["temper_cells"]]], dtype=float)
    out["B(cells,temper)"] = np.linalg.solve(B, [t5, t6])
    return out


def predict(models, m, slab_ref):
    """Predicted s/propose under both calibrations (slab-scaled cells)."""
    cells_eff = m["cells"] * m["slab"] / slab_ref
    a, b = models["A(cells,bands)"]
    tA = a * cells_eff + b * m["n_bands"]
    a, b = models["B(cells,temper)"]
    tB = a * cells_eff + b * m["temper_cells"]
    return tA, tB


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalogue", default=DEF_CAT)
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--tobs-target", type=float, default=90 * 86400.0)
    ap.add_argument("--wavelet-bounds", type=float, nargs=2,
                    default=(3600.0, 4400.0))
    ap.add_argument("--min-freq", type=float, default=1e-4)
    ap.add_argument("--max-freq", type=float, default=2.5e-2)
    ap.add_argument("--ntemps", type=int, default=12,
                    help="VGB ladder rungs (VGB_NTEMPS default)")
    ap.add_argument("--nwalkers", type=int, default=16,
                    help="pinned by cells=occ*ntemps*nwalkers matching both "
                         "measured anchors exactly")
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0, 12.0],
                    help="preset width caps (WDM layers) to sweep")
    args = ap.parse_args()

    from lisatools.globalfit.stock.erebor.vgb import vgb_preset_band_edges

    f0, amp = load_catalogue(args.catalogue)
    Nf, Nt, ldf, Tobs = production_wdm(
        args.dt, args.tobs_target, tuple(args.wavelet_bounds))
    print(f"catalogue: {len(f0)} VGBs in [{f0.min():.4e}, {f0.max():.4e}] Hz")
    print(f"WDM grid: Nf={Nf} Nt={Nt} layer_df={ldf:.4e} Hz "
          f"Tobs={Tobs:.6g} s  (ntemps={args.ntemps} nwalkers={args.nwalkers})")

    grids = {}

    # ---- baseline (a): production uniform per-layer grid, stride 2 ----
    start, end = vgb_span(f0, ldf, 1, args.min_freq, args.max_freq)
    grids["v5(uniform, stride2)"] = (uniform_edges(start, end, ldf, 1), 2)

    # ---- baseline (b): v6 narrow bands (GB_SUBBAND_DIVISOR=8), stride 9 ----
    grids["v6(div8, stride9)"] = (uniform_edges(start, end, ldf, 8), 9)

    # ---- preset grids over the width-cap sweep ----
    preset_meta = {}
    for cap in args.caps:
        name = f"preset(cap={cap:g})"
        edges, meta = vgb_preset_band_edges(
            f0, Tobs, ldf, start_freq=start, end_freq=end,
            width_cap_layers=cap, amps=amp, unit_stride=2,
            validate=True, return_meta=True,
        )
        grids[name] = (edges, 2)
        preset_meta[name] = meta

    metrics = {
        name: grid_metrics(edges, f0, stride, args.ntemps, args.nwalkers,
                           Tobs, ldf)
        for name, (edges, stride) in grids.items()
    }

    # anchor validation
    print("\n-- anchor validation (model cells vs measured) --")
    for name, anchor in ANCHORS.items():
        m = metrics[name]
        flag = "OK" if m["cells"] == anchor["cells"] else "MISMATCH"
        print(f"  {name:24s} model cells={m['cells']:6d}  "
              f"measured={anchor['cells']:6d}  [{flag}]  "
              f"(occ={m['occ']} bands x {args.ntemps * args.nwalkers})")

    models = calibrate(metrics["v5(uniform, stride2)"],
                       metrics["v6(div8, stride9)"])
    for k, (a, b) in models.items():
        print(f"  calibration {k}: alpha={a:.3e} s/cell, beta={b:.3e} s/unit")
    slab_ref = metrics["v5(uniform, stride2)"]["slab"]

    hdr = (f"{'grid':22s} {'bands':>5s} {'strd':>4s} {'occ':>3s} "
           f"{'cells':>5s} {'popU':>4s} {'rnds':>4s} {'slab':>4s} "
           f"{'fills(skip)':>11s} {'fills(no)':>9s} {'sep@strd':>8s} "
           f"{'minS':>4s} {'t_pred[s]':>16s}")
    print("\n-- efficiency report --")
    print(hdr)
    for name, m in metrics.items():
        tA, tB = predict(models, m, slab_ref)
        meas = ANCHORS.get(name)
        tstr = (f"meas {meas['t']:.1f}" if meas
                else f"{tA:5.1f} / {tB:5.1f}")
        print(f"{name:22s} {m['n_bands']:5d} {m['stride']:4d} {m['occ']:3d} "
              f"{m['cells']:5d} {m['units_populated']:4d} {m['rounds']:4d} "
              f"{m['slab']:4d} {m['fills_skip']:11d} {m['fills_noskip']:9d} "
              f"{str(m['sep_passes']):>8s} {str(m['min_safe_stride']):>4s} "
              f"{tstr:>16s}")

    print("\nnotes:")
    print(" * fills model: proposal cells + tempering rows;"
          " measured fill_slots counters also include buffer-cache"
          " effects the model does not track -- ratios, not absolutes.")
    print(" * preset grids: all source windows sit at ODD band indices ->"
          " stride-2 unit 1 carries every proposal; unit 0 holds only"
          " empty fillers (get_subset returns None; with"
          " GB_TEMPER_SKIP_EMPTY=1 its tempering rows skip fills too).")
    print(" * slab = shared band_slab_Nf (layers): per-cell fill/ll traffic"
          " scales with it; it follows the WIDEST band, which is why the"
          " preset builder width-caps windows AND fillers.")
    for name, meta in preset_meta.items():
        wins = meta["window_sources"]
        sizes = sorted((len(w) for w in wins), reverse=True)
        print(f" * {name}: {meta['num_windows']} windows, cluster sizes"
              f" {sizes} (clusters serialize within their window; the"
              " per-leaf stretch handles this).")


if __name__ == "__main__":
    main()
