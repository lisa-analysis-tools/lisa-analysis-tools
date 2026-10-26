"""Stage timing of the direct-to-WDM EMRI template (GPU or CPU): where do the milliseconds go?

Wraps the stages of one EMRIDirectWDM call with timers (a device synchronise before each
timer stops, so GPU time lands on the stage that launched it) and prints per-template totals
next to the production template's wall time on the same grid.

    python scripts/emri/emri_direct_stage_timing.py --backend cuda13x --dt 2.5 --reps 5 \
        --catalog /path/emri_cat_mojito_lite_processed_MT.hdf5 \
        --direct-table wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5 --thresh 1e-3,1e-5

Sweeps in ONE process (FEW and the table load once): ``--response-grid sparse,pixels`` (the
dense response's grid), ``--lookup kernel,python`` (the fused C++/CUDA lookup-sum kernel vs the
Python tracer + table + scatter-add) and ``--chunk-rows 1,2,4,8,16,32`` (rows per response call,
timed over ``--batch-rows`` rows each). ``--band-hz 2.5e-4,2.5e-2`` builds the batch on the
run's active band only (as the fit's adapter does). Each (threshold, grid, chunk) prints its
stage split and, on a GPU, the cupy memory-pool footprint of that batch; ``--out`` appends one JSON
line per run and a summary table (ms per template vs rows per call) closes each threshold.

``--chunk-batch 1`` (the default HERE) builds a batch's plunge-chunk rows in one call (the
batched path the speed test measures); ``0`` builds them one per call, the library and global-fit
default (``EMRIDirectWDM(chunk_batch=...)``, ``EMRI_DIRECT_CHUNK_BATCH``); the ``[split]`` lines
and the JSON line carry ``chunk_batch``.

The lookup / plunge-chunk split: ``--nt-sub`` sets the plunge-chunk length in time layers
(``EMRIDirectWDM(Nt_sub=...)``; default: ``EMRI_DIRECT_NT_SUB``, else 128), or ``auto`` /
``auto:<floor>`` (one length per call, the shortest holding the call's longest tail in one
window, never below the floor); ``--tail-skip-px S`` looks a row whose tail spans at most S pixels
up through its end instead (``EMRIDirectWDM(tail_skip_px=...)``; default:
``EMRI_DIRECT_TAIL_SKIP_PX``, else 0 = never). After every timed call (single template and each
batch) a ``[split]`` line gives the generator's ``last_stats`` in the ``[EMRI_DIRECT] leaf N:``
line's wording plus the skipped tails, ``chunk rows A (tail median L px, W windows skipped S)
lookup px P chunk px C dropped px D`` (a skipped row has no tail entry), its ``nt_sub=`` the length
(``auto(eff N)``: the call's choice, ``last_stats["nt_sub_eff"]``), and the JSON line carries
``nt_sub`` (``"auto"`` in auto mode), ``nt_sub_floor``, ``nt_sub_eff``, ``tail_skip_px``,
``tail_skipped``, ``chunk_rows``, ``windows``, ``tail_median_px``, ``lookup_pixels``,
``chunk_pixels`` and ``dropped_pixels``: the speed test shows the split without a global-fit run. The stage list has
the dense plunge tail (``_dense_td``: the dense channels of every (row, window) pair) as its own
line inside stage 5. In auto mode with ``--chunk-batch 1`` a batch's window length is that of its
longest tail, the single template's its own, so the ``[parity]`` lines (which name both) also carry
that length's edge contamination difference (128-layer windows: ~2e-6 of the peak); with
``--chunk-batch 0`` every chunk row is its own call.

Phase walls: after the single call and after every timed batch a ``[walls]`` line gives the
generator's ``last_stats["walls"]`` of that call (exclusive wall seconds per phase, each boundary
device-synchronised, so GPU time lands in the phase that launched it; the batch's summed over its
``_batch_chunk`` calls) in ms per row::

    [walls] grid=sparse/kernel thr=0.001 nt_sub=128 chunk_batch=1 batch of 32 (8 per call): capture a
    ms/row, response b (pad, setup, kernel, splines), lookup c, tail d (setup, kernel), transform e,
    splice f, plan p, alloc x, total g (unattributed u); response points padded P (rows p_min..p_max);
    per call: first c1 ms, rest median c2 ms (n calls)

``capture`` = the row loop (FEW, tracks, grids, ``_chunk_item``), ``response`` = ``_dense_response``
/ ``run_response`` (its ``walls_detail`` sub-parts: ``pad_grid`` to the common length, the kernel
object's construction and uploads, the kernel, the output splines), ``lookup`` = the fused kernel or
the Python tracer + table, ``tail`` = ``tail_td(pairs)`` (sub-parts: kernel objects, channels
kernel), ``transform`` = ``wdm_chunk_of_td``, ``splice`` = the index splices, ``plan`` = handoffs,
pixel ranges, track subsets and the pair plan, ``alloc`` = the output array; ``response points
padded`` is the chunk rows' common response grid length against their own lengths; ``per call``
(batches only) the wall time of each ``_batch_chunk`` call (``last_stats["walls_per_call"]``): the
warm-up before a timed batch is ``rows[:min(2, chunk)]`` and the GPU pool is freed before it, so a
first call far above the rest is first-use cost (allocations, new cuFFT plan shapes), not the batch.
The JSON line carries ``walls_ms_per_row``, ``walls_detail_ms_per_row``, ``n_response_points``,
``n_response_points_rows_min`` / ``_max`` and ``walls_per_call_ms``. The phase whose per-row cost
grows from the single call to the batch is the batch's regression.

Batch parity (the NUMBERS of the batched path; "mm vs production" covers only the single template,
which is a one-row batch): after every timed batch, one UNTIMED ``direct.batch(rows[:chunk],
chunk_rows=chunk)`` (no ``consume``) is compared with each of those rows built alone, one line per
row (laptop, 2026-10-25: ``--response spline --lookup python --dt 20 --days 16``, CD1L EMRI 1's
plunge 10 d into the window, ``START_OFFSET_S=43891200``)::

    [parity] grid=sparse/python thr=0.01 nt_sub=128 batch of 3 (3 per call) row 1: max|batch -
    single| / max|single| 0.00e+00, mm 0.00e+00 (single: tail 60 px, 1 windows, trajectory end
    +2.98 h vs row 0) -- pass: ~1e-6 relative, ...

``max|batch[k] - single[:, m_lo:m_hi]| / max|single[:, m_lo:m_hi]|`` and the noise-weighted
mismatch of the two on the run box (the ``mm vs production`` helper). Pass: ~1e-6 relative, the
spline-padding level (the dense batch pads every row's response grid to the common length), not
1e-12; the spline response gives each row its own response, so there the batch is bitwise the
single template.
The JSON line carries the worst row as ``parity_max_rel`` and ``parity_mm_max``. So that the batch's
rows do not all hand off and end on one pixel (the information-matrix rows move the source by
~1e-6 relative), batch rows 1 and 2 are copies of the source with p0 shifted to move its plunge
by +3 and -5 WDM layers (hours; from a secant through one trial FEW call), the rest are the
original rows.

SNR and the MOJITO DATA: per threshold it prints the source's optimal SNR (production and each
direct template) on the fit's run box -- 0.25-25 mHz, ``--edge`` pixels cropped at each end --
against the scirdv1 XYZ sensitivity plus the fitted hyperbolic-tangent galactic foreground at
this window's Tobs (``--foreground on``, default; ``off`` = instrument only). When the source's mojito L1 brick
is found (``--orbits auto``, the default, or ``l1``: ``--l1-dir``, MOJITO_LIGHT_PATH/data/EMRI/L1,
or recursively under MOJITO_DATA_PATH / MOJITO_INFO_PATH / the catalogue's root), the templates
are built on the injection's orbits and window and each is also scored against its source-only,
noise-free stream (``--data auto``; ``off`` skips it): data SNR, mismatch (noise weighted, no
time/phase maximisation), logL = -1/2 <d-h|d-h> and the opt/data SNR ratio. The data includes
every mode, so a template's mismatch carries its mode truncation (EMRI 1, 6 months, eps 1e-3:
residual SNR^2 ~3, docs/emri-direct-wdm.md).
"""
import argparse
import collections
import functools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_batch_speed as B  # noqa: E402  (source loader, grid conventions)
import emri_tof_xyz_threeway as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--thresh", default="1e-3,1e-5")
    ap.add_argument("--mode-batch", type=int, default=64)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--orbits", choices=("auto", "l1", "equal-arm"), default="auto")
    ap.add_argument("--direct-table", required=True)
    ap.add_argument("--batch-rows", type=int, default=0,
                    help="also time EMRIDirectWDM.batch over this many parameter rows")
    ap.add_argument("--chunk-rows", default="16",
                    help="rows per response call in the batch; a comma list sweeps them (e.g. 1,2,4,8,16,32)")
    ap.add_argument("--response-grid", default="sparse",
                    help="dense response grid(s): sparse, pixels, or a comma list of both")
    ap.add_argument("--lookup", default="kernel",
                    help="lookup path(s): kernel, python, or a comma list of both")
    ap.add_argument("--band-hz", default="",
                    help="f_lo,f_hi [Hz]: batch output restricted to these layers (default: all)")
    ap.add_argument("--data", choices=("auto", "off"), default="auto",
                    help="score the templates against the source's mojito L1 stream when its brick is used")
    ap.add_argument("--foreground", choices=("on", "off"), default="on",
                    help="add the fitted tanh galactic foreground at this window's Tobs to the SNR/data sensitivity")
    ap.add_argument("--edge", type=int, default=60,
                    help="pixels cropped at each grid end for the SNR box (the launcher's EDGE_CROP_WAVELETS)")
    ap.add_argument("--out", default=None, help="append one JSON line per (threshold, grid, lookup, chunk) here")
    ap.add_argument("--response", choices=("spline", "dense"), default="dense",
                    help="TDI-on-the-fly response: 'dense' (TDDenseTDIonTheFly: exact phases, geometry shared "
                         "across harmonics; needs a backend built with it) or 'spline' (TDTDIonTheFly)")
    ap.add_argument("--nt-sub", default=None,
                    help="plunge-chunk length in time layers, or 'auto' / 'auto:<floor>' (one length per "
                         "call holding its longest tail in one window; EMRIDirectWDM Nt_sub; default: "
                         "EMRI_DIRECT_NT_SUB, else 128)")
    ap.add_argument("--tail-skip-px", type=int, default=None,
                    help="rows whose plunge tail spans at most this many pixels (handoff to last pixel) are "
                         "looked up through their end instead (EMRIDirectWDM tail_skip_px; default: "
                         "EMRI_DIRECT_TAIL_SKIP_PX, else 0 = never)")
    ap.add_argument("--chunk-batch", type=int, choices=(0, 1), default=1,
                    help="1 (default here): a batch's plunge-chunk rows built in ONE call (the batched "
                         "path); 0: one per call, the library and global-fit default "
                         "(EMRIDirectWDM chunk_batch / EMRI_DIRECT_CHUNK_BATCH)")
    args = ap.parse_args()

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.response import tdionfly as TF
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import emritdionfly as EF
    from lisatools.sources.emri import wdm_direct as WD
    from lisatools.sources.emri.response import get_emri_response_wrapper
    import lisatools.wdm_het as WH

    gpu = args.backend != "cpu"
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    T = collections.defaultdict(float)
    C = collections.Counter()
    LABELS = []

    def wrap(owner, name, label):
        f = getattr(owner, name)
        LABELS.append(label)

        @functools.wraps(f)
        def g(*a, **k):
            sync()
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                sync()
                T[label] += time.perf_counter() - t0
                C[label] += 1
        import inspect
        static = isinstance(owner, type) and isinstance(inspect.getattr_static(owner, name), staticmethod)
        setattr(owner, name, staticmethod(g) if static else g)

    nf = int(round(B.LAYER_DT / args.dt))
    nt = int(round(args.days * 86400.0 / B.LAYER_DT))
    n = nf * nt
    params, data_t0, orb = B.load_source(args.src, args.backend, n * args.dt, catalog=args.catalog,
                                         l1_dir=args.l1_dir, orbits=args.orbits)
    box = B.RunBox(nf, nt, args.dt, args.backend, edge=args.edge, foreground=args.foreground == "on")
    if args.data == "auto" and B.LAST_BRICK is not None:
        d_td = B.load_l1_data(B.LAST_BRICK, args.dt, n)
        if d_td is not None:
            box.set_data(d_td, B.LAST_BRICK)
            del d_td
            print(f"[data] {args.days:g} d: mojito L1 {box.brick} (source only, noise-free): data SNR "
                  f"{box.data_snr:.3f} ({box.describe()})", flush=True)
    elif args.data == "auto":
        print("[data] no mojito L1 brick used (equal-arm orbits): no data comparison", flush=True)
    fit = erebor.get_stock("all_sources")
    off = data_t0 - W.REF
    oi = int(round(off / args.dt))
    tdi = TDIConfig(fit.general.tdi_gen_str, force_backend=args.backend)
    wg = get_emri_response_wrapper(Tobs=(n + oi) * args.dt + 4e4, dt=args.dt, t_start=W.REF,
                                   t0_shift_to_data=off - oi * args.dt, tdi_config=tdi,
                                   tdi_chan=fit.general.tdi_chan, order=fit.emri.response_order,
                                   force_backend=args.backend, orbits=orb)
    gen = wg.waveform_gen.waveform_generator
    tds = TDSettings(n, args.dt, t0=0.0, force_backend=args.backend)
    wdm = WDMSettings(nf, nt, args.dt, force_backend=args.backend)
    table = WDMLookupTable.from_file(args.direct_table, force_backend=args.backend)
    grids = [g.strip() for g in args.response_grid.split(",") if g.strip()]
    lookups = [g.strip() for g in args.lookup.split(",") if g.strip()]
    chunks = [int(c) for c in str(args.chunk_rows).split(",") if c.strip()]
    band = None
    if args.band_hz:
        f_lo, f_hi = (float(v) for v in args.band_hz.split(","))
        band = (max(0, int(np.floor(f_lo / wdm.layer_df))), min(nf, int(np.ceil(f_hi / wdm.layer_df)) + 1))
    directs = {f"{g}/{lk}": WD.EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=tdi, t_start=W.REF,
                                             data_t0=data_t0, mode_batch=args.mode_batch, force_backend=args.backend,
                                             response=args.response, response_grid=g, lookup=lk, f_band=band,
                                             Nt_sub=args.nt_sub, tail_skip_px=args.tail_skip_px,
                                             chunk_batch=bool(args.chunk_batch))
               for g in grids for lk in lookups}
    tail_skip_px = next(iter(directs.values())).tail_skip_px            # resolved (argument > env > 0)
    nt_sub = next(iter(directs.values())).Nt_sub                       # resolved (argument > env > 128), or "auto"
    nt_floor = next(iter(directs.values())).nt_sub_floor

    def nt_label(st):
        """The plunge window length of a call for the [split] / [walls] lines: the fixed length, or
        ``auto(eff N)`` with the call's choice (``stats["nt_sub_eff"]``; ``auto:F(eff N)`` for a
        floor F other than 128)."""
        if nt_sub != "auto":
            return str(nt_sub)
        return f"{'auto' if nt_floor == 128 else f'auto:{nt_floor}'}(eff {st.get('nt_sub_eff', '-')})"
    pool = cp.get_default_memory_pool() if gpu else None
    m_lo, m_hi = band if band is not None else (0, nf)
    xp = cp if gpu else np

    def plunge_shifted_rows(direct, p, layers=(3.0, -5.0)):
        """Rows of ``p`` with p0 shifted so the trajectory ends ``layers`` WDM layers later /
        earlier (a secant through one trial FEW call on the knots: d t_end / d p0), so a batch's
        rows hand off and end on DIFFERENT pixels; a trajectory that outlasts the FEW span (no
        plunge to move) gets a fixed 3e-5 relative p0 step per layer instead."""
        kw = dict(mode_selection_threshold=float(args.thresh.split(",")[0]))

        def t_end(q):
            return float(np.asarray(direct._few_knots(q, kw).t_arr)[-1])
        e0, dp = t_end(p), 1e-5 * p[3]
        q = list(p)
        q[3] += dp
        slope = (t_end(q) - e0) / dp
        moves = abs(slope * dp) > 1.0
        out = []
        for L in layers:
            q = list(p)
            q[3] += L * wdm.layer_dt / slope if moves else L * 3e-5 * p[3]
            out.append([float(v) for v in q])
        how = (f"plunge moved by {', '.join(f'{L:+g}' for L in layers)} layers (d t_end / d p0 {slope:.4g} s)"
               if moves else "no plunge inside the FEW span: fixed 3e-5 relative p0 steps per layer")
        return out, how

    def embed(band_arr):
        """A ``(nch, n_m, Nt)`` band template on the full grid (zeros outside ``[m_lo, m_hi)``)."""
        band_arr = xp.asarray(band_arr)
        full = xp.zeros((band_arr.shape[0], nf, nt))
        full[:, m_lo:m_hi] = band_arr
        return full

    def parity(direct, label, sub, chunk, thr):
        """UNTIMED: ONE ``direct.batch(sub, chunk_rows=chunk)`` (no ``consume``) against each row
        built alone (``direct(*row)``, the single-template path), one ``[parity]`` line per row:
        ``max|batch[k] - single[:, m_lo:m_hi]| / max|single[:, m_lo:m_hi]|`` and the noise-weighted
        mismatch of the two on the box (``box.match``, the mm vs production metric). Returns
        ``(max rel, max mm)`` over the rows that are not zero."""
        arr = direct.batch(sub, chunk_rows=chunk, mode_selection_threshold=thr)
        st_par = dict(direct.last_stats)
        rels, mms, e_ref = [], [], None
        for k, p in enumerate(sub):
            one = direct(*p, mode_selection_threshold=thr)
            st1 = direct.last_stats
            e_k = float(np.asarray(direct._last_holder.t_arr)[-1])
            e_ref = e_k if e_ref is None else e_ref
            s_band = xp.asarray(one.arr)[:, m_lo:m_hi]
            del one
            b = xp.asarray(arr[k])
            peak = float(xp.max(xp.abs(s_band)))
            if peak > 0:
                rel = float(xp.max(xp.abs(b - s_band))) / peak
                mm = box.match(embed(b), embed(s_band))
                rels.append(rel)
                mms.append(mm)
                res = f"max|batch - single| / max|single| {rel:.2e}, mm {mm:.2e}"
            else:
                res = "single template is zero (ends before the window): " + (
                    "batch row zero too" if float(xp.max(xp.abs(b))) == 0 else "BATCH ROW NOT ZERO")
            single_nt = f", nt_sub {nt_label(st1)}" if nt_sub == "auto" else ""
            print(f"[parity] grid={label} thr={thr:g} nt_sub={nt_label(st_par)} batch of {len(sub)} ({chunk} per "
                  f"call) row {k}: {res} (single: tail {st1.get('tail_layers', 0)} px, {st1.get('windows', 0)} "
                  f"windows{single_nt}, trajectory end {(e_k - e_ref) / 3600.0:+.2f} h vs row 0) -- pass: ~1e-6 "
                  f"relative, the spline-padding level (the dense batch pads each row's response grid to "
                  f"the common length), not 1e-12", flush=True)
            del s_band, b
        del arr
        if pool is not None:
            pool.free_all_blocks()
        return (max(rels) if rels else float("nan")), (max(mms) if mms else float("nan"))

    shifted, shift_how = [], ""
    if args.batch_rows > 0:
        shifted, shift_how = plunge_shifted_rows(next(iter(directs.values())), list(params))
        print(f"[parity] batch rows 1-{len(shifted)}: p0-shifted copies of the source ({shift_how}); the "
              f"other rows are the +/- 1e-6 information-matrix steps and jittered walkers", flush=True)

    wrap(WD.EMRIDirectWDM, "_mode_list", "1 mode list (FEW call #1 + handoff check)")
    wrap(type(gen), "__call__", "  FEW generator calls (both)")
    wrap(EF.EMRITDIonFly, "__call__", "2 TOF total (FEW call #2 + feed + response)")
    wrap(EF.EMRITDIonFly, "mode_amp_phase", "  TOF feed: mode amp/phase")
    wrap(EF.EMRITDIonFly, "run_response", "  TOF run_response (input splines + kernel + output)")
    wrap(WD.EMRIDirectWDM, "_dense_response", "  DENSE response (kernel + output splines)")
    wrap(TF.TDTDIonTheFly, "__init__", "  TOF input splines")
    wrap(TF.TDTDIonTheFly, "__call__", "  TOF response kernel + output splines")
    wrap(WD.EMRIDirectWDM, "_handoff_may_trip", "  handoff bound (3 fundamentals)")
    wrap(WD, "harmonic_tracks_from_holder", "3 harmonic tracks")
    wrap(WD, "tracer_from_tof_output", "4 tracer (output spline evals)")
    wrap(WD, "accumulate_harmonic_batch", "5 lookup + plunge tail (accumulate_harmonic_batch)")
    wrap(type(table), "get_wdm_coeffs", "  table get_wdm_coeffs")
    wrap(WD, "lookup_sum_kernel", "  FUSED lookup-sum kernel")
    wrap(WD.EMRIDirectWDM, "_dense_td", "  dense TD tail")
    wrap(WH, "wdm_chunk_of_td", "6 plunge chunks")

    def prod(thr):
        h = wg(*params, mode_selection_threshold=thr)
        xp = cp if gpu else np
        h = xp.stack([xp.asarray(c) for c in h]) if isinstance(h, (list, tuple)) else xp.atleast_2d(h)
        return TDSignal(h[:3, oi:oi + n], tds).transform(wdm).arr

    def record(**rec):
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def chunk_split(st):
        """The lookup / plunge-chunk split of a call's ``last_stats``: the ``[EMRI_DIRECT] leaf N:``
        line's wording and the JSON fields. A single call is one row (a chunk row when it
        transformed windows); a batch counts its batched chunk rows plus any built alone."""
        tail = st.get("tail_layers", [])
        if isinstance(tail, (int, np.integer)):
            rows = 1 if st.get("windows", 0) else 0
            tail = [int(tail)] if rows else []
        else:
            rows = int(st.get("chunk_rows", 0)) + int(st.get("alone", 0))
        med = int(np.median(tail)) if len(tail) else 0
        f = dict(chunk_rows=rows, windows=int(st.get("windows", 0)), tail_median_px=med,
                 lookup_pixels=int(st.get("lookup_pixels", 0)), chunk_pixels=int(st.get("chunk_pixels", 0)),
                 dropped_pixels=int(st.get("dropped_pixels", 0)), nt_sub_eff=st.get("nt_sub_eff"),
                 tail_skipped=int(st.get("tail_skipped", 0)))
        txt = (f"chunk rows {rows} (tail median {med} px, {f['windows']} windows skipped {f['tail_skipped']}) "
               f"lookup px {f['lookup_pixels']} chunk px {f['chunk_pixels']} dropped px {f['dropped_pixels']}")
        return txt, f

    WALLS = ("capture", "response", "lookup", "tail_dense", "transform", "splice", "plan", "alloc")

    def walls_split(st, n_rows):
        """The phase walls of a call's ``last_stats`` (``walls``: exclusive seconds per phase of the
        whole batch / single call, device-synchronised; ``walls_detail``: sub-parts) in ms per row:
        the ``[walls]`` line's text and the JSON fields."""
        w, d = st.get("walls") or {}, st.get("walls_detail") or {}
        ms = {k: 1e3 * w.get(k, 0.0) / n_rows for k in WALLS}
        total = 1e3 * w.get("total", 0.0) / n_rows
        other = total - sum(1e3 * v / n_rows for k, v in w.items() if k != "total")
        det = {k: 1e3 * v / n_rows for k, v in d.items()}
        pts, pts_rows = st.get("n_response_points"), list(st.get("n_response_points_rows") or [])
        rows_txt = f"rows {min(pts_rows)}..{max(pts_rows)}" if pts_rows else "no chunk rows"
        calls = [1e3 * v for v in st.get("walls_per_call") or []]   # batch only: each _batch_chunk call [ms]
        calls_txt = (f"; per call: first {calls[0]:.0f} ms, rest median {float(np.median(calls[1:])):.0f} ms "
                     f"({len(calls)} calls)") if len(calls) > 1 else ""
        g = det.get
        txt = (f"capture {ms['capture']:.1f} ms/row, response {ms['response']:.1f} (pad {g('response_pad', 0):.1f}, "
               f"setup {g('response_setup', 0):.1f}, kernel {g('response_kernel', 0):.1f}, splines "
               f"{g('response_splines', 0):.1f}), lookup {ms['lookup']:.1f}, tail {ms['tail_dense']:.1f} (setup "
               f"{g('tail_setup', 0):.1f}, kernel {g('tail_kernel', 0):.1f}), transform {ms['transform']:.1f}, "
               f"splice {ms['splice']:.1f}, plan {ms['plan']:.1f}, alloc {ms['alloc']:.1f}, total {total:.1f} "
               f"(unattributed {other:.1f}); response points padded {pts if pts is not None else '-'} ({rows_txt})"
               f"{calls_txt}")
        f = dict(walls_ms_per_row=dict(ms, total=total, unattributed=other), walls_detail_ms_per_row=det,
                 n_response_points=pts, n_response_points_rows_min=min(pts_rows) if pts_rows else None,
                 n_response_points_rows_max=max(pts_rows) if pts_rows else None, walls_per_call_ms=calls)
        return txt, f

    base = dict(src=args.src, days=args.days, dt=args.dt, nf=nf, nt=nt, backend=args.backend,
                response=args.response, nt_sub=nt_sub, nt_sub_floor=nt_floor, tail_skip_px=tail_skip_px,
                chunk_batch=args.chunk_batch, band=band, brick=box.brick, tobs_s=box.tobs,
                foreground=box.foreground, noise=box.describe(),
                data_snr=None if box.d is None else box.data_snr)

    def scored(label, full):
        """Optimal SNR, the noise-weighted mismatch to production (a direct template) and the data
        score of one full-grid template; prints the [data] line."""
        snr[label] = box.snr(full)
        if label != "production":
            mmp[label] = box.match(full, h_prod)
        if box.d is not None:
            vs[label] = box.score(full)
            v = vs[label]
            print(f"[data] {args.days:g} d thr={thr:g} {label}: mm {v['mm']:.3e} logL {v['logL']:.4g} "
                  f"opt/data SNR {v['snr_ratio']:.5f} (data SNR {box.data_snr:.3f})", flush=True)

    for thr in [float(x) for x in args.thresh.split(",")]:
        snr, vs, mmp = {}, {}, {}
        h_prod = prod(thr)                                               # + warm-up
        scored("production", h_prod)
        print(f"\n[snr] {args.days:g} d thr={thr:g}: production optimal SNR {snr['production']:.3f} "
              f"({box.describe()})", flush=True)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            prod(thr)
        sync()
        t_prod = (time.perf_counter() - t0) / args.reps
        summary = {}
        for grid, direct in directs.items():
            scored(grid, direct(*params, mode_selection_threshold=thr).arr)   # + warm-up
            sync()
            T.clear()
            C.clear()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                direct(*params, mode_selection_threshold=thr)
            sync()
            t_dir = (time.perf_counter() - t0) / args.reps
            print(f"\n[stages] response={args.response} grid/lookup={grid} nt_sub={nt_sub} thr={thr:g} "
                  f"modes={direct.last_stats.get('modes')} "
                  f"n_fine={direct.last_stats.get('n_fine')} backend={args.backend} grid Nf={nf} Nt={nt} dt={args.dt} "
                  f"({args.days:g} d): direct {t_dir * 1e3:.0f} ms, production {t_prod * 1e3:.0f} ms (per template); "
                  f"SNR direct {snr[grid]:.3f} production {snr['production']:.3f}; mm vs production "
                  f"{mmp[grid]:.3e}", flush=True)
            split_txt, split = chunk_split(direct.last_stats)
            print(f"[split] grid={grid} thr={thr:g} nt_sub={nt_label(direct.last_stats)} "
                  f"chunk_batch={args.chunk_batch} single: {split_txt}", flush=True)
            walls_txt, walls = walls_split(direct.last_stats, 1)               # the last timed call
            print(f"[walls] grid={grid} thr={thr:g} nt_sub={nt_label(direct.last_stats)} single: {walls_txt}",
                  flush=True)
            for k in [lab for lab in LABELS if lab in T]:
                print(f"  {k:48s} {T[k] / args.reps * 1e3:8.1f} ms  ({C[k] // args.reps} calls)", flush=True)
            stages = {k.strip(): T[k] / args.reps * 1e3 for k in LABELS if k in T}
            record(**base, **split, **walls, thr=thr, grid=grid, chunk=0, rows=1, modes=direct.last_stats.get("modes"),
                   n_response=direct.last_stats.get("n_fine"), ms_per_template=t_dir * 1e3,
                   production_ms=t_prod * 1e3, stages_ms=stages, snr_direct=snr[grid],
                   snr_production=snr["production"], snr_edge=args.edge, mm_vs_production=mmp[grid],
                   data_direct=vs.get(grid), data_production=vs.get("production"))
            summary[(grid, 0)] = t_dir * 1e3
            if args.batch_rows <= 0:
                continue
            n_rows = max(args.batch_rows, max(chunks))
            base_rows = B.batch_rows(params, n_rows)
            # rows 1.. : the p0-shifted copies (their plunge, handoff and tail end differ by hours)
            rows = (base_rows[:1] + shifted + base_rows[1:])[:n_rows]
            sink = lambda idx, arr: None                                 # noqa: E731 (timing only)
            for chunk in chunks:
                direct.batch(rows[:min(2, chunk)], chunk_rows=chunk, consume=sink, mode_selection_threshold=thr)
                sync()
                if pool is not None:
                    pool.free_all_blocks()
                T.clear()
                C.clear()
                t0 = time.perf_counter()
                direct.batch(rows, chunk_rows=chunk, consume=sink, mode_selection_threshold=thr)
                sync()
                t_b = time.perf_counter() - t0
                st_b = dict(direct.last_stats)                           # before the parity run replaces it
                mem = pool.total_bytes() / 1e9 if pool is not None else float("nan")
                print(f"[stages] grid={grid} thr={thr:g} BATCH of {len(rows)} ({chunk} per call): {t_b * 1e3:.0f} ms "
                      f"total = {t_b / len(rows) * 1e3:.0f} ms per template (production {t_prod * 1e3:.0f}); "
                      f"gpu pool {mem:.1f} GB {({k: v for k, v in st_b.items() if not k.startswith('walls')})}",
                      flush=True)
                split_txt, split = chunk_split(st_b)
                print(f"[split] grid={grid} thr={thr:g} nt_sub={nt_label(st_b)} chunk_batch={args.chunk_batch} "
                      f"BATCH of {len(rows)} ({chunk} per call): {split_txt}", flush=True)
                walls_txt, walls = walls_split(st_b, len(rows))
                print(f"[walls] grid={grid} thr={thr:g} nt_sub={nt_label(st_b)} chunk_batch={args.chunk_batch} "
                      f"batch of {len(rows)} ({chunk} per call): {walls_txt}", flush=True)
                for k in [lab for lab in LABELS if lab in T]:
                    print(f"  {k:48s} {T[k] / len(rows) * 1e3:8.1f} ms per template  ({C[k]} calls)", flush=True)
                stages = {k.strip(): T[k] / len(rows) * 1e3 for k in LABELS if k in T}
                # the batched path's NUMBERS, untimed: the first `chunk` rows batched vs each alone
                p_rel, p_mm = parity(direct, grid, rows[:chunk], chunk, thr)
                record(**base, **split, **walls, thr=thr, grid=grid, chunk=chunk, rows=len(rows),
                       modes=st_b.get("modes"), n_response=st_b.get("n_response"),
                       ms_per_template=t_b / len(rows) * 1e3, production_ms=t_prod * 1e3, gpu_pool_gb=mem,
                       stages_ms=stages, snr_direct=snr[grid], snr_production=snr["production"],
                       snr_edge=args.edge, mm_vs_production=mmp[grid], data_direct=vs.get(grid),
                       data_production=vs.get("production"), parity_max_rel=p_rel, parity_mm_max=p_mm)
                summary[(grid, chunk)] = t_b / len(rows) * 1e3
        cols = [0] + (chunks if args.batch_rows > 0 else [])
        head = "  ".join(f"{('single' if c == 0 else f'{c}/call'):>8s}" for c in cols)
        print(f"\n[summary] {args.days:g} d thr={thr:g}: ms per template (production {t_prod * 1e3:.0f})\n"
              f"  {'grid/lookup':16s}{head}", flush=True)
        for grid in directs:
            print(f"  {grid:16s}" + "  ".join(f"{summary.get((grid, c), float('nan')):8.0f}" for c in cols), flush=True)
        print(f"  optimal SNR ({box.describe()}): production "
              f"{snr['production']:.3f}; " + ", ".join(f"{g} {snr[g]:.3f} (ratio {snr[g] / snr['production']:.6f}, "
                                                       f"mm vs production {mmp[g]:.2e})" for g in directs), flush=True)
        if vs:
            print(f"  vs mojito data {box.brick} (data SNR {box.data_snr:.3f}): "
                  + "; ".join(f"{g} mm {v['mm']:.3e} logL {v['logL']:.4g} SNR ratio {v['snr_ratio']:.5f}"
                              for g, v in vs.items()), flush=True)


if __name__ == "__main__":
    main()
