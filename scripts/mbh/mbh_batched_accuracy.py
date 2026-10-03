#!/usr/bin/env python
"""Accuracy of the batched, windowed MBH template against the stock templates on one grid.

The accuracy step of scripts/mbh/mbh_speed_durations.sh; its speed step is
mbh_batched_gpu_benchmark.py on the SAME grid. This script imports the
benchmark's builders (``parse_args`` + presets, ``build_context``,
``build_stock_gen``, ``build_windowed_adapter``), so grid, source placement,
orbits, window and generator configuration cannot drift between the two steps.
Every benchmark grid / source / orbits / window flag is accepted here
(``--smoke``, ``--nt``, ``--dt``, ``--edge-crop``, ``--source-id``,
``--catalogue``, ``--orbits-file``, ``--merger-day``, ``--window-*-days``,
``--order``, ``--acc-tol``, ``--foreground``, ...); its sweep flags are accepted and
ignored.

Per source, for the truth row and NEAR-truth rows (t_plunge +/- 0.3 s, distance
x (1 +/- 5e-4), phi_ref +/- 2e-3 rad: the gating rows of
mbh_batched_mojito_check.py), the batched template (all rows in ONE batched
generator call, t_plunge shifted by the snap exactly as MBHBatchedLikeMove
does) is compared with two STOCK references:

* ``stock90`` the move's own cross-check generator (``MBH_LIKELIHOOD=batched``'s
  residual rebuild and ``_verify_prev_logl``): PhenomTHMTDIWaveform on the
  lattice-SNAPPED epoch, T = window_before (90 d), the batched response order,
  through ``SnappedEpochMBHGen``. THE ACCEPTANCE GATES ON THIS ONE.
* ``prod``    the production stock default (``MBH_LIKELIHOOD=full``): the
  unsnapped epoch, T = MBH_DEFAULT_WAVEFORM_DURATION (YRSID/12 = 30.44 d),
  response order 8. Information only: the batched model carries 90 d of
  inspiral, so its difference from a 30.44-d template is a MODEL difference
  (the inspiral 30-90 d before merger), not a batching error.

For a reference R the data are R's own noiseless truth template on the run grid
(so logL_R(truth) = 0), noise-weighted like the benchmark's containers (mbh_harness.py:
the EMRI harness's ``RunBox`` noise, scirdv1 XYZ PLUS the
FittedHyperbolicTangentGalacticForeground at Tobs = Nf*Nt*dt; ``--foreground off`` drops
the foreground), scored through the container slicing (the batched template on its
sub-box), in the move's convention (-1/2 <d|d> over the active box + <d|h> - 1/2 <h|h>
over the template's box):

  dlogL      logL(batched) - logL(R), per row
  mismatch   1 - <R|b> / sqrt(<R|R><b|b>) over the kept box
  kept err   max |b - R| / max |R| over the kept layers (R's maximum over the active box)
  outside    R's power in the active box outside the kept box: plain (sum of squares)
             and noise-weighted (1 - <R_box|R_box> / <R|R>)
  ||delta||  noise-weighted norm of b - R over the kept box

Acceptance per source (truth + near rows, vs stock90): |dlogL| <= --acc-tol (0.5, the
move's MBH_CHECK_LL_TOL) and mismatch < --mm-tol (1e-6); one line
``[accuracy] src <id> <days> d ...: PASS|FAIL`` per run. ``--jsonl`` APPENDS one JSON
line per (row, reference) and one ``kind=accuracy_source`` line; ``--strict`` exits 3 on
FAIL.

    # cluster (one GPU), the merger-centred 120-d grid (merger 100 d in), mojito MBHB id 17,
    # equal-arm orbits
    python scripts/mbh/mbh_batched_accuracy.py --backend cuda13x --nt 2880 --merger-day 100 \\
        --source-id 17 --jsonl OUT/accuracy.jsonl --strict
    # the source's own mojito brick: its L1 orbits (windowed ltt read) and its time frame
    # (the grid centred on the catalogue merger, shifted into the file near its ends)
    MOJITO_DATA_PATH=/shared/data/mojito_cache python scripts/mbh/mbh_batched_accuracy.py \\
        --backend cuda13x --nt 2880 --merger-day 100 --source-id 16 --orbits auto \\
        --jsonl OUT/accuracy.jsonl --strict
    # laptop smoke (the benchmark's tiny CPU grid, truth + one near row, < 3 GB RSS)
    python scripts/mbh/mbh_batched_accuracy.py --smoke --backend cpu

JSON (the EMRI harness's names where the quantity is shared): every line carries
``brick``, ``tobs_s``, ``foreground`` (bool), ``noise`` (RunBox.describe()), ``data_snr``
(None: no mojito stream is read here -- the driver's data step scores it) and
``differs_from_emri``; a row line carries ``snr_batched`` / ``snr_ref`` and ``vs_ref`` =
{mm = 1 - <R|b>/sqrt(<R|R><b|b>) over the ACTIVE box, logL, snr_ratio, snr}: EMRI's
``data`` dict with the reference template as the data; the source line carries
``mm_vs_production`` (truth row: batched vs the production template, noise-weighted on
the active box, no maximisation) and ``snr_stock90`` (the source's optimal SNR here).

Memory: the references are full-grid stock templates (the 90-d phentax + response,
the full-grid WDM transform: the stock column of the benchmark), one at a time; the
data / template boxes are 3 x Nf_active x Nt_active (~178 x Nt) arrays; the batched
call holds all rows (3 by default) on the window lattice (~3.8 M samples per row at
dt 2.5 s, independent of the duration).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0

#: per-mode defaults of the accuracy-only flags (CLI > preset), as the benchmark's PRESETS
ACC_PRESETS = {
    "full": dict(near=2, prod_T_days="default"),
    # the smoke stands a 1-d generation window in for the production 30.44 d: the
    # laptop's phentax runs un-jitted, where a 30-d waveform costs GBs and minutes
    # (a 6-h one is all zero after the generator's 15000-s onset warm-up zeroing)
    "smoke": dict(near=1, prod_T_days="1"),
}


def parse_acc_args(argv):
    """``(accuracy-only flags, the rest)``; the rest goes to the benchmark's parser."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False,
        epilog="Every other flag is mbh_batched_gpu_benchmark.py's (see its --help).",
    )
    p.add_argument("--near", type=int, choices=(0, 1, 2),
                   help="near-truth rows: 2 = +/- (default), 1 = + only (smoke), 0 = truth only")
    p.add_argument("--near-dt", type=float, default=0.3, help="near rows: t_plunge offset [s]")
    p.add_argument("--near-dist", type=float, default=5e-4, help="near rows: fractional distance offset")
    p.add_argument("--near-phi", type=float, default=2e-3, help="near rows: phi_ref offset [rad]")
    p.add_argument("--prod-T-days",
                   help="production reference phentax T: 'default' (MBH_DEFAULT_WAVEFORM_DURATION = "
                        "YRSID/12 = 30.44 d) or days (smoke: 0.25)")
    p.add_argument("--prod-order", type=int, default=8,
                   help="production reference response order (MBH_RESPONSE_ORDER default 8)")
    p.add_argument("--skip-prod", action="store_true", help="only the gating stock90 reference")
    p.add_argument("--mm-tol", type=float, default=1e-6, help="acceptance: mismatch vs stock90 below this")
    p.add_argument("--jsonl", help="APPEND one JSON line per (row, reference) + one per source here")
    p.add_argument("--strict", action="store_true", help="exit 3 when the acceptance FAILS")
    return p.parse_known_args(argv)


def _backend_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--backend" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--backend="):
            return a.split("=", 1)[1]
    return "cuda13x"  # the benchmark's default


def _configure_process(backend):
    """mbh_batched_gpu_benchmark.py's process environment, set BEFORE numpy / jax
    import (keep the two equal): one thread, JAX allocating on demand; on the CPU
    backend JAX on the CPU with one XLA thread for execution and codegen."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_cpu_multi_thread_eigen=false --xla_cpu_parallel_codegen_split_count=1",
        )


if __name__ == "__main__":
    _configure_process(_backend_from_argv(sys.argv[1:]))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import mbh_batched_gpu_benchmark as B  # noqa: E402  (builders; ARGS stays None on import)
from lisatools.analysiscontainer import AnalysisContainer  # noqa: E402
from lisatools.domains import WDMSignal  # noqa: E402
from lisatools.utils.utility import asnumpy  # noqa: E402


def fval(x):
    """Host float of a (possibly device, possibly complex) scalar."""
    return float(np.real(asnumpy(x)))


def accuracy_rows(truth, acc):
    """Row 0 = truth; then the near rows (+ first)."""
    rows, kinds = [np.array(truth, dtype=np.float64)], ["truth"]
    for sgn in (+1.0, -1.0)[: int(acc.near)]:
        r = np.array(truth, dtype=np.float64)
        r[10] += sgn * acc.near_dt       # t_plunge [s]
        r[4] *= 1.0 + sgn * acc.near_dist  # distance, fractional
        r[5] += sgn * acc.near_phi        # phi_ref [rad]
        rows.append(r)
        kinds.append("near")
    return np.stack(rows), kinds


def row_metrics(ac_data, dd, sens, h_ref, h_b, box):
    """Batched ``h_b`` (kept box) vs reference ``h_ref`` (active box); the data of
    ``ac_data`` (``<d|d>`` = ``dd``) are the reference's truth template.

    Besides the MBH acceptance quantities, ``vs_ref`` = the EMRI harness's ``data`` dict
    with d = R(truth) over the active box and h = the batched template: ``mm`` = 1 -
    <d|h>/sqrt(<d|d><h|h>) (so it also charges the reference's power outside the kept box,
    unlike ``mismatch``), ``logL`` = -1/2<d-h|d-h>, ``snr_ratio`` = sqrt(<h|h>/<d|d>),
    ``snr`` = sqrt<h|h>."""
    xp = h_ref.settings.xp
    r_box = xp.ascontiguousarray(h_ref.arr[..., box])
    hr = np.asarray(asnumpy(h_ref.arr))
    hb = np.asarray(asnumpy(h_b.arr))
    scale = float(np.abs(hr).max())
    p = hr * hr
    p_tot = float(p.sum())
    ll_ref = fval(ac_data.template_likelihood(h_ref))
    ll_b = fval(ac_data.template_likelihood(h_b))
    dh_b = fval(ac_data.template_inner_product(h_b))
    ac_ref = AnalysisContainer(h_ref, sens)
    hh_ref = fval(ac_ref.inner_product())
    hh_ref_box = fval(ac_ref.template_inner_product(WDMSignal(r_box, h_b.settings)))
    overlap = fval(ac_ref.template_inner_product(h_b, normalize=True))
    snr_b = fval(ac_ref.template_snr(h_b)[0])
    dnorm = fval(ac_data.template_snr(WDMSignal(h_b.arr - r_box, h_b.settings))[0])
    nan = float("nan")
    data_snr = float(np.sqrt(max(dd, 0.0)))
    return dict(
        logL_ref=ll_ref, logL_batched=ll_b, dlogL=ll_b - ll_ref, mismatch=1.0 - overlap,
        kept_layer_rel_err=float(np.abs(hb - hr[..., box]).max()) / scale if scale > 0 else nan,
        ref_power_outside_box=float(p_tot - p[..., box].sum()) / p_tot if p_tot > 0 else nan,
        ref_snr2_outside_box=1.0 - hh_ref_box / hh_ref if hh_ref > 0 else nan,
        delta_norm=dnorm, snr_ref=float(np.sqrt(max(hh_ref, 0.0))), snr_batched=snr_b,
        # an all-zero reference (e.g. a generation window inside the onset zeroing): its
        # ratios are meaningless; flagged instead of read
        degenerate_reference=bool(hh_ref <= 0.0),
        vs_ref=dict(mm=1.0 - dh_b / (data_snr * snr_b) if data_snr > 0 and snr_b > 0 else nan,
                    logL=ll_b, snr_ratio=snr_b / data_snr if data_snr > 0 else nan, snr=snr_b),
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    acc, rest = parse_acc_args(argv)
    args = B.parse_args(rest)
    mode = "smoke" if args.smoke else "full"
    for k, v in ACC_PRESETS[mode].items():
        if getattr(acc, k) is None:
            setattr(acc, k, v)
    B.start_watchdog(args.mem_cap_gb)
    t_start = time.perf_counter()

    try:
        ctx = B.build_context(args)
    except B.NotAdmitted as exc:
        # a brick-placed window the merger is not admitted to: a skip, never a failure
        days = float(args.nf) * float(args.nt) * float(args.dt) / DAY
        print(f"[accuracy] src {args.source_id} {days:g} d (Nt {args.nt}): SKIPPED -- {exc}", flush=True)
        if acc.jsonl:
            line = dict(script="mbh_batched_accuracy", kind="accuracy_source", status="skipped_not_admitted",
                        mode=mode, backend=args.backend, days=days, Nf=int(args.nf), Nt=int(args.nt),
                        dt=float(args.dt), source_id=args.source_id, reason=str(exc), brick=None,
                        tobs_s=days * DAY, foreground=args.foreground == "on", noise=None, data_snr=None,
                        differs_from_emri=list(B.DIFFERS_FROM_EMRI))
            with open(acc.jsonl, "a") as f:
                f.write(json.dumps(line, default=B._json_default) + "\n")
        print("DONE", flush=True)
        return 0
    B.mark(f"accuracy: mode={mode} backend={args.backend} grid Nf={ctx.Nf} Nt={ctx.Nt} dt={ctx.dt} "
           f"({ctx.Tobs / DAY:.2f} d), active t [{ctx.wdm.ind_min_t}, {ctx.wdm.ind_max_t}]; source "
           f"{ctx.cat_src}; merger {(ctx.t_merge_abs - ctx.data_t0) / DAY:.3f} d in ({ctx.placement_note}); "
           f"snap {ctx.snap:+.6f} s; {ctx.orbits_desc}")

    rows, kinds = accuracy_rows(ctx.truth, acc)   # every row gates (truth + near)

    # ---- batched: all rows in ONE windowed call, t_plunge - snap (the move's _generate)
    windowed, adapter = B.build_windowed_adapter(ctx)
    geom = B.mbh_window_layers(ctx.wdm, ctx.t_merge_abs, ctx.W_before, ctx.W_after, ctx.W_pad, ctx.W_margin)
    adapter.set_window(geom["n_start"], geom["Nt_keep"], geom["n_pad_lo"], geom["n_pad_hi"])
    rel0 = geom["n_start"] - int(ctx.wdm.ind_min_t)
    box = slice(rel0, rel0 + geom["Nt_keep"])
    geometry = dict(geom, **adapter.geometry)
    geometry.update(
        merger_layer=(ctx.t_merge_abs - ctx.data_t0) / ctx.layer,
        box_clamped_lo=bool(geom["n_start"] == int(ctx.wdm.ind_min_t)),
        box_clamped_hi=bool(geom["n_start"] + geom["Nt_keep"] == int(ctx.wdm.ind_max_t) + 1),
    )
    B.mark(f"kept layers [{geom['n_start']}, {geom['n_start'] + geom['Nt_keep']}) segment "
           f"[{adapter.geometry['s0']}, {adapter.geometry['s0'] + adapter.geometry['Nt_seg']}); clamped "
           f"lo/hi {geometry['box_clamped_lo']}/{geometry['box_clamped_hi']}")
    p = rows.copy()
    p[:, 10] -= ctx.snap
    gen_b = B.cache_relief(adapter, args)
    hb_all, t_b = B.timed(ctx, lambda: gen_b(*p.T))
    h_b = [WDMSignal(hb_all.arr[i], hb_all.settings) for i in range(rows.shape[0])]
    B.mark(f"batched: {rows.shape[0]} rows in one call, {t_b:.3g} s (first call: JIT included)")

    # ---- references: the gating stock90, then (information) the production default
    refs = [("stock90", dict(T_days=ctx.W_before / DAY, order=int(args.order), epoch="snapped", gating=True))]
    if not acc.skip_prod:
        T_prod = (float(B.MBH_DEFAULT_WAVEFORM_DURATION) if acc.prod_T_days == "default"
                  else float(acc.prod_T_days) * DAY)
        refs.append(("prod", dict(T_days=T_prod / DAY, order=int(acc.prod_order), epoch="unsnapped", gating=False)))
    # the benchmark's noise (mbh_harness.py): the EMRI harness's RunBox on this grid, XYZ2
    # scirdv1 + the fitted tanh galactic foreground at Tobs = Nf*Nt*dt unless --foreground off
    sens = ctx.box.sens
    B.mark(f"noise weighting: {ctx.noise_label}")
    base = dict(
        script="mbh_batched_accuracy", mode=mode, backend=args.backend, days=ctx.Tobs / DAY, Nf=ctx.Nf,
        Nt=ctx.Nt, dt=ctx.dt, edge_crop=int(args.edge_crop), source_id=args.source_id,
        catalogue_source=ctx.cat_src, orbits=ctx.orbits_desc, placement=ctx.placement_note,
        brick=None if ctx.brick is None else os.path.basename(ctx.brick), start_offset_s=ctx.start_offset_s,
        merger_day=(ctx.t_merge_abs - ctx.data_t0) / DAY, order=int(args.order),
        decimate=int(ctx.decimate),
        window_days=dict(before=ctx.W_before / DAY, after=ctx.W_after / DAY, pad=ctx.W_pad / DAY,
                         margin=ctx.W_margin / DAY),
        Nt_keep=int(geom["Nt_keep"]), n_start=int(geom["n_start"]), **ctx.noise,
        data_snr=None,   # no mojito stream is read here (the driver's data step scores it)
        differs_from_emri=list(B.DIFFERS_FROM_EMRI),
    )
    lines, results = [], {}
    for name, meta in refs:
        raw = B.build_stock_gen(ctx, ctx.t0_snapped if name == "stock90" else ctx.epoch,
                                meta["T_days"] * DAY, meta["order"])
        gen = B.SnappedEpochMBHGen(raw, ctx.snap) if name == "stock90" else raw
        sg = B.cache_relief(gen.get_signals_for_residuals, args)
        h_ref, t_ref = [], []
        for i in range(rows.shape[0]):
            h, t = B.timed(ctx, lambda: sg(*rows[i]))
            h_ref.append(h)
            t_ref.append(t)
        B.evict_stock_gen(raw)
        raw = gen = sg = None
        gc.collect()
        data = WDMSignal(ctx.xp.array(h_ref[0].arr, copy=True), h_ref[0].settings)
        ac_data = AnalysisContainer(data, sens)
        dd = fval(ac_data.inner_product())
        results[name] = []
        for i in range(rows.shape[0]):
            m = row_metrics(ac_data, dd, sens, h_ref[i], h_b[i], box)
            m.update(row=i, row_kind=kinds[i], gating=bool(meta["gating"]))
            results[name].append(m)
            lines.append(dict(base, kind="accuracy_row", ref=name, ref_T_days=meta["T_days"],
                              ref_order=meta["order"], ref_epoch=meta["epoch"], ref_s=t_ref[i], **m))
            print(f"[accuracy] src {args.source_id} row {i} [{kinds[i]}] vs {name} "
                  f"(T {meta['T_days']:.4g} d, order {meta['order']}): dlogL {m['dlogL']:+.3e} "
                  f"(logL ref {m['logL_ref']:+.3e}, batched {m['logL_batched']:+.3e}) | mm {m['mismatch']:.3e} "
                  f"| kept err {m['kept_layer_rel_err']:.2e} | outside box {m['ref_power_outside_box']:.2e} "
                  f"(SNR^2 {m['ref_snr2_outside_box']:.2e}) | ||delta|| {m['delta_norm']:.3e} | SNR ref/batched "
                  f"{m['snr_ref']:.2f}/{m['snr_batched']:.2f} | mm over the active box {m['vs_ref']['mm']:.3e} "
                  f"| stock {t_ref[i]:.3g} s"
                  + (" | DEGENERATE reference (all zero)" if m["degenerate_reference"] else ""), flush=True)
        del data, ac_data, h_ref
        gc.collect()

    gate = [m for m in results["stock90"] if m["gating"]]
    worst_dll = max(abs(m["dlogL"]) for m in gate)
    worst_mm = max(m["mismatch"] for m in gate)
    # an all-zero gating reference cannot pass: its mismatch is undefined
    ok = bool(worst_dll <= float(args.acc_tol) and worst_mm < float(acc.mm_tol)
              and not any(m["degenerate_reference"] for m in gate))
    info = ""
    if "prod" in results:
        info = (f"; vs production stock (T {refs[1][1]['T_days']:.4g} d, order {acc.prod_order}, information): "
                f"max|dlogL| {max(abs(m['dlogL']) for m in results['prod']):.3e}, max mm "
                f"{max(m['mismatch'] for m in results['prod']):.3e}")
        if any(m["degenerate_reference"] for m in results["prod"]):
            info += " [DEGENERATE: the production reference is all zero; ignore these numbers]"
    verdict = "PASS" if ok else "FAIL"
    snr = results["stock90"][0]["snr_ref"]           # the source's optimal SNR at this duration
    mm_prod = None if "prod" not in results else results["prod"][0]["vs_ref"]["mm"]
    print(f"[accuracy] src {args.source_id} {ctx.Tobs / DAY:g} d (Nt {ctx.Nt}, SNR {snr:.1f}"
          + (f", window decimated {ctx.decimate}x" if ctx.decimate != 1 else "") + f"): {verdict} -- "
          f"vs stock90 (snapped, T {ctx.W_before / DAY:g} d) on {len(gate)} truth+near rows: max|dlogL| "
          f"{worst_dll:.3e} (tol {float(args.acc_tol):g}), max mm {worst_mm:.3e} (tol {float(acc.mm_tol):g})"
          f"{info}" + ("" if mm_prod is None else f"; mm_vs_production {mm_prod:.3e}"), flush=True)
    lines.append(dict(
        base, kind="accuracy_source", status="ok", passed=ok,
        gate=dict(dlogL_tol=float(args.acc_tol), mm_tol=float(acc.mm_tol)), snr_stock90=snr,
        max_abs_dlogL_stock90=worst_dll, max_mismatch_stock90=worst_mm,
        max_abs_dlogL_prod=None if "prod" not in results else max(abs(m["dlogL"]) for m in results["prod"]),
        max_mismatch_prod=None if "prod" not in results else max(m["mismatch"] for m in results["prod"]),
        # the EMRI harness's fast-vs-production key: truth row, batched vs the production
        # template, noise-weighted over the active box, no maximisation
        mm_vs_production=mm_prod,
        rows=rows, row_kinds=kinds, geometry=geometry, batched_s=t_b, batched_rows=int(rows.shape[0]),
        wall_s=time.perf_counter() - t_start, peak_rss_gb=B.rss_gb(),
    ))
    if acc.jsonl:
        with open(acc.jsonl, "a") as f:
            for line in lines:
                f.write(json.dumps(line, default=B._json_default) + "\n")
        B.mark(f"appended {len(lines)} lines to {acc.jsonl}")
    B.mark(f"done in {time.perf_counter() - t_start:.1f} s")
    print("DONE", flush=True)
    return 3 if (acc.strict and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
