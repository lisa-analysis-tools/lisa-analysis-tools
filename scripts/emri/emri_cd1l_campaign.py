"""CD1L EMRI campaign: mismatch / logL / SNR of the EMRI templates against the mojito data.

For ONE CD1L EMRI (catalogue row ``--src``, 0-7) over ONE observation span (``--duration``
6mo = 180 d or 24mo = 720 d, the production TOBS convention: 15552000 s per 6 months),
build up to three templates on the data grid and score each in the WDM domain against the
per-source mojito L1 stream (noise-free, source only):

* ``prod``   the production response (get_emri_response_wrapper: SPECIAL frame, ICRS L1
             orbits, Lagrange order 40, REF-anchored; the 6mo CD1L base signal);
* ``tof``    TDI-on-the-fly (EMRITDIonFly, icrs_special), dense TD -> WDM;
* ``direct`` EMRIDirectWDM (n_ref lookup + plunge chunk; needs ``--table``).

Grid: DT=20 s (the data is decimated 2.5 s -> 20 s; EMRI content is below 25 mHz), Nf=180,
so layer_dt = 3600 s as in production. The 6mo grid is Nt=4320 (= the production Nt),
24mo is Nt=17280. The window starts START_OFFSET_S after the data start (the production
wrapper zeroes a 3e4 s buffer at its ends). Edge crop: 20 layers each end, as the
production WDM grid (min_time/max_time).

Per template (all WDM, SciRD v1 XYZ sensitivity, no time/phase maximisation):
  logL = -1/2 <d-h|d-h>   (0 for a perfect template: the data is source-only)
  logL0 = -1/2 <d|d>, snr_data = sqrt(<d|d>), snr_opt, snr_det (template_snr),
  mm_data = 1 - <d|h>/sqrt(<d|d><h|h>)  (noise weighted)
  flat per-channel mismatch + norm ratio vs the data (the amplitude safeguard),
and pairwise tof-vs-prod / direct-vs-prod / direct-vs-tof (noise-weighted mm, dlogL, flat
per-channel mm + norm ratio). Wall time and peak RSS per template.

One JSON line per (src, duration, thresh) is APPENDED to ``--out``. Run one source and
one duration per process (each FEW generator reads its 5.1 GB amplitude file whole: ~6 GB
transient); the driver ``emri_cd1l_campaign.sh`` loops them serially.

Memory / accuracy knobs (env; the long-window behaviour of tof/direct is what this
campaign is for):
  MOJITO_LIGHT_PATH   dir with catalogues/ and data/EMRI/L1/ (mojito light v1.0.0)
  START_OFFSET_S      window start after the data start [s] (default 5e4)
  TOF_FINE_DT         fine trajectory spacing fed to TOF and direct [s] (default 300)
  MODE_BATCH          modes per TOF / direct batch (default 16)
  EVAL_CHUNK          TOF dense-eval chunk [samples] (default 4096)
  RSS_LIMIT_GB        hard kill (exit 42) above this peak RSS; 0 = off (default 0)
"""
import argparse
import gc
import json
import os
import platform
import resource
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_tof_xyz_threeway as W  # noqa: E402  (loader, production wrapper, TOF helpers)

NF, DT = 180, 20.0
# frequency bands for the residual power left after subtracting a template [Hz] (None = grid edge)
BANDS = [(None, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2), (1e-2, None)]
DURATIONS = {"6mo": 15552000.0, "24mo": 4 * 15552000.0}
EDGE_LAYERS = 20


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if platform.system() == "Darwin" else r / 1e6   # bytes on macOS, kB on Linux


def flat_stats(a, b):
    """Per channel flat 1 - Re(O) and |a|/|b| (arrays already cropped to the active band)."""
    mm, amp = [], []
    for c in range(3):
        x, y = a[c], b[c]
        den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        mm.append(1.0 - float(np.sum(x * y)) / den if den > 0 else float("nan"))
        ny = float(np.linalg.norm(y))
        amp.append(float(np.linalg.norm(x)) / ny if ny > 0 else float("nan"))
    return mm, amp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, required=True, help="CD1L EMRI catalogue row (0-7)")
    ap.add_argument("--duration", required=True,
                    help="6mo, 24mo, or '<days>d' (whole hours; e.g. 16d for a laptop smoke run)")
    ap.add_argument("--templates", default="prod,tof,direct", help="comma list of prod,tof,direct")
    ap.add_argument("--thresh", default="1e-3", help="comma list of FEW mode_selection_threshold (production 1e-3)")
    ap.add_argument("--table", default=None, help="WDM lookup table h5 (Nf=180, dt=20); required for direct")
    ap.add_argument("--out", default="emri_cd1l_campaign.jsonl")
    args = ap.parse_args()

    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    bad = set(templates) - {"prod", "tof", "direct"}
    if bad:
        ap.error(f"unknown templates {sorted(bad)}")
    if "direct" in templates and not args.table:
        ap.error("--table is required for the direct template")

    if args.duration in DURATIONS:
        tobs = DURATIONS[args.duration]
    elif args.duration.endswith("d"):
        tobs = round(float(args.duration[:-1]) * 24) * NF * DT
    else:
        ap.error(f"--duration must be one of {sorted(DURATIONS)} or '<days>d'")
    n_win = int(round(tobs / DT))
    nt = n_win // NF
    assert nt * NF == n_win
    fine_dt = float(os.environ.get("TOF_FINE_DT", "300"))
    W.N_WIN = n_win
    W.N_FINE = max(1024, int(n_win * DT / fine_dt))
    W.MODE_BATCH = int(os.environ.get("MODE_BATCH", "16"))
    W.EVAL_CHUNK = int(os.environ.get("EVAL_CHUNK", "4096"))
    W.START_OFFSET_S = float(os.environ.get("START_OFFSET_S", "5e4"))
    if os.environ.get("MOJITO_LIGHT_PATH"):
        W.PATH = os.environ["MOJITO_LIGHT_PATH"]
    rss_limit = float(os.environ.get("RSS_LIMIT_GB", "0"))
    if rss_limit > 0:
        W.watchdog(rss_limit)

    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.datacontainer import DataResidualArray
    from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    t_start = time.perf_counter()
    params, data, data_t0, orb = W.load(args.src)
    # templates are built on the FULL grid (the direct assembly needs every pixel) and scored
    # on the production-style cropped grid: crop() slices a full array to wdm's active band
    wdm_full = WDMSettings(NF, nt, DT, force_backend="cpu")
    wdm = WDMSettings(NF, nt, DT, min_time=EDGE_LAYERS * NF * DT, max_time=(nt - EDGE_LAYERS) * NF * DT,
                      force_backend="cpu")
    tds = TDSettings(n_win, DT, t0=0.0, force_backend="cpu")

    def crop(arr, s=None):
        s = wdm if s is None else s
        out = np.ascontiguousarray(np.asarray(arr)[:, s.active_slice_f, s.active_slice_t])
        assert out.shape[1:] == (s.Nf_active, s.Nt_active), (out.shape, s.Nf_active, s.Nt_active)
        return out

    d_full = np.asarray(TDSignal(data, tds).transform(wdm_full).arr)
    d_arr = crop(d_full)
    del data
    gc.collect()
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1")
    ac_data = AnalysisContainer(DataResidualArray(WDMSignal(d_arr, wdm)), sens)
    dd = float(np.real(ac_data.inner_product()))
    # per-band containers: the residual power a subtracted template leaves, by frequency
    band_ac = []
    for lo, hi in BANDS:
        sb = WDMSettings(NF, nt, DT, min_freq=lo, max_freq=hi, min_time=EDGE_LAYERS * NF * DT,
                         max_time=(nt - EDGE_LAYERS) * NF * DT, force_backend="cpu")
        band_ac.append((sb, AnalysisContainer(DataResidualArray(WDMSignal(crop(d_full, sb), sb)),
                                              XYZ2SensitivityMatrix(sb, model="scirdv1"))))
    del d_full
    base = dict(src=args.src, duration=args.duration, tobs_s=tobs, dt=DT, nf=NF, nt=nt,
                edge_layers=EDGE_LAYERS, window_start_after_ref_s=data_t0 - W.REF,
                tof_fine_dt=fine_dt, n_fine=W.N_FINE, mode_batch=W.MODE_BATCH,
                host=socket.gethostname(), load_wall_s=time.perf_counter() - t_start,
                snr_data=float(np.sqrt(dd)), logL0=-0.5 * dd, bands_hz=BANDS,
                data_snr2_bands=[float(np.real(ac.inner_product())) for _, ac in band_ac])
    print(f"[campaign] src={args.src} {args.duration}: Nt={nt} snr_data={np.sqrt(dd):.3f} "
          f"start REF+{data_t0 - W.REF:.0f}s", flush=True)

    wg = offset_int = gen = None
    if templates:   # the ONE FEW construction; the TOF and direct paths reuse its generator
        wg, offset_int, gen = W.legacy_wrapper(orb, data_t0)

    direct_gen = None
    if "direct" in templates:
        from lisatools.domains import WDMLookupTable
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM
        table = WDMLookupTable.from_file(args.table, force_backend="cpu")
        direct_gen = EMRIDirectWDM(gen, table, wdm_full, orbits=orb, tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                                   t_start=W.REF, data_t0=data_t0, n_fine=W.N_FINE, mode_batch=W.MODE_BATCH)

    for thr in [float(x) for x in args.thresh.split(",")]:
        row = dict(base, thresh=thr)
        arrs = {}
        for tag in templates:
            t0 = time.perf_counter()
            try:
                if tag == "prod":
                    h_full = np.asarray(TDSignal(W.legacy_td(params, wg, offset_int, thr), tds).transform(wdm_full).arr)
                    row["prod_nmodes"] = int(getattr(gen, "num_modes_kept", -1))
                elif tag == "tof":
                    td, nsub, n_in = W.tof_td(params, orb, data_t0, thr, gen)
                    row["tof_nsub"], row["tof_inside_samples"] = nsub, n_in
                    h_full = np.asarray(TDSignal(td, tds).transform(wdm_full).arr)
                    del td
                else:
                    h_full = np.asarray(direct_gen(*params, mode_selection_threshold=thr).arr)
                    row.update({f"direct_{k}": v for k, v in direct_gen.last_stats.items()})
                h = crop(h_full)
            except Exception as exc:   # record and keep going: this is a debugging campaign
                row[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"
                row[f"{tag}_traceback"] = traceback.format_exc()[-4000:]
                print(f"[campaign] {tag} FAILED: {row[f'{tag}_error']}", flush=True)
                continue
            row[f"{tag}_wall_s"] = time.perf_counter() - t0
            row[f"{tag}_peak_rss_gb"] = peak_rss_gb()
            arrs[tag] = h
            hs = WDMSignal(h, wdm)
            opt, det = ac_data.template_snr(hs)
            ll = float(np.real(ac_data.template_likelihood(hs)))
            hh = float(np.real(opt)) ** 2
            row[f"{tag}_logL"] = ll
            row[f"{tag}_resid_snr2"] = -2.0 * ll                # power left in the residual
            row[f"{tag}_resid_snr2_bands"] = [
                -2.0 * float(np.real(ac.template_likelihood(WDMSignal(crop(h_full, sb), sb)))) for sb, ac in band_ac]
            del h_full
            row[f"{tag}_snr_opt"] = float(np.real(opt))
            row[f"{tag}_snr_det"] = float(np.real(det))
            row[f"{tag}_snr_ratio"] = float(np.real(opt)) / np.sqrt(dd)
            dh = 0.5 * (dd + hh + 2 * ll)                       # <d|h> from -1/2<d-h|d-h>
            row[f"{tag}_mm_data"] = 1.0 - dh / np.sqrt(dd * hh) if hh > 0 else float("nan")
            row[f"{tag}_flat_mm_data"], row[f"{tag}_flat_amp_data"] = flat_stats(h, d_arr)
            print(f"[campaign] thr={thr:g} {tag}: logL={ll:+.4f} resid_snr2 bands="
                  f"{'/'.join(f'{x:.3g}' for x in row[f'{tag}_resid_snr2_bands'])} mm_data={row[f'{tag}_mm_data']:.3e} "
                  f"snr_opt/data={row[f'{tag}_snr_ratio']:.6f} wall={row[f'{tag}_wall_s']:.0f}s "
                  f"rss={row[f'{tag}_peak_rss_gb']:.1f}GB", flush=True)
            gc.collect()

        for a, b in (("tof", "prod"), ("direct", "prod"), ("direct", "tof")):
            if a in arrs and b in arrs:
                ac_b = AnalysisContainer(DataResidualArray(WDMSignal(arrs[b], wdm)), sens)
                bb = float(np.real(ac_b.inner_product()))
                opt_a, _ = ac_b.template_snr(WDMSignal(arrs[a], wdm))
                ll = float(np.real(ac_b.template_likelihood(WDMSignal(arrs[a], wdm))))
                aa = float(np.real(opt_a)) ** 2
                ab = 0.5 * (bb + aa + 2 * ll)
                row[f"mm_{a}_{b}"] = 1.0 - ab / np.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
                row[f"dlogL_{a}_{b}"] = row[f"{a}_logL"] - row[f"{b}_logL"]
                row[f"flat_mm_{a}_{b}"], row[f"flat_amp_{a}_{b}"] = flat_stats(arrs[a], arrs[b])
                print(f"[campaign] thr={thr:g} {a} vs {b}: mm={row[f'mm_{a}_{b}']:.3e} "
                      f"dlogL={row[f'dlogL_{a}_{b}']:+.4e}", flush=True)
        row["total_wall_s"] = time.perf_counter() - t_start
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
