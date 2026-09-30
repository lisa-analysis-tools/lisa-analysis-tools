"""Per-EMRI mismatch of the direct-to-WDM template against the CD1L data (Task A9).

For ONE CD1L EMRI (catalogue row ``--src``) on a WDM grid with the production layer
duration (3600 s; Nf=180 at dt=20 s, matching the laptop lookup table) starting
START_OFFSET_S after the data start, compare in the WDM domain, per channel X, Y, Z:

* ``direct``: :class:`lisatools.sources.emri.wdm_direct.EMRIDirectWDM` (the fast
  template: n_ref lookup + plunge chunk, no dense time grid for the inspiral);
* ``tof``: the same TDI-on-the-fly response, dense TD -> WDM;
* ``prod``: the production response (get_emri_response_wrapper, the 6mo base signal);
* ``data``: the mojito L1 source-only stream, decimated 2.5 s -> 20 s.

Mismatch 1 - Re(O) (flat, NO maximisation) over pixels at least EDGE layers from the
window ends (production crops edges too). Appends one JSON line per (src, thresh).
Env: MOJITO_LIGHT_PATH, NT (layers, default 384 = 16 d), THRESH (default "1e-3").
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_tof_xyz_threeway as W  # noqa: E402

NF, DT = 180, 20.0
NT = int(os.environ.get("NT", "384"))
EDGE = 32


def mm(a, b):
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return 1.0 - float(np.sum(a * b)) / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="emri_direct_wdm_mismatch.jsonl")
    args = ap.parse_args()

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import EMRITDIonFly
    from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

    W.watchdog(float(os.environ.get("WT_RSS_LIMIT_GB", "5")))
    W.N_WIN = NF * NT                         # samples at DT: the WDM grid span
    params, data, data_t0, orb = W.load(args.src)
    REF = W.REF
    wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
    tds = TDSettings(NF * NT, DT, force_backend="cpu")
    table = WDMLookupTable.from_file(args.table, force_backend="cpu")
    tdi = TDIConfig("2nd generation", force_backend="cpu")
    span = NF * NT * DT
    h_data = np.asarray(TDSignal(data, tds).transform(wdm).arr)

    wg, offset_int, gen = W.legacy_wrapper(orb, data_t0)      # the ONE FEW construction
    n = np.arange(NT)
    act = (n >= EDGE) & (n < NT - EDGE)
    for thr in [float(x) for x in os.environ.get("THRESH", "1e-3").split(",")]:
        row = dict(src=args.src, thresh=thr, nt=NT, dt=DT, window_days=span / 86400,
                   window_start_after_ref_s=data_t0 - REF)
        t0 = time.perf_counter()
        direct = EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=tdi, t_start=REF, data_t0=data_t0,
                               mode_batch=16)
        h_dir = np.asarray(direct(*params, mode_selection_threshold=thr).arr)
        row["wall_direct_s"] = time.perf_counter() - t0
        row.update({f"direct_{k}": v for k, v in direct.last_stats.items()})
        modes = direct._mode_list(params, {"mode_selection_threshold": thr})

        t0 = time.perf_counter()
        tg = data_t0 + np.arange(NF * NT) * DT
        td = np.zeros((3, tg.size))
        for j in range(0, len(modes), 16):
            fly = EMRITDIonFly(gen, orb, tdi, DT, data_t0 - REF + span + 2000.0, REF, frame="icrs_special",
                               n_fine=direct.n_fine, t_fine_window=(data_t0, data_t0 + span))
            out = fly(*params, mode_selection=modes[j:j + 16])
            x = np.asarray(out.x)
            ins = np.flatnonzero((tg > x[:, 0].max()) & (tg < x[:, -1].min()))
            for i in range(0, ins.size, 4096):
                sl = ins[i:i + 4096]
                td[:, sl] += np.real(np.sum(np.asarray(out.eval_tdi(tg[sl])), axis=0))
            del out, fly
            gc.collect()
        h_tof = np.asarray(TDSignal(td, tds).transform(wdm).arr)
        row["wall_tof_dense_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        h_prod = np.asarray(TDSignal(W.legacy_td(params, wg, offset_int, thr), tds).transform(wdm).arr)
        row["wall_prod_s"] = time.perf_counter() - t0

        for tag, a, b in (("direct_data", h_dir, h_data), ("prod_data", h_prod, h_data), ("tof_data", h_tof, h_data),
                          ("direct_prod", h_dir, h_prod), ("direct_tof", h_dir, h_tof), ("tof_prod", h_tof, h_prod)):
            row[f"mm_{tag}"] = [mm(a[c][:, act], b[c][:, act]) for c in range(3)]
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
