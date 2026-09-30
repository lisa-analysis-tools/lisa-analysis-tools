"""Per-EMRI injection-set table for the TDI-on-the-fly sign-off (Task A0.6).

For ONE CD1L EMRI (catalogue row ``--src``) on a SHORT window, and for each
mode threshold, append one JSON line with:

* ``dlogl_tof`` / ``dlogl_leg``: SciRD ``logL(h) - logL(d)`` (``logL(d) = 0``),
  and ``dlogl_tof_minus_leg``;
* per channel X, Y, Z: ``1 - Re(O)`` for tof-vs-data, leg-vs-data and
  tof-vs-leg (flat, Tukey 0.1, NO time/phase maximisation);
* opt/data SNR for both templates, data SNR, number of TOF subs.

The legacy template is the production recipe (``get_emri_response_wrapper``),
i.e. the plan's base signal; the TOF is ``EMRITDIonFly(frame="icrs_special",
n_fine=N_FINE)``. Run one source per process, sequentially (laptop: one job at
a time), e.g.::

    for s in 0 1 2 3 4 5 6 7; do run.sh scripts/emri/emri_tof_injection_set.py --src $s; done

Env (shared with emri_tof_xyz_threeway.py): MOJITO_LIGHT_PATH (dir holding
catalogues/ and data/EMRI/L1/; the laptop cache has ONLY row 1, 5.7 GB per brick),
N_WIN (4096 at DT=20 s; 65536 = 15.2 d),
START_OFFSET_S (5e4), N_FINE (1024), THRESH (default "1e-3,1e-7": the 6mo
production EMRI_EPS and a converged reference).
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("THRESH", "1e-3,1e-7")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy.signal.windows import tukey  # noqa: E402

import emri_tof_xyz_threeway as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, required=True, help="catalogue ROW (0-based)")
    ap.add_argument("--out", default="emri_tof_injection_set.jsonl")
    args = ap.parse_args()

    W.watchdog(float(os.environ.get("WT_RSS_LIMIT_GB", "5")))
    params, data, data_t0, orb = W.load(args.src)
    win = tukey(W.N_WIN, 0.1)
    ff = np.fft.rfftfreq(W.N_WIN, d=W.DT)
    wg, offset_int, gen = W.legacy_wrapper(orb, data_t0)   # the ONE FEW construction
    for thr in W.THRESHES:
        t0 = time.perf_counter()
        tof, nsub, n_in = W.tof_td(params, orb, data_t0, thr, gen)
        t_tof = time.perf_counter() - t0
        t0 = time.perf_counter()
        leg = W.legacy_td(params, wg, offset_int, thr)
        t_leg = time.perf_counter() - t0
        dl_tof, r_tof = W.likelihood(tof, data, win)
        dl_leg, r_leg = W.likelihood(leg, data, win)
        row = dict(
            src=args.src, thresh=thr, n_win=W.N_WIN, dt=W.DT, start_offset_s=W.START_OFFSET_S,
            window_start_after_ref_s=data_t0 - W.REF, n_fine=W.N_FINE, tof_subs=nsub,
            tof_inside=n_in, dlogl_tof=dl_tof, dlogl_leg=dl_leg,
            dlogl_tof_minus_leg=dl_tof - dl_leg, opt_over_data_tof=r_tof, opt_over_data_leg=r_leg,
            wall_tof_s=t_tof, wall_leg_s=t_leg,
        )
        for tag, a, b in (("tof_data", tof, data), ("leg_data", leg, data), ("tof_leg", tof, leg)):
            row[f"mm_{tag}"] = [float(W.mismatch(a[c], b[c], win, ff)) for c in range(3)]
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
