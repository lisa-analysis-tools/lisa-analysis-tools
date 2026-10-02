#!/usr/bin/env python
"""Laptop gate for the SOBBH direct-to-WDM lookup scorer (docs/sobbh-wdm-lookup.md).

Grid: the production layer duration (3600 s) sampled at 20 s (Nf=180; Nyquist 25 mHz), NT
layers (1024 = 42.7 d, 4320 = 6 months), synthetic epoch 0.5 yr. For each catalogue-like
source: the lookup template vs the batched response's own dense TD->WDM (flat per-channel
mismatch 1 - Re(O) with NO maximisation, norm ratio), the noise-weighted mismatch and dlogL
(scirdv1, XYZ), the SNR; then wall times per batch of ``--rows`` rows for the lookup comp's
get_ll / fill and the chunked comp's get_ll on the same residual, plus the lnL agreement of
both comps against the exact container inner products.

Run (one Python process at a time on the laptop):
    .wtenv/run.sh scripts/sobbh/sobbh_lookup_gate.py --table $TABLE --nt 1024 --rows 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np

NF, DT = 180, 20.0
EDGE = 24

SOURCES = np.array(
    [
        # (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)
        [36.0, 29.0, 0.1, -0.2, 0.6e9, 4.5e-3, 0.3, 0.8, 1.1, 2.0, 0.4],
        [50.0, 40.0, 0.3, 0.3, 1.0e9, 8.0e-3, 2.2, 2.1, 0.4, 4.4, -0.7],
        [25.0, 20.0, -0.4, 0.0, 0.3e9, 1.2e-2, 4.0, 1.4, 2.5, 0.7, 1.1],
        [60.0, 55.0, 0.1, 0.2, 0.8e9, 1.5e-2, 1.1, 1.2, 0.7, 3.1, 0.2],
        [20.0, 10.0, 0.0, 0.5, 0.5e9, 6.0e-3, 5.5, 0.3, 1.8, 5.9, -1.2],
        [80.0, 75.0, 0.6, 0.6, 2.0e9, 1.8e-2, 0.9, 1.9, 0.1, 1.5, 0.9],
    ]
)


def mm_flat(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
    return 1.0 - ab / np.sqrt(aa * bb), np.sqrt(aa / bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table",
        default=os.environ.get(
            "SOBBH_LOOKUP_TABLE_PATH",
            "/Users/mkatz/Research/lisa_sprint_2026/"
            "wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5",
        ),
    )
    ap.add_argument("--nt", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--eval-dt", type=float, default=600.0)
    ap.add_argument("--out", default="sobbh_lookup_gate.jsonl")
    ap.add_argument("--no-chunked", action="store_true")
    args = ap.parse_args()

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.diagnostic import inner_product
    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations
    from lisatools.utils.constants import YRSID_SI

    nt = int(args.nt)
    nobs = NF * nt
    t0 = int(0.5 * YRSID_SI / DT) * DT
    ref = float(t0)
    orbits = EqualArmlengthOrbits(force_backend="cpu")
    tdi = TDIConfig("2nd generation", force_backend="cpu")
    wdm = WDMSettings(NF, nt, DT, t0=t0, min_freq=2e-3, max_freq=2.4e-2, force_backend="cpu")
    table = WDMLookupTable.from_file(args.table, force_backend="cpu")
    comp = SOBBHLookupComputations(
        wdm,
        ref,
        table,
        orbits=orbits,
        tdi_config=tdi,
        tdi_type="XYZ",
        n_grid=2048,
        buffer_time=5000.0,
        eval_dt=args.eval_dt,
        num_m_layers=2,
        interp="cubic",
        row_batch=args.rows,
        force_backend="cpu",
        d_d=0.0,
    )
    direct = comp.direct
    grid_t = np.arange(nobs) * DT + t0
    tds = TDSettings(nobs, DT, force_backend="cpu")
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1")

    print(
        f"grid Nf={NF} Nt={nt} dt={DT} layer_dt={wdm.layer_dt} s; table "
        f"{os.path.basename(args.table)}"
    )
    tag = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows_out = []
    h_tof = []
    for i, src in enumerate(SOURCES):
        t_a = time.perf_counter()
        out = direct.tof.build(src[None, :], float(grid_t[0]), float(grid_t[-1]))
        td = np.asarray(out.eval_tdi(grid_t))[0]
        truth = np.asarray(TDSignal(td, tds).transform(wdm).arr)
        t_tof = time.perf_counter() - t_a
        t_a = time.perf_counter()
        got = np.asarray(direct.dense(src[None, :])[0].arr)
        t_look = time.perf_counter() - t_a
        sl = slice(EDGE, nt - EDGE)
        mms = [mm_flat(got[c, :, sl], truth[c, :, sl]) for c in range(3)]
        ac = AnalysisContainer(WDMSignal(truth, wdm), sens)
        h_sig = WDMSignal(got, wdm)
        hh_t = float(np.real(inner_product(WDMSignal(truth, wdm), WDMSignal(truth, wdm), psd=sens)))
        hh_l = float(np.real(inner_product(h_sig, h_sig, psd=sens)))
        dh = float(np.real(ac.template_inner_product(h_sig)))
        # interior-only (EDGE-trimmed) noise-weighted mismatch: isolates the gap between the
        # flat interior `mm` (unweighted, interior-only) and the all-pixel `mm_w` (weighted,
        # including the dense transform's unreliable grid-end pixels).
        truth_int = truth.copy()
        truth_int[..., :EDGE] = 0
        truth_int[..., nt - EDGE :] = 0
        got_int = got.copy()
        got_int[..., :EDGE] = 0
        got_int[..., nt - EDGE :] = 0
        ac_int = AnalysisContainer(WDMSignal(truth_int, wdm), sens)
        h_sig_int = WDMSignal(got_int, wdm)
        hh_t_int = float(
            np.real(inner_product(WDMSignal(truth_int, wdm), WDMSignal(truth_int, wdm), psd=sens))
        )
        hh_l_int = float(np.real(inner_product(h_sig_int, h_sig_int, psd=sens)))
        dh_int = float(np.real(ac_int.template_inner_product(h_sig_int)))
        mm_w_int = 1.0 - dh_int / np.sqrt(hh_t_int * hh_l_int)
        row = dict(
            src=i,
            f_low=float(src[5]),
            snr=float(np.sqrt(hh_t)),
            mm=[m for m, _ in mms],
            ratio=[r for _, r in mms],
            mm_w=1.0 - dh / np.sqrt(hh_t * hh_l),
            mm_w_int=mm_w_int,
            dlogL=-0.5 * (hh_t + hh_l - 2.0 * dh),
            t_tof_dense_s=t_tof,
            t_lookup_s=t_look,
            stats=dict(direct.last_stats),
            tag=tag,
        )
        rows_out.append(row)
        h_tof.append(truth)
        print(
            f"src {i} f_low {src[5]:.4f} snr {row['snr']:.1f}  mm X/Y/Z "
            f"{row['mm'][0]:.2e}/{row['mm'][1]:.2e}/{row['mm'][2]:.2e}  ratio "
            f"{row['ratio'][0]:.5f}/{row['ratio'][1]:.5f}/{row['ratio'][2]:.5f}  mm_w "
            f"{row['mm_w']:.2e}  mm_w_int {row['mm_w_int']:.2e}  dlogL {row['dlogL']:.3e}  "
            f"tof-dense {t_tof:.1f}s lookup {t_look:.1f}s"
        )

    # ---- scoring timings: both comps, one residual, the same batch of rows --------------
    data = h_tof[0] + 0.5 * h_tof[1]
    ac = AnalysisContainer(WDMSignal(data.copy(), wdm), XYZ2SensitivityMatrix(wdm, model="scirdv1"))
    aca = AnalysisContainerArray([ac])
    batch = np.tile(SOURCES[0], (args.rows, 1))
    rng = np.random.default_rng(1)
    batch[:, 5] += rng.uniform(-0.3, 0.3, args.rows) * float(wdm.layer_df)
    batch[:, 6] = rng.uniform(0, 2 * np.pi, args.rows)
    idx = np.zeros(args.rows, dtype=np.int32)
    exact = []
    for r in batch:
        hb = np.asarray(
            TDSignal(
                np.asarray(
                    direct.tof.build(r[None, :], float(grid_t[0]), float(grid_t[-1])).eval_tdi(
                        grid_t
                    )
                )[0],
                tds,
            )
            .transform(wdm)
            .arr
        )
        hs = WDMSignal(hb, wdm)
        exact.append(
            float(np.real(ac.template_inner_product(hs)))
            - 0.5 * float(np.real(inner_product(hs, hs, psd=ac.sens_mat)))
        )
    exact = np.asarray(exact)
    timing = {}
    for label in ("lookup_get_ll_warm", "lookup_get_ll"):
        t_a = time.perf_counter()
        ll_look = np.asarray(comp.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx))
        timing[label] = time.perf_counter() - t_a
    buf = np.zeros(3 * int(wdm.Nf_active) * int(wdm.Nt_active))
    t_a = time.perf_counter()
    comp.fill_global_wdm(batch, buf, data_index=idx, factors=np.ones(args.rows))
    timing["lookup_fill"] = time.perf_counter() - t_a
    timing["lookup_vs_exact_max_abs"] = float(np.abs(ll_look - exact).max())
    if not args.no_chunked:
        from bbhx.sobbhcomps import SOBBHWDMComputations

        ch = SOBBHWDMComputations(
            wdm,
            t_ref=ref,
            Nt_sub=32,
            n_pad=4,
            N_sparse=256,
            N_cp_sig=0,
            N_cp_orbit=0,
            orbits=orbits,
            tdi_config="2nd generation",
            force_backend="cpu",
            d_d=0.0,
            tdi_type="XYZ",
        )
        for label in ("chunked_get_ll_warm", "chunked_get_ll"):
            t_a = time.perf_counter()
            ll_ch = np.asarray(
                ch.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx, m_band_half_width=3)
            )
            timing[label] = time.perf_counter() - t_a
        timing["chunked_vs_exact_max_abs"] = float(np.abs(ll_ch - exact).max())
    timing["rows"] = args.rows
    timing["nt"] = nt
    print("timing:", json.dumps(timing, indent=1))
    with open(args.out, "a") as fp:
        for row in rows_out:
            fp.write(json.dumps(dict(nt=nt, eval_dt=args.eval_dt, **row)) + "\n")
        fp.write(json.dumps(dict(timing=timing, tag=tag)) + "\n")


if __name__ == "__main__":
    sys.exit(main())
