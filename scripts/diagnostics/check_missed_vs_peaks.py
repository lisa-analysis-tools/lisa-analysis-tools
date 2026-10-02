#!/usr/bin/env python
"""Did the F-stat refit SEE the detectable sources the 6mo search is missing?

Run ON THE CLUSTER next to a refit's ``fstat_grid_peaks_stacked.npz`` (the
file is too large to move; this reads only its two small members). The 350
targets are embedded: every catalogue GB with SNR > 7 (frozen set, iteration-12
noise) at 3-5.5 mHz that fewer than two cold walkers held at stored row 63 of
the 6mo v9 store (job 675, 2026-10-02). For each target it finds the nearest
peak of the refit and reports whether one sits within 1 / 2 / 4 frequency bins,
split by band, by SNR and by whether one walker already held the source.

    python check_missed_vs_peaks.py \\
        /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gb_fstat_fit/shared_search/epoch_0034/fstat_grid_peaks_stacked.npz \\
        [more epoch files ...] [--truth gb_truth_3to21.npz]

Reading:
  * most targets WITH a peak within 2 bins -> the grid saw them; the loss is in
    the birth draw / acceptance (candidate flood, weights, mismatch);
  * most targets WITHOUT a peak -> the peak floor / grid resolution lose them
    before any birth is proposed.
With ``--truth`` it also reports the peak list's purity in 3-5.5 mHz: the
fraction of peaks within 2 bins of ANY SNR > 7 catalogue source.
Output is a short printout plus ``missed_vs_peaks_<epoch>.csv`` (350 rows).
"""
import os
import sys

import numpy as np

TOBS = 15552000.0                      # 6-month window [s]
BIN_MHZ = 1e3 / TOBS                   # one FD bin in mHz (6.43e-5)

# f0 [mHz], SNR (iteration-12 noise), walkers holding it at row 63 (0 or 1)
TARGETS = """
3.0078813 7.44 0;3.0139885 7.22 0;3.0199807 7.29 0;3.0259674 7.19 0;3.0276246 8.82 0;3.0349296 7.74 0;3.0433736 8.76 0;3.0445358 12.51 1;3.0495511 7.64 1;3.0495867 8.13 1;3.0518528 8.50 0;3.0624825 8.04 0;3.0799334 8.62 0;3.0824364 7.34 0;3.0868716 7.15 0;3.1073733 7.22 1;3.1075910 7.94 1;3.1117121 7.44 0;3.1150966 7.50 0;3.1182658 7.72 1;3.1202237 8.90 1;3.1432580 8.72 0;3.1492727 8.78 1;3.1526009 7.99 0;3.1611396 9.62 0;3.1773936 7.14 0;3.1793369 7.12 0;3.1847677 7.64 0;3.1877286 9.87 1;3.1992419 7.69 1;3.2058225 7.80 0;3.2061565 7.88 0;3.2093507 11.12 0;3.2248957 7.72 0;3.2312286 9.09 0;3.2326430 7.56 0;3.2347946 9.09 0;3.2458715 10.53 1;3.2476141 8.11 0;3.2606079 8.20 0;3.2765788 8.07 1;3.2918406 8.08 0;3.3054684 9.72 1;3.3095147 7.86 0;3.3124232 12.59 1;3.3155713 7.10 1;3.3170143 7.31 0;3.3208691 9.05 1;3.3245844 7.61 1;3.3273401 9.02 0;3.3296220 7.78 0;3.3301802 7.44 1;3.3325930 9.11 1;3.3338757 9.24 1;3.3380846 7.58 0;3.3390906 9.37 0;3.3395896 7.20 0;3.3445965 7.52 1;3.3512155 7.35 1;3.3560054 8.17 1;3.3802821 7.22 1;3.3938763 7.71 1;3.3958864 9.54 1;3.4001017 7.45 1;3.4008033 7.51 0;3.4016593 7.12 0;3.4157343 11.76 1;3.4187037 7.49 1;3.4424102 7.48 0;3.4438698 7.75 1;3.4447825 7.28 1;3.4468095 8.87 1;3.4505889 7.94 0;3.4535812 14.79 1;3.4591333 7.58 0;3.4592330 7.65 0;3.4630955 9.39 0;3.4661021 13.45 1;3.4692815 7.35 1;3.4728055 7.80 0;3.4751401 9.63 1;3.4841018 7.65 1;3.4922610 8.36 1;3.5077463 9.00 1;3.5131690 7.28 1;3.5142456 10.05 0;3.5178587 9.01 0;3.5200457 7.26 0;3.5234084 9.62 0;3.5235416 14.95 0;3.5327322 7.82 0;3.5360220 7.49 1;3.5376421 7.96 0;3.5400623 8.04 1;3.5473528 7.69 0;3.5528377 9.74 0;3.5541368 9.06 1;3.5574720 7.18 1;3.5627447 8.25 1;3.5671275 9.47 1;3.5785307 7.91 0;3.5790909 8.50 1;3.5806889 9.62 1;3.5830017 9.67 0;3.5842878 8.95 0;3.5858128 8.46 0;3.5909868 9.46 0;3.6061713 8.75 1;3.6104382 11.35 0;3.6134804 9.67 0;3.6134978 10.87 0;3.6179974 7.19 0;3.6214514 7.81 0;3.6315828 10.71 1;3.6322268 9.28 1;3.6361284 8.78 0;3.6421756 9.22 1;3.6455734 7.83 0;3.6460946 7.55 1;3.6480341 9.80 1;3.6515421 8.32 0;3.6545608 7.87 0;3.6659980 7.23 0;3.6741360 7.54 1;3.6819393 7.74 0;3.6850379 7.11 0;3.6948365 8.64 0;3.6961404 7.69 1;3.7071573 7.68 0;3.7260547 7.20 1;3.7365196 7.48 0;3.7396896 9.42 0;3.7397691 8.90 0;3.7405972 7.31 0;3.7408948 7.92 1;3.7438085 7.80 1;3.7448547 7.03 0;3.7453094 8.50 0;3.7521780 7.63 0;3.7533539 9.01 0;3.7543059 8.47 0;3.7561217 7.33 1;3.7600315 8.23 1;3.7635222 9.73 1;3.7637112 7.69 1;3.7638492 7.05 0;3.7764648 7.76 0;3.7818720 7.34 0;3.7897070 9.90 1;3.7976816 8.54 0;3.7998738 9.24 0;3.8002846 7.58 0;3.8102662 7.21 1;3.8187760 7.88 0;3.8235924 11.06 0;3.8247050 8.85 0;3.8329974 9.46 1;3.8346344 7.16 0;3.8347280 8.28 0;3.8356406 8.90 1;3.8455184 8.20 1;3.8460855 7.44 1;3.8506113 8.52 0;3.8510660 8.06 0;3.8520008 9.12 0;3.8538434 10.84 0;3.8539516 11.65 0;3.8545959 8.28 0;3.8612249 7.30 0;3.8644995 7.43 0;3.8669912 7.64 1;3.8712520 7.45 0;3.8744456 8.07 0;3.8759348 7.54 0;3.8761511 7.66 1;3.8764054 8.61 0;3.8802352 7.82 0;3.8842906 9.95 1;3.8869096 7.53 1;3.8882508 8.23 0;3.8900464 7.75 0;3.8944719 7.13 0;3.9056861 7.20 0;3.9063861 8.97 0;3.9073674 13.06 1;3.9123876 8.46 0;3.9125985 8.12 1;3.9144669 8.99 0;3.9157782 7.96 0;3.9198935 8.33 0;3.9227300 8.71 0;3.9233674 8.19 0;3.9282900 7.23 0;3.9287573 9.70 0;3.9384200 8.92 0;3.9568509 7.20 0;3.9744995 8.20 1;3.9773554 11.03 0;3.9813405 7.46 1;3.9826043 9.36 0;3.9861802 7.77 0;4.0118688 7.59 0;4.0166269 8.07 0;4.0387996 10.62 0;4.0390867 14.09 1;4.0391376 15.94 0;4.0436863 8.18 1;4.0458454 7.13 1;4.0522039 9.25 0;4.0625629 10.11 0;4.0665728 7.30 0;4.0773125 8.36 0;4.0800610 7.45 0;4.0824034 7.09 1;4.0833484 7.99 0;4.0952079 7.03 1;4.1017035 11.59 0;4.1033543 9.76 1;4.1039643 8.72 1;4.1070079 7.65 0;4.1226621 10.29 1;4.1375632 7.50 0;4.1406550 10.28 0;4.1432558 7.40 1;4.1433756 7.93 0;4.1445972 8.59 0;4.1462571 8.32 0;4.1468028 7.05 1;4.1472557 10.50 1;4.1624029 8.11 0;4.1738477 7.30 0;4.1748261 8.82 0;4.1858611 7.64 0;4.1872483 10.49 0;4.1892190 9.34 1;4.1910500 7.39 0;4.1915291 7.18 1;4.1944517 8.22 0;4.2055441 7.74 0;4.2085434 7.28 0;4.2255052 9.21 0;4.2261615 7.23 0;4.2347342 7.57 1;4.2356094 9.81 0;4.2467264 7.47 0;4.2515601 9.55 0;4.2547575 12.90 1;4.2555652 7.59 0;4.2653764 8.98 0;4.2704864 8.79 0;4.2818770 7.39 0;4.2832798 7.45 0;4.2839119 7.36 1;4.2849947 8.66 0;4.2881176 8.05 0;4.2901134 7.20 0;4.2921617 8.61 0;4.3028074 8.50 1;4.3078185 7.23 0;4.3104747 7.47 0;4.3270510 9.50 0;4.3274419 7.09 0;4.3290914 8.79 0;4.3341151 7.39 1;4.3451414 8.88 1;4.3540540 7.61 0;4.3577421 7.24 0;4.3579906 9.47 0;4.3711921 7.78 0;4.3768214 9.18 0;4.3853209 8.27 0;4.3855454 7.69 0;4.3877541 8.62 0;4.4055745 8.19 0;4.4127928 9.09 0;4.4328968 7.91 0;4.4351986 9.82 0;4.4432210 11.86 0;4.4465546 8.63 1;4.4489243 7.90 1;4.4509005 9.11 0;4.4555279 11.16 1;4.4635628 9.68 1;4.4681395 7.98 0;4.4784524 9.21 0;4.4915460 7.81 0;4.5099903 7.93 1;4.5141403 10.33 1;4.5237203 7.19 0;4.5312790 7.61 1;4.5322514 7.68 0;4.5355534 7.27 0;4.5445448 7.09 0;4.5588271 8.82 1;4.5660645 9.26 1;4.5782991 8.59 0;4.5897676 7.54 1;4.5998461 15.12 1;4.6616432 7.63 1;4.6664244 13.42 1;4.6777206 8.42 0;4.6970367 7.90 1;4.6989569 9.76 0;4.7007391 8.10 0;4.7272224 9.11 1;4.7281028 11.04 1;4.7346889 7.05 0;4.7391910 7.97 1;4.7552910 8.00 1;4.7942845 7.59 0;4.8010724 7.58 0;4.8077679 7.68 1;4.8205150 8.62 0;4.8322698 7.05 0;4.8497764 8.16 0;4.8498092 7.51 0;4.8873600 7.87 0;4.8889151 8.44 1;4.8902145 13.48 1;4.8940817 13.29 1;4.8957206 7.31 0;4.9008340 12.40 0;4.9013549 9.04 1;4.9133844 7.88 0;4.9319402 11.29 0;4.9360007 7.70 1;4.9383339 7.01 1;4.9628608 8.01 0;4.9874476 7.85 0;5.0113940 8.21 0;5.0327658 10.38 1;5.0846511 8.69 1;5.0868171 8.10 0;5.1319328 11.25 0;5.1518061 7.82 0;5.1558500 12.06 1;5.1890686 7.80 1;5.1928911 8.14 1;5.1939322 7.86 1;5.2042228 7.33 0;5.2273545 7.05 1;5.2282480 12.54 0;5.2386835 7.46 1;5.2613172 8.87 1;5.2707101 9.10 0;5.2766613 9.11 1;5.3959466 7.81 1;5.4050282 7.09 0;5.4144065 8.31 1;5.4565180 7.75 0
"""


def _targets():
    rows = [r.split() for r in TARGETS.replace("\n", "").split(";") if r.strip()]
    a = np.array([[float(x) for x in r] for r in rows])
    return a[:, 0], a[:, 1], a[:, 2].astype(int)


def _nearest(peaks_f0, f0):
    ps = np.sort(peaks_f0)
    j = np.searchsorted(ps, f0)
    near = np.full(f0.shape, np.inf)
    idx = np.zeros(f0.shape, dtype=int)
    for off in (-1, 0):
        jj = np.clip(j + off, 0, ps.size - 1)
        d = np.abs(ps[jj] - f0)
        m = d < near
        near[m] = d[m]
        idx[m] = jj[m]
    return near, ps[idx]


def main(argv):
    truth = None
    paths = []
    it = iter(argv)
    for a in it:
        if a == "--truth":
            truth = next(it)
        else:
            paths.append(a)
    if not paths:
        print(__doc__)
        return 2
    tf0, tsnr, held = _targets()
    print(f"{tf0.size} targets at 3-5.5 mHz: {int((held == 0).sum())} held by no walker, "
          f"{int((held == 1).sum())} by one; one FD bin = {BIN_MHZ:.3e} mHz")
    any_truth = None
    if truth is not None:
        T = np.load(truth)
        f_all = np.asarray(T["f0"], float).reshape(-1)
        f_all = f_all * 1e3 if f_all.max() < 1.0 else f_all
        s_all = np.asarray(T["snr"], float).reshape(-1)
        any_truth = np.sort(f_all[s_all > 7.0])
    for path in paths:
        d = np.load(path, allow_pickle=False)           # members load lazily
        pf = np.asarray(d["peak_f0_mHz"], float).reshape(-1)
        pF = np.asarray(d["peak_F"], float).reshape(-1)
        order = np.argsort(pf)
        pf, pF = pf[order], pF[order]
        tag = os.path.basename(os.path.dirname(path)) or path
        near_mhz, near_f = _nearest(pf, tf0)
        # F of the nearest peak (same ordering as pf)
        jn = np.searchsorted(pf, near_f)
        jn = np.clip(jn, 0, pf.size - 1)
        Fn = pF[jn]
        bins = near_mhz / BIN_MHZ
        # The peak list at 3-4.5 mHz is DENSE (tens per 17 uHz band), so an f0
        # coincidence alone proves little: report the CHANCE rate (the fraction
        # of bins in the group that lie within the window of some peak) next to
        # the hit rate, and a STRONG match that also asks the peak's F to be at
        # least the target's own expected F under the floor's noise. Expected
        # F = SNR^2 / 2 with the iteration-12 SNR; under the nudged noise a
        # 3.5-4.5 mHz source scores ~1.4-1.9x that, so F_ratio >= 1 is the
        # conservative test for a nudged refit and F_ratio >= 0.5 for a
        # released one.
        Fexp = tsnr ** 2 / 2.0
        Fratio = Fn / Fexp
        print(f"\n=== {tag}: {pf.size} peaks total, "
              f"{int(((pf >= 3.0) & (pf < 5.5)).sum())} in 3-5.5 mHz "
              f"(F floor in file: {pF.min():.1f}) ===")
        print("group           n | within 1 bin: hit%  chance% | within 2 bins: hit%  chance% | "
              "strong (<=1 bin & F>=Fexp)%  (F>=0.5Fexp)% | median F_near/Fexp")
        groups = [("3.0-3.5", 3.0, 3.5), ("3.5-4.0", 3.5, 4.0), ("4.0-4.5", 4.0, 4.5),
                  ("4.5-5.0", 4.5, 5.0), ("5.0-5.5", 5.0, 5.5), ("ALL 3-5.5", 3.0, 5.5)]

        def _chance(lo, hi, w_bins):
            """fraction of FD bins in [lo, hi) within w_bins of some peak"""
            sel = pf[(pf >= lo - 1e-3) & (pf < hi + 1e-3)]
            if sel.size == 0:
                return 0.0
            grid = np.arange(lo, hi, BIN_MHZ)
            nm, _ = _nearest(sel, grid)
            return float(np.mean(nm / BIN_MHZ <= w_bins))

        for name, lo, hi in groups:
            m = (tf0 >= lo) & (tf0 < hi)
            if not m.any():
                continue
            s1 = m & (bins <= 1.0) & (Fratio >= 1.0)
            s5 = m & (bins <= 1.0) & (Fratio >= 0.5)
            print(f"{name:12s} {m.sum():4d} | {100 * np.mean(bins[m] <= 1):15.0f}% {100 * _chance(lo, hi, 1.0):7.0f}% "
                  f"| {100 * np.mean(bins[m] <= 2):15.0f}% {100 * _chance(lo, hi, 2.0):7.0f}% "
                  f"| {100 * s1.sum() / m.sum():22.0f}% {100 * s5.sum() / m.sum():14.0f}% "
                  f"| {np.median(Fratio[m & (bins <= 1.0)]) if np.any(m & (bins <= 1.0)) else float('nan'):8.2f}")
        for name, m in (("held by NO walker", held == 0), ("held by ONE walker", held == 1),
                        ("SNR 7-8", tsnr < 8), ("SNR 8-10", (tsnr >= 8) & (tsnr < 10)), ("SNR > 10", tsnr >= 10)):
            print(f"  {name:20s} n={m.sum():4d}: within 1 bin {100 * np.mean(bins[m] <= 1):4.0f}%, "
                  f"strong(F>=Fexp) {100 * np.mean((bins[m] <= 1) & (Fratio[m] >= 1)):4.0f}%, "
                  f"strong(F>=0.5Fexp) {100 * np.mean((bins[m] <= 1) & (Fratio[m] >= 0.5)):4.0f}%")
        if any_truth is not None:
            sel = (pf >= 3.0) & (pf < 5.5)
            nm, _ = _nearest(any_truth, pf[sel])
            pur = np.mean(nm / BIN_MHZ <= 2.0)
            print(f"  purity: {100 * pur:.0f}% of the {int(sel.sum())} peaks in 3-5.5 mHz lie within 2 bins "
                  f"of SOME SNR > 7 catalogue source (resolved or not)")
        out = f"missed_vs_peaks_{tag}.csv"
        np.savetxt(out, np.column_stack([tf0, tsnr, held, bins, Fn]), delimiter=",",
                   header="f0_mHz,snr_it12,walkers_holding,nearest_peak_bins,nearest_peak_F", fmt="%.7f,%.2f,%d,%.2f,%.2f")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
