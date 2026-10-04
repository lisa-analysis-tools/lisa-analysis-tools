"""Summary tables + plots for a gb_speed_durations.sh output directory.

Usage: python gb_speed_plots.py <OUT dir>

Figures (PNG, in <OUT>):
  speed_scaling.png   -- us per scored row vs rows per call, one panel per duration
  speed_vs_days.png   -- us per row at the largest row count vs observation duration
  gate_accuracy.png   -- delta-vs-delta error vs true |dlnL| (tier line max(0.1, T/100))
  mojito_accuracy.png -- |dlogL vs production| and mm vs data per mojito source
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENGINES = ("chunked", "sighet_reim", "sighet_ampph", "lookup")
#: categorical slots 1-4 (reference palette, light), fixed order per engine
COLOR = {"chunked": "#2a78d6", "sighet_reim": "#eb6834", "sighet_ampph": "#1baf7a",
         "lookup": "#eda100"}
MARK = {"chunked": "o", "sighet_reim": "s", "sighet_ampph": "^", "lookup": "D"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip().startswith("{")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def speed(out, recs):
    if not recs:
        return
    days = sorted({r["days"] for r in recs})
    print("\n== SPEED (us per scored row, warm; sig-het setup per reference in brackets)")
    for d in days:
        rows = sorted({r["rows"] for r in recs if r["days"] == d})
        hdr = f"{'engine':>13} " + " ".join(f"{n:>11}" for n in rows)
        print(f"-- {d:.0f} d  ({recs[0]['backend']}, nf {recs[0]['nf']}, dt {recs[0]['dt']})\n{hdr}")
        for e in ENGINES:
            line = []
            for n in rows:
                rr = [r for r in recs if r["days"] == d and r["rows"] == n and r["engine"] == e]
                if rr:
                    s = f"{rr[0]['us_per_row']:.3g}"
                    if rr[0].get("setup_s_per_ref"):
                        s += f"[{1e3 * rr[0]['setup_s_per_ref']:.0f}ms]"
                    line.append(f"{s:>11}")
                else:
                    line.append(f"{'-':>11}")
            if any(x.strip() != "-" for x in line):
                print(f"{e:>13} " + " ".join(line))
        for e in ENGINES:
            rr = [r for r in recs if r["days"] == d and r["engine"] == e
                  and "max_abs_dll_vs_chunked" in r]
            if rr:
                print(f"   {e}: max |dll vs chunked| over rows = "
                      f"{max(r['max_abs_dll_vs_chunked'] for r in rr):.3g}")
    fig, axes = plt.subplots(1, len(days), figsize=(4.2 * len(days), 3.6), squeeze=False,
                             facecolor=SURFACE)
    for ax, d in zip(axes[0], days):
        for e in ENGINES:
            rr = sorted((r for r in recs if r["days"] == d and r["engine"] == e),
                        key=lambda r: r["rows"])
            if rr:
                ax.loglog([r["rows"] for r in rr], [r["us_per_row"] for r in rr], lw=2,
                          marker=MARK[e], ms=8, color=COLOR[e], label=e)
        ax.set_xlabel("rows per call")
        ax.set_ylabel("us per row (warm)")
        ax.set_title(f"{d:.0f} d", fontsize=10)
        _style(ax)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("GB scoring cost vs batch size", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speed_scaling.png"), dpi=150)
    plt.close(fig)
    if len(days) > 1:
        fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=SURFACE)
        for e in ENGINES:
            pts = []
            for d in days:
                rr = [r for r in recs if r["days"] == d and r["engine"] == e]
                if rr:
                    big = max(rr, key=lambda r: r["rows"])
                    pts.append((d, big["us_per_row"]))
            if pts:
                ax.loglog(*zip(*pts), lw=2, marker=MARK[e], ms=8, color=COLOR[e], label=e)
        ax.set_xlabel("observation duration [d]")
        ax.set_ylabel("us per row at the largest batch")
        ax.set_title("GB scoring cost vs duration", fontsize=10)
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "speed_vs_days.png"), dpi=150)
        plt.close(fig)


def gate(out, recs):
    deltas = [r for r in recs if r.get("kind") == "delta"]
    for r in recs:
        if r.get("kind") == "summary":
            print(f"\n== GATE {r['days']:.0f} d (SNR {r['rho']:.0f}): tier pass fraction / "
                  f"max eps / max eps/T / max |anchor|")
            for e, s in r["engines"].items():
                print(f"{e:>13}: {s['tier_pass_frac']:6.1%}  {s['max_eps']:9.3g}  "
                      f"{s['max_eps_over_T']:9.3g}  {s['max_abs_anchor']:9.3g}")
    tmpl = [r for r in recs if r.get("kind") == "template"]
    if tmpl:
        print("\n== GATE template level vs dense truth: worst mm / worst |ratio-1|")
        for e in ("chunked", "lookup"):
            rr = [r[e] for r in tmpl if e in r]
            if rr:
                print(f"{e:>13}: {max(x['mm'] for x in rr):.3g}  "
                      f"{max(abs(x['ratio'] - 1) for x in rr):.3g}")
    if not deltas:
        return
    days = sorted({r["days"] for r in deltas})
    fig, axes = plt.subplots(1, len(days), figsize=(4.2 * len(days), 3.8), squeeze=False,
                             facecolor=SURFACE)
    for ax, d in zip(axes[0], days):
        rr = [r for r in deltas if r["days"] == d]
        T = np.array([r["T"] for r in rr])
        for e in ENGINES:
            if e in rr[0]:
                eps = np.array([max(r[e]["eps"], 1e-12) for r in rr])
                ax.loglog(T, eps, ls="none", marker=MARK[e], ms=6, color=COLOR[e], label=e,
                          mec=SURFACE, mew=1.0)
        tt = np.logspace(np.log10(max(T.min(), 1e-3)), np.log10(T.max() * 1.5), 50)
        ax.loglog(tt, np.maximum(0.1, tt / 100), color=INK2, lw=1, ls="--", label="tier bar")
        ax.set_xlabel("true |dlnL| (T)")
        ax.set_ylabel("|D_engine - D_truth|")
        ax.set_title(f"{d:.0f} d", fontsize=10)
        _style(ax)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("GB delta-vs-delta accuracy vs dense truth", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gate_accuracy.png"), dpi=150)
    plt.close(fig)


def mojito(out, recs):
    if not recs:
        return
    print("\n== MOJITO: per source -- data SNR | mm vs data per engine | dlogL vs production")
    for r in recs:
        mm = " ".join(f"{e}={r[e]['data']['mm']:.2e}" for e in ENGINES if e in r)
        dl = " ".join(f"{e}={r[e]['dlogL_vs_production']:+.1e}" for e in ENGINES
                      if e in r and "dlogL_vs_production" in r[e])
        print(f"{r['source']:>3} {r['src']:>10} {r['days']:5.0f} d f0 {1e3 * r['f0']:7.4f} mHz "
              f"SNR {r['data_snr']:7.2f} nb {r['neighbours_subtracted']} | {mm} | {dl}")
    labels = [f"{r['source']} {r['src']} {r['days']:.0f}d" for r in recs]
    x = np.arange(len(recs))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(6, 0.6 * len(recs)), 6.4), sharex=True,
                                 facecolor=SURFACE)
    w = 0.8 / len(ENGINES)
    for k, e in enumerate(ENGINES):
        if e not in recs[0]:
            continue
        mm = [max(r[e]["data"]["mm"], 1e-14) for r in recs]
        a1.bar(x + (k - 1.5) * w, mm, width=w * 0.9, color=COLOR[e], label=e, log=True)
        if e != "chunked":
            dl = [max(abs(r[e].get("dlogL_vs_production", 0.0)), 1e-12) for r in recs]
            a2.bar(x + (k - 1.5) * w, dl, width=w * 0.9, color=COLOR[e], label=e, log=True)
    a1.set_ylabel("mismatch vs data")
    a2.set_ylabel("|dlogL vs chunked|")
    a2.set_xticks(x)
    a2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    for ax in (a1, a2):
        _style(ax)
    a1.legend(fontsize=8, frameon=False, ncol=4)
    fig.suptitle("Mojito GB / VGB: engine templates vs the data", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mojito_accuracy.png"), dpi=150)
    plt.close(fig)


def main():
    out = sys.argv[1]
    speed(out, _load(os.path.join(out, "speed.jsonl")))
    gate(out, _load(os.path.join(out, "gate.jsonl")))
    mojito(out, _load(os.path.join(out, "mojito.jsonl")))
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
