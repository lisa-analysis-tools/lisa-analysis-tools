"""Markdown tables from agent5_error_budget.py JSON outputs.

    python agent5_summarize.py agent5_out/res_2d.json agent5_out/res_30d.json ...
"""

import json
import sys


def fmt(x):
    return "nan" if x != x else f"{x:.1e}"


def main(paths):
    for p in paths:
        for res in json.load(open(p)):
            print(f"\n#### {res['config']} ({res['backend']}, nf={res['nf']}, N={res['N']})\n")
            notes = []
            if "R_mirror_mismatch" in res:
                notes.append(f"S_gal/R mirror vs compiled kernel: {res['R_mirror_mismatch']:.1e}")
            if "moment0_helper_vs_legacy_band" in res:
                notes.append(f"moment-0 helper vs legacy conv: {res['moment0_helper_vs_legacy_band']:.1e}")
            if notes:
                print("; ".join(notes) + "\n")
            print("| family | variant | entry med | entry p99 | entry max (f) | white med | white p99 |"
                  " white max (f) | dll med | dll max | max dll f<1e-4 / 1e-4..1e-3 / >1e-3 |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in res["rows"]:
                print(f"| {r['family']} | {r['variant']} | {fmt(r['median'])} | {fmt(r['p99'])} |"
                      f" {fmt(r['max'])} ({r['f_at_max']:.2e}) | {fmt(r['w_median'])} | {fmt(r['w_p99'])} |"
                      f" {fmt(r['w_max'])} ({r['w_f_at_max']:.2e}) | {fmt(r['dll_median'])} |"
                      f" {fmt(r['dll_max'])} | {' / '.join(fmt(x) for x in r.get('dll_band_max', [float('nan')] * 3))} |")


if __name__ == "__main__":
    main(sys.argv[1:])
