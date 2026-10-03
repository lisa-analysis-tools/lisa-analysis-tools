"""Markdown tables from mbh_cd1l_campaign.py's JSON lines (latest row per window/src).

usage: python mbh_cd1l_campaign_summary.py results.jsonl
"""
import json
import sys

TEMPLATES = ("prod", "prod90", "batched", "tof")
PAIRS = (("batched", "prod90"), ("batched", "prod"), ("prod90", "prod"),
         ("tof", "prod"), ("tof", "prod90"))
BANDS = ("<1", "1-5", "5-15", "15-25")


def f(v, fmt):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "/".join(format(x, fmt) for x in v)
    if isinstance(v, dict):
        return "/".join(format(v[b], fmt) if b in v else "-" for b in BANDS)
    return format(v, fmt)


def main(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[(r["window"], r["src"])] = r
    keys = sorted(rows)
    ok = [k for k in keys if rows[k].get("status") == "ok"]
    # rows before 2026-10-02 carry no noise record: they were SciRD v1 XYZ alone
    noise = sorted({rows[k].get("noise", "scirdv1 XYZ, no galactic foreground (unrecorded)") for k in ok})
    print("Noise weighting: " + ("; ".join(noise) or "-") + "\n")

    print("## Windows (data; band SNR over WDM layers <1 / 1-5 / 5-15 / 15-25 mHz)\n")
    print("| window | src | merger d in window | snr_data | data band SNR | batched kept box d "
          "| merger layer | clamp lo/hi | outside box/data | prod90 power outside box | load s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ok:
        r = rows[k]
        g = r.get("batched_geometry", {})
        kb = g.get("kept_box_days_in_window", [0.0, 0.0])
        geo = (f"{kb[0]:.2f}..{kb[1]:.2f} | {g['merger_layer']:.1f} | "
               f"{int(g['box_clamped_lo'])}/{int(g['box_clamped_hi'])} | "
               f"{int(g['outside_box'])}/{int(g['outside_data'])}") if "n_start" in g else \
            f"{g.get('error', '-')[:40]} | - | - | -"
        print(f"| {r['window']} | {r['src']} | {r['placement']['merger_days_in_window']:.2f} | "
              f"{r['snr_data']:.2f} | {f(r.get('band_snr_data'), '.3g')} | {geo} | "
              f"{f(r.get('prod90_power_outside_box'), '.1e')} | {f(r.get('load_wall_s'), '.0f')} |")

    print("\n## Templates vs data (WDM, SciRD v1 XYZ; logL = -1/2<d-h|d-h>, 0 is perfect)\n")
    print("| window | src | snr_data | tmpl | logL | mm_data | snr_opt/data | flat amp X/Y/Z "
          "| resid SNR <1/1-5/5-15/15-25 mHz | wall s | RSS GB | error |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ok:
        r = rows[k]
        for t in TEMPLATES:
            if f"{t}_logL" not in r and f"{t}_error" not in r:
                continue
            print(f"| {r['window']} | {r['src']} | {r['snr_data']:.2f} | {t} | "
                  f"{f(r.get(f'{t}_logL'), '+.4f')} | {f(r.get(f'{t}_mm_data'), '.2e')} | "
                  f"{f(r.get(f'{t}_snr_ratio'), '.6f')} | {f(r.get(f'{t}_flat_amp_data'), '.6f')} | "
                  f"{f(r.get(f'{t}_band_resid_snr'), '.3g')} | "
                  f"{f(r.get(f'{t}_wall_s'), '.0f')} | {f(r.get(f'{t}_peak_rss_gb'), '.2f')} | "
                  f"{r.get(f'{t}_error', '')[:60]} |")

    print("\n## Template vs template (noise-weighted mm, dlogL = logL_a - logL_b, ||a-b|| noise weighted)\n")
    print("| window | src | pair | mm | dlogL | norm(a-b) | flat mm X/Y/Z | flat amp X/Y/Z |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ok:
        r = rows[k]
        for a, b in PAIRS:
            if f"mm_{a}_{b}" in r:
                print(f"| {r['window']} | {r['src']} | {a}-{b} | {r[f'mm_{a}_{b}']:.2e} | "
                      f"{r[f'dlogL_{a}_{b}']:+.3e} | {r[f'delta_norm_{a}_{b}']:.3e} | "
                      f"{f(r[f'flat_mm_{a}_{b}'], '.1e')} | {f(r[f'flat_amp_{a}_{b}'], '.6f')} |")

    skipped = [k for k in keys if rows[k].get("status") != "ok"]
    print("\n## Skipped (production admission: window start <= t_merge < window end + buffer)\n")
    print("| window | src | status | merger d in window | merger d after window end | window [d in file] |")
    print("|---|---|---|---|---|---|")
    for k in skipped:
        r = rows[k]
        p = r["placement"]
        lo = p["window_start_days_after_file_start"]
        print(f"| {r['window']} | {r['src']} | {r['status']} | {p['merger_days_in_window']:.2f} | "
              f"{p['merger_days_after_window_end']:.2f} | {lo:.2f}..{lo + p['window_days']:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1])
