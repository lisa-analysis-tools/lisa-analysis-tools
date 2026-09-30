"""Markdown tables from emri_cd1l_campaign.py's JSON lines (latest row per src/duration/thresh).

usage: python emri_cd1l_campaign_summary.py results.jsonl
"""
import json
import sys


def f(v, fmt):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "/".join(format(x, fmt) for x in v)
    return format(v, fmt)


def main(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[(r["duration"], r["thresh"], r["src"])] = r
    keys = sorted(rows)
    print("## Templates vs data (WDM, SciRD v1; logL = -1/2<d-h|d-h>, 0 is perfect)\n")
    print("| dur | thresh | src | snr_data | tmpl | logL | mm_data | snr_opt/data | flat amp X/Y/Z | wall s | RSS GB | error |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = rows[k]
        for t in ("prod", "tof", "direct"):
            if f"{t}_logL" not in r and f"{t}_error" not in r:
                continue
            print(f"| {r['duration']} | {r['thresh']:g} | {r['src']} | {r['snr_data']:.2f} | {t} | "
                  f"{f(r.get(f'{t}_logL'), '+.3f')} | {f(r.get(f'{t}_mm_data'), '.2e')} | "
                  f"{f(r.get(f'{t}_snr_ratio'), '.5f')} | {f(r.get(f'{t}_flat_amp_data'), '.5f')} | "
                  f"{f(r.get(f'{t}_wall_s'), '.0f')} | {f(r.get(f'{t}_peak_rss_gb'), '.1f')} | "
                  f"{r.get(f'{t}_error', '')[:60]} |")
    print("\n## Template vs template\n")
    print("| dur | thresh | src | pair | mm | dlogL | flat mm X/Y/Z | flat amp X/Y/Z |")
    print("|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = rows[k]
        for a, b in (("tof", "prod"), ("direct", "prod"), ("direct", "tof")):
            if f"mm_{a}_{b}" in r:
                print(f"| {r['duration']} | {r['thresh']:g} | {r['src']} | {a}-{b} | {r[f'mm_{a}_{b}']:.2e} | "
                      f"{r[f'dlogL_{a}_{b}']:+.3e} | {f(r[f'flat_mm_{a}_{b}'], '.1e')} | "
                      f"{f(r[f'flat_amp_{a}_{b}'], '.5f')} |")


if __name__ == "__main__":
    main(sys.argv[1])
