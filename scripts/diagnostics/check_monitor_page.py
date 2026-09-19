#!/usr/bin/env python
"""What is and is not in a generated monitor page.

    python scripts/diagnostics/check_monitor_page.py PAGE.html [PAGE2.html ...]

"Some plots are missing" has three different causes and the page records
each one differently. This tells them apart:

  EMBEDDED     ``<img src="data:image/png;base64,...">`` -- a rendered panel.
               Plots are NEVER written to disk; ``fig_b64()`` inlines them,
               which is why a page is 6-7 MB and works offline.
  GALLERIES    JS payloads swapped by a dropdown (the VGB corner set, the
               per-source views) -- NOT ``<img>`` tags, so a naive ``<img>``
               grep undercounts by all of them.
  PLACEHOLDER  ``img()`` fell back because the panel's key never reached
               ``IMGS``: ``fig_b64()`` did not run because its INPUT was not
               in the store. Expected for a snapshot tar -- the F-stat
               comb/peaks caches are not in the archive, only ``DONE.json``.
  MISSING      a whole section skipped WITH A STATED REASON. These are the
               ones worth reading; everything else is bookkeeping.

Reference, from pages built off the snapshot tars on 2026-09-18:

    3-month it 210 : 26 embedded, 3 placeholders, missing=3
    6-month it  47 : 26 embedded, 3 placeholders, missing=3

Materially fewer embedded panels than that means the generator lost data it
normally has -- read the MISSING reasons first, they usually say why.
"""

from __future__ import annotations

import re
import sys


def report(path: str) -> None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            s = f.read()
    except OSError as exc:
        print(f"could not read {path}: {exc}")
        return

    print(f"=== {path}")
    print(f"    size         : {len(s) // 1024} KB")

    embedded = len(re.findall(r'<img src="data:image/png;base64,', s))
    print(f"    embedded     : {embedded} panels")

    # Gallery payloads: each entry carries a "label" and a "png". Counting the
    # pairs avoids the other "label" keys elsewhere in the page.
    gal = len(re.findall(r'"label":\s*"[^"]*",\s*"png":', s))
    if not gal:                      # key order is not guaranteed
        gal = len(re.findall(r'"png":\s*"', s))
    print(f"    gallery slides: {gal} (JS payload, not <img>)")

    ph = re.findall(r"plot unavailable in this snapshot: ([^<]*)", s)
    print(f"    placeholders : {len(ph)}")
    for p in ph:
        print(f"        - {p.strip()}")

    # MISSING entries render as <li> items; take the ones that read like the
    # generator's own reasons rather than page furniture.
    lis = re.findall(r"<li>(.*?)</li>", s, flags=re.S)
    reasons = [re.sub(r"<[^>]+>", "", x).strip() for x in lis]
    reasons = [r for r in reasons
               if len(r) > 60 and ("skipped" in r or "unavailable" in r
                                   or "no " in r or "not " in r)]
    print(f"    MISSING      : {len(reasons)}")
    for r in reasons:
        r = " ".join(r.split())
        print(f"        - {r[:160]}{'...' if len(r) > 160 else ''}")
    print()


def main(argv=None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__)
        return 1
    for p in args:
        report(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
