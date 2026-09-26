"""``python -m lisatools.globalfit.monitor [--snapshot] RUN_DIR [OUT.html]``

The drop-in for ``python scripts/diagnostics/gf_monitor_gen.py RUN_DIR
OUT.html``. Positional and thin on purpose: every runbook that exists
already uses those two arguments.

``--snapshot`` also builds ``<RUN_DIR>_snapshot.tar.gz``, which is what
the saver rank asks for.

Note this spawns a child interpreter to do the render -- see the package
docstring; it is what makes the page identical to the old invocation,
not an implementation detail to optimise away.
"""

import sys
import time

from . import build_monitor, default_out_path
from .snapshot import build_snapshot


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want_snapshot = "--snapshot" in argv
    if want_snapshot:
        argv.remove("--snapshot")
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m lisatools.globalfit.monitor "
              "[--snapshot] RUN_DIR [OUT.html]")
        return 0 if argv else 2

    run_dir = argv[0]
    out = argv[1] if len(argv) > 1 else default_out_path(run_dir)
    st = time.perf_counter()
    build_monitor(run_dir, out)
    print(f"[monitor] wrote {out} in {time.perf_counter() - st:.1f} s")
    if want_snapshot:
        st = time.perf_counter()
        tar = build_snapshot(run_dir)
        print(f"[monitor] wrote {tar} in {time.perf_counter() - st:.1f} s"
              if tar else "[monitor] snapshot FAILED (see warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
