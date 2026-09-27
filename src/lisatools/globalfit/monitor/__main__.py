"""Build the monitor page, and optionally the snapshot tar, in one command.

    python -m lisatools.globalfit.monitor [--snapshot] RUN_DIR [OUT.html]

Two directions, two entry points:

* **here** -- a LIVE run directory -> page (and with ``--snapshot``, the
  tarball too). This is the cluster-side command.
* :mod:`lisatools.globalfit.monitor.from_tar` -- a downloaded tarball ->
  page. That one CONSUMES a tar; it does not make one.

``--snapshot`` gives you both artifacts from one invocation:

    python -m lisatools.globalfit.monitor --snapshot /path/to/gf_prod_run

writes ``<RUN_DIR>/gf_monitor.html`` and then
``<RUN_DIR>_snapshot.tar.gz``.

⚠ THE ORDER IS LOAD-BEARING, not incidental. The page is written INTO the
run directory first, so the tar that follows contains it -- ship the
tarball and the report travels with the data it describes. Build the tar
first and you get a snapshot of a run with no report in it, which is the
whole point of doing them together. ``tests/test_monitor_main_cli.py``
pins the ordering.

No environment is required: ``build_monitor`` resolves the mojito tree
itself (see :func:`lisatools.globalfit.monitor.resolve_mojito_path`),
including from the run's own settings.
"""

import sys
import time

from . import build_monitor, default_out_path
from .snapshot import build_snapshot

USAGE = ("usage: python -m lisatools.globalfit.monitor [--snapshot] "
         "RUN_DIR [OUT.html]\n"
         "       --snapshot  also build <RUN_DIR>_snapshot.tar.gz, which "
         "will CONTAIN the page\n"
         "       (a downloaded tarball goes the other way: python -m "
         "lisatools.globalfit.monitor.from_tar SNAP.tar.gz)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want_snapshot = "--snapshot" in argv
    if want_snapshot:
        argv.remove("--snapshot")
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2

    run_dir = argv[0]
    # Defaulting INTO the run directory is what lets --snapshot carry the
    # page; an explicit OUT elsewhere is honoured, and then the tar simply
    # will not contain it.
    out = argv[1] if len(argv) > 1 else default_out_path(run_dir)
    st = time.perf_counter()
    # ONE PYTHON (user 2026-09-26). Safe since the font-size pin made
    # in-process byte-identical to the child-interpreter render.
    build_monitor(run_dir, out, in_process=True)
    print(f"[monitor] wrote {out} in {time.perf_counter() - st:.1f} s")

    if want_snapshot:
        st = time.perf_counter()
        tar = build_snapshot(run_dir)
        if tar:
            print(f"[monitor] wrote {tar} in {time.perf_counter() - st:.1f} s")
        else:
            print("[monitor] snapshot FAILED (see warnings above); the page "
                  "was still written", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
