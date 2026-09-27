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

BOTH ARTIFACTS LAND BESIDE THE RUN FOLDER (user ruling 2026-09-26), not
inside it::

    /data/gf_prod_6mo_v9_4gpu/                    <- the run
    /data/gf_prod_6mo_v9_4gpu_monitor.html        <- the page
    /data/gf_prod_6mo_v9_4gpu_snapshot.tar.gz     <- the tar

⚠ This REVERSES an earlier guarantee: the tar used to contain the page,
because the page was written inside the directory being archived. It no
longer does. Copy both if the report has to travel with the data.

No environment is required: ``build_monitor`` resolves the mojito tree
itself (see :func:`lisatools.globalfit.monitor.resolve_mojito_path`),
including from the run's own settings.
"""

import sys
import time

from . import build_monitor, default_out_path, describe_run
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
    # Beside the run directory, not inside it -- see the module docstring.
    out = argv[1] if len(argv) > 1 else default_out_path(run_dir)
    _state = describe_run(run_dir)
    if _state:
        print(f"[monitor] run state: {_state}")
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
