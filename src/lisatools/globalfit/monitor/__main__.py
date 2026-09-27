"""Build the monitor page, and optionally the snapshot tar, in one command.

    python -m lisatools.globalfit.monitor [--snapshot] [--build-truth] \\
        RUN_DIR [OUT.html]

Two directions, two entry points:

* **here** -- a LIVE run directory -> page (and with ``--snapshot``, the
  tarball too). This is the cluster-side command.
* :mod:`lisatools.globalfit.monitor.from_tar` -- a downloaded tarball ->
  page. That one CONSUMES a tar; it does not make one.

BOTH ARTIFACTS LAND BESIDE THE RUN FOLDER (user ruling 2026-09-26), not
inside it::

    /data/gf_prod_6mo_v9_4gpu/                    <- the run
    /data/gf_prod_6mo_v9_4gpu_monitor.html        <- the page
    /data/gf_prod_6mo_v9_4gpu_snapshot.tar.gz     <- the tar

⚠ This REVERSES an earlier guarantee: the tar used to contain the page,
because the page was written inside the directory being archived. It no
longer does. Copy both if the report has to travel with the data.

``--build-truth`` generates the detectability truth set into the run
directory first. Anything after ``--`` goes straight to ``build_truth``
(``--catalogue``, ``--l1-brick``, ``--flo`` ...), though normally none of
it is needed: the mojito tree is resolved from ``MOJITO_INFO_PATH``, the
run's own settings, the legacy ``MOJITO_*`` vars or the standard
cluster/laptop paths, and handed to build_truth for both the catalogue
and the L1 orbit bricks.
"""

import argparse
import sys
import time

from . import (build_monitor, build_truth_set, check_truth,
               default_out_path, describe_run)
from .snapshot import build_snapshot

USAGE = ("usage: python -m lisatools.globalfit.monitor [--snapshot] "
         "[--build-truth] RUN_DIR [OUT.html] [-- BUILD_TRUTH_ARGS...]\n"
         "       --snapshot     also build <RUN_DIR>_snapshot.tar.gz\n"
         "       --build-truth  generate the truth set into RUN_DIR first\n"
         "       (a downloaded tarball goes the other way: python -m "
         "lisatools.globalfit.monitor.from_tar SNAP.tar.gz)")


def _parser():
    """A REAL parser (2026-09-27).

    This used to be ad-hoc list surgery: strip the flags it knew about,
    then take ``argv[0]`` as the run dir and ``argv[1]`` as the output
    path. So an unrecognised flag did not error -- it silently BECAME THE
    OUTPUT PATH. ``--build-truth --catalogue /path/...`` set
    ``out="--catalogue"``, which is how a real cluster invocation went
    wrong. argparse rejects what it does not know, which is the point.
    """
    ap = argparse.ArgumentParser(
        prog="python -m lisatools.globalfit.monitor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out", nargs="?", default=None,
                    help="output .html (default: <RUN_DIR>_monitor.html, "
                         "beside the run folder)")
    ap.add_argument("--snapshot", action="store_true",
                    help="also build <RUN_DIR>_snapshot.tar.gz beside it")
    ap.add_argument("--build-truth", action="store_true",
                    help="generate the detectability truth set into the run "
                         "directory first when it is missing or was built "
                         "for a different Tobs. Tens of minutes, CPU-only.")
    ap.add_argument("--subprocess", action="store_true",
                    help="render in a child interpreter. Output identical.")
    # FIRST-CLASS, not just pass-through. These are the two overrides
    # anyone actually reaches for, and requiring `-- --catalogue X` for
    # them is a trap: the natural spelling produced "unrecognized
    # arguments" on a real cluster invocation (2026-09-27).
    ap.add_argument("--catalogue", default=None,
                    help="GB catalogue hdf5 (or the directory holding it) "
                         "for --build-truth. Normally unnecessary.")
    ap.add_argument("--l1-brick", default=None,
                    help="explicit mojito L1 .h5 for the injected orbits "
                         "used by --build-truth. Normally unnecessary.")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # SPLIT ON `--` BEFORE PARSING, do not use argparse.REMAINDER.
    # REMAINDER absorbs everything from the first token it does not
    # recognise onward -- INCLUDING flags this parser defines. With it,
    # `RUN_DIR --build-truth -- --catalogue X` parsed build_truth as
    # FALSE and handed "--build-truth" to build_truth instead. It also
    # meant an unknown flag was swallowed rather than rejected, which is
    # the behaviour this parser exists to provide.
    if "--" in argv:
        _i = argv.index("--")
        argv, extra = argv[:_i], argv[_i + 1:]
    else:
        extra = []
    a = _parser().parse_args(argv)
    for _flag, _val in (("--catalogue", a.catalogue),
                        ("--l1-brick", a.l1_brick)):
        if _val:
            extra += [_flag, _val]
    run_dir = a.run_dir
    out = a.out or default_out_path(run_dir)

    _truth, _tnote = check_truth(run_dir)
    if _truth:
        print(f"[monitor] truth set: {_tnote}")
    elif a.build_truth:
        print(f"[monitor] truth set: {_tnote} -- BUILDING one now "
              "(tens of minutes, CPU-only)")
        try:
            build_truth_set(run_dir, extra_argv=extra)
            print(f"[monitor] truth set: {check_truth(run_dir)[1]}")
        except Exception as e:                    # noqa: BLE001
            print(f"[monitor] truth build FAILED ({type(e).__name__}: {e})",
                  file=sys.stderr)
    else:
        print(f"[monitor] WARNING: {_tnote} -- no detectability overlays. "
              "Pass --build-truth to generate one.", file=sys.stderr)
        if extra:
            # Otherwise these vanish without a word, which is how you spend
            # an hour wondering why --catalogue had no effect.
            print(f"[monitor] NOTE: {extra} was given but --build-truth was "
                  "not, so it does nothing.", file=sys.stderr)

    _state = describe_run(run_dir)
    if _state:
        print(f"[monitor] run state: {_state}")

    st = time.perf_counter()
    build_monitor(run_dir, out, in_process=not a.subprocess)
    print(f"[monitor] wrote {out} in {time.perf_counter() - st:.1f} s")

    if a.snapshot:
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
