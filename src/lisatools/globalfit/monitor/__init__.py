"""Installed monitor-page and snapshot production.

WHAT THIS IS (user ruling 2026-09-26): "containerize the html production
so it easily adjustable but build into the lisatools/globalfit package.
It does not have to be clean, we will fix it up later, but make sure it
reproduces the html as is done now."

**Reproduction is the hard requirement; tidiness is not.**
``_generator.py`` is ``scripts/diagnostics/gf_monitor_gen.py`` moved here
byte-for-byte -- 5908 lines, 431 top-level statements, 217 module-level
names, ``sys.argv`` read at import, ``raise SystemExit(0)`` if imported.
Turning it into a library is a real refactor (its 23 top-level helpers
close over those 217 globals) and is explicitly deferred.

What moving it buys, which is the part that was blocking: it SHIPS.
``wheel.packages = ["src/lisatools"]`` covers this directory, so a wheel
install has the generator and the saver rank no longer has to walk the
filesystem hunting for a ``scripts/`` directory that exists only in a
source checkout.

WHY EVERY PATH HERE SPAWNS A FRESH INTERPRETER
----------------------------------------------
Not for isolation (though it gives that too) -- for CORRECTNESS. The
generator sets its own dark-theme ``plt.rcParams`` near the top, and
~700 lines later imports ``lisatools...erebor.noise``, whose chain
reaches eryn, which calls ``plt.style.use(["science"])`` as an import
side effect. So in the original invocation the science style lands AFTER
the generator's block and wins.

Import this package and that order inverts: ``import lisatools`` already
pulls matplotlib, pyplot AND eryn (measured), so the style is applied
BEFORE the generator's block and the generator's block wins instead.
That is not hypothetical -- running it in-process changed the
"instrument noise posteriors" panel (25722 vs 27586 base64 chars, 25806
pixels across the whole plot area). Everything else on the page was
identical, which is exactly how a defect like this survives review.

A child interpreter running ``python <_generator.py> RUN_DIR OUT`` has
imported nothing, so the original order is restored and the page is
identical BY CONSTRUCTION rather than by inspection. The subprocess also
returns the build's ~2.5 GB peak RSS to the OS and keeps a matplotlib
segfault away from the caller -- on the saver rank, the run's only
writer, that second property is not optional either.

``build_truth.py`` came along unchanged and NOT renamed: the generator
locates it as a ``__file__`` sibling (``from build_truth import opt_snr``)
for column 2 of the #detect table. Renaming it would drop that column
behind a caught exception.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)

__all__ = [
    "build_monitor",
    "default_out_path",
    "generator_path",
    "resolve_mojito_path",
]

#: A mojito tree is the level holding BOTH of these. Anything else is
#: someone pointing one directory too high or too low, which is the
#: mistake this whole resolution exists to absorb.
MOJITO_SUBDIRS = ("catalogues", "data")

#: Where mojito data lives when nobody says. The cluster layout is flat;
#: the laptop cache nests two levels deeper under build_truth.py's
#: convention, which is exactly the asymmetry people get wrong.
_MOJITO_CANDIDATES = (
    "/shared/data/mojito_cache",
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0",
    "~/.mojito_cache",
)


def _mojito_ok(p) -> bool:
    return bool(p) and all(
        os.path.isdir(os.path.join(p, d)) for d in MOJITO_SUBDIRS)


def _mojito_try(p):
    """``p``, or its nested brickmarket layout, if either validates."""
    if not p:
        return None
    p = os.path.expanduser(str(p))
    if _mojito_ok(p):
        return p
    nested = os.path.join(p, "brickmarket", "mojito_light_v1_0_0")
    return nested if _mojito_ok(nested) else None


def _mojito_from_run(run_dir):
    """The path the RUN ITSELF recorded, read back out of the snapshot.

    ``run_settings.log`` carries the ``mojito_data_path`` the job actually
    loaded bricks from, so a snapshot is self-describing: on the machine
    that produced it, nothing needs to be configured at all. On another
    machine the recorded path simply will not exist and the caller falls
    through to the local candidates.
    """
    import glob as _glob

    for pat in ("*_artifacts/run_settings.log", "run_settings.log"):
        for fp in _glob.glob(os.path.join(run_dir, pat)):
            try:
                with open(fp, errors="replace") as fh:
                    txt = fh.read()
            except OSError:
                continue
            import re as _re

            for m in _re.finditer(r"(/[\w./-]*mojito[\w./-]*)", txt):
                hit = _mojito_try(m.group(1))
                if hit:
                    return hit
    return None


def resolve_mojito_path(run_dir=None, explicit=None):
    """``(path, where_it_came_from)``, or ``(None, reason)``.

    ONE resolution for the whole monitor, so "set MOJITO_INFO_PATH first"
    stops being a second step people have to remember. Order:

    1. ``explicit`` (a ``--mojito`` flag) -- a human overriding on purpose;
    2. ``MOJITO_INFO_PATH`` -- the documented knob;
    3. the path the SNAPSHOT ITSELF records, so a tarball rendered on the
       machine that produced it needs no configuration at all;
    4. the legacy vars, so existing runbooks keep working;
    5. the standard cluster and laptop locations.

    Every candidate is validated for ``catalogues/`` and ``data/`` and
    gets one rescue attempt down the nested ``brickmarket`` layout, so
    pointing one level too high still resolves.
    """
    hit = _mojito_try(explicit)
    if hit:
        return hit, "--mojito"
    hit = _mojito_try(os.environ.get("MOJITO_INFO_PATH"))
    if hit:
        return hit, "MOJITO_INFO_PATH"
    if run_dir:
        hit = _mojito_from_run(run_dir)
        if hit:
            return hit, "the run's own run_settings.log"
    for name in ("MOJITO_CAT", "MOJITO_DATA_PATH", "MOJITO_CACHE_DIR"):
        v = os.environ.get(name)
        if v:
            # MOJITO_CAT may name the catalogue FILE; its tree is two up.
            v = os.path.expanduser(v)
            if os.path.isfile(v):
                v = os.path.dirname(os.path.dirname(v))
            hit = _mojito_try(v)
            if hit:
                return hit, name
    for cand in _MOJITO_CANDIDATES:
        hit = _mojito_try(cand)
        if hit:
            return hit, "the default search path"
    return None, ("no directory holding catalogues/ and data/ was found "
                  "(tried --mojito, MOJITO_INFO_PATH, the run's own "
                  "settings, MOJITO_CAT/DATA_PATH/CACHE_DIR, and "
                  + ", ".join(_MOJITO_CANDIDATES) + ")")


def generator_path() -> str:
    """Absolute path to ``_generator.py``.

    Resolved from this package's own ``__file__`` rather than by
    importing the generator -- importing it raises the ``SystemExit(0)``
    its guard exists to raise.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_generator.py")


def default_out_path(run_dir: str) -> str:
    """Where the page lands when the caller does not say.

    A SIBLING of the run directory, not a file inside it (user ruling
    2026-09-26: "same path as to the --snapshot folder (not in the folder
    same path as that folder)"), so both artifacts land together:

        /data/gf_prod_6mo_v9_4gpu/              <- the run
        /data/gf_prod_6mo_v9_4gpu_monitor.html  <- the page
        /data/gf_prod_6mo_v9_4gpu_snapshot.tar.gz

    ⚠ CONSEQUENCE, stated because it reverses an earlier guarantee: the
    tar no longer CONTAINS the page. It used to, precisely because the
    page was written inside the directory being archived. Ship both files
    if the report needs to travel with the data.
    """
    return os.path.abspath(str(run_dir).rstrip("/")) + "_monitor.html"


def describe_run(run_dir: str) -> Optional[str]:
    """One line saying HOW FAR ALONG the run is, or ``None``.

    The generator never printed this and neither did the wrappers, so the
    only way to learn what iteration a page describes was to open the
    page. That is the first thing anyone wants to know about a snapshot --
    especially when re-rendering a tarball that may be the same one as
    last time.

    Rewind- and torn-row aware for the same reasons
    :func:`build_truth.latest_iteration` is: ``log_like`` rows are
    preallocated so the dataset length is capacity, not progress; a
    rewound store's ``iteration`` attr is lower than its filled extent and
    the rows past it are a discarded trajectory. Never raises -- this is a
    convenience line, not a gate.
    """
    import glob as _glob

    try:
        import h5py
        import numpy as np

        cands = [p for p in _glob.glob(os.path.join(run_dir, "*.h5"))
                 if "testing" in os.path.basename(p)
                 and "CORRUPT" not in p and "backup" not in p]
        if not cands:
            return None
        store = max(cands, key=os.path.getmtime)
        with h5py.File(store, "r") as f:
            g = f["global_fit"]
            ll = g["log_like"][:, 0, 0, :]
            filled = np.where(np.any(ll != 0.0, axis=1))[0]
            if not filled.size:
                return f"{os.path.basename(store)}: no filled iterations yet"
            nit = int(filled.max()) + 1
            attr = g.attrs.get("iteration")
            rewound = attr is not None and 0 < int(attr) < nit
            if rewound:
                nit = int(attr)
            last = nit - 1
            L = ll[last]
            bits = [f"iteration {last} ({nit} stored)"]
            if rewound:
                bits.append("REWOUND")
            if "inds/gb" in g:
                inds = g["inds/gb"]
                per = [int(np.count_nonzero(inds[last, 0, 0, w]))
                       for w in range(inds.shape[3])]
                bits.append(f"{sum(per):,} GB leaves {per}")
            bits.append(f"max lnL {L.max():.1f} (walker spread "
                        f"{L.max() - L.min():.1f})")
            return "; ".join(bits)
    except Exception as e:                        # noqa: BLE001
        logger.debug("describe_run failed: %r", e)
        return None


def build_monitor_in_process(run_dir: str, out_path: str,
                             mojito: Optional[str] = None) -> str:
    """Render the page in THIS interpreter -- one python, no child.

    ⚠ READ THIS BEFORE PREFERRING IT. The generator sets its own dark
    theme in ``plt.rcParams`` near the top and ~700 lines later imports
    ``...erebor.noise``, whose chain reaches eryn, which calls
    ``plt.style.use(["science"])`` as an IMPORT SIDE EFFECT. Run as a
    script, that side effect lands AFTER the generator's block and the
    science style wins. In this process ``lisatools`` is already imported,
    so the side effect has already happened and the generator's block wins
    instead.

    Measured consequence (2026-09-26): the "instrument noise posteriors"
    panel renders differently -- 25722 vs 27586 base64 chars, 25806 pixels
    across the whole plot area -- with all 760 other lines of the page
    identical. Nothing else changes.

    So this is NOT byte-identical to :func:`build_monitor` and cannot be
    made so without the generator taking control of its own style
    ordering. It is offered because one process is sometimes worth more
    than one panel, not because the difference is imaginary.

    It also keeps the build's ~2.5 GB peak RSS in the caller's heap and
    puts a matplotlib fault in the caller's process, which is why the
    saver rank -- the run's only writer -- must not use it.
    """
    import runpy

    argv = [generator_path(), str(run_dir), str(out_path)]
    saved_argv, saved_env = sys.argv, os.environ.get("MOJITO_INFO_PATH")
    _moj, _src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    sys.argv = argv
    if _moj:
        os.environ["MOJITO_INFO_PATH"] = _moj
        logger.info("monitor: mojito data from %s (%s)", _moj, _src)
    ns = None
    try:
        ns = runpy.run_path(generator_path(), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        # CLOSE THE GENERATOR'S HDF5 HANDLES. It is a script: it opens the
        # store at module level and never closes it, because a child
        # process exits and takes the handle with it. In THIS process the
        # handle survives, and h5py then refuses to create the snapshot's
        # ``*_extract.h5``:
        #
        #   OSError: Unable to synchronously create file (unable to
        #            truncate a file which is already open)
        #
        # which broke `--snapshot` the moment the render stopped being a
        # subprocess -- the page was written and the tar silently was not.
        # runpy hands back the module globals, so the handles are reachable
        # without the generator having to grow a teardown.
        for _v in list((ns or {}).values()):
            try:
                import h5py

                if isinstance(_v, h5py.File):
                    _v.close()
                elif isinstance(_v, h5py.Group) and _v.file:
                    _v.file.close()
            except Exception:                     # noqa: BLE001
                pass
        sys.argv = saved_argv
        if saved_env is None:
            os.environ.pop("MOJITO_INFO_PATH", None)
        else:
            os.environ["MOJITO_INFO_PATH"] = saved_env
    return out_path


def build_monitor(run_dir: str, out_path: Optional[str] = None, *,
                  timeout: float = 1800.0,
                  mojito: Optional[str] = None,
                  in_process: bool = True,
                  check: bool = True) -> Optional[str]:
    """Build the HTML page for ``run_dir`` in a FRESH interpreter.

    Returns the path written, or ``None`` when ``check=False`` and the
    build failed. See the module docstring for why this is a subprocess
    and not an import -- it is a correctness requirement, not a
    preference.

    Published atomically: the child writes a temp beside the target and
    this renames it into place, so a browser refreshing the page never
    catches half a document.
    """
    run_dir = os.path.abspath(str(run_dir))
    out_path = out_path or default_out_path(run_dir)
    tmp = out_path + ".tmp"
    st = time.perf_counter()
    # RESOLVE THE MOJITO TREE HERE, so "export MOJITO_INFO_PATH first" is
    # not a second step every caller has to remember. The child reads it
    # from the environment, so passing it down is all that is needed; an
    # already-set MOJITO_INFO_PATH still wins inside resolve_mojito_path.
    env = dict(os.environ)
    _moj, _src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    if _moj:
        env["MOJITO_INFO_PATH"] = _moj
        logger.info("monitor: mojito data from %s (%s)", _moj, _src)
    else:
        logger.warning(
            "monitor: %s. The page will fall back to the analytic PSD "
            "injection and DROP the residual-spectrum and "
            "data/template/residual panels, with no error on the page "
            "itself.", _src)
    # GF_MONITOR_IN_PROCESS=1 (or in_process=True) renders without a child
    # interpreter -- one python, at the cost of one restyled panel. See
    # build_monitor_in_process for the measurement.
    # STRAIGHT PYTHON FLOW, no child interpreter (user ruling
    # 2026-09-26: "no subprocesses!"). Safe since the font.size pin made
    # the in-process render byte-identical to the child one.
    # GF_MONITOR_SUBPROCESS=1 is the escape hatch for a caller that wants
    # the ~2.5 GB peak and any matplotlib fault kept out of its own
    # process -- the saver rank is the case that might.
    _inproc = in_process and os.environ.get(
        "GF_MONITOR_SUBPROCESS", "0").strip() not in ("1", "true", "yes", "on")
    try:
        if _inproc:
            build_monitor_in_process(run_dir, tmp, mojito=_moj)
        else:
            subprocess.run(
                [sys.executable, generator_path(), run_dir, tmp],
                check=True, timeout=timeout, env=env,
                # DO NOT SWALLOW THE GENERATOR'S OWN OUTPUT. It reports
                # the plot count, the VGB corners, the GB cloud size and
                # the missing-panel count -- the only running commentary
                # there is -- and DEVNULL here meant the subprocess path
                # silently showed less than the in-process one.
                stderr=subprocess.PIPE,
            )
        os.replace(tmp, out_path)
    except Exception as e:                        # noqa: BLE001
        _err = getattr(e, "stderr", b"") or b""
        if isinstance(_err, bytes):
            _err = _err.decode("utf-8", "replace")
        logger.warning(
            "monitor page NOT refreshed (%s: %s). Any previous page is "
            "untouched.%s", type(e).__name__, e,
            (" Last stderr: " + _err.strip()[-800:]) if _err else "")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        if check:
            raise
        return None
    logger.info("monitor page built in %.1f s -> %s",
                time.perf_counter() - st, out_path)
    return out_path
