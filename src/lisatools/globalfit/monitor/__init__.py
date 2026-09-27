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

    Beside the run it describes, so a snapshot tar of the run directory
    carries its own page.
    """
    return os.path.join(os.path.abspath(run_dir), "gf_monitor.html")


def build_monitor(run_dir: str, out_path: Optional[str] = None, *,
                  timeout: float = 1800.0,
                  mojito: Optional[str] = None,
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
    try:
        subprocess.run(
            [sys.executable, generator_path(), run_dir, tmp],
            check=True, timeout=timeout, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
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
