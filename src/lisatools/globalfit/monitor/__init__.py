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
]


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
    try:
        subprocess.run(
            [sys.executable, generator_path(), run_dir, tmp],
            check=True, timeout=timeout,
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
