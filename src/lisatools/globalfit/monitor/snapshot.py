"""Snapshot tar production, importable from the installed package.

This is ``scripts/fstat_proposal/make_snapshots.sh`` expressed in Python
so the saver rank can build a snapshot without a source checkout, a shell,
or a ``cd`` to the repo root (the shell script does all three).

THE SHELL SCRIPT IS DELIBERATELY LEFT ALONE (user instruction 2026-09-26:
"try not to touch any other code"). So yes, there are two implementations
of this recipe for now. That is a known debt, written down here rather
than discovered later: if you change the exclusion list, change it in
BOTH, and the test ``SnapshotMatchesTheShellRecipeTest`` compares the two
so the pair cannot drift silently.

The recipe, unchanged:

  1. reduce the live store with :func:`._store_extract.extract` (the big
     per-iteration chains keep only the last ``keep`` rows, cold chains
     ``cold_keep``), writing ``<store>_extract.h5``;
  2. take every file in the run dir EXCEPT the full h5 stores, backups,
     dissect dumps, rendered diagnostics, the fstat grid payloads, the
     midit checkpoint, and any prior archive;
  3. above ``log_cap_mb`` replace the DEBUG run log with a grep-filtered
     full-history file plus a raw tail (it grows ~0.5 MB/iteration and is
     the one file that outgrows the budget);
  4. ``tar -czf <run_dir>_snapshot.tar.gz``.
"""

from __future__ import annotations

import os
import re
import tarfile
import time
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)

__all__ = ["build_snapshot", "LOG_KEEP_PATTERN"]

#: Line families the filtered run log keeps. Byte-identical to the
#: ``grep -aE`` alternation in make_snapshots.sh -- the test compares them.
LOG_KEEP_PATTERN = (
    r"\[SAVE\]|\[GB_ACCEPT|\[GB_OBS_BASIS|\[GB_INMODEL_TRACE|MISMATCH|"
    r"\[GB_TEMPER_CHECK|\[GB_CAP|\[GB_ORTHO|\[stageB\]|epoch .* done in|"
    r"\[FSTAT_CTR|\[GB_TIMING|\[GB_BAND_SHUTOFF|\[GB_BAND_REVIVE|"
    r"\[V8-NOISE|\[COARSE_AUDIT|\[GB_TRUST|\[GB_VERT|\[unequal-arm|"
    r"\[galfor-modulation|\[LADDER|\[MIDIT_CKPT|WARNING|ERROR|CRITICAL|"
    r"Traceback|ll AUDIT|DELTA-vs-DELTA|sig-het engine resolved|"
    r"\[GB_SWEEP|\[LOGMIRROR|\[GB_REPLACE"
)

#: Name fragments that never enter a snapshot, whatever else matches.
_EXCLUDE_PATH_PARTS = ("fstat_grid_parts", "/dissect/", "_artifacts/diagnostics/")
_EXCLUDE_SUFFIXES = (".bak", ".tmp", ".tar.gz", ".zip",
                     "_midit_checkpoint.pkl")


def _live_store(run_dir: str) -> Optional[str]:
    """Newest ``*testing*.h5`` that is not a backup / corrupt / extract.

    Same selection as the shell's ``ls -t | grep -v``, and the same
    selection the monitor page makes -- a snapshot whose extract came from
    a different store than the page describes would be quietly incoherent.
    """
    cands = []
    for fn in os.listdir(run_dir):
        if not fn.endswith(".h5") or "testing" not in fn:
            continue
        if any(t in fn for t in ("backup", "CORRUPT", "_extract")):
            continue
        p = os.path.join(run_dir, fn)
        cands.append((os.path.getmtime(p), p))
    return max(cands)[1] if cands else None


def _filter_run_log(run_dir: str, log_cap_mb: int):
    """Above the cap, write the filtered + tail pair. Returns the raw log
    to EXCLUDE from the archive, or ``None`` to ship it whole."""
    import glob as _glob

    hits = _glob.glob(os.path.join(run_dir, "*_artifacts", "globalfit_run.log"))
    if not hits:
        return None
    runlog = hits[0]
    mb = os.path.getsize(runlog) // 1048576
    if mb <= log_cap_mb:
        return None
    logger.info("run log %d MB > %d MB -- shipping filtered + tail",
                mb, log_cap_mb)
    keep = re.compile(LOG_KEEP_PATTERN)
    base = runlog[:-4]
    try:
        with open(runlog, "rb") as fh, open(base + "_filtered.log", "wb") as out:
            for line in fh:
                if keep.search(line.decode("utf-8", "replace")):
                    out.write(line)
        with open(runlog, "rb") as fh:
            fh.seek(max(0, os.path.getsize(runlog) - 20_000_000))
            with open(base + "_tail.log", "wb") as out:
                out.write(fh.read())
    except Exception as e:                       # noqa: BLE001
        logger.warning("run-log filtering failed (%s: %s); shipping it whole",
                       type(e).__name__, e)
        return None
    return runlog


def _members(run_dir: str, include_fstat: bool, skip_raw_log: Optional[str]):
    out = []
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            rel = "/" + os.path.relpath(p, run_dir)
            if any(part in rel for part in _EXCLUDE_PATH_PARTS):
                continue
            if any(fn.endswith(s) for s in _EXCLUDE_SUFFIXES):
                continue
            if ".h5.bak" in fn:
                continue
            # Full stores out, the reduced extract in.
            if fn.endswith(".h5") and not fn.endswith("_extract.h5"):
                continue
            if not include_fstat and "gb_fstat_fit" in rel and fn != "DONE.json":
                continue
            if skip_raw_log and os.path.abspath(p) == os.path.abspath(skip_raw_log):
                continue
            out.append(p)
    return out


def build_snapshot(run_dir: str, out_path: Optional[str] = None, *,
                   keep: int = 5, cold_keep: int = 12,
                   include_fstat: bool = False,
                   log_cap_mb: int = 200) -> Optional[str]:
    """Build ``<run_dir>_snapshot.tar.gz``. Returns the path, or ``None``.

    Never raises: this runs on the writer rank, where a failed diagnostic
    must not become a failed run. A ``None`` return with a warning is the
    contract.
    """
    run_dir = os.path.abspath(str(run_dir).rstrip("/"))
    out_path = out_path or (run_dir + "_snapshot.tar.gz")
    st = time.perf_counter()
    try:
        store = _live_store(run_dir)
        if store is None:
            logger.warning("snapshot: no live *testing*.h5 under %s", run_dir)
            return None
        from ._store_extract import extract

        extract(store, store[:-3] + "_extract.h5", keep,
                cold_keep=cold_keep)
        skip = _filter_run_log(run_dir, log_cap_mb)
        members = _members(run_dir, include_fstat, skip)
        # Write to a temp and rename: a consumer polling for the tar must
        # never pick up a partial archive. Same reasoning as the page and
        # the running backup copy.
        tmp = out_path + ".tmp"
        with tarfile.open(tmp, "w:gz") as tf:
            for p in members:
                try:
                    tf.add(p, arcname=os.path.relpath(
                        p, os.path.dirname(run_dir)))
                except (FileNotFoundError, OSError) as e:
                    # A live file can vanish or change mid-archive. Benign:
                    # the append-only log gives a consistent prefix and the
                    # extract h5 is written atomically.
                    logger.debug("snapshot: skipping %s (%s)", p, e)
        os.replace(tmp, out_path)
    except Exception as e:                       # noqa: BLE001 -- see docstring
        logger.warning("snapshot NOT built (%s: %s). The run is unaffected.",
                       type(e).__name__, e)
        try:
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
        except Exception:
            pass
        return None
    logger.info("snapshot %s (%.1f MB) in %.1f s", out_path,
                os.path.getsize(out_path) / 1048576.0,
                time.perf_counter() - st)
    return out_path
