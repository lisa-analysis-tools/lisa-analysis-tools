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


def _reduce_log(path: str, tail_bytes: int) -> bool:
    """Write ``<base>_filtered.log`` + ``<base>_tail.log`` beside ``path``.

    True if both were written and the raw file can be skipped.
    """
    keep = re.compile(LOG_KEEP_PATTERN)
    base = path[:-4] if path.endswith(".log") else path
    try:
        with open(path, "rb") as fh, open(base + "_filtered.log", "wb") as out:
            for line in fh:
                if keep.search(line.decode("utf-8", "replace")):
                    out.write(line)
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - tail_bytes))
            with open(base + "_tail.log", "wb") as out:
                out.write(fh.read())
    except Exception as e:                       # noqa: BLE001
        logger.warning("log filtering failed for %s (%s: %s); shipping it "
                       "whole", path, type(e).__name__, e)
        return False
    return True


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
    return runlog if _reduce_log(runlog, 20_000_000) else None


def _filter_all_logs(run_dir: str, cap_mb: int, tail_mb: int):
    """Short mode: reduce EVERY oversized ``*.log``, not just the run log.

    ⚠ THE SLURM LOGS ARE THE BULK. ``_filter_run_log`` only ever looked
    at ``*_artifacts/globalfit_run.log``; the per-job
    ``slurm_stdout_<id>.log`` files sit in the run directory root and
    shipped WHOLE. On 6mo v9 that was a single 386 MB member, which is
    most of why a full snapshot is ~330 MB compressed. A short snapshot
    is supposed to be the log information, not the log volume, so each
    oversized log becomes its filtered + tail pair.
    """
    out = set()
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if not fn.endswith(".log"):
                continue
            if fn.endswith(("_filtered.log", "_tail.log")):
                continue          # our own products from a previous run
            p = os.path.join(root, fn)
            try:
                mb = os.path.getsize(p) // 1048576
            except OSError:
                continue
            if mb <= cap_mb:
                continue
            if _reduce_log(p, int(tail_mb) * 1_000_000):
                out.add(os.path.abspath(p))
    if out:
        logger.info("short snapshot: reduced %d oversized log(s) to "
                    "filtered + %d MB tail", len(out), tail_mb)
    return out


def _members(run_dir: str, include_fstat: bool, skip_raw_log,
             max_file_mb: Optional[int] = None):
    """Archive members. ``skip_raw_log`` is one path or a set of them.

    ``max_file_mb`` drops anything bigger (short mode). The reduced
    extract ``.h5`` is EXEMPT: it is "the most recent state", i.e. the
    point of the snapshot, and it is already bounded by ``keep`` /
    ``cold_keep`` rather than by a byte count.
    """
    if skip_raw_log is None:
        skip = set()
    elif isinstance(skip_raw_log, (set, frozenset, list, tuple)):
        skip = {os.path.abspath(p) for p in skip_raw_log}
    else:
        skip = {os.path.abspath(skip_raw_log)}
    out = []
    dropped = []
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
            if os.path.abspath(p) in skip:
                continue
            # EXEMPT from the size cap: the reduced products we just went
            # to the trouble of making. The extract IS "the most recent
            # state" and is bounded by keep/cold_keep; a _filtered.log is
            # bounded by the keep-pattern and a _tail.log by tail_mb.
            # Without this the cap silently discards the filtered log of
            # a big job and leaves only its tail -- caught on a real 6mo
            # run dir, where slurm_stdout_654_filtered.log was built and
            # then dropped while its tail survived.
            if max_file_mb is not None and not fn.endswith(
                    ("_extract.h5", "_filtered.log", "_tail.log")):
                try:
                    mb = os.path.getsize(p) / 1048576.0
                except OSError:
                    continue
                if mb > max_file_mb:
                    dropped.append((rel, mb))
                    continue
            out.append(p)
    if dropped:
        dropped.sort(key=lambda x: -x[1])
        logger.info(
            "short snapshot: dropped %d file(s) over %d MB (%s%s)",
            len(dropped), max_file_mb,
            ", ".join(f"{r} {m:.0f} MB" for r, m in dropped[:4]),
            ", ..." if len(dropped) > 4 else "")
    return out


def build_snapshot(run_dir: str, out_path: Optional[str] = None, *,
                   keep: int = 5, cold_keep: int = 12,
                   include_fstat: bool = False,
                   log_cap_mb: int = 200,
                   short: bool = False,
                   max_file_mb: Optional[int] = None,
                   tail_mb: int = 5) -> Optional[str]:
    """Build ``<run_dir>_snapshot.tar.gz``. Returns the path, or ``None``.

    ``short=True`` builds ``<run_dir>_short.tar.gz`` instead: log
    information and the most recent state, and nothing big. Concretely
    it overrides the defaults to ``keep=1``, ``cold_keep=1``,
    ``log_cap_mb=20`` and ``max_file_mb=5``, and reduces EVERY oversized
    ``*.log`` to its filtered + tail pair rather than only the run log.
    ``include_fstat`` stays off, as it already is by default. Any of
    those can still be passed explicitly to override the preset.

    What that removes, measured on 6mo v9: the per-job
    ``slurm_stdout_*.log`` (386 MB for job 650, shipped whole before
    because only ``*_artifacts/globalfit_run.log`` was ever filtered),
    ``gb_truth_3to21.npz`` (78 MB), and the five-iteration extract in
    favour of a one-iteration one. A full snapshot of that run is
    ~330 MB compressed.

    Never raises: this runs on the writer rank, where a failed diagnostic
    must not become a failed run. A ``None`` return with a warning is the
    contract.
    """
    run_dir = os.path.abspath(str(run_dir).rstrip("/"))
    if short:
        # Presets, not hard-codes: an explicit kwarg still wins. Compared
        # against the signature defaults so "the caller said 5" and "the
        # caller said nothing" are distinguishable without sentinels.
        if keep == 5:
            keep = 1
        if cold_keep == 12:
            cold_keep = 1
        if log_cap_mb == 200:
            log_cap_mb = 20
        if max_file_mb is None:
            max_file_mb = 5
        # ⚠ NO DEAD BAND. A log bigger than max_file_mb but smaller than
        # log_cap_mb would be left unreduced and then dropped by the
        # size cap -- silently losing the whole file instead of keeping
        # its filtered + tail pair. Reduce at the SMALLER of the two so
        # every log that is too big to ship whole gets reduced first.
        log_cap_mb = min(int(log_cap_mb), int(max_file_mb))
    out_path = out_path or (
        run_dir + ("_short.tar.gz" if short else "_snapshot.tar.gz"))
    st = time.perf_counter()
    try:
        store = _live_store(run_dir)
        if store is None:
            logger.warning("snapshot: no live *testing*.h5 under %s", run_dir)
            return None
        from ._store_extract import extract

        extract(store, store[:-3] + "_extract.h5", keep,
                cold_keep=cold_keep)
        if short:
            skip = _filter_all_logs(run_dir, log_cap_mb, tail_mb)
        else:
            skip = _filter_run_log(run_dir, log_cap_mb)
        members = _members(run_dir, include_fstat, skip,
                           max_file_mb=max_file_mb)
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
