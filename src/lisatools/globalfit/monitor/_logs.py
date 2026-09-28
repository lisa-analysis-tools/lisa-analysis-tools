"""Run-log discovery, importable without running the generator.

``_generator.py`` is a standalone script that ``raise SystemExit(0)`` on
import, so anything it defines is unreachable to an importer. This one
helper needs to be reachable -- ``tests/test_diagnostics_multirank.py``
calls it through the ``scripts/diagnostics/gf_monitor_gen.py`` entry
point, and that file is now a shim with no body of its own.

The function itself is moved verbatim from the generator, not copied:
``gf_run_log_digest.py`` already carries a hand-synced second copy, and a
third would be worse than a shared one.
"""

from __future__ import annotations

import os
import re
import sys

__all__ = ["discover_run_logs"]


def discover_run_logs(run_dir):
    """Every rank's run log under ``run_dir``: the head's ``globalfit_run.log``
    first, then ``globalfit_run.rank<k>.log`` files sorted by rank number.

    Each rank writes its own log file under the walker-block layout
    (``run.py::_rank_log_filenames``); concatenating them (head first) is
    what lets the regex scans below (e.g. ``RJ_SPLIT_RE``) see every rank's
    ``[GB_ACCEPT rj-split]`` lines, not just the head's (Plan 5 Task 4).

    The walk is RECURSIVE and deterministic (directories and file names
    sorted), identical to ``gf_run_log_digest.py``'s copy of this helper. A
    second file for a rank already seen (the same run unpacked twice under
    ``run_dir``) is NOT silently dropped -- the first one found wins and the
    duplicate is named on stderr.
    """
    found = {}
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if fn == "globalfit_run.log":
                key = -1
            else:
                m = re.match(r"^globalfit_run\.rank(\d+)\.log$", fn)
                if m is None:
                    continue
                key = int(m.group(1))
            path = os.path.join(root, fn)
            if key in found:
                label = "head" if key < 0 else f"rank {key}"
                print(
                    f"# WARNING: duplicate {label} run log {path}; keeping {found[key]}",
                    file=sys.stderr,
                )
                continue
            found[key] = path

    # ---- slurm stdout: the SUPERSET, and the only source of some lines ---
    #
    # ⚠ WITHOUT THIS THE PAGE IS BLIND ON THE CLUSTER. Measured on the 6mo
    # v9 tars (2026-09-28): ``slurm_stdout_<job>.log`` is a verbatim
    # superset of ``globalfit_run.log`` plus the per-rank logs (3000/3000
    # and 2960/3000 sampled lines found in it), AND it alone carries
    # ``[GF_TIMING]``, ``[V9-SEED]``, the ``[r4/saver]`` lines and the
    # stage table -- those never reach the run logs at all. So:
    #
    #  * ``GFT_RE`` could never match on a cluster-built page, which is
    #    why the "search efficiency" panel has been dead there, and
    #  * a page built from a snapshot whose head log was shipped as the
    #    filtered + tail pair (``globalfit_run_filtered.log`` /
    #    ``_tail.log``, which the names above do not match) saw NO head
    #    text at all, and a short tar gave it nothing whatsoever.
    #
    # Appended LAST so the head log keeps priority for everything it does
    # carry, and only the NEWEST stdout is taken: older ones are dead jobs
    # whose lines would be replayed as if current.
    _stdouts = []
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if re.match(r"^slurm_stdout_\d+\.log$", fn):
                p = os.path.join(root, fn)
                try:
                    _stdouts.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    out = [found[k] for k in sorted(found)]
    if _stdouts:
        out.append(max(_stdouts)[1])
    elif not out:
        # Neither kind present: say so rather than silently rendering a
        # page with every log-derived panel empty.
        print(f"# WARNING: no run log and no slurm_stdout_*.log under "
              f"{run_dir}; every log-derived panel will be empty",
              file=sys.stderr)
    return out
