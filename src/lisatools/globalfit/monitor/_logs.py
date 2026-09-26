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
    return [found[k] for k in sorted(found)]
