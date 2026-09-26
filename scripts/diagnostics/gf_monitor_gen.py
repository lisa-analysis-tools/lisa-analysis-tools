#!/usr/bin/env python
"""Compatibility shim -- the generator now lives in the package.

Moved 2026-09-26 to ``lisatools.globalfit.monitor._generator`` so it ships
in the wheel and the saver rank can reach it without hunting for a
``scripts/`` directory that only exists in a source checkout. This file
stays so every existing runbook, ``snapshot_to_html.sh``, and
muscle-memory invocation keeps working unchanged:

    python scripts/diagnostics/gf_monitor_gen.py RUN_DIR OUT.html

is still exactly equivalent to

    python -m lisatools.globalfit.monitor RUN_DIR OUT.html

Nothing is reimplemented here -- both forward into the same module, so
there is no second copy of the page logic to drift.
"""
import sys

# discover_run_logs comes from ._logs, NOT from ._generator: the generator
# raises SystemExit(0) on import by design, so importing it here would
# never bind the name. Tests reach the helper through THIS path.
from lisatools.globalfit.monitor._logs import discover_run_logs  # noqa: F401

if __name__ == "__main__":
    from lisatools.globalfit.monitor import build_monitor

    _run_dir = sys.argv[1] if len(sys.argv) > 1 else "prod3mo/gf_prod_3mo"
    _out = sys.argv[2] if len(sys.argv) > 2 else "gf_monitor.html"
    build_monitor(_run_dir, _out)
