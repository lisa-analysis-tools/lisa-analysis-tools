#!/usr/bin/env python
"""Compatibility shim -- build_truth now lives in the package.

Moved 2026-09-26 to ``lisatools.globalfit.monitor.build_truth`` alongside
the generator, which locates it as a ``__file__`` sibling to compute
column 2 of the #detect table. The CLI is unchanged:

    OMP_NUM_THREADS=1 python scripts/diagnostics/build_truth.py STORE.h5 ...
"""
import sys

from lisatools.globalfit.monitor.build_truth import *        # noqa: F401,F403
from lisatools.globalfit.monitor.build_truth import main

if __name__ == "__main__":
    sys.exit(main())
