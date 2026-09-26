#!/usr/bin/env python
"""Compatibility shim -- the store extractor now lives in the package.

Moved 2026-09-26 to ``lisatools.globalfit.monitor._store_extract`` so the
saver rank can build a snapshot without a source checkout.
``make_snapshots.sh`` still calls this path and still works unchanged.
"""
import sys

from lisatools.globalfit.monitor._store_extract import *     # noqa: F401,F403
from lisatools.globalfit.monitor._store_extract import extract, main

if __name__ == "__main__":
    sys.exit(main())
