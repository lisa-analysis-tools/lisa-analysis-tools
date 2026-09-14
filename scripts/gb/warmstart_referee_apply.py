"""Thin CLI shim -- the code lives in the installed package.

User ruling 2026-09-14: "All the warmstart code should be part of the
globalfit/lisatools package. Any separate scripts should just call that
installed code." Equivalent invocation:
``python -m lisatools.globalfit.warmstart.referee_apply ...``.
"""

from lisatools.globalfit.warmstart.referee_apply import main

if __name__ == "__main__":
    raise SystemExit(main())
