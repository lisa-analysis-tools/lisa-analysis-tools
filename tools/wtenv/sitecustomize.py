"""Run a git WORKTREE's lisatools instead of the editable-installed main checkout.

Usage (from the worktree root, any machine with the dev stack installed):

    export PYTHONPATH=$PWD/tools/wtenv LAT_WORKTREE_SRC=$PWD/src
    python -m unittest tests.test_emri_wdm_direct
    bash scripts/emri/emri_speed_durations.sh      # the scripts inherit the env

Why: ``pip install -e .`` (scikit-build-core) installs a redirecting meta-path
finder that hard-maps every ``lisatools.*`` module to the checkout it was
installed from.  Meta-path finders beat ``sys.path``, so ``PYTHONPATH`` alone
cannot shadow it.  With ``LAT_WORKTREE_SRC`` set, this ``sitecustomize`` strips
that finder, drops the installed checkout's ``src`` from ``sys.path`` and puts
the worktree's ``src`` first.  The compiled backend packages
(``lisatools_backend_cpu`` / ``lisatools_backend_cuda13x`` in site-packages)
are separate top-level packages and keep being imported from there, so a
pure-Python worktree needs no rebuild.  Unset ``LAT_WORKTREE_SRC`` and nothing
happens.  The running production jobs are untouched: their checkout, their
editable install and their imports are not modified.
"""

import os
import sys

_wt = os.environ.get("LAT_WORKTREE_SRC")
if _wt:
    _wt = os.path.abspath(_wt)
    # 1. the redirecting finder of the editable install
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if getattr(type(f), "__module__", "") != "_lisaanalysistools_editable"
    ]
    # 2. any other checkout's src on sys.path (an explicit LAT_MAIN_SRC, or any
    #    entry that holds a lisatools package and is not the worktree)
    _main = os.environ.get("LAT_MAIN_SRC")
    _keep = []
    for p in sys.path:
        q = os.path.abspath(p or ".")
        if q == _wt:
            continue
        if _main and q == os.path.abspath(_main):
            continue
        if q.endswith(os.sep + "src") and os.path.isdir(os.path.join(q, "lisatools")):
            continue
        _keep.append(p)
    sys.path[:] = [_wt] + _keep
