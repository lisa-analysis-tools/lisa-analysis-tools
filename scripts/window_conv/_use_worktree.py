"""Make ``import lisatools`` resolve to this worktree's ``src/lisatools``.

The shared venv has an editable scikit-build-core install of the *main* tree
(``site-packages/_editable_skbc_lisaanalysistools.{pth,py}``): a
``ScikitBuildRedirectingFinder`` sits at the front of ``sys.meta_path`` and maps
every ``lisatools.*`` module name to a file under
``erebor_org_setup/lisa-analysis-tools/src``. This shim

1. strips all ``lisatools`` / ``lisatools.*`` entries from that finder's
   ``known_source_files`` and ``submodule_search_locations`` (the compiled
   ``lisatools_backend_cpu`` / ``lisatools_backend_cuda12x`` wheel entries are
   left alone, so the HEAD C++ backends still load from site-packages);
2. puts ``<worktree>/src`` at the front of ``sys.path`` so the regular path
   finder picks up the worktree package;
3. imports ``lisatools`` and asserts it came from the worktree.

Import this module FIRST (before anything that imports lisatools, gbgpu, ...).
Nothing on disk is modified; the effect is per-process.
"""

from __future__ import annotations

import os
import sys

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKTREE_SRC = os.path.join(WORKTREE, "src")
_PKG = "lisatools"


def _is_lisatools(name: str) -> bool:
    return name == _PKG or name.startswith(_PKG + ".")


def _neutralize_editable_finder() -> int:
    n_patched = 0
    for finder in sys.meta_path:
        if type(finder).__name__ != "ScikitBuildRedirectingFinder":
            continue
        ksf = getattr(finder, "known_source_files", None)
        if not ksf or not any(_is_lisatools(k) for k in ksf):
            continue
        finder.known_source_files = {k: v for k, v in ksf.items() if not _is_lisatools(k)}
        finder.submodule_search_locations = {
            k: v for k, v in finder.submodule_search_locations.items() if not _is_lisatools(k)
        }
        finder.pkgs = frozenset(p for p in finder.pkgs if not _is_lisatools(p))
        n_patched += 1
    return n_patched


if any(_is_lisatools(m) for m in sys.modules):
    _bad = sorted(m for m in sys.modules if _is_lisatools(m))[:3]
    _f = getattr(sys.modules.get(_PKG), "__file__", "") or ""
    if not _f.startswith(WORKTREE_SRC):
        raise ImportError(
            f"_use_worktree imported after lisatools was already loaded from {_f!r} "
            f"({_bad}); import it first."
        )

N_FINDERS_PATCHED = _neutralize_editable_finder()
if sys.path[:1] != [WORKTREE_SRC]:
    if WORKTREE_SRC in sys.path:
        sys.path.remove(WORKTREE_SRC)
    sys.path.insert(0, WORKTREE_SRC)

import lisatools  # noqa: E402

assert os.path.abspath(lisatools.__file__).startswith(WORKTREE_SRC + os.sep), (
    f"lisatools resolved to {lisatools.__file__}, not the worktree {WORKTREE_SRC}"
)
