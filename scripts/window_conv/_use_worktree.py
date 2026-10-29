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

Native backends from the worktree (opt-in)
------------------------------------------
By default the compiled backends still come from the shared site-packages
(HEAD C++). With ``WCONV_NATIVE=1`` in the environment the
``lisatools_backend_cpu`` / ``lisatools_backend_cuda12x`` entries are also
stripped from the editable finder and the isolated prefix built by
``build_native.sh`` (``scripts/window_conv/.wconv_env/site``) is put on
``sys.path``; both ``pycppdetector`` modules are imported eagerly and asserted
to live in that prefix (``NATIVE`` is then True). The env var is the only
supported switch: ``import lisatools`` eagerly imports
``lisatools_backend_cuda12x.pycppdetector``, so ``use_native()`` cannot be
called after this module has been imported (it raises).
"""

from __future__ import annotations

import os
import sys

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKTREE_SRC = os.path.join(WORKTREE, "src")
_PKG = "lisatools"
NATIVE_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wconv_env", "site")
NATIVE_BACKENDS = ("lisatools_backend_cpu", "lisatools_backend_cuda12x")
NATIVE = False


def _is_lisatools(name: str) -> bool:
    return name == _PKG or name.startswith(_PKG + ".")


def _is_native_backend(name: str) -> bool:
    return any(name == b or name.startswith(b + ".") for b in NATIVE_BACKENDS)


def _neutralize_editable_finder(pred=_is_lisatools) -> int:
    """Drop every module name matching ``pred`` from the editable finder(s)."""
    n_patched = 0
    for finder in sys.meta_path:
        if type(finder).__name__ != "ScikitBuildRedirectingFinder":
            continue
        ksf = getattr(finder, "known_source_files", None) or {}
        kwf = getattr(finder, "known_wheel_files", None) or {}
        if not any(pred(k) for k in (*ksf, *kwf)):
            continue
        finder.known_source_files = {k: v for k, v in ksf.items() if not pred(k)}
        finder.known_wheel_files = {k: v for k, v in kwf.items() if not pred(k)}
        finder.submodule_search_locations = {
            k: v for k, v in finder.submodule_search_locations.items() if not pred(k)
        }
        finder.pkgs = frozenset(p for p in finder.pkgs if not pred(p))
        n_patched += 1
    return n_patched


def use_native() -> None:
    """Load the compiled backends from the worktree build (``build_native.sh``)."""
    global NATIVE
    if NATIVE:
        return
    loaded = sorted(m for m in sys.modules if _is_native_backend(m))
    if loaded:
        raise ImportError(
            f"use_native() called after {loaded} were imported (import lisatools loads them); "
            "set WCONV_NATIVE=1 before importing _use_worktree instead."
        )
    for b in NATIVE_BACKENDS:
        so_dir = os.path.join(NATIVE_PREFIX, b)
        if not os.path.isdir(so_dir) or not any(
            f.startswith("pycppdetector.") for f in os.listdir(so_dir)
        ):
            raise ImportError(f"{so_dir} has no pycppdetector; run scripts/window_conv/build_native.sh")
    _neutralize_editable_finder(_is_native_backend)
    if NATIVE_PREFIX in sys.path:
        sys.path.remove(NATIVE_PREFIX)
    sys.path.insert(0, NATIVE_PREFIX)
    import importlib

    for b in NATIVE_BACKENDS:
        mod = importlib.import_module(b + ".pycppdetector")
        f = os.path.abspath(mod.__file__)
        assert f.startswith(os.path.join(NATIVE_PREFIX, b) + os.sep), (
            f"{b}.pycppdetector resolved to {f}, not the worktree build in {NATIVE_PREFIX}"
        )
    NATIVE = True


if any(_is_lisatools(m) for m in sys.modules):
    _bad = sorted(m for m in sys.modules if _is_lisatools(m))[:3]
    _f = getattr(sys.modules.get(_PKG), "__file__", "") or ""
    if not _f.startswith(WORKTREE_SRC):
        raise ImportError(
            f"_use_worktree imported after lisatools was already loaded from {_f!r} "
            f"({_bad}); import it first."
        )

N_FINDERS_PATCHED = _neutralize_editable_finder()
if os.environ.get("WCONV_NATIVE", "0") not in ("", "0"):
    use_native()  # inserts NATIVE_PREFIX at sys.path[0]; WORKTREE_SRC goes in front below
if sys.path[:1] != [WORKTREE_SRC]:
    if WORKTREE_SRC in sys.path:
        sys.path.remove(WORKTREE_SRC)
    sys.path.insert(0, WORKTREE_SRC)

import lisatools  # noqa: E402

assert os.path.abspath(lisatools.__file__).startswith(WORKTREE_SRC + os.sep), (
    f"lisatools resolved to {lisatools.__file__}, not the worktree {WORKTREE_SRC}"
)
