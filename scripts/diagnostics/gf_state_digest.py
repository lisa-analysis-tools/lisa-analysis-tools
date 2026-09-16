#!/usr/bin/env python
"""Cluster-gate state digest: per-array sha1 of a global-fit HDF store.

``python scripts/diagnostics/gf_state_digest.py <store.h5> [<store2.h5> ...]``
opens each store with :class:`~lisatools.globalfit.hdfbackend.GFHDFBackend`,
takes ``backend.get_last_sample()``, and prints one line per array as
``name shape sha1`` (name-sorted) -- the cluster-gate tool for diffing two
runs' final state bit-for-bit (Plan 5 Task 4 of the multi-rank walker-block
port; see ``docs/multirank-cluster-gates.md`` Step 1). With two (or more)
stores it also prints a diff summary of which array names differ against
the first store, and exits 1 if any store differs (0 otherwise).

Sub-state arrays (GB band_info, per-leaf temperature ladders, ...) are
included when a branch name matches this script's small built-in registry
below (``gb``/``vgb`` -> GB, ``mbh`` -> MBH, ``emri`` -> EMRI, ``sobbh`` ->
SOBBH; any other branch found under ``sub_backend/`` in the file falls back
to the generic ``ModuleSubBackend``/``ModuleSubState``). A plain HDF5 file
does not self-describe which Python classes wrote its sub-backends -- that
mapping lives on the run's config, not the file -- so a branch whose data
does not fit the guessed class is skipped with a one-line warning on
stderr rather than crashing the whole digest.
"""
from __future__ import annotations

import hashlib
import sys

import h5py
import numpy as np

from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GBHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
    ModuleSubBackend,
    SOBBHHDFBackend,
)
from lisatools.globalfit.state import EMRIState, GBState, MBHState, ModuleSubState, SOBBHState
from lisatools.utils.utility import asnumpy

#: branch name -> (sub-backend class, sub-state class); any other branch
#: found under ``sub_backend/`` in the file falls back to the generic pair.
_KNOWN_SUB_BACKENDS = {
    "gb": (GBHDFBackend, GBState),
    "vgb": (GBHDFBackend, GBState),
    "mbh": (MBHHDFBackend, MBHState),
    "emri": (EMRIHDFBackend, EMRIState),
    "sobbh": (SOBBHHDFBackend, SOBBHState),
}
_GENERIC_SUB_BACKEND = (ModuleSubBackend, ModuleSubState)


def array_sha1(arr) -> str:
    """``sha1(16)`` of an array's raw bytes, host-side (device arrays pulled)."""
    return hashlib.sha1(np.ascontiguousarray(asnumpy(arr)).tobytes()).hexdigest()[:16]


def _discover_sub_backend_names(path):
    """Branch names with a ``sub_backend/<name>`` group in this store, if any."""
    try:
        with h5py.File(path, "r") as f:
            for group_name in ("global_fit", "mcmc"):  # "mcmc" = pre-rename legacy
                if group_name in f and "sub_backend" in f[group_name]:
                    return list(f[group_name]["sub_backend"].keys())
    except OSError:
        pass
    return []


def open_backend(path):
    """A :class:`GFHDFBackend` for ``path`` with sub-backends wired in for
    every branch this script recognizes (best-effort; see module docstring).
    """
    names = _discover_sub_backend_names(path)
    if not names:
        return GFHDFBackend(path)
    sub_backend = {}
    sub_state_bases = {}
    for name in names:
        backend_cls, state_cls = _KNOWN_SUB_BACKENDS.get(name, _GENERIC_SUB_BACKEND)
        sub_backend[name] = backend_cls
        sub_state_bases[name] = state_cls
    return GFHDFBackend(path, sub_backend=sub_backend, sub_state_bases=sub_state_bases)


def state_arrays(state):
    """``{name: array}`` for every array on a :class:`GFState` last-sample.

    ``log_like``/``log_prior``/``betas`` at top level; ``coords/<branch>``/
    ``inds/<branch>`` per branch; ``substate/<branch>/<array>`` for every
    array a branch's sub-state exposes via ``static_arrays()`` /
    ``storage_arrays()`` (skipped, with a warning on stderr, for a branch
    whose sub-state does not match the class this script guessed for it).
    """
    out = {}
    for base_name in ("log_like", "log_prior", "betas"):
        arr = getattr(state, base_name, None)
        if arr is not None:
            out[base_name] = asnumpy(arr)
    for name, arr in state.branches_coords.items():
        out[f"coords/{name}"] = asnumpy(arr)
    for name, arr in state.branches_inds.items():
        out[f"inds/{name}"] = asnumpy(arr)
    sub_states = getattr(state, "sub_states", None) or {}
    for name, sub in sub_states.items():
        if sub is None:
            continue
        try:
            arrays = dict(sub.static_arrays())
            arrays.update(sub.storage_arrays())
        except Exception as exc:  # noqa: BLE001 - best-effort, never abort the digest
            print(f"# WARNING: substate {name!r} unreadable ({exc}); skipped", file=sys.stderr)
            continue
        for arr_name, arr in arrays.items():
            out[f"substate/{name}/{arr_name}"] = asnumpy(arr)
    return out


def digest_store(path):
    """``{array name: (shape, sha1)}`` for one store's last sample, name-sorted."""
    backend = open_backend(path)
    state = backend.get_last_sample()
    arrays = state_arrays(state)
    return {
        name: (tuple(np.shape(arrays[name])), array_sha1(arrays[name]))
        for name in sorted(arrays)
    }


def _print_digest(label, digest):
    print(f"=== {label} ===")
    for name in sorted(digest):
        shape, sha1 = digest[name]
        print(f"{name} {shape} {sha1}")


def main(argv):
    if not argv:
        print("usage: gf_state_digest.py <store.h5> [<store2.h5> ...]", file=sys.stderr)
        return 2
    digests = [digest_store(path) for path in argv]
    for path, digest in zip(argv, digests):
        _print_digest(path, digest)
    if len(digests) < 2:
        return 0
    mismatch = False
    for other_path, other_digest in zip(argv[1:], digests[1:]):
        names = sorted(set(digests[0]) | set(other_digest))
        diffs = [name for name in names if digests[0].get(name) != other_digest.get(name)]
        print(f"\n=== diff: {argv[0]} vs {other_path} ===")
        if diffs:
            mismatch = True
            for name in diffs:
                print(f"  {name}: {digests[0].get(name)} != {other_digest.get(name)}")
        else:
            print("  identical")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
