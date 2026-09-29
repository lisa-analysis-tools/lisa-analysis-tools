#!/usr/bin/env python
"""Rebuild a TORN ``global_fit/recipe`` group in a global-fit store.

3-month job 666: the store resumed, loaded every chain and sub-backend and
reached "initial log likelihood", then died in ``add_recipe`` with

    KeyError: 'Unable to synchronously open object (message not aligned)'

which is HDF5 reporting a malformed OBJECT HEADER on that one group. A
recipe step group is tiny, but its attrs are rewritten at every stage
transition, and an attr write rewrites the header -- so a kill or
MPI-abort in that window tears it while the multi-GB datasets around it
stay perfectly readable.

⚠ THE RUNNING BACKUP COPY IS NOT A WAY OUT, and that is not bad luck:
``_atomic_backup_copy`` uses ``shutil.copyfile`` (a BYTE copy) and then
validates with ``_validate_resume_readable``, which walks ONLY
``global_fit/sub_backend``. It never opens the recipe group. So a torn
recipe header is copied faithfully into the backup, passes validation,
and ``promote_backup_if_store_unreadable`` never fires either.

We do not need it. The recipe group holds five tiny groups whose correct
contents are known, so we REBUILD rather than restore -- which costs zero
stored iterations.

RUN WITH THE JOB DOWN. Works on a copy and never modifies the original.

    python repair_recipe_group.py /path/to/gf_prod_3mo_testing.h5
    python repair_recipe_group.py /path/to/store.h5 --apply   # swap it in
"""
import argparse
import os
import shutil
import sys

import h5py

# The live recipe, in order. ``order num`` is 1-based and add_recipe
# asserts it equals the step's index in the in-memory recipe + 1, so the
# ORDER here must match the launcher's stage list exactly.
RECIPE = [
    ("noise_search", True),
    ("gb_search_1", False),
    ("gb_search_2", False),
    ("gb_search_3", False),
    ("full_pe", False),
]
# Steps that finished, and the stored iteration they finished at. Only
# noise_search has ever completed on this run (measured at store
# iterations 106 and 115; identical both times).
COMPLETED_ITERATION = {"noise_search": 24}


def rebuild(path):
    """Delete the torn group and write a fresh, correct one."""
    with h5py.File(path, "a") as f:
        root = f["global_fit"]
        if "recipe" in root:
            # Unlinking does not read the torn object header. If HDF5 still
            # refuses, the object is unreachable in place and you need the
            # salvage-copy route in the docstring of --help.
            del root["recipe"]
        grp = root.create_group("recipe")
        for i, (name, status) in enumerate(RECIPE, start=1):
            step = grp.create_group(name)
            step.attrs["status"] = bool(status)
            step.attrs["order num"] = i
            if name in COMPLETED_ITERATION:
                step.attrs["completed_iteration"] = int(
                    COMPLETED_ITERATION[name])
        root.attrs["has_recipe"] = True


def verify(path):
    """Read it back exactly the way ``add_recipe`` will."""
    with h5py.File(path, "r") as f:
        root = f["global_fit"]
        it = int(root.attrs["iteration"])
        assert bool(root.attrs["has_recipe"]) is True, "has_recipe not set"
        grp = root["recipe"]
        out = []
        for i, (name, status) in enumerate(RECIPE, start=1):
            assert name in grp, f"{name} missing"
            step = grp[name]                      # the read that was failing
            assert bool(step.attrs["status"]) == status, name
            assert int(step.attrs["order num"]) == i, name
            out.append((name, bool(step.attrs["status"]),
                        int(step.attrs["completed_iteration"])
                        if "completed_iteration" in step.attrs else None))
        # and confirm the rest of the store still reads
        ll = root["log_like"][it - 1, 0, 0, :]
        return it, out, ll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--apply", action="store_true",
                    help="swap the repaired file into place (keeps the "
                         "original as <store>_CORRUPT.h5)")
    a = ap.parse_args()

    src = os.path.abspath(a.store)
    work = src[:-3] + "_REPAIRED.h5"
    print(f"[repair] copying {src}\n      -> {work}")
    shutil.copyfile(src, work)

    print("[repair] rebuilding global_fit/recipe ...")
    rebuild(work)

    print("[repair] verifying ...")
    it, steps, ll = verify(work)
    print(f"[repair] OK. iteration={it}")
    for name, status, comp in steps:
        print(f"           {name:<14} status={status!s:<5} "
              f"completed_iteration={comp}")
    print(f"[repair] cold log_like at row {it - 1}: {ll}")

    if not a.apply:
        print(f"\n[repair] DRY RUN -- repaired file left at {work}\n"
              f"         rerun with --apply to swap it in.")
        return 0

    corrupt = src[:-3] + "_CORRUPT.h5"
    print(f"[repair] keeping the original as {corrupt}")
    os.replace(src, corrupt)
    os.replace(work, src)
    print(f"[repair] {src} is repaired. Relaunch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
