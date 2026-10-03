"""Print the last saved iteration of a global-fit run folder. Nothing else.

    python scripts/diagnostics/gf_last_iteration.py /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu
    82

``-v`` adds the store it read and what the last row was saved after::

    python scripts/diagnostics/gf_last_iteration.py <run_dir> -v
    82  gf_prod_6mo_testing.h5  saved_after=noise_ratchet_search

Reads one attribute (``global_fit/iteration``) without file locking, so it is
safe on a store the saver is writing to. Picks the run's live store: the
newest ``*.h5`` in the folder that is not a running backup, a pre-migration
copy or a quarantined file. A path to an ``.h5`` works too.
"""

import glob
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SKIP = ("backup", "CORRUPT", "pre_rerung", "midit", ".stale")


def find_store(path):
    if os.path.isfile(path):
        return path
    cands = [p for p in glob.glob(os.path.join(path, "*.h5"))
             if not any(s in os.path.basename(p) for s in SKIP)]
    if not cands:
        raise FileNotFoundError(f"no run store (*.h5) in {path}")
    return max(cands, key=os.path.getmtime)


def last_iteration(path, group="global_fit"):
    """``(iteration, store_path, saved_after)``; ``saved_after`` is None when absent."""
    import h5py

    store = find_store(path)
    with h5py.File(store, "r") as f:
        g = f[group] if group in f else f
        it = int(g.attrs["iteration"])
        after = None
        if it > 0 and "saved_after" in g and g["saved_after"].shape[0] >= it:
            raw = g["saved_after"][it - 1]
            after = raw.decode() if isinstance(raw, bytes) else str(raw)
    return it, store, after


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv
    args = [a for a in argv if a != "-v"]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: gf_last_iteration.py <run_dir | store.h5> [-v]")
        return 2
    it, store, after = last_iteration(args[0])
    if verbose:
        print(f"{it}  {os.path.basename(store)}  saved_after={after}")
    else:
        print(it)
    return 0


if __name__ == "__main__":
    sys.exit(main())
