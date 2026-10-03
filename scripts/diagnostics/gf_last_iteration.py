"""Print the last saved iteration of a global-fit run folder and the stage it is in.

    python scripts/diagnostics/gf_last_iteration.py /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu
    82  stage=gb_search_3

``-v`` adds the stage's place in the recipe, where it started, the last row's
leg, the ratchet stamp and the store it read::

    python scripts/diagnostics/gf_last_iteration.py <run_dir> -v
    82  stage=gb_search_3 (step 4 of 5; started at row 47, 35 rows in; done: gb_search_seed, gb_search_1, gb_search_2)  saved_after=noise_ratchet_search  store=gf_prod_6mo_testing.h5

The stage is the first recipe step (by ``order num``) whose ``status`` is not
complete -- what a resume would run. Reads attrs only, without file locking,
so it is safe on a store the saver is writing to. Picks the run's live store:
the newest ``*.h5`` in the folder that is not a running backup, a
pre-migration copy or a quarantined file. A path to an ``.h5`` works too.
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


def _attr(a, key, default=None):
    v = a.get(key, default)
    if v is None:
        return None
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:  # noqa: BLE001
        return v


def current_stage(g):
    """The recipe step a resume would run: ``dict(name, order, total, start, done, ratchet_done)``.

    ``None`` when the store carries no recipe group. ``name`` is None when
    every step is complete (the recipe has finished).
    """
    if "recipe" not in g:
        return None
    steps = []
    for name in g["recipe"]:
        a = g["recipe"][name].attrs
        steps.append(dict(name=name, order=int(_attr(a, "order num", 0) or 0),
                          status=bool(_attr(a, "status", False)),
                          start=_attr(a, "start_iteration"),
                          completed=_attr(a, "completed_iteration"),
                          ratchet_done=_attr(a, "galfor_ratchet_done")))
    steps.sort(key=lambda s: s["order"])
    done = [s["name"] for s in steps if s["status"]]
    cur = next((s for s in steps if not s["status"]), None)
    return dict(name=None if cur is None else cur["name"],
                order=None if cur is None else cur["order"], total=len(steps),
                start=None if cur is None else cur["start"],
                ratchet_done=None if cur is None else cur["ratchet_done"], done=done)


def last_iteration(path, group="global_fit"):
    """``(iteration, store_path, saved_after, stage)``; ``saved_after`` / ``stage`` None when absent."""
    import h5py

    store = find_store(path)
    with h5py.File(store, "r") as f:
        g = f[group] if group in f else f
        it = int(g.attrs["iteration"])
        after = None
        if it > 0 and "saved_after" in g and g["saved_after"].shape[0] >= it:
            raw = g["saved_after"][it - 1]
            after = raw.decode() if isinstance(raw, bytes) else str(raw)
        stage = current_stage(g)
    return it, store, after, stage


def describe_stage(it, stage, verbose):
    if stage is None:
        return "stage=none (no recipe group)"
    if stage["name"] is None:
        return f"stage=FINISHED (all {stage['total']} steps complete)"
    s = f"stage={stage['name']}"
    if not verbose:
        return s
    bits = [f"step {stage['order']} of {stage['total']}"]
    if stage["start"] is not None:
        bits.append(f"started at row {int(stage['start'])}, {it - int(stage['start'])} rows in")
    if stage["ratchet_done"]:
        bits.append("galfor ratchet DONE (stamped)")
    if stage["done"]:
        bits.append("done: " + ", ".join(stage["done"]))
    return s + " (" + "; ".join(bits) + ")"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv
    args = [a for a in argv if a != "-v"]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: gf_last_iteration.py <run_dir | store.h5> [-v]")
        return 2
    it, store, after, stage = last_iteration(args[0])
    line = f"{it}  {describe_stage(it, stage, verbose)}"
    if verbose:
        line += f"  saved_after={after}  store={os.path.basename(store)}"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
