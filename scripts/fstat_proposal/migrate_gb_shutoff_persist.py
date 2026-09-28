"""Give a global-fit store the per-(walker, band) RJ shutoff datasets.

WHY THIS EXISTS. ``PerBranchHDFBackend.reset`` creates a branch's datasets
ONCE, from a template, when the file is first written. ``save_step`` then
does::

    for name, arr in arrays.items():
        if name not in grp:
            continue                      # <-- silent
        grp[name][iteration] = arr

so any band array added to the code AFTER a store was created is dropped on
every save, with no warning, forever. That is what happened to the whole
per-(walker, band) RJ shutoff family: the v9 stores were created before it
existed, and nothing has ever reached disk.

Measured on ``gf_prod_6mo_v9_4gpu`` (2026-09-27): the ``sub_backend/gb``
group holds ``band_rj_shutoff`` (the 1-D per-band valve) but NONE of
``band_rj_shutoff_w``, ``band_shutoff_w_step``, ``band_shutoff_best_w``,
``band_shutoff_streak_w``. The run keeps the valve alive across restarts
only through the mid-iteration checkpoint pickle; the h5 -- which is what
the monitor reads and what a clean resume reads -- has never seen it.

WHAT IT WRITES, in place after a ``.bak`` copy, into
``<name>/sub_backend/gb``:

* ``band_rj_shutoff_w``     (step, nw, nb) bool    -> False   (all open)
* ``band_shutoff_w_step``   (step, 1)      int64   -> -1      (no step yet)
* ``band_shutoff_best_w``   (step, nw, nb) float64 -> -inf    (empty window)
* ``band_shutoff_streak_w`` (step, nw, nb) int64   -> 0       (empty window)
* ``band_cold_logl_max_w``  (step, nw, nb) float64 -> -inf    (no history)
* ``band_cold_logl_w``      (step, nw, nb) float64 -> -inf    (telemetry)
* ``band_shutoff_reset_w``  (step, nw, nb) int64   -> 0       (telemetry)
* ``band_cold_logl_peak_w`` (step, nw, nb) float64 -> -inf    (cycle peak)

``--with-stage`` adds the per-(walker, band) search-stage latch
(``band_stage_w`` / ``band_stage_occ_last_w`` / ``band_stage_streak_w`` /
``band_stage``) and ``--with-cap-w`` the per-walker leaf-cap family, both of
which are missing from the same stores for the same reason. Neither is on by
default: this script's job is the shutoff valve the user asked for, and each
extra family is a separate behaviour change to opt into knowingly.

BACKFILL IS NEUTRAL, NOT INVENTED. Every historical row gets the fresh-state
value. The migration cannot know what the valve held at iteration 40 and
does not pretend to: what matters is that the datasets EXIST, so that from
the next save onward the live values are written. The one row anything reads
back is the last one, and a neutral last row means "all open, window empty"
-- the permissive direction, which cannot lose a source. The run re-earns
one window (``GB_SEARCH_BAND_SHUTOFF_CONV_ITER`` + 1 iterations) and then
keeps it across every later restart, which is the whole point.

``maxshape[0]`` is ``None`` on every dataset created here, because
``PerBranchHDFBackend.grow`` resizes every non-static dataset in the group
along axis 0. A fixed first axis would make the next ``grow()`` raise and
take the run down.

RUN IT WITH THE JOB DOWN. HDF5 file locking will normally refuse a writer
while the run holds the file, but a store on a filesystem with locking
disabled would let two writers in and corrupt it. The script refuses a file
touched within ``--live-window`` seconds unless ``--force`` is given.

Usage::

    # one store
    python scripts/fstat_proposal/migrate_gb_shutoff_persist.py \\
        /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5

    # both production runs, look first
    python scripts/fstat_proposal/migrate_gb_shutoff_persist.py --dry-run \\
        /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5 \\
        /shared/data/global_fit_output/gf_prod_3mo_v9_2gpu/gf_prod_3mo_testing.h5

Idempotent: a dataset that already exists is left exactly as it is and
reported as ``present``.
"""
import argparse
import os
import shutil
import sys
import time

import h5py
import numpy as np

#: ``(name, axes, dtype, fill)``. ``axes`` names the per-iteration shape in
#: terms of the group's own attrs, so one table drives both the shape and
#: the validation and the two cannot disagree.
SHUTOFF_FAMILY = (
    ("band_rj_shutoff_w",     ("nwalkers", "num_bands"), np.bool_,   False),
    ("band_shutoff_w_step",   ("one",),                  np.int64,   -1),
    ("band_shutoff_best_w",   ("nwalkers", "num_bands"), np.float64, -np.inf),
    ("band_shutoff_streak_w", ("nwalkers", "num_bands"), np.int64,   0),
    # The ALL-TIME cold-chain lnL max per (walker, band). Never released,
    # not by a recipe step and not by a restart -- it is the reference the
    # valve judges against (user ruling 2026-09-27).
    ("band_cold_logl_max_w",  ("nwalkers", "num_bands"), np.float64, -np.inf),
    # THIS iteration's per-(walker, sub-band) cold lnL. Pure telemetry --
    # nothing reads it back -- but it is the only recorded history of the
    # statistic the valve judges, and without it "have the sub-band lnLs
    # converged?" cannot be answered from a snapshot at all.
    ("band_cold_logl_w",      ("nwalkers", "num_bands"), np.float64, -np.inf),
    # How often each pair has had its streak zeroed -- separates "keeps
    # resetting" from "simply still young".
    ("band_shutoff_reset_w",  ("nwalkers", "num_bands"), np.int64,   0),
    # WITHIN-ITERATION peak, accumulated at the end of every in-model
    # repeat group from the sub-band buffer and consumed once per
    # iteration by the gate. This is the array that stops the valve
    # judging against a stale mid-cycle sample.
    ("band_cold_logl_peak_w", ("nwalkers", "num_bands"), np.float64, -np.inf),
)

STAGE_FAMILY = (
    # COARSE is 0, which is also the fresh value: a store that never
    # measured the latch must not hand a walker the relaxed FINE floor.
    ("band_stage_w",            ("nwalkers", "num_bands"), np.int8,  0),
    ("band_stage_occ_last_w",   ("nwalkers", "num_bands"), np.int64, -1),
    ("band_stage_streak_w",     ("nwalkers", "num_bands"), np.int64, 0),
    ("band_stage",              ("num_bands",),            np.int8,  0),
)

CAP_W_FAMILY = (
    # -1 is the cap's "disarmed" sentinel; the first cap-enabled RJ move
    # arms it to leaf_cap_start, exactly as it does for a fresh state.
    ("cap_cell_leaf_cap_w", ("nwalkers", "num_cap_cells"), np.float64, -1.0),
    ("cap_cell_iters_w",    ("nwalkers", "num_cap_cells"), np.float64, 0.0),
    ("cap_cell_best_ll_w",  ("nwalkers", "num_cap_cells"), np.float64, -np.inf),
    ("band_best_ll_w",      ("nwalkers", "num_bands"),     np.float64, -np.inf),
)


def _root_group(f):
    """The one top-level sampler group (``global_fit`` on every store)."""
    names = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    if len(names) != 1:
        raise SystemExit(
            f"expected exactly one top-level group, found {names!r}. This "
            f"does not look like a global-fit store.")
    return f[names[0]]


def _plan(grp, families):
    """``[(name, per_iter_shape, dtype, fill, status)]`` for one gb group."""
    attrs = {k: int(v) for k, v in grp.attrs.items()
             if np.isscalar(v) or np.ndim(v) == 0}
    attrs["one"] = 1
    # The step axis comes from a SIBLING dataset, never from the file's
    # ``iteration`` attr: grow() pre-allocates rows ahead of the iteration
    # counter, and a short new dataset would desync the group the moment
    # the next save wrote past its end.
    sibling = None
    for cand in ("band_rj_shutoff", "band_occ_streak", "band_leaf_cap"):
        if cand in grp:
            sibling = grp[cand]
            break
    if sibling is None:
        raise SystemExit(
            f"{grp.name}: no per-iteration band dataset to take the step "
            f"axis from; refusing to guess it.")
    nstep = int(sibling.shape[0])

    out = []
    for name, axes, dtype, fill in families:
        try:
            shape = tuple(attrs[a] for a in axes)
        except KeyError as exc:
            raise SystemExit(
                f"{grp.name}: group attr {exc.args[0]!r} is missing, so "
                f"{name!r} cannot be sized. Refusing rather than guessing.")
        if name in grp:
            got = tuple(grp[name].shape[1:])
            status = "present" if got == shape else f"PRESENT-BUT-{got}"
        else:
            status = "create"
        out.append((name, nstep, shape, dtype, fill, status))
    return out


def migrate(path, families, dry_run=False, backup=True, live_window=300.0,
            force=False, compression="gzip", compression_opts=4):
    if not os.path.exists(path):
        raise SystemExit(f"no such store: {path}")
    age = time.time() - os.path.getmtime(path)
    if age < live_window and not force and not dry_run:
        raise SystemExit(
            f"{path} was modified {age:.0f} s ago (< --live-window "
            f"{live_window:.0f} s), so the run is probably still writing "
            f"it. Stop the job and re-run, or pass --force if you are "
            f"certain it is down. Two writers on one HDF5 file corrupt it.")

    with h5py.File(path, "r") as f:
        root = _root_group(f)
        if "sub_backend" not in root or "gb" not in root["sub_backend"]:
            raise SystemExit(f"{path}: no 'sub_backend/gb' group.")
        plan = _plan(root["sub_backend"]["gb"], families)
        iteration = int(root.attrs.get("iteration", -1))

    print(f"\n=== {path}")
    print(f"    iteration attr: {iteration}")
    todo = [p for p in plan if p[-1] == "create"]
    for name, nstep, shape, dtype, fill, status in plan:
        print(f"    {status:>18}  {name:<24} "
              f"({nstep}, {', '.join(str(s) for s in shape)}) "
              f"{np.dtype(dtype).name} fill={fill}")
    odd = [p for p in plan if p[-1].startswith("PRESENT-BUT-")]
    if odd:
        raise SystemExit(
            f"{path}: {[p[0] for p in odd]} already exist with a DIFFERENT "
            f"per-iteration shape than this run's grid implies. That is a "
            f"grid mismatch, not a missing-dataset problem; migrate the "
            f"band grid first (migrate_gb_band_edges.py) and re-run.")
    if not todo:
        print("    nothing to do -- every dataset is already present.")
        return 0
    if dry_run:
        print(f"    [dry run] would create {len(todo)} dataset(s).")
        return len(todo)

    if backup:
        bak = path + ".bak"
        if os.path.exists(bak):
            raise SystemExit(
                f"{bak} already exists; move or delete it first so an "
                f"earlier backup is never silently overwritten.")
        print(f"    copying {os.path.getsize(path) / 1e6:.0f} MB -> {bak}")
        shutil.copy2(path, bak)

    with h5py.File(path, "r+") as f:
        grp = _root_group(f)["sub_backend"]["gb"]
        for name, nstep, shape, dtype, fill, status in todo:
            full = (nstep,) + shape
            # Chunk on ONE step at a time: every write is a single
            # iteration's slab, which is exactly how the backend writes
            # every other band array.
            chunks = (1,) + shape
            # ``fillvalue``, not an explicit write. An unwritten HDF5 chunk
            # is not stored at all and reads back as the fill, so the four
            # datasets add ~0 bytes to a 127 MB store instead of the
            # ~170 MB a materialised (2013, 4, 1232) float64 + int64 pair
            # would cost. -inf round-trips correctly as a fill value
            # (checked on this h5py/HDF5 build before relying on it).
            d = grp.create_dataset(
                name, shape=full, dtype=dtype,
                maxshape=(None,) + shape, chunks=chunks,
                compression=compression, compression_opts=compression_opts,
                fillvalue=np.asarray(fill, dtype=dtype)[()],
            )
            back = d[full[0] - 1]
            ok = (np.all(np.isneginf(back)) if fill == -np.inf
                  else np.all(back == fill))
            if not ok:
                raise SystemExit(
                    f"{name}: the fill did not take (read back "
                    f"{np.ravel(back)[:3]}, wanted {fill}). Refusing to "
                    f"leave a dataset whose unwritten rows read as "
                    f"something other than the fresh state.")
            print(f"    created {name} {full} {np.dtype(dtype).name} "
                  f"fill={fill}")
    print(f"    done: {len(todo)} dataset(s) created.")
    return len(todo)


def make_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stores", nargs="+", help="global-fit .h5 store(s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be created, touch nothing")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy (not recommended)")
    ap.add_argument("--with-stage", action="store_true",
                    help="also add the per-(walker, band) search-stage latch")
    ap.add_argument("--with-cap-w", action="store_true",
                    help="also add the per-walker leaf-cap family")
    ap.add_argument("--live-window", type=float, default=300.0,
                    help="refuse a store modified this recently (seconds)")
    ap.add_argument("--force", action="store_true",
                    help="override the live-store refusal. The job must be "
                         "down; two writers corrupt the file.")
    return ap


def main(argv=None):
    a = make_parser().parse_args(argv)
    families = list(SHUTOFF_FAMILY)
    if a.with_stage:
        families += list(STAGE_FAMILY)
    if a.with_cap_w:
        families += list(CAP_W_FAMILY)
    total = 0
    for path in a.stores:
        total += migrate(path, tuple(families), dry_run=a.dry_run,
                         backup=not a.no_backup,
                         live_window=a.live_window, force=a.force)
    print(f"\n{total} dataset(s) "
          f"{'would be ' if a.dry_run else ''}created across "
          f"{len(a.stores)} store(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
