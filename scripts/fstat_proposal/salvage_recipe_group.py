#!/usr/bin/env python
"""Salvage a store whose ``global_fit/recipe`` has a TORN step header.

6mo v9, 2026-10-04: replica_pe converged at stored iteration 139 and the
recipe transition rewrote ``global_fit/recipe/replica_pe``'s attrs; that
object header is now unreadable ("message not aligned") while every other
object, including the last row of every dataset, reads fine. Snapshots fail
("object visitation failed") because they visit every object.

WHY A SALVAGE COPY AND NOT ``del``. Unlinking the torn group (or its parent)
makes HDF5 decrement the torn object's reference count, which lives in the
very header that is damaged. Copying everything EXCEPT the recipe group into
a fresh file never touches it. ``Group.copy`` is ``H5Ocopy``: chunking,
filters, dtypes and unlimited max shapes come across unchanged.

The recipe is then REBUILT from the readable step groups' own attrs plus the
explicit ``--set`` patches for the torn one, in ``--order`` (``order num`` =
1-based position, which ``add_recipe`` asserts).

NEVER modifies the original. Default is a dry run; ``--apply`` writes
``<store>_salvaged.h5`` beside it and verifies it object by object (shape,
dtype, max shape, chunks, compression, attrs, last row of every growing
dataset) and reads the recipe back the way ``add_recipe`` does. Swapping it
in is a manual ``mv`` it prints.

    python salvage_recipe_group.py STORE.h5 \\
        --order gb_search_seed,gb_search_1,gb_search_2,gb_search_3,replica_pe,full_pe \\
        --set replica_pe:status=true,completed_iteration=139 \\
        --set full_pe:status=false,start_iteration=139 --apply
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

RECIPE = "recipe"
ROOT = "global_fit"


def _parse_value(v):
    lv = v.strip().lower()
    if lv in ("true", "false"):
        return lv == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_sets(items):
    """``["name:k=v,k=v", ...]`` -> ``{name: {k: v}}``."""
    out = {}
    for it in items or []:
        name, _, rest = it.partition(":")
        if not name or not rest:
            raise ValueError(f"--set wants NAME:key=value[,key=value], got {it!r}")
        d = out.setdefault(name.strip(), {})
        for kv in rest.split(","):
            k, _, v = kv.partition("=")
            if not k or not _:
                raise ValueError(f"bad key=value {kv!r} in {it!r}")
            d[k.strip()] = _parse_value(v)
    return out


def read_steps(src_path, order):
    """Readable step attrs, and the steps that could not be read."""
    attrs, unreadable = {}, []
    with h5py.File(src_path, "r") as f:
        rec = f[ROOT][RECIPE]
        for name in order:
            try:
                attrs[name] = dict(rec[name].attrs)
            except Exception as e:  # noqa: BLE001 -- the torn one, by design
                unreadable.append((name, repr(e)))
                attrs[name] = {}
        extra = [k for k in rec.keys() if k not in order]
    return attrs, unreadable, extra


def plan_recipe(attrs, order, sets):
    plan = {}
    for i, name in enumerate(order, start=1):
        a = dict(attrs.get(name, {}))
        a.update(sets.get(name, {}))
        a["order num"] = i
        if "status" not in a:
            raise ValueError(f"step {name!r} has no readable status and no --set for it")
        plan[name] = a
    return plan


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def salvage(src_path, dst_path, plan):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        _copy_attrs(src, dst)
        for top in src.keys():
            if top != ROOT:
                src.copy(src[top], dst, name=top)
        sg = src[ROOT]
        dg = dst.create_group(ROOT)
        _copy_attrs(sg, dg)
        for k in sg.keys():
            if k == RECIPE:
                continue                      # never opened: holds the torn header
            sg.copy(sg[k], dg, name=k)
        rec = dg.create_group(RECIPE)
        for name, a in plan.items():
            g = rec.create_group(name)
            for k, v in a.items():
                g.attrs[k] = v
        dg.attrs["has_recipe"] = True


def _same_attrs(a, b):
    if set(a.attrs.keys()) != set(b.attrs.keys()):
        return False
    for k in a.attrs.keys():
        x, y = a.attrs[k], b.attrs[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        elif x != y:
            return False
    return True


def verify(src_path, dst_path, plan):
    """Every object outside the recipe identical in layout, attrs and last row."""
    problems = []
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "r") as dst:
        it = int(src[ROOT].attrs["iteration"])

        def walk(sg, dg, path):
            if not _same_attrs(sg, dg):
                problems.append(f"{path}: attrs differ")
            for k in sg.keys():
                if path == f"/{ROOT}" and k == RECIPE:
                    continue
                q = f"{path}/{k}"
                if k not in dg:
                    problems.append(f"{q}: missing"); continue
                so, do = sg[k], dg[k]
                if isinstance(so, h5py.Group):
                    walk(so, do, q); continue
                for prop in ("shape", "dtype", "maxshape", "chunks", "compression",
                             "compression_opts"):
                    if getattr(so, prop) != getattr(do, prop):
                        problems.append(f"{q}: {prop} {getattr(so, prop)} != {getattr(do, prop)}")
                if not _same_attrs(so, do):
                    problems.append(f"{q}: attrs differ")
                if so.ndim and so.maxshape[0] is None and so.shape[0] >= it > 0:
                    if not np.array_equal(so[it - 1], do[it - 1], equal_nan=so.dtype.kind == "f"):
                        problems.append(f"{q}[{it - 1}]: last row differs")

        walk(src, dst, "")
        rec = dst[ROOT][RECIPE]
        assert bool(dst[ROOT].attrs["has_recipe"]) is True
        for i, (name, a) in enumerate(plan.items(), start=1):
            g = rec[name]
            assert int(g.attrs["order num"]) == i, name
            assert bool(g.attrs["status"]) == bool(a["status"]), name
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store")
    ap.add_argument("--order", required=True,
                    help="comma-separated step names in recipe order")
    ap.add_argument("--set", action="append", default=[],
                    help="NAME:key=value[,key=value] patch (repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="write <store>_salvaged.h5 and verify it")
    args = ap.parse_args(argv)
    order = [s.strip() for s in args.order.split(",") if s.strip()]
    sets = parse_sets(args.set)
    attrs, unreadable, extra = read_steps(args.store, order)
    if extra:
        print(f"refusing: the store's recipe has steps not in --order: {extra}")
        return 2
    plan = plan_recipe(attrs, order, sets)
    for name, err in unreadable:
        print(f"UNREADABLE {name}: {err[:120]}")
    for name, a in plan.items():
        shown = {k: v for k, v in a.items() if k != "move_order"}
        print(f"plan {name}: {shown}")
    dst = args.store[:-3] + "_salvaged.h5"
    if not args.apply:
        print(f"dry run; --apply writes {dst}")
        return 0
    if os.path.exists(dst):
        print(f"refusing: {dst} exists")
        return 2
    salvage(args.store, dst, plan)
    problems = verify(args.store, dst, plan)
    if problems:
        print(f"VERIFY FAILED ({len(problems)}):")
        for p in problems[:40]:
            print("  ", p)
        return 1
    print(f"OK: {dst} matches the store outside the recipe; recipe rebuilt.")
    torn = args.store[:-3] + "_TORN.h5"
    print("swap it in with the job DOWN:")
    print(f"  mv {args.store} {torn} && mv {dst} {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
