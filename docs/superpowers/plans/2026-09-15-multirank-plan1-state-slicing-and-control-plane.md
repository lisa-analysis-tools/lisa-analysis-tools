# Multi-rank global fit, Plan 1: walker slicing + control plane (WP0, WP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two dependency-free foundations of the one-rank-per-GPU port: walker-block slicing/merging of `GFState`, and the communication package (rank roles + layout + device pinning, the head-side fan-out + compute-side service loop, and an in-process fake communicator so all of it is unit-tested on a laptop with no MPI).

**Architecture:** Every walker-axis array of the state can be cut into equal static blocks and merged back (ladders copied by value, delta counters summed). A `WalkerBlockLayout` resolves roles (head / compute / saver) and each rank's device list before the build; `WalkerFanout.run(...)` on the head ships per-rank payloads, runs the head's own block, gathers replies and merges, and is a direct call with zero copies when there is one compute rank; `ComputeService.serve()` on compute ranks dispatches commands to `move.gf_serve(...)`. `FakeWorld` runs N ranks as threads in one process with pickle-copy transport so the same code paths are exercised without mpi4py.

**Tech Stack:** Python 3.12, numpy, eryn `State`/`BranchSupplemental`, `unittest`, mpi4py (lazy import only; NOT installed on the laptop), threading/queue/pickle for the fake communicator.

**Spec:** `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` (sections Decisions, Architecture, WP0, WP1). Plans 2+ (run.py integration, recipe, addremove/PSD, GB, scripts) build on the interfaces defined here.

## Global Constraints

- Work ONLY in the worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank` (branch `multirank-walker-blocks`). Never touch the main `dev` checkout.
- **No `git commit` and no `git push`** (user policy: commits happen only on the user's explicit instruction, per session). Every task ends with a "ready to commit" report instead of a commit.
- Tests run through the worktree runner so the editable install of the main tree is shadowed (its meta-path finder beats `PYTHONPATH`):
  ```sh
  cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank
  .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest tests.test_<name> -v; tail -n 25 .wtenv/<name>.log
  ```
  The runner waits for the 1-minute load average to drop below 6, runs under `nice -n 10` with every thread pool pinned to 1, and writes stdout+stderr to the log. One python process at a time (8 GB laptop). CPU only. Tiny fixtures only.
- `mpi4py` is NOT importable on the laptop: no module in this plan may import it at module import time. Import it lazily inside functions and only when the communicator is not a fake.
- Repo rules: `black` line length 100, `isort` black profile; never store an array module (`self.xp = cp`) on an instance; guard any `__getattr__` against dunder probing; no new settings files; env knob = capitalized attribute name.
- New package: `src/lisatools/globalfit/communication/` with `__init__.py`, `walkerslice.py`, `fakecomm.py`, `ranks.py`, `fanout.py`. Nothing else in the tree changes except `state.py` (Task 1) and `moves/globalfitmove.py` (Task 7).
- `state.py` semantics to preserve: `initialize_tempered` allocates the tempered block; `delta_counter_names` are per-iteration deltas zeroed after each save; `betas_attr_name` is `"betas"` on the base (flat ladder) and `"betas_all"` on `PerLeafLadderState` (per-leaf ladder, walker axis LAST on `log_like`/`log_prior`); `GBState.band_info` is NOT part of the tempered block and is never sliced by this plan.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/lisatools/globalfit/state.py` (modify, `ModuleSubState` ~line 396-553, `PerLeafLadderState` ~1283-1350) | `walker_axes`, `_bare_like`, `slice_walkers`, `merge_walkers` on the sub-state classes |
| `src/lisatools/globalfit/communication/__init__.py` (create) | re-exports the public names of the package |
| `src/lisatools/globalfit/communication/walkerslice.py` (create) | `slice_state` / `merge_state` for a whole `GFState` (main branches, supplementals with the `walker_inds` remap, sub-states) |
| `src/lisatools/globalfit/communication/fakecomm.py` (create) | `FakeWorld` / `FakeComm` / `FakeCommNull` / `FakeAbort`: in-process N-rank simulator of the mpi4py subset used |
| `src/lisatools/globalfit/communication/ranks.py` (create) | `RankRole`, `RankPlacement`, `WalkerBlockLayout`, `resolve_roles`, `build_layout`, `select_rank_device`, `prepare_rank`, `derive_rank_seed`, `install_mpi_abort_on_error`, `rank_tag`, `prefix_stdout` |
| `src/lisatools/globalfit/communication/fanout.py` (create) | `WalkerFanout` (head), `ComputeService` (compute ranks), `RemoteWorkerError`, `concat_blocks`, command/reply schemas |
| `src/lisatools/globalfit/moves/globalfitmove.py` (modify, `MoveBuildContext` line 68-89, `GlobalFitMove` line 230) | `layout`/`fanout`/`rank`/`state_local` context fields auto-filled from `ctx.curr`; `GlobalFitMove.gf_serve` default |
| `tests/test_walkerslice_roundtrip.py` (create) | Tasks 1-2 |
| `tests/test_fakecomm.py` (create) | Task 3 |
| `tests/test_rank_layout.py` (create) | Tasks 4-5 |
| `tests/test_fanout_passthrough.py`, `tests/test_fanout_fakecomm.py` (create) | Task 6 |
| `tests/test_move_build_context_layout.py` (create) | Task 7 |

---

### Task 1: Sub-state walker slicing (`ModuleSubState.slice_walkers` / `merge_walkers`)

**Files:**
- Modify: `src/lisatools/globalfit/state.py` (insert after `_copy_tempered_from`, which ends at line 553; and inside `PerLeafLadderState` after `betas_attr_name = "betas_all"` at line 1338)
- Test: `tests/test_walkerslice_roundtrip.py`

**Interfaces:**
- Consumes: `ModuleSubState.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim, coords=None, inds=None)`, `tempered_initialized`, `delta_counter_names`, `betas_attr_name` (existing).
- Produces: `ModuleSubState.walker_axes: dict[str, int]`; `ModuleSubState._bare_like() -> ModuleSubState`; `ModuleSubState.slice_walkers(w0: int, w1: int) -> ModuleSubState`; `ModuleSubState.merge_walkers(part, w0: int, w1: int, *, sum_counters: bool = True) -> None`. `PerLeafLadderState` overrides `walker_axes` (log_like/log_prior axis 2) and `_bare_like` (carries `betas_all` by value).

- [ ] **Step 1: Write the failing test**

Create `tests/test_walkerslice_roundtrip.py`:

```python
"""Walker-block slice/merge round trips for the multi-rank fan-out (WP0)."""

import unittest

import numpy as np
from eryn.state import BranchSupplemental

from lisatools.globalfit.state import GBState, GFState, MBHState, ModuleSubState

NTEMPS, NWALKERS, NLEAVES, NDIM = 3, 6, 2, 4


def _fill(sub, rng):
    """Random-fill every tempered array so a lost or misplaced column is detectable."""
    for name in sub.tempered_array_names:
        arr = getattr(sub, name, None)
        if arr is None:
            continue
        if arr.dtype == bool:
            arr[...] = rng.random(arr.shape) > 0.3
        elif np.issubdtype(arr.dtype, np.integer):
            arr[...] = rng.integers(0, 50, size=arr.shape)
        else:
            arr[...] = rng.standard_normal(arr.shape)


def _make_sub(cls, rng, **kwargs):
    sub = cls(None, **kwargs)
    sub.initialize_tempered(NTEMPS, NWALKERS, NLEAVES, NDIM)
    _fill(sub, rng)
    return sub


class SubStateSliceMergeTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _check_roundtrip(self, sub):
        half = NWALKERS // 2
        left = sub.slice_walkers(0, half)
        right = sub.slice_walkers(half, NWALKERS)

        # slices: right geometry, zeroed delta counters, walker columns copied
        self.assertEqual(left.nwalkers, half)
        self.assertEqual(right.nwalkers, NWALKERS - half)
        for name in sub.delta_counter_names:
            self.assertTrue(np.all(getattr(left, name) == 0), name)
        for name, axis in sub.walker_axes.items():
            src = getattr(sub, name, None)
            if src is None:
                continue
            np.testing.assert_array_equal(
                getattr(left, name), np.take(src, np.arange(0, half), axis=axis), err_msg=name
            )
            np.testing.assert_array_equal(
                getattr(right, name), np.take(src, np.arange(half, NWALKERS), axis=axis), err_msg=name
            )

        # pretend each body counted something, then merge into a zeroed twin
        for name in sub.delta_counter_names:
            getattr(left, name)[...] = 1
            getattr(right, name)[...] = 2
        twin = sub._bare_like()
        twin.initialize_tempered(NTEMPS, NWALKERS, NLEAVES, NDIM)
        for name in sub.delta_counter_names:
            getattr(twin, name)[...] = getattr(sub, name)
        twin.merge_walkers(left, 0, half)
        twin.merge_walkers(right, half, NWALKERS)
        for name, axis in sub.walker_axes.items():
            src = getattr(sub, name, None)
            if src is None:
                continue
            np.testing.assert_array_equal(getattr(twin, name), src, err_msg=name)
        for name in sub.delta_counter_names:
            np.testing.assert_array_equal(getattr(twin, name), getattr(sub, name) + 3, err_msg=name)
        return left, right, twin

    def test_base_substate_flat_ladder_by_value(self):
        sub = _make_sub(ModuleSubState, self.rng)
        sub.betas = np.linspace(1.0, 0.1, NTEMPS)
        left, right, twin = self._check_roundtrip(sub)
        np.testing.assert_array_equal(left.betas, sub.betas)
        self.assertIsNot(left.betas, sub.betas)
        # merge never writes a ladder back
        left.betas[:] = -1.0
        twin.merge_walkers(left, 0, NWALKERS // 2)
        self.assertFalse(np.any(twin.betas == -1.0))

    def test_per_leaf_ladder_walker_axis_last(self):
        betas_all = np.tile(np.linspace(1.0, 0.05, NTEMPS), (NLEAVES, 1))
        sub = _make_sub(MBHState, self.rng, betas_all=betas_all)
        self.assertEqual(sub.log_like.shape, (NLEAVES, NTEMPS, NWALKERS))
        left, right, twin = self._check_roundtrip(sub)
        self.assertEqual(left.log_like.shape, (NLEAVES, NTEMPS, NWALKERS // 2))
        np.testing.assert_array_equal(left.betas_all, betas_all)
        self.assertIsNot(left.betas_all, sub.betas_all)
        self.assertEqual(left.num_mbhs, NLEAVES)

    def test_gb_substate_never_touches_band_info(self):
        sub = _make_sub(GBState, self.rng)
        left, right, twin = self._check_roundtrip(sub)
        self.assertFalse(hasattr(left, "_band_info"))

    def test_bad_blocks_raise(self):
        sub = _make_sub(ModuleSubState, self.rng)
        with self.assertRaises(ValueError):
            sub.slice_walkers(4, 2)
        with self.assertRaises(ValueError):
            sub.slice_walkers(0, NWALKERS + 1)
        with self.assertRaises(ValueError):
            sub.merge_walkers(sub.slice_walkers(0, 2), 0, 3)

    def test_uninitialized_substate_slices_to_bare(self):
        sub = ModuleSubState(None)
        part = sub.slice_walkers(0, 1)
        self.assertFalse(part.tempered_initialized)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank
.wtenv/wt_run.sh $PWD/src .wtenv/t1.log python -m unittest tests.test_walkerslice_roundtrip -v; tail -n 25 .wtenv/t1.log
```
Expected: FAIL / ERROR with `AttributeError: 'ModuleSubState' object has no attribute 'slice_walkers'` (and `walker_axes`).

- [ ] **Step 3: Implement the sub-state methods**

In `src/lisatools/globalfit/state.py`, insert immediately after `_copy_tempered_from` (after line 553, before the `_reseed_state_arrays` comment block) inside `ModuleSubState`:

```python
    # ------------------------------------------------------------------
    # Walker-block slicing (multi-rank fan-out). Every tempered array with
    # a walker axis is sliced/merged by column. Ladders and the
    # per-iteration delta counters have NO walker axis: a slice carries the
    # ladder BY VALUE and never writes it back (the head owns ladders), and
    # its counters start at zero so a fan-out body's counts come back as
    # deltas that merge SUM-adds.
    # ------------------------------------------------------------------

    #: walker axis of every tempered array that has one (absent = no walker
    #: axis). Subclasses with a different layout override the dict.
    walker_axes: dict = {
        "coords": 1,
        "inds": 1,
        "log_like": 1,
        "log_prior": 1,
        "d_h": 0,
        "h_h": 0,
    }

    def _bare_like(self):
        """A fresh, empty instance of this class (no tempered block yet)."""
        return type(self)(None)

    def slice_walkers(self, w0: int, w1: int):
        """Copy walkers ``[w0, w1)`` of the tempered block into a fresh instance.

        Returns a bare instance when this sub-state has no tempered block.
        """
        part = self._bare_like()
        if not self.tempered_initialized:
            return part
        w0, w1 = int(w0), int(w1)
        if not (0 <= w0 < w1 <= int(self.nwalkers)):
            raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {self.nwalkers}).")
        part.initialize_tempered(
            self.ntemps,
            w1 - w0,
            self.nleaves_max,
            self.ndim,
            coords=self.coords[:, w0:w1],
            inds=self.inds[:, w0:w1],
        )
        for name, axis in self.walker_axes.items():
            if name in ("coords", "inds"):
                continue
            src = getattr(self, name, None)
            if src is None:
                continue
            getattr(part, name)[...] = np.take(src, np.arange(w0, w1), axis=axis)
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is not None:
            part.betas = np.array(self.betas, copy=True)
        return part

    def merge_walkers(self, part, w0: int, w1: int, *, sum_counters: bool = True):
        """Write ``part``'s walker columns back into ``[w0, w1)``.

        Ladders are NOT written back; delta counters are SUM-added when
        ``sum_counters`` (a slice starts from zero, so its counters are the
        deltas of the command that produced it).
        """
        if not self.tempered_initialized or not getattr(part, "tempered_initialized", False):
            return
        w0, w1 = int(w0), int(w1)
        if not (0 <= w0 < w1 <= int(self.nwalkers)):
            raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {self.nwalkers}).")
        if int(part.nwalkers) != w1 - w0:
            raise ValueError(
                f"slice has {part.nwalkers} walkers but the block [{w0}, {w1}) has {w1 - w0}."
            )
        for name, axis in self.walker_axes.items():
            dst = getattr(self, name, None)
            src = getattr(part, name, None)
            if dst is None or src is None:
                continue
            index = [slice(None)] * dst.ndim
            index[axis] = slice(w0, w1)
            dst[tuple(index)] = src
        if sum_counters:
            for name in self.delta_counter_names:
                dst = getattr(self, name, None)
                src = getattr(part, name, None)
                if dst is not None and src is not None:
                    dst[...] += src
```

In `PerLeafLadderState`, insert right after `betas_attr_name = "betas_all"` (line 1338):

```python
    # per-leaf log_like / log_prior are (nleaves_max, ntemps, nwalkers): walker axis LAST
    walker_axes = {**ModuleSubState.walker_axes, "log_like": 2, "log_prior": 2}

    def _bare_like(self):
        betas_all = None if self.betas_all is None else np.array(self.betas_all, copy=True)
        return type(self)(None, betas_all=betas_all)
```

- [ ] **Step 4: Run the test to verify it passes**

Run the same command as Step 2. Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Run the existing sub-state suite to prove nothing regressed**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t1b.log python -m unittest tests.test_gf_substate_roundtrip tests.test_module_substate_reseed -v; tail -n 15 .wtenv/t1b.log
```
Expected: all `OK`.

- [ ] **Step 6: Report ready to commit (do not commit)**

Run `git status --short` and report: "ready to commit: src/lisatools/globalfit/state.py, tests/test_walkerslice_roundtrip.py".

---

### Task 2: `GFState` slicing and merging (`communication/walkerslice.py`)

**Files:**
- Create: `src/lisatools/globalfit/communication/__init__.py`, `src/lisatools/globalfit/communication/walkerslice.py`
- Test: `tests/test_walkerslice_roundtrip.py` (append a second test class)

**Interfaces:**
- Consumes: Task 1 (`slice_walkers`, `merge_walkers`); eryn `State.__init__(coords, inds, branch_supplemental, supplemental, log_like, log_prior, betas, blobs, ...)`; `BranchSupplemental(obj_info, base_shape, copy)` with `.holder`, `.base_shape`, `__getitem__`, `__setitem__`; `GFState(coords, ..., sub_state_bases=None)`.
- Produces: `slice_state(full: GFState, w0: int, w1: int, *, sub_states="all" | list[str]) -> GFState` (copies; `supplemental["walker_inds"]` remapped to `tile(arange(B), (ntemps, 1))`; `part.sub_states[name]` is a sliced sub-state or `None`; `part.sub_state_bases` copied); `merge_state(full: GFState, part: GFState, w0: int, w1: int) -> None` (walker columns written back, sub-state counters summed, ladders and the head's global `walker_inds` untouched); constant `WALKER_INDS_KEY = "walker_inds"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walkerslice_roundtrip.py` (after the imports add `from lisatools.globalfit.communication.walkerslice import merge_state, slice_state` and `from tests.test_gf_substate_roundtrip import BRANCH_SHAPES, NTEMPS as RT_NTEMPS, NWALKERS as RT_NWALKERS, make_state`):

```python
class GFStateSliceMergeTest(unittest.TestCase):
    """Whole-state slice/merge over the roundtrip fixture (4 walkers -> two blocks of 2)."""

    def setUp(self):
        self.rng = np.random.default_rng(99)
        self.state = make_state(self.rng)
        nt, nw = RT_NTEMPS, RT_NWALKERS
        self.state.supplemental = BranchSupplemental(
            {
                "walker_inds": np.tile(np.arange(nw), (nt, 1)),
                "aux": self.rng.standard_normal((nt, nw, 2)),
            },
            base_shape=(nt, nw),
        )
        self.state.branches["psd"].branch_supplemental = BranchSupplemental(
            {"tag": self.rng.integers(0, 9, (nt, nw, 1))}, base_shape=(nt, nw, 1)
        )
        for sub in self.state.sub_states.values():
            for name in ("d_h", "h_h"):
                getattr(sub, name)[...] = self.rng.standard_normal(getattr(sub, name).shape)
            for name in sub.delta_counter_names:
                arr = getattr(sub, name)
                arr[...] = self.rng.integers(0, 5, arr.shape)
        self.ref = GFState(self.state, copy=True)

    def test_slice_geometry_and_walker_inds_remap(self):
        part = slice_state(self.state, 2, 4)
        for name, br in part.branches.items():
            self.assertEqual(br.nwalkers, 2)
            np.testing.assert_array_equal(br.coords, self.ref.branches[name].coords[:, 2:4])
            np.testing.assert_array_equal(br.inds, self.ref.branches[name].inds[:, 2:4])
        np.testing.assert_array_equal(
            part.supplemental.holder["walker_inds"], np.tile(np.arange(2), (RT_NTEMPS, 1))
        )
        np.testing.assert_array_equal(
            part.supplemental.holder["aux"], self.ref.supplemental.holder["aux"][:, 2:4]
        )
        np.testing.assert_array_equal(
            part.branches["psd"].branch_supplemental.holder["tag"],
            self.ref.branches["psd"].branch_supplemental.holder["tag"][:, 2:4],
        )
        np.testing.assert_array_equal(part.log_like, self.ref.log_like[:, 2:4])
        np.testing.assert_array_equal(part.log_prior, self.ref.log_prior[:, 2:4])
        np.testing.assert_array_equal(part.betas, self.ref.betas)
        mbh = part.sub_states["mbh"]
        self.assertEqual(mbh.nwalkers, 2)
        self.assertEqual(mbh.log_like.shape, (BRANCH_SHAPES["mbh"][0], RT_NTEMPS, 2))
        np.testing.assert_array_equal(mbh.d_h, self.ref.sub_states["mbh"].d_h[2:4])
        self.assertEqual(part.sub_state_bases, self.state.sub_state_bases)
        # a slice is a copy: mutating it never reaches the full state
        part.branches["gb"].coords[...] = 123.0
        part.supplemental.holder["aux"][...] = 5.0
        np.testing.assert_array_equal(self.state.branches["gb"].coords, self.ref.branches["gb"].coords)
        np.testing.assert_array_equal(self.state.supplemental.holder["aux"], self.ref.supplemental.holder["aux"])

    def test_sub_state_filter(self):
        part = slice_state(self.state, 0, 2, sub_states=["mbh"])
        self.assertIsNotNone(part.sub_states["mbh"])
        for name in ("gb", "emri", "sobbh", "psd"):
            self.assertIsNone(part.sub_states[name])
        none = slice_state(self.state, 0, 2, sub_states=[])
        self.assertTrue(all(v is None for v in none.sub_states.values()))

    def test_slice_survives_the_gfstate_copy_path(self):
        part = slice_state(self.state, 0, 2)
        twin = GFState(part, copy=True)
        np.testing.assert_array_equal(twin.sub_states["mbh"].coords, part.sub_states["mbh"].coords)
        self.assertEqual(twin.branches["gb"].nwalkers, 2)

    def test_merge_roundtrip_restores_columns_and_sums_counters(self):
        target = GFState(self.state, copy=True)
        for br in target.branches.values():
            br.coords[...] = 0.0
            br.inds[...] = False
        target.log_like[...] = 0.0
        target.log_prior[...] = 0.0
        target.supplemental.holder["aux"][...] = 0.0
        target.branches["psd"].branch_supplemental.holder["tag"][...] = -1
        for sub in target.sub_states.values():
            sub.coords[...] = 0.0
            sub.d_h[...] = 0.0
        left = slice_state(self.state, 0, 2)
        right = slice_state(self.state, 2, 4)
        for part in (left, right):
            for sub in part.sub_states.values():
                for name in sub.delta_counter_names:
                    getattr(sub, name)[...] = 1
        merge_state(target, left, 0, 2)
        merge_state(target, right, 2, 4)

        for name, br in target.branches.items():
            np.testing.assert_array_equal(br.coords, self.ref.branches[name].coords)
            np.testing.assert_array_equal(br.inds, self.ref.branches[name].inds)
        np.testing.assert_array_equal(target.log_like, self.ref.log_like)
        np.testing.assert_array_equal(target.log_prior, self.ref.log_prior)
        np.testing.assert_array_equal(target.supplemental.holder["aux"], self.ref.supplemental.holder["aux"])
        # the head's walker_inds stay GLOBAL ids (a slice's remapped ids never come back)
        np.testing.assert_array_equal(
            target.supplemental.holder["walker_inds"], self.ref.supplemental.holder["walker_inds"]
        )
        np.testing.assert_array_equal(
            target.branches["psd"].branch_supplemental.holder["tag"],
            self.ref.branches["psd"].branch_supplemental.holder["tag"],
        )
        for name, sub in target.sub_states.items():
            ref = self.ref.sub_states[name]
            for aname in sub.walker_axes:
                if getattr(ref, aname, None) is not None:
                    np.testing.assert_array_equal(getattr(sub, aname), getattr(ref, aname), err_msg=f"{name}.{aname}")
            for cname in sub.delta_counter_names:
                np.testing.assert_array_equal(getattr(sub, cname), getattr(ref, cname) + 2, err_msg=f"{name}.{cname}")
        # ladders untouched
        np.testing.assert_array_equal(target.sub_states["mbh"].betas_all, self.ref.sub_states["mbh"].betas_all)
        np.testing.assert_array_equal(target.sub_states["psd"].betas, self.ref.sub_states["psd"].betas)
        np.testing.assert_array_equal(
            target.sub_states["gb"].band_info["band_temps"], self.ref.sub_states["gb"].band_info["band_temps"]
        )

    def test_gb_band_info_is_never_sliced(self):
        part = slice_state(self.state, 0, 2)
        self.assertFalse(hasattr(part.sub_states["gb"], "_band_info"))

    def test_bad_blocks_raise(self):
        with self.assertRaises(ValueError):
            slice_state(self.state, 3, 3)
        with self.assertRaises(ValueError):
            merge_state(self.state, slice_state(self.state, 0, 2), 0, 3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t2.log python -m unittest tests.test_walkerslice_roundtrip -v; tail -n 25 .wtenv/t2.log
```
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.communication'`.

- [ ] **Step 3: Create the package and the slicing module**

Create `src/lisatools/globalfit/communication/__init__.py` (Tasks 3-6 extend the export list; keep it exactly in sync with what exists):

```python
"""Multi-rank communication layer of the global fit.

One MPI rank per GPU (general in both directions via ``gpus_per_rank`` /
``ranks_per_gpu``), equal static walker blocks, a head rank that sequences
the recipe and also computes block 0, computation ranks that serve
head-directed commands, and the unchanged saver rank. See
``docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md``.

``mpi4py`` is imported lazily and only for real communicators; the
:mod:`fakecomm` simulator runs the same code in one process for tests.
"""

from .walkerslice import WALKER_INDS_KEY, merge_state, slice_state

__all__ = ["WALKER_INDS_KEY", "merge_state", "slice_state"]
```

Create `src/lisatools/globalfit/communication/walkerslice.py`:

```python
"""Walker-block slicing of a :class:`~lisatools.globalfit.state.GFState`.

A slice is a self-contained ``GFState`` over ``B = w1 - w0`` walkers: every
walker-axis array cut by column (copies), ladders by value, sub-state delta
counters zeroed, and ``supplemental["walker_inds"]`` REMAPPED to ``0..B-1``
because a rank's ACA has exactly ``B`` rows. :func:`merge_state` writes a
slice's walker columns back and SUM-adds the counters; ladders and the
head's global ``walker_inds`` are never written back.
"""

from __future__ import annotations

import numpy as np
from eryn.state import BranchSupplemental

from ..state import GFState

WALKER_INDS_KEY = "walker_inds"


def _nwalkers_of(state) -> int:
    first = next(iter(state.branches.values()))
    return int(first.nwalkers)


def _walker_block(w0, w1, nwalkers):
    w0, w1 = int(w0), int(w1)
    if not (0 <= w0 < w1 <= int(nwalkers)):
        raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {nwalkers}).")
    return w0, w1


def _slice_supp(supp, w0, w1):
    """Column-slice every array a ``BranchSupplemental`` holds (copies)."""
    if supp is None:
        return None
    sliced = {
        name: np.array(values, copy=True)
        for name, values in supp[(slice(None), slice(w0, w1))].items()
    }
    base_shape = (int(supp.base_shape[0]), w1 - w0) + tuple(supp.base_shape[2:])
    return BranchSupplemental(sliced, base_shape=base_shape, copy=False)


def slice_state(full, w0, w1, *, sub_states="all"):
    """A ``GFState`` holding walkers ``[w0, w1)`` of ``full`` (all copies).

    Args:
        full: the head's full ``GFState``.
        w0, w1: the walker block.
        sub_states: ``"all"`` slices every tempered sub-state; a list of
            branch names slices only those (the rest are ``None``); ``[]``
            slices none.
    """
    w0, w1 = _walker_block(w0, w1, _nwalkers_of(full))
    B = w1 - w0
    coords = {name: np.array(br.coords[:, w0:w1], copy=True) for name, br in full.branches.items()}
    inds = {name: np.array(br.inds[:, w0:w1], copy=True) for name, br in full.branches.items()}
    branch_supps = {
        name: _slice_supp(br.branch_supplemental, w0, w1) for name, br in full.branches.items()
    }
    supp = _slice_supp(full.supplemental, w0, w1)
    if supp is not None and WALKER_INDS_KEY in supp.holder:
        ntemps = int(supp.holder[WALKER_INDS_KEY].shape[0])
        supp.holder[WALKER_INDS_KEY] = np.tile(np.arange(B), (ntemps, 1))

    def _cols(arr):
        return None if arr is None else np.array(arr[:, w0:w1], copy=True)

    part = GFState(
        coords,
        inds=inds,
        branch_supplemental=branch_supps,
        supplemental=supp,
        log_like=_cols(full.log_like),
        log_prior=_cols(full.log_prior),
        betas=None if full.betas is None else np.array(full.betas, copy=True),
        blobs=_cols(full.blobs),
        sub_state_bases=None,
    )
    part.sub_state_bases = dict(getattr(full, "sub_state_bases", None) or {})
    wanted = set(full.branches) if sub_states == "all" else set(sub_states)
    full_subs = getattr(full, "sub_states", None) or {}
    part.sub_states = {}
    for name in full.branches:
        sub = full_subs.get(name)
        if name in wanted and sub is not None and getattr(sub, "tempered_initialized", False):
            part.sub_states[name] = sub.slice_walkers(w0, w1)
        else:
            part.sub_states[name] = None
    return part


def merge_state(full, part, w0, w1):
    """Write ``part``'s walker columns back into ``full[:, w0:w1]``.

    Sub-state delta counters are SUM-added; ladders and the head's global
    ``supplemental["walker_inds"]`` are untouched.
    """
    w0, w1 = _walker_block(w0, w1, _nwalkers_of(full))
    if _nwalkers_of(part) != w1 - w0:
        raise ValueError(
            f"slice has {_nwalkers_of(part)} walkers but the block [{w0}, {w1}) has {w1 - w0}."
        )
    for name, br in full.branches.items():
        pbr = part.branches[name]
        br.coords[:, w0:w1] = pbr.coords
        br.inds[:, w0:w1] = pbr.inds
        if br.branch_supplemental is not None and pbr.branch_supplemental is not None:
            br.branch_supplemental[(slice(None), slice(w0, w1))] = pbr.branch_supplemental.holder
    for name in ("log_like", "log_prior", "blobs"):
        dst = getattr(full, name, None)
        src = getattr(part, name, None)
        if dst is not None and src is not None:
            dst[:, w0:w1] = src
    if full.supplemental is not None and part.supplemental is not None:
        payload = {k: v for k, v in part.supplemental.holder.items() if k != WALKER_INDS_KEY}
        full.supplemental[(slice(None), slice(w0, w1))] = payload
    full_subs = getattr(full, "sub_states", None) or {}
    part_subs = getattr(part, "sub_states", None) or {}
    for name, sub in full_subs.items():
        psub = part_subs.get(name)
        if sub is not None and psub is not None:
            sub.merge_walkers(psub, w0, w1)
```

- [ ] **Step 4: Run the test to verify it passes**

Run the Step 2 command. Expected: `Ran 11 tests ... OK`.

- [ ] **Step 5: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/communication/__init__.py, src/lisatools/globalfit/communication/walkerslice.py, tests/test_walkerslice_roundtrip.py".

---

### Task 3: In-process fake communicator (`communication/fakecomm.py`)

**Files:**
- Create: `src/lisatools/globalfit/communication/fakecomm.py`
- Modify: `src/lisatools/globalfit/communication/__init__.py` (exports)
- Test: `tests/test_fakecomm.py`

**Interfaces:**
- Produces: `FakeWorld(size, nodes=None, timeout=60.0)` with `.comm(rank) -> FakeComm`, `.run(fn, ranks=None, timeout=None) -> dict[rank, result]` (thread per rank; re-raises the first non-abort failure as `RuntimeError` with the remote traceback text), `.abort(code, rank)`, `.aborted`; `FakeComm` with `Get_rank`, `Get_size`, `Get_processor_name`, `send(obj, dest, tag=0)`, `isend(...) -> request with .wait()`, `recv(source, tag=0)`, `iprobe(source, tag=0)`, `bcast(obj, root=0)`, `allgather(obj)`, `barrier()`, `Split(color, key=0)`, `Split_type(split_type, key=0)`, `Abort(code=1)`, `Free()`, and the class attributes `COMM_TYPE_SHARED`, `UNDEFINED`; `FakeCommNull` (every attribute raises `RuntimeError`); `FakeAbort(RuntimeError)`.
- Transport semantics: `send` pickle-copies (a caller's later mutation never reaches the receiver; unpicklable payloads fail loudly).

- [ ] **Step 1: Write the failing test**

Create `tests/test_fakecomm.py`:

```python
"""The in-process fake communicator behaves like the mpi4py subset the global fit uses."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeAbort, FakeWorld


class FakeCommTest(unittest.TestCase):
    def test_send_recv_pickle_copies(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            if rank == 0:
                payload = {"a": np.arange(3)}
                comm.send(payload, dest=1)
                payload["a"][0] = 99  # must NOT reach rank 1
                return comm.recv(source=1)
            got = comm.recv(source=0)
            comm.send(int(got["a"][0]), dest=0)
            return got

        out = world.run(fn)
        self.assertEqual(out[0], 0)
        np.testing.assert_array_equal(out[1]["a"], [0, 1, 2])

    def test_isend_iprobe(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            if rank == 0:
                req = comm.isend("hello", dest=1)
                req.wait()
                return comm.recv(source=1)
            while not comm.iprobe(source=0):
                pass
            msg = comm.recv(source=0)
            comm.send(msg + "!", dest=0)
            return msg

        out = world.run(fn)
        self.assertEqual(out, {0: "hello!", 1: "hello"})

    def test_bcast_and_allgather(self):
        world = FakeWorld(3)

        def fn(rank, comm):
            state = comm.bcast({"x": rank} if rank == 0 else None, root=0)
            return state["x"], comm.allgather(rank * 10)

        out = world.run(fn)
        for r in range(3):
            self.assertEqual(out[r], (0, [0, 10, 20]))

    def test_split_type_by_node_and_split_by_colour(self):
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])

        def fn(rank, comm):
            node = comm.Split_type(comm.COMM_TYPE_SHARED, key=rank)
            color = comm.UNDEFINED if rank == 4 else 0
            sub = comm.Split(color, key=rank)
            sub_info = None if rank == 4 else (sub.Get_rank(), sub.Get_size())
            return (comm.Get_processor_name(), node.Get_rank(), node.Get_size(), sub_info)

        out = world.run(fn)
        self.assertEqual(out[0], ("node0", 0, 3, (0, 4)))
        self.assertEqual(out[2], ("node0", 1, 3, (2, 4)))
        self.assertEqual(out[4], ("node0", 2, 3, None))
        self.assertEqual(out[1], ("node1", 0, 2, (1, 4)))
        self.assertEqual(out[3], ("node1", 1, 2, (3, 4)))

    def test_null_comm_raises(self):
        world = FakeWorld(1)
        with self.assertRaises(RuntimeError):
            world.run(lambda r, c: c.Split(c.UNDEFINED, key=r).Get_rank())

    def test_abort_unblocks_pending_recv_and_reports_root_cause(self):
        world = FakeWorld(2, timeout=5.0)

        def fn(rank, comm):
            if rank == 0:
                raise ValueError("boom")
            return comm.recv(source=0)  # would block forever without the abort

        with self.assertRaises(RuntimeError) as cm:
            world.run(fn)
        self.assertIn("boom", str(cm.exception))
        self.assertIsNotNone(world.aborted)

    def test_explicit_abort(self):
        world = FakeWorld(2, timeout=5.0)

        def fn(rank, comm):
            if rank == 1:
                comm.Abort(3)
            return comm.recv(source=1)

        with self.assertRaises(RuntimeError) as cm:
            world.run(fn)
        self.assertIn("Abort", str(cm.exception))
        self.assertEqual(world.aborted, (3, 1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t3.log python -m unittest tests.test_fakecomm -v; tail -n 20 .wtenv/t3.log
```
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.communication.fakecomm'`.

- [ ] **Step 3: Implement the fake communicator**

Create `src/lisatools/globalfit/communication/fakecomm.py`:

```python
"""In-process stand-in for the mpi4py communicator subset the global fit uses.

``FakeWorld(size)`` runs one Python thread per rank; ``FakeComm`` implements
``Get_rank / Get_size / Get_processor_name / send / isend / recv / iprobe /
bcast / allgather / barrier / Split / Split_type / Abort / Free`` with
pickle-copy transport, so the SAME fan-out code runs on a laptop with no
MPI installed, in one python process. A test harness, not a performance
tool.
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
import traceback

#: mirrors of the mpi4py constants the layout code needs (only identity matters)
COMM_TYPE_SHARED = 1
UNDEFINED = -32766


class FakeAbort(RuntimeError):
    """Raised in every rank's thread once any rank aborts the world."""


def _pcopy(obj):
    return pickle.loads(pickle.dumps(obj))


class _Request:
    def wait(self):
        return None

    def test(self):
        return True, None


class _Group:
    """Shared plumbing of one communicator: inboxes + collective slots."""

    def __init__(self, members):
        self.members = tuple(members)  # world ranks, in local-rank order
        n = len(self.members)
        self.inbox = {(d, s): queue.Queue() for d in range(n) for s in range(n)}
        self.barrier = threading.Barrier(n) if n > 0 else None
        self.slots = [None] * n


class FakeCommNull:
    """What ``Split`` returns to a rank with colour ``UNDEFINED``."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError("FakeCommNull: this rank is not a member of the communicator")


class FakeComm:
    COMM_TYPE_SHARED = COMM_TYPE_SHARED
    UNDEFINED = UNDEFINED

    def __init__(self, world, group, world_rank):
        self._world = world
        self._group = group
        self._world_rank = int(world_rank)
        self._rank = group.members.index(self._world_rank)

    # -- topology ---------------------------------------------------------
    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return len(self._group.members)

    def Get_processor_name(self):
        return self._world.node_name(self._world_rank)

    def Free(self):
        return None

    # -- point to point ---------------------------------------------------
    def send(self, obj, dest, tag=0):
        self._check_abort()
        self._group.inbox[(int(dest), self._rank)].put(_pcopy(obj))

    def isend(self, obj, dest, tag=0):
        self.send(obj, dest, tag)
        return _Request()

    def recv(self, source, tag=0):
        q = self._group.inbox[(self._rank, int(source))]
        while True:
            self._check_abort()
            try:
                return q.get(timeout=0.02)
            except queue.Empty:
                continue

    def iprobe(self, source, tag=0):
        return not self._group.inbox[(self._rank, int(source))].empty()

    # -- collectives (every member calls them in the same order) ----------
    def _wait_barrier(self):
        self._check_abort()
        try:
            self._group.barrier.wait(timeout=self._world.timeout)
        except threading.BrokenBarrierError:
            raise FakeAbort(
                f"rank {self._world_rank}: collective broken (abort or timeout)"
            ) from None

    def _exchange(self, value):
        group = self._group
        group.slots[self._rank] = value
        self._wait_barrier()
        out = list(group.slots)
        self._wait_barrier()  # nobody overwrites the slots before everyone has read
        return out

    def bcast(self, obj, root=0):
        out = self._exchange(_pcopy(obj) if self._rank == int(root) else None)
        return _pcopy(out[int(root)])

    def allgather(self, obj):
        return [_pcopy(v) for v in self._exchange(_pcopy(obj))]

    def barrier(self):
        self._wait_barrier()

    Barrier = barrier

    def Split(self, color, key=0):
        colors = self._exchange((color, key))
        gid = self._exchange(self._world.next_group_id() if self._rank == 0 else None)[0]
        if color == UNDEFINED:
            return FakeCommNull()
        order = sorted(
            (i for i, (c, _k) in enumerate(colors) if c == color),
            key=lambda i: (colors[i][1], self._group.members[i]),
        )
        members = tuple(self._group.members[i] for i in order)
        return FakeComm(self._world, self._world.group(gid, members), self._world_rank)

    def Split_type(self, split_type, key=0):
        return self.Split(color=self._world.node_of(self._world_rank), key=key)

    # -- failure ----------------------------------------------------------
    def Abort(self, errorcode=1):
        self._world.abort(errorcode, self._world_rank)
        raise FakeAbort(f"rank {self._world_rank} called Abort({errorcode})")

    def _check_abort(self):
        if self._world.aborted is not None:
            code, who = self._world.aborted
            raise FakeAbort(f"rank {self._world_rank}: world aborted by rank {who} (code {code})")


class FakeWorld:
    """``size`` ranks on ``nodes`` (e.g. ``[0, 1, 0, 1, 0]``), one thread each."""

    def __init__(self, size, nodes=None, timeout=60.0):
        self.size = int(size)
        self.nodes = list(nodes) if nodes is not None else [0] * self.size
        if len(self.nodes) != self.size:
            raise ValueError("nodes must have one entry per rank")
        self.timeout = float(timeout)
        self.aborted = None
        self._groups = {}
        self._lock = threading.Lock()
        self._gid = 0
        self._world_group = self.group(0, tuple(range(self.size)))

    def node_of(self, rank):
        return int(self.nodes[int(rank)])

    def node_name(self, rank):
        return f"node{self.node_of(rank)}"

    def next_group_id(self):
        with self._lock:
            self._gid += 1
            return self._gid

    def group(self, gid, members):
        with self._lock:
            key = (int(gid), tuple(members))
            if key not in self._groups:
                self._groups[key] = _Group(members)
            return self._groups[key]

    def comm(self, rank):
        return FakeComm(self, self._world_group, int(rank))

    def abort(self, code, rank):
        self.aborted = (int(code), int(rank))
        with self._lock:
            for g in self._groups.values():
                if g.barrier is not None:
                    g.barrier.abort()

    def run(self, fn, ranks=None, timeout=None):
        """Run ``fn(rank, comm)`` on one thread per rank; return ``{rank: result}``.

        Re-raises the first NON-abort failure (the root cause) as a
        ``RuntimeError`` carrying the remote traceback text, after aborting
        the world so no other thread stays blocked.
        """
        ranks = list(range(self.size)) if ranks is None else list(ranks)
        results, errors = {}, {}

        def _target(r):
            try:
                results[r] = fn(r, self.comm(r))
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[r] = (exc, traceback.format_exc())
                if self.aborted is None:
                    self.abort(1, r)

        threads = [
            threading.Thread(target=_target, args=(r,), name=f"fake-rank-{r}", daemon=True)
            for r in ranks
        ]
        for t in threads:
            t.start()
        deadline = time.time() + (self.timeout if timeout is None else float(timeout))
        for t in threads:
            t.join(max(0.0, deadline - time.time()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive and not errors:
            self.abort(1, -1)
            raise TimeoutError(f"FakeWorld.run: ranks still running at timeout: {alive}")
        if errors:
            root = [r for r, (e, _) in errors.items() if not isinstance(e, FakeAbort)]
            r = min(root) if root else min(errors)
            exc, tb = errors[r]
            raise RuntimeError(f"rank {r} failed:\n{tb}") from exc
        return results
```

Add to `communication/__init__.py`:

```python
from .fakecomm import FakeAbort, FakeComm, FakeCommNull, FakeWorld
```
and extend `__all__` with `"FakeAbort", "FakeComm", "FakeCommNull", "FakeWorld"`.

- [ ] **Step 4: Run the test to verify it passes**

Run the Step 2 command. Expected: `Ran 7 tests ... OK`.

- [ ] **Step 5: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/communication/fakecomm.py, src/lisatools/globalfit/communication/__init__.py, tests/test_fakecomm.py".

---

### Task 4: Rank roles and the walker-block layout (`communication/ranks.py`, part 1)

**Files:**
- Create: `src/lisatools/globalfit/communication/ranks.py`
- Modify: `src/lisatools/globalfit/communication/__init__.py`
- Test: `tests/test_rank_layout.py`

**Interfaces:**
- Consumes: a communicator (real or `FakeComm`) with `Get_size`, `Get_rank`, `Split_type`, `allgather`, `Split`, and optionally `Get_processor_name`; `numpy.random.SeedSequence`.
- Produces: `RankRole` (HEAD, COMPUTE, SAVER, SPARE); `RankPlacement(rank, role, node, local_index, devices: tuple, device_slot, w0, w1)` (frozen); `WalkerBlockLayout(size, head_rank, saver_rank, compute_ranks, nwalkers, block, placements, gpus_per_rank, ranks_per_gpu, legacy)` with `n_compute`, `worker_ranks`, `is_single()`, `role_of(rank)`, `block_of(rank) -> (w0, w1)`, `local_gpus(rank) -> list[int] | None`, `fanout_rank(rank) -> int`, `ranks_on_device(node, device) -> tuple`, `make_fanout_comm(comm)`, `describe() -> str`, `digest() -> str`; `resolve_roles(size, main_rank=0) -> (head, saver, compute_ranks)`; `build_layout(comm, nwalkers, gpu_pool, *, gpus_per_rank=1, ranks_per_gpu=1, main_rank=0, legacy=None) -> WalkerBlockLayout` (`legacy=None` reads env `GF_LEGACY_RANK_LAYOUT`); `derive_rank_seed(base_seed, layout, rank) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rank_layout.py`:

```python
"""Rank roles + walker-block layout resolved identically on every rank (FakeComm)."""

import unittest

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import (
    RankRole,
    build_layout,
    derive_rank_seed,
    resolve_roles,
)


def _layouts(world, nwalkers, pool, **kwargs):
    return world.run(lambda r, c: build_layout(c, nwalkers, pool, legacy=False, **kwargs))


class ResolveRolesTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(resolve_roles(1), (0, 0, (0,)))
        self.assertEqual(resolve_roles(2), (0, 0, (0, 1)))
        self.assertEqual(resolve_roles(3), (0, 2, (0, 1)))
        self.assertEqual(resolve_roles(5), (0, 4, (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            resolve_roles(0)


class BuildLayoutTest(unittest.TestCase):
    def test_two_nodes_one_rank_per_gpu(self):
        # cyclic placement: ranks 0,2,4 on node0, ranks 1,3 on node1; rank 4 = saver
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])
        outs = _layouts(world, 8, [0, 1])
        lay = outs[0]
        for r in range(5):
            self.assertEqual(outs[r].describe(), lay.describe())
            self.assertEqual(outs[r].digest(), lay.digest())
        self.assertEqual(lay.compute_ranks, (0, 1, 2, 3))
        self.assertEqual(lay.saver_rank, 4)
        self.assertEqual(lay.worker_ranks, (1, 2, 3))
        self.assertEqual(lay.block, 2)
        self.assertFalse(lay.is_single())
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[2].devices, (1,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual(lay.placements[3].devices, (1,))
        self.assertEqual([lay.block_of(r) for r in lay.compute_ranks], [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual(lay.role_of(4), RankRole.SAVER)
        self.assertEqual(lay.block_of(4), (0, 0))
        self.assertEqual(lay.local_gpus(3), [1])
        self.assertEqual(lay.fanout_rank(3), 3)
        self.assertEqual(lay.placements[4].local_index, 2)

    def test_ranks_per_gpu_share_one_device(self):
        world = FakeWorld(3)  # head + one compute rank + saver, ONE GPU
        lay = _layouts(world, 6, [0], ranks_per_gpu=2)[0]
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual((lay.placements[0].device_slot, lay.placements[1].device_slot), (0, 1))
        self.assertEqual(lay.ranks_on_device("node0", 0), (0, 1))
        self.assertEqual(lay.block, 3)

    def test_gpus_per_rank_owns_two_devices(self):
        world = FakeWorld(2)  # saver aliased to the head: compute = (0, 1)
        lay = _layouts(world, 4, [0, 1, 2, 3], gpus_per_rank=2)[0]
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.placements[1].devices, (2, 3))
        self.assertEqual(lay.local_gpus(1), [2, 3])

    def test_errors(self):
        world = FakeWorld(3)
        with self.assertRaises(RuntimeError):
            _layouts(world, 7, [0, 1])  # 7 % 2 != 0
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0])  # two compute ranks on one GPU without ranks_per_gpu
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=2, ranks_per_gpu=2)
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=0)

    def test_single_rank_cpu(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertTrue(lay.is_single())
        self.assertIsNone(lay.local_gpus(0))
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.role_of(0), RankRole.HEAD)
        self.assertEqual(lay.saver_rank, 0)

    def test_legacy_layout_keeps_todays_roles(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 5, [0, 1], legacy=True))[0]
        self.assertTrue(lay.legacy)
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.block, 5)
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.role_of(1), RankRole.SPARE)
        self.assertEqual(lay.role_of(2), RankRole.SAVER)


class SeedTest(unittest.TestCase):
    def test_distinct_and_deterministic(self):
        lay = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        seeds = [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks])
        self.assertNotEqual(seeds, [derive_rank_seed(103210, lay, r) for r in lay.compute_ranks])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t4.log python -m unittest tests.test_rank_layout -v; tail -n 20 .wtenv/t4.log
```
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.communication.ranks'`.

- [ ] **Step 3: Implement roles, layout and seeds**

Create `src/lisatools/globalfit/communication/ranks.py`:

```python
"""Rank roles, walker-block layout, per-rank device pinning and seeds.

Roles: HEAD (rank 0: sequences the recipe, owns the full host state, AND
computes walker block 0), COMPUTE (one device list + one walker block each),
SAVER (highest rank, unchanged; aliased to the head below 3 ranks). The
mapping between ranks and GPUs is general in both directions:
``gpus_per_rank`` (a rank owns several devices, sharded in-process) and
``ranks_per_gpu`` (several ranks share one device); at most one exceeds 1.

``GF_LEGACY_RANK_LAYOUT=1`` restores today's roles (one compute rank owning
the whole pool, other non-saver ranks are stopped SPARE ranks).

``mpi4py`` is imported lazily and only for real communicators.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import os
import sys
import threading
import traceback

import numpy as np

LEGACY_ENV = "GF_LEGACY_RANK_LAYOUT"


class RankRole(enum.Enum):
    HEAD = "head"
    COMPUTE = "compute"
    SAVER = "saver"
    SPARE = "spare"  # legacy layout only


@dataclasses.dataclass(frozen=True)
class RankPlacement:
    rank: int
    role: RankRole
    node: str
    local_index: int
    #: the rank's device list in the per-node pool's numbering (empty on CPU);
    #: several ranks list the same device when ranks_per_gpu > 1
    devices: tuple
    #: this rank's index among the ranks sharing its device
    device_slot: int
    w0: int
    w1: int


@dataclasses.dataclass(frozen=True)
class WalkerBlockLayout:
    size: int
    head_rank: int
    saver_rank: int
    compute_ranks: tuple
    nwalkers: int
    block: int
    placements: dict
    gpus_per_rank: int = 1
    ranks_per_gpu: int = 1
    legacy: bool = False

    @property
    def n_compute(self) -> int:
        return len(self.compute_ranks)

    @property
    def worker_ranks(self) -> tuple:
        return tuple(r for r in self.compute_ranks if r != self.head_rank)

    def is_single(self) -> bool:
        return self.n_compute == 1

    def role_of(self, rank) -> RankRole:
        return self.placements[int(rank)].role

    def block_of(self, rank) -> tuple:
        p = self.placements[int(rank)]
        return p.w0, p.w1

    def local_gpus(self, rank):
        devices = self.placements[int(rank)].devices
        return list(devices) if devices else None

    def fanout_rank(self, rank) -> int:
        """This rank's index in the fan-out communicator (== compute-rank order)."""
        return self.compute_ranks.index(int(rank))

    def ranks_on_device(self, node, device) -> tuple:
        return tuple(
            r
            for r, p in sorted(self.placements.items())
            if p.node == str(node) and int(device) in p.devices and p.role != RankRole.SAVER
        )

    def make_fanout_comm(self, comm):
        """Split ``comm`` into the head+compute communicator (saver: null comm)."""
        rank = int(comm.Get_rank())
        color = 0 if rank in self.compute_ranks else _undefined(comm)
        return comm.Split(color, key=rank)

    def describe(self) -> str:
        head = (
            f"walker-block layout: size={self.size} n_compute={self.n_compute} "
            f"nwalkers={self.nwalkers} block={self.block} "
            f"gpus_per_rank={self.gpus_per_rank} ranks_per_gpu={self.ranks_per_gpu}"
            f"{' LEGACY' if self.legacy else ''}"
        )
        lines = [head]
        for r in range(self.size):
            p = self.placements[r]
            lines.append(
                f"  r{r:<3d} {p.role.value:<7s} node={p.node} local={p.local_index} "
                f"devices={list(p.devices)} slot={p.device_slot} walkers=[{p.w0},{p.w1})"
            )
        return "\n".join(lines)

    def digest(self) -> str:
        return hashlib.sha256(self.describe().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# communicator shims (mpi4py only when the comm is real)
# --------------------------------------------------------------------------


def _comm_type_shared(comm):
    value = getattr(comm, "COMM_TYPE_SHARED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.COMM_TYPE_SHARED


def _undefined(comm):
    value = getattr(comm, "UNDEFINED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.UNDEFINED


def _proc_name(comm) -> str:
    fn = getattr(comm, "Get_processor_name", None)
    if fn is not None:
        return str(fn())
    from mpi4py import MPI

    return str(MPI.Get_processor_name())


# --------------------------------------------------------------------------
# roles + layout
# --------------------------------------------------------------------------


def resolve_roles(size, main_rank=0):
    """``(head, saver, compute_ranks)``: saver = highest non-head rank at
    ``size >= 3`` (aliased to the head below that, as today); every other
    rank computes, the head included. No spares."""
    size = int(size)
    if size < 1:
        raise ValueError("communicator size must be >= 1")
    head = int(main_rank)
    if not (0 <= head < size):
        raise ValueError(f"main_rank {head} is not a rank of a size-{size} communicator")
    others = [r for r in range(size) if r != head]
    saver = others[-1] if size >= 3 else head
    compute = tuple(r for r in range(size) if r == head or r != saver)
    return head, saver, compute


def build_layout(
    comm,
    nwalkers,
    gpu_pool,
    *,
    gpus_per_rank=1,
    ranks_per_gpu=1,
    main_rank=0,
    legacy=None,
):
    """Resolve the identical layout on every rank (one collective ``allgather``).

    ``gpu_pool`` is the PER-NODE device list (the ``GPUS`` setting). Compute
    ranks on a node are ordered by world rank and assigned blocked:
    ``gpus_per_rank = k > 1`` -> devices ``pool[i*k:(i+1)*k]``;
    ``ranks_per_gpu = m > 1`` -> device ``pool[i // m]``, slot ``i % m``.
    """
    if legacy is None:
        legacy = os.environ.get(LEGACY_ENV, "0") == "1"
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    head, saver, compute = resolve_roles(size, main_rank)
    k, m = int(gpus_per_rank), int(ranks_per_gpu)
    if k < 1 or m < 1:
        raise ValueError("gpus_per_rank and ranks_per_gpu must both be >= 1")
    if k > 1 and m > 1:
        raise ValueError("at most one of gpus_per_rank / ranks_per_gpu may exceed 1")
    pool = [int(g) for g in (gpu_pool or [])]
    if legacy:
        compute = (head,)
    nwalkers = int(nwalkers)
    n_compute = len(compute)
    if nwalkers % n_compute:
        raise ValueError(
            f"nwalkers={nwalkers} is not divisible by the compute-rank count {n_compute}: "
            "equal walker blocks are required (pick NWALKERS as a multiple of it)."
        )
    block = nwalkers // n_compute

    if size == 1 or not hasattr(comm, "Split_type"):
        table = [(_proc_name(comm), 0)]
    else:
        node_comm = comm.Split_type(_comm_type_shared(comm), key=rank)
        table = comm.allgather((_proc_name(comm), int(node_comm.Get_rank())))
        free = getattr(node_comm, "Free", None)
        if free is not None:
            free()

    by_node = {}
    for r in range(size):
        by_node.setdefault(str(table[r][0]), []).append(r)

    placements = {}
    for node, ranks in by_node.items():
        comp_here = [r for r in ranks if r in compute]
        if pool and not legacy:
            capacity = len(pool) * m // k
            if len(comp_here) > capacity:
                raise ValueError(
                    f"node {node}: {len(comp_here)} compute ranks but the per-node GPU pool "
                    f"{pool} supports at most {capacity} "
                    f"(gpus_per_rank={k}, ranks_per_gpu={m})"
                )
        for r in ranks:
            local_index = int(table[r][1])
            if r == head:
                role = RankRole.HEAD
            elif r == saver and saver != head:
                role = RankRole.SAVER
            elif r in compute:
                role = RankRole.COMPUTE
            else:
                role = RankRole.SPARE
            if role in (RankRole.SAVER, RankRole.SPARE):
                # builds on a device like everyone else, then releases it
                devices = (pool[local_index % len(pool)],) if pool else ()
                placements[r] = RankPlacement(r, role, node, local_index, devices, 0, 0, 0)
                continue
            i = comp_here.index(r)
            if not pool:
                devices, slot = (), 0
            elif legacy:
                devices, slot = tuple(pool), 0
            elif k > 1:
                devices, slot = tuple(pool[i * k : (i + 1) * k]), 0
            else:
                devices, slot = (pool[i // m],), i % m
            bi = compute.index(r)
            placements[r] = RankPlacement(
                r, role, node, local_index, devices, slot, bi * block, (bi + 1) * block
            )
    return WalkerBlockLayout(
        size=size,
        head_rank=head,
        saver_rank=saver,
        compute_ranks=tuple(compute),
        nwalkers=nwalkers,
        block=block,
        placements=placements,
        gpus_per_rank=k,
        ranks_per_gpu=m,
        legacy=bool(legacy),
    )


def derive_rank_seed(base_seed, layout, rank) -> int:
    """Per-compute-rank seed: ``SeedSequence(base).spawn(n_compute)[i]``, deterministic."""
    sequence = np.random.SeedSequence([int(base_seed), 0x5AFE])
    children = sequence.spawn(layout.n_compute)
    child = children[layout.fanout_rank(rank)]
    return int(child.generate_state(1, dtype=np.uint32)[0])
```

Add to `communication/__init__.py`:

```python
from .ranks import (
    LEGACY_ENV,
    RankPlacement,
    RankRole,
    WalkerBlockLayout,
    build_layout,
    derive_rank_seed,
    resolve_roles,
)
```
and extend `__all__` accordingly.

- [ ] **Step 4: Run the test to verify it passes**

Run the Step 2 command. Expected: `Ran 8 tests ... OK`.

- [ ] **Step 5: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/communication/ranks.py, src/lisatools/globalfit/communication/__init__.py, tests/test_rank_layout.py".

---

### Task 5: Device pinning, `prepare_rank`, abort hook, rank tags (`communication/ranks.py`, part 2)

**Files:**
- Modify: `src/lisatools/globalfit/communication/ranks.py` (append), `src/lisatools/globalfit/communication/__init__.py`
- Test: `tests/test_rank_layout.py` (append two test classes)

**Interfaces:**
- Consumes: Task 4; a "fit" object with `.general.nwalkers`, `.general.gpus`, optional `.general.gpus_per_rank` / `.general.ranks_per_gpu`, `.main_rank`, `.built` (as `StockGlobalFit` exposes: `stock/base.py:728` `built`, `:339-340` `head_rank`/`main_rank`, `stock/erebor/fit.py:61,287` `nwalkers`/`gpus`).
- Produces: `select_rank_device(layout, rank, *, environ=None, device_count_fn=None, set_device_fn=None, logger=None) -> (gpus: list[int] | None, mode: str)` with modes `"cpu"`, `"legacy"`, `"visible"`, `"setdevice"`; `prepare_rank(fit, comm, *, logger=None, environ=None, device_count_fn=None, set_device_fn=None) -> WalkerBlockLayout` (idempotent; sets `fit.general.gpus`, `fit.rank_layout`, `fit.rank_device_mode`; raises if the fit is already built with size > 1); `install_mpi_abort_on_error(comm)`; `rank_tag(layout, rank) -> str` (`"r0/head"`, `"r1/c1"`, `"r2/saver"`, `"r1/spare"`); `prefix_stdout(tag)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rank_layout.py` (add `from lisatools.globalfit.communication.ranks import prepare_rank, rank_tag, select_rank_device` to the imports):

```python
class SelectRankDeviceTest(unittest.TestCase):
    def _layout(self):
        return FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]

    def test_visible_mode_narrows_env_and_renumbers(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_visible_mode_maps_through_a_prenarrowed_env(self):
        env = {"CUDA_VISIBLE_DEVICES": "2,3"}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_no_runtime_probe_still_counts_as_visible(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(), 0, environ=env, device_count_fn=lambda: None, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_fallback_setdevice_when_the_runtime_is_already_up(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        pinned = []
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 2, set_device_fn=pinned.append
        )
        self.assertEqual((gpus, mode), ([1], "setdevice"))
        self.assertEqual(pinned, [1])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")

    def test_cpu_and_legacy(self):
        cpu = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertEqual(select_rank_device(cpu, 0, environ={}), (None, "cpu"))
        legacy = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=True))[0]
        env = {}
        self.assertEqual(select_rank_device(legacy, 0, environ=env), ([0, 1], "legacy"))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_pool_index_outside_visible_set_raises(self):
        env = {"CUDA_VISIBLE_DEVICES": "5"}
        with self.assertRaises(ValueError):
            select_rank_device(
                self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
            )


class _General:
    def __init__(self):
        self.nwalkers = 4
        self.gpus = [0, 1]
        self.gpus_per_rank = 1
        self.ranks_per_gpu = 1


class _Fit:
    main_rank = 0
    built = False

    def __init__(self):
        self.general = _General()


class PrepareRankTest(unittest.TestCase):
    def test_sets_local_gpus_layout_and_mode_on_every_rank(self):
        def fn(rank, comm):
            fit = _Fit()
            env = {}
            lay = prepare_rank(
                fit, comm, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
            )
            again = prepare_rank(fit, comm, environ=env)  # idempotent: no second layout
            return (
                fit.general.gpus,
                fit.rank_device_mode,
                env.get("CUDA_VISIBLE_DEVICES"),
                lay.block_of(rank),
                fit.rank_layout is lay and again is lay,
                rank_tag(lay, rank),
            )

        out = FakeWorld(3).run(fn)
        self.assertEqual(out[0], ([0], "visible", "0", (0, 2), True, "r0/head"))
        self.assertEqual(out[1], ([0], "visible", "1", (2, 4), True, "r1/c1"))
        self.assertEqual(out[2][:3], ([0], "visible", "0"))
        self.assertEqual(out[2][5], "r2/saver")

    def test_refuses_to_run_after_build_with_several_ranks(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.built = True
            return prepare_rank(fit, comm, environ={}, device_count_fn=lambda: 1)

        with self.assertRaises(RuntimeError):
            FakeWorld(2).run(fn)

    def test_cpu_fit_keeps_gpus_none(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.general.gpus = None
            prepare_rank(fit, comm, environ={})
            return fit.general.gpus, fit.rank_device_mode

        self.assertEqual(FakeWorld(2).run(fn)[1], (None, "cpu"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t5.log python -m unittest tests.test_rank_layout -v; tail -n 25 .wtenv/t5.log
```
Expected: `ImportError: cannot import name 'prepare_rank'`.

- [ ] **Step 3: Implement device pinning, `prepare_rank`, the abort hook and the tags**

Append to `src/lisatools/globalfit/communication/ranks.py`:

```python
# --------------------------------------------------------------------------
# device pinning (must run BEFORE any CUDA initialisation on the rank)
# --------------------------------------------------------------------------

_CUDART_NAMES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.dylib",
)


def _load_cudart():
    import ctypes

    for name in _CUDART_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cudart_device_count():
    """``cudaGetDeviceCount`` via ctypes, or ``None`` when no runtime is loadable."""
    import ctypes

    lib = _load_cudart()
    if lib is None:
        return None
    count = ctypes.c_int(0)
    if lib.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return None
    return int(count.value)


def _cudart_set_device(device):
    lib = _load_cudart()
    if lib is None:
        raise RuntimeError("cudaSetDevice fallback requested but no CUDA runtime is loadable")
    if lib.cudaSetDevice(int(device)) != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed")


def select_rank_device(
    layout,
    rank,
    *,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
    logger=None,
):
    """Pin this process to its devices. Returns ``(gpus, mode)``.

    ``gpus`` is what ``general_info.gpus`` must become on this rank
    (``None`` on CPU). Modes: ``"cpu"``; ``"legacy"`` (pool untouched);
    ``"visible"`` (``CUDA_VISIBLE_DEVICES`` narrowed, the rank sees its
    devices as ``0..k-1``); ``"setdevice"`` (the runtime was already
    initialised, e.g. by a CUDA-aware MPI, so the env is restored and
    ``cudaSetDevice`` pins the first device; the rank keeps the pool ids).
    """
    environ = os.environ if environ is None else environ
    device_count_fn = _cudart_device_count if device_count_fn is None else device_count_fn
    set_device_fn = _cudart_set_device if set_device_fn is None else set_device_fn
    placement = layout.placements[int(rank)]
    if not placement.devices:
        return None, "cpu"
    if layout.legacy:
        return list(placement.devices), "legacy"

    previous = environ.get("CUDA_VISIBLE_DEVICES")
    if previous:
        # pool ids index the CURRENTLY visible set (Slurm may already have narrowed it)
        visible = [v.strip() for v in previous.split(",") if v.strip()]
        try:
            physical = [visible[d] for d in placement.devices]
        except IndexError:
            raise ValueError(
                f"rank {rank}: device pool ids {list(placement.devices)} exceed the visible "
                f"set CUDA_VISIBLE_DEVICES={previous!r}"
            ) from None
    else:
        physical = [str(d) for d in placement.devices]
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)

    count = device_count_fn()
    if count is None or count == len(placement.devices):
        return list(range(len(placement.devices))), "visible"

    # the runtime saw the pool before we narrowed the env: fall back to cudaSetDevice
    if previous is None:
        environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environ["CUDA_VISIBLE_DEVICES"] = previous
    set_device_fn(placement.devices[0])
    environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if logger is not None:
        logger.warning(
            "rank %d: CUDA runtime already initialised (%d devices visible); pinned device %d "
            "with cudaSetDevice instead of CUDA_VISIBLE_DEVICES",
            rank,
            count,
            placement.devices[0],
        )
    return list(placement.devices), "setdevice"


def prepare_rank(
    fit,
    comm,
    *,
    logger=None,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
):
    """THE driver hook: resolve the layout and pin this rank's device BEFORE ``fit.build()``.

    Idempotent (returns the stored layout on a second call). Reads the
    pre-build settings ``fit.general.nwalkers`` / ``.gpus`` (the per-node
    pool) / ``.gpus_per_rank`` / ``.ranks_per_gpu``; writes the rank-local
    ``fit.general.gpus``, ``fit.rank_layout`` and ``fit.rank_device_mode``.
    """
    layout = getattr(fit, "rank_layout", None)
    if layout is not None:
        return layout
    if int(comm.Get_size()) > 1 and bool(getattr(fit, "built", False)):
        raise RuntimeError(
            "prepare_rank must run BEFORE fit.build(): the build allocates on the device"
        )
    general = fit.general
    pool = list(general.gpus) if getattr(general, "gpus", None) else []
    layout = build_layout(
        comm,
        int(general.nwalkers),
        pool,
        gpus_per_rank=int(getattr(general, "gpus_per_rank", 1) or 1),
        ranks_per_gpu=int(getattr(general, "ranks_per_gpu", 1) or 1),
        main_rank=int(getattr(fit, "main_rank", 0) or 0),
    )
    gpus, mode = select_rank_device(
        layout,
        int(comm.Get_rank()),
        environ=environ,
        device_count_fn=device_count_fn,
        set_device_fn=set_device_fn,
        logger=logger,
    )
    if pool:
        general.gpus = gpus
    fit.rank_layout = layout
    fit.rank_device_mode = mode
    if logger is not None:
        logger.info("%s\nrank %d device mode: %s", layout.describe(), comm.Get_rank(), mode)
    return layout


# --------------------------------------------------------------------------
# failure + logging helpers
# --------------------------------------------------------------------------


def install_mpi_abort_on_error(comm):
    """Route any rank's uncaught exception (main thread or threads) through ``comm.Abort``.

    Under ``mpiexec`` a crashed rank otherwise leaves the survivors blocked in
    their receive loops for the whole allocation. Returns ``comm`` when it
    installed the hooks, ``None`` for a single-process run (normal Python
    exception behaviour is kept there).
    """
    if comm is None or int(comm.Get_size()) < 2:
        return None
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    def _hook(exc_type, exc, tb):
        try:
            print(
                f"\n[MPI-ABORT] rank {rank} of {size} raised {exc_type.__name__}: {exc}\n"
                "[MPI-ABORT] aborting ALL ranks so the job fails fast instead of hanging.",
                file=sys.stderr,
                flush=True,
            )
            if exc_type is not KeyboardInterrupt:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - never mask the abort
            pass
        finally:
            try:
                comm.Abort(1)
            except Exception:  # noqa: BLE001
                os._exit(1)

    sys.excepthook = _hook
    if hasattr(threading, "excepthook"):

        def _thread_hook(args):
            _hook(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = _thread_hook
    return comm


def rank_tag(layout, rank) -> str:
    role = layout.role_of(rank)
    if role == RankRole.HEAD:
        return f"r{int(rank)}/head"
    if role == RankRole.SAVER:
        return f"r{int(rank)}/saver"
    if role == RankRole.SPARE:
        return f"r{int(rank)}/spare"
    return f"r{int(rank)}/c{layout.fanout_rank(rank)}"


class _PrefixedStream:
    """Line-prefixing wrapper for a text stream (worker-rank stdout)."""

    def __init__(self, stream, prefix):
        self._stream = stream
        self._prefix = prefix
        self._at_line_start = True

    def write(self, text):
        out = []
        for chunk in str(text).splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._prefix)
            out.append(chunk)
            self._at_line_start = chunk.endswith("\n")
        self._stream.write("".join(out))

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._stream, name)


def prefix_stdout(tag):
    """Prefix every stdout line of this process with ``[tag] `` (idempotent)."""
    if not isinstance(sys.stdout, _PrefixedStream):
        sys.stdout = _PrefixedStream(sys.stdout, f"[{tag}] ")
    return sys.stdout
```

Add `install_mpi_abort_on_error`, `prefix_stdout`, `prepare_rank`, `rank_tag`, `select_rank_device` to the `ranks` import in `communication/__init__.py` and to `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

Run the Step 2 command. Expected: `Ran 17 tests ... OK`.

- [ ] **Step 5: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/communication/ranks.py, src/lisatools/globalfit/communication/__init__.py, tests/test_rank_layout.py".

---

### Task 6: Head-side fan-out and compute-side service loop (`communication/fanout.py`)

**Files:**
- Create: `src/lisatools/globalfit/communication/fanout.py`
- Modify: `src/lisatools/globalfit/communication/__init__.py`
- Test: `tests/test_fanout_passthrough.py`, `tests/test_fanout_fakecomm.py`

**Interfaces:**
- Consumes: Task 4 (`WalkerBlockLayout.block_of`, `worker_ranks`, `fanout_rank`, `is_single`, `digest`); a fan-out communicator (`isend/recv/send/allgather`); move objects exposing `gf_serve(op, payload, clock, model)`.
- Produces:
  - `WalkerFanout(comm, layout, rank, *, model=None, logger=None)` with `.single`, `.is_head`, `.clock` (dict: `iteration, stage, stage_kind, move, call_index, seq, seed_base`), `.enter_stage(name, kind)`, `.note_iteration(i)`, `.run(op, *, move=None, per_rank_payload, local_body, merge, shared=None)`, `.allgather_walker_vector(local_1d) -> np.ndarray`, `.ping()`, `.stop()`.
  - `run` contract: `per_rank_payload(rank, w0, w1) -> payload` (head only); `local_body(payload, model) -> result` (every compute rank, its own block; on the head with `self.model`); `merge(results: dict[world_rank -> result])`; single mode = `merge({head: local_body(per_rank_payload(head, 0, N), model)})` with NO communicator access and no copies; multi mode = `isend` per worker, head computes its block, `recv` replies in rank order, `RemoteWorkerError` on any `ok=False`, then `merge`.
  - `ComputeService(comm, layout, rank, *, registry, model=None, builtins=None, logger=None)` with `.handle(cmd) -> reply` and `.serve() -> int` (commands served before `{"op": "stop"}`); dispatch `builtins[op](payload, clock, model)` first, else `registry[cmd["move"]].gf_serve(op, payload, clock, model)` after stamping `move.gf_stage_kind = clock["stage_kind"]`; builtin `"ping"` always present.
  - `RemoteWorkerError(rank, op, move, message, remote_traceback)`; `concat_blocks(results, layout) -> np.ndarray`; `STOP_OP = "stop"`.
  - Schemas: `cmd = {"seq", "op", "move", "clock", "payload", "shared"}`, `reply = {"seq", "rank", "ok", "result", "wall_s", "error": None | {"type", "msg", "traceback"}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fanout_passthrough.py`:

```python
"""With one compute rank the fan-out is a direct call: no comm, no copies."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import WalkerFanout
from lisatools.globalfit.communication.ranks import build_layout


class _NeverComm:
    """A communicator whose every use is a test failure."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"comm.{name} was touched in single mode")


class PassthroughTest(unittest.TestCase):
    def setUp(self):
        self.layout = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertTrue(self.layout.is_single())

    def test_single_is_a_direct_call_with_the_same_objects(self):
        fo = WalkerFanout(_NeverComm(), self.layout, 0, model="MODEL")
        fo.enter_stage("pe", "pe_kind")
        payload = {"x": np.arange(4)}
        seen = {}

        def body(p, model):
            seen["payload"] = p
            seen["model"] = model
            seen["clock"] = dict(fo.clock)
            return {"y": p["x"] * 2}

        out = fo.run(
            "op",
            move="m",
            per_rank_payload=lambda r, w0, w1: payload,
            local_body=body,
            merge=lambda results: results,
        )
        self.assertIs(seen["payload"], payload)
        self.assertEqual(seen["model"], "MODEL")
        self.assertEqual(seen["clock"]["stage"], "pe")
        self.assertEqual(list(out), [0])
        np.testing.assert_array_equal(out[0]["y"], [0, 2, 4, 6])

    def test_allgather_and_stop_and_ping_are_local(self):
        fo = WalkerFanout(_NeverComm(), self.layout, 0)
        np.testing.assert_array_equal(fo.allgather_walker_vector(np.arange(4)), np.arange(4))
        fo.stop()
        self.assertEqual(fo.ping(), {0: self.layout.digest()})


if __name__ == "__main__":
    unittest.main()
```

Create `tests/test_fanout_fakecomm.py`:

```python
"""Head-directed commands over the fake communicator: round trip, errors, stop, isolation."""

import time
import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import (
    ComputeService,
    RemoteWorkerError,
    WalkerFanout,
    concat_blocks,
)
from lisatools.globalfit.communication.ranks import RankRole, build_layout


class _StubMove:
    gf_move_name = "stub"

    def __init__(self):
        self.calls = []
        self.gf_stage_kind = None

    def gf_serve(self, op, payload, clock, model):
        self.calls.append((op, self.gf_stage_kind, clock["call_index"]))
        if op == "boom":
            raise ValueError("kaboom")
        return {"rank_sum": float(np.sum(payload["x"])), "model": model}


def _run_world(size, nodes, nwalkers, head_fn, saver_fn=None):
    world = FakeWorld(size, nodes=nodes)
    stubs = {}

    def fn(rank, comm):
        layout = build_layout(comm, nwalkers, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return saver_fn(rank, comm) if saver_fn else "saver-idle"
        model = f"model-{rank}"
        if role == RankRole.HEAD:
            fo = WalkerFanout(fcomm, layout, rank, model=model)
            fo.enter_stage("pe", "pe_kind")
            try:
                return head_fn(fo, layout)
            finally:
                fo.stop()
        stubs[rank] = _StubMove()
        service = ComputeService(fcomm, layout, rank, registry={"stub": stubs[rank]}, model=model)
        return service.serve()

    return world.run(fn), stubs


class FanoutFakeCommTest(unittest.TestCase):
    def test_round_trip_in_walker_order_and_stage_stamp(self):
        def head(fo, layout):
            def payload(rank, w0, w1):
                return {"x": np.arange(w0, w1)}

            def body(p, model):
                return {"rank_sum": float(np.sum(p["x"])), "model": model}

            out = fo.run("score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r)
            out2 = fo.run("score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r)
            return {r: rep["rank_sum"] for r, rep in out.items()}, out[0]["model"], out[1]["model"], out2 == out

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        sums, m0, m1, same = out[0]
        self.assertEqual(sums, {0: 1.0, 1: 5.0})
        self.assertEqual((m0, m1), ("model-0", "model-1"))
        self.assertTrue(same)
        self.assertEqual(out[1], 2)  # the worker served two commands before stop
        self.assertEqual(out[2], "saver-idle")
        self.assertEqual(stubs[1].calls, [("score", "pe_kind", 1), ("score", "pe_kind", 2)])

    def test_worker_error_surfaces_on_the_head_with_traceback(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "boom",
                    move="stub",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            err = cm.exception
            return err.rank, err.op, err.move, "kaboom" in err.remote_traceback

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], (1, "boom", "stub", True))
        self.assertEqual(out[1], 1)

    def test_unknown_move_is_an_error_reply_not_a_hang(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "score",
                    move="nope",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            return cm.exception.error["type"]

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], "KeyError")

    def test_saver_never_sees_fanout_traffic(self):
        def saver(rank, comm):
            time.sleep(0.2)
            return comm.iprobe(source=0)

        def head(fo, layout):
            return fo.run(
                "score",
                move="stub",
                per_rank_payload=lambda r, w0, w1: {"x": np.ones(2)},
                local_body=lambda p, m: {"rank_sum": 2.0},
                merge=lambda r: sorted(r),
            )

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head, saver_fn=saver)
        self.assertEqual(out[0], [0, 1])
        self.assertFalse(out[2])

    def test_ping_checks_layout_digests(self):
        (out, _stubs) = _run_world(3, [0, 0, 0], 4, lambda fo, layout: fo.ping())
        self.assertEqual(set(out[0].values()), {out[0][0]})
        self.assertEqual(sorted(out[0]), [0, 1])

    def test_allgather_walker_vector_is_a_setup_phase_collective(self):
        world = FakeWorld(3)

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return None
            fo = WalkerFanout(fcomm, layout, rank)
            w0, w1 = layout.block_of(rank)
            return fo.allgather_walker_vector(np.arange(w0, w1) * 10.0)

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out[1], out[0])

    def test_concat_blocks_follows_compute_rank_order(self):
        layout = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        results = {3: np.array([6, 7]), 0: np.array([0, 1]), 2: np.array([4, 5]), 1: np.array([2, 3])}
        np.testing.assert_array_equal(concat_blocks(results, layout), np.arange(8))

    def test_run_on_a_worker_raises(self):
        layout = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[1]
        fo = WalkerFanout(None, layout, 1)
        with self.assertRaises(RuntimeError):
            fo.run("x", per_rank_payload=lambda r, w0, w1: None, local_body=lambda p, m: None, merge=lambda r: r)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t6.log python -m unittest tests.test_fanout_passthrough tests.test_fanout_fakecomm -v; tail -n 20 .wtenv/t6.log
```
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.communication.fanout'`.

- [ ] **Step 3: Implement the fan-out and the service loop**

Create `src/lisatools/globalfit/communication/fanout.py`:

```python
"""Head-directed fan-out over walker blocks + the compute-rank service loop.

Sampling phase: the head calls :meth:`WalkerFanout.run` inside a proposal;
each computation rank receives one command, runs the same body on its own
walker block and replies; the head merges. With ONE compute rank ``run`` is
a direct call (no communicator, no pickling, no copies). Setup phase:
:meth:`WalkerFanout.allgather_walker_vector` is a symmetric collective every
compute rank calls at the same program point.

Command / reply schemas (host numpy, pickled by mpi4py ``send``/``recv``)::

    cmd   = {"seq", "op", "move", "clock", "payload", "shared"}
    reply = {"seq", "rank", "ok", "result", "wall_s",
             "error": None | {"type", "msg", "traceback"}}
"""

from __future__ import annotations

import collections
import time
import traceback

import numpy as np

STOP_OP = "stop"
PING_OP = "ping"


class RemoteWorkerError(RuntimeError):
    """A computation rank's command failed; carries the remote traceback text."""

    def __init__(self, rank, op, move, error):
        self.rank = int(rank)
        self.op = op
        self.move = move
        self.error = dict(error or {})
        self.remote_traceback = self.error.get("traceback", "")
        super().__init__(
            f"rank {self.rank} failed op={op!r} move={move!r}: "
            f"{self.error.get('type')}: {self.error.get('msg')}\n{self.remote_traceback}"
        )


def concat_blocks(results, layout):
    """Concatenate per-rank 1-D results in compute-rank (== walker) order."""
    return np.concatenate([np.asarray(results[r]) for r in layout.compute_ranks])


def _reply(seq, rank, ok, result, wall_s, error=None):
    return {"seq": seq, "rank": rank, "ok": ok, "result": result, "wall_s": wall_s, "error": error}


class WalkerFanout:
    """Head-side fan-out (usable on every compute rank for the collectives)."""

    def __init__(self, comm, layout, rank, *, model=None, logger=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.model = model
        self.logger = logger
        self.head = layout.head_rank
        self.is_head = self.rank == self.head
        self.single = layout.is_single()
        self.seq = 0
        self.clock = {
            "iteration": 0,
            "stage": None,
            "stage_kind": None,
            "move": None,
            "call_index": 0,
            "seq": 0,
            "seed_base": None,
        }
        self._call_index = collections.Counter()

    # -- clock -----------------------------------------------------------
    def enter_stage(self, name, kind):
        self.clock["stage"] = name
        self.clock["stage_kind"] = kind

    def note_iteration(self, iteration):
        self.clock["iteration"] = int(iteration)

    # -- commands --------------------------------------------------------
    def run(self, op, *, move=None, per_rank_payload, local_body, merge, shared=None):
        if not self.is_head:
            raise RuntimeError("WalkerFanout.run is head-only; compute ranks serve commands")
        key = (move, op)
        self._call_index[key] += 1
        clock = dict(self.clock, move=move, call_index=self._call_index[key], seq=self.seq)
        w0, w1 = self.layout.block_of(self.head)
        if self.single:
            self.clock.update(clock)
            return merge({self.head: local_body(per_rank_payload(self.head, w0, w1), self.model)})

        self.seq += 1
        clock["seq"] = self.seq
        self.clock.update(clock)
        requests = []
        for r in self.layout.worker_ranks:
            rw0, rw1 = self.layout.block_of(r)
            cmd = {
                "seq": self.seq,
                "op": op,
                "move": move,
                "clock": clock,
                "payload": per_rank_payload(r, rw0, rw1),
                "shared": shared,
            }
            requests.append(self.comm.isend(cmd, dest=self.layout.fanout_rank(r)))
        t0 = time.perf_counter()
        local = local_body(per_rank_payload(self.head, w0, w1), self.model)
        head_wall = time.perf_counter() - t0
        for req in requests:
            req.wait()
        replies = {self.head: _reply(self.seq, self.head, True, local, head_wall)}
        for r in self.layout.worker_ranks:
            replies[r] = self.comm.recv(source=self.layout.fanout_rank(r))
        failures = [rep for rep in replies.values() if not rep["ok"]]
        if failures:
            if self.logger is not None:
                for rep in failures:
                    self.logger.error(
                        "rank %d failed op=%r move=%r:\n%s",
                        rep["rank"],
                        op,
                        move,
                        (rep.get("error") or {}).get("traceback", ""),
                    )
            first = failures[0]
            raise RemoteWorkerError(first["rank"], op, move, first.get("error"))
        bad_seq = [rep["rank"] for rep in replies.values() if rep["seq"] != self.seq]
        if bad_seq:
            raise RuntimeError(f"fan-out sequence mismatch from ranks {bad_seq} (op={op!r})")
        if self.logger is not None:
            worst = max(rep["wall_s"] for rep in replies.values())
            self.logger.debug(
                "[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f", op, move, head_wall, worst
            )
        return merge({r: rep["result"] for r, rep in replies.items()})

    def ping(self):
        """Every compute rank's layout digest, keyed by world rank; raises on mismatch."""
        digests = self.run(
            PING_OP,
            per_rank_payload=lambda r, w0, w1: None,
            local_body=lambda payload, model: self.layout.digest(),
            merge=lambda results: results,
        )
        if len(set(digests.values())) != 1:
            raise RuntimeError(f"rank layouts disagree: {digests}")
        return digests

    def stop(self):
        if self.single or not self.is_head:
            return
        for r in self.layout.worker_ranks:
            self.comm.send(
                {"seq": self.seq, "op": STOP_OP, "move": None, "clock": dict(self.clock),
                 "payload": None, "shared": None},
                dest=self.layout.fanout_rank(r),
            )

    # -- setup-phase collective ------------------------------------------
    def allgather_walker_vector(self, local_1d):
        """Concatenate every compute rank's 1-D block vector in walker order (collective)."""
        local = np.asarray(local_1d)
        if self.single:
            return local
        parts = self.comm.allgather(local)  # fan-out comm ranks == compute-rank order
        return np.concatenate([np.asarray(p) for p in parts])


class ComputeService:
    """Compute-rank command loop: ``recv`` is the clock, ``{"op": "stop"}`` exits."""

    def __init__(self, comm, layout, rank, *, registry, model=None, builtins=None, logger=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.registry = dict(registry)
        self.model = model
        self.logger = logger
        self.builtins = {PING_OP: lambda payload, clock, model: self.layout.digest()}
        self.builtins.update(builtins or {})
        self._head_local = layout.fanout_rank(layout.head_rank)

    def handle(self, cmd):
        seq = cmd.get("seq")
        op = cmd["op"]
        move_name = cmd.get("move")
        clock = cmd.get("clock") or {}
        t0 = time.perf_counter()
        try:
            if op in self.builtins:
                result = self.builtins[op](cmd.get("payload"), clock, self.model)
            else:
                move = self.registry[move_name]
                if "stage_kind" in clock:
                    move.gf_stage_kind = clock["stage_kind"]
                result = move.gf_serve(op, cmd.get("payload"), clock, self.model)
            return _reply(seq, self.rank, True, result, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - reported to the head, never swallowed
            if self.logger is not None:
                self.logger.exception("rank %d: command %r for move %r failed", self.rank, op, move_name)
            error = {"type": type(exc).__name__, "msg": str(exc), "traceback": traceback.format_exc()}
            return _reply(seq, self.rank, False, None, time.perf_counter() - t0, error)

    def serve(self):
        served = 0
        while True:
            cmd = self.comm.recv(source=self._head_local)
            if cmd.get("op") == STOP_OP:
                return served
            reply = self.handle(cmd)
            self.comm.send(reply, dest=self._head_local)
            served += 1
```

Add to `communication/__init__.py`:

```python
from .fanout import PING_OP, STOP_OP, ComputeService, RemoteWorkerError, WalkerFanout, concat_blocks
```
and extend `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command. Expected: `Ran 10 tests ... OK`.

- [ ] **Step 5: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/communication/fanout.py, src/lisatools/globalfit/communication/__init__.py, tests/test_fanout_passthrough.py, tests/test_fanout_fakecomm.py".

---

### Task 7: Move-side hooks (`MoveBuildContext` fields, `GlobalFitMove.gf_serve`)

**Files:**
- Modify: `src/lisatools/globalfit/moves/globalfitmove.py:68-89` (`MoveBuildContext`), `:230-260` (`GlobalFitMove`)
- Test: `tests/test_move_build_context_layout.py`

**Interfaces:**
- Produces: `MoveBuildContext.layout`, `.fanout`, `.rank`, `.state_local` (all default `None`; `__post_init__` fills `layout`/`fanout`/`rank` from `ctx.curr.rank_layout` / `ctx.curr.fanout` / `ctx.curr.rank` when those exist and the field was left `None`); `GlobalFitMove.gf_serve(op, payload, clock, model)` raising `NotImplementedError` naming the move. Plan 2 sets `curr.rank_layout` / `curr.fanout` / `curr.rank` / `state_local` in `run.py`; Plans 3-4 override `gf_serve`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_move_build_context_layout.py`:

```python
"""MoveBuildContext exposes the rank layout / fan-out; GlobalFitMove.gf_serve defaults to 'not served'."""

import unittest

from lisatools.globalfit.moves.globalfitmove import GlobalFitMove, MoveBuildContext


class _Curr:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class MoveBuildContextLayoutTest(unittest.TestCase):
    def _ctx(self, curr, **overrides):
        return MoveBuildContext(
            recipe=None, engine_info=None, curr=curr, acs=None, priors={}, state=None, **overrides
        )

    def test_fields_default_to_none_without_a_layout(self):
        ctx = self._ctx(_Curr())
        self.assertIsNone(ctx.layout)
        self.assertIsNone(ctx.fanout)
        self.assertIsNone(ctx.rank)
        self.assertIsNone(ctx.state_local)

    def test_fields_auto_fill_from_curr(self):
        curr = _Curr(rank_layout="LAYOUT", fanout="FANOUT", rank=3)
        ctx = self._ctx(curr)
        self.assertEqual((ctx.layout, ctx.fanout, ctx.rank), ("LAYOUT", "FANOUT", 3))

    def test_explicit_fields_win(self):
        curr = _Curr(rank_layout="LAYOUT", fanout="FANOUT", rank=3)
        ctx = self._ctx(curr, layout="MINE", rank=0, state_local="SLICE")
        self.assertEqual((ctx.layout, ctx.fanout, ctx.rank, ctx.state_local), ("MINE", "FANOUT", 0, "SLICE"))


class _PlainMove(GlobalFitMove):
    gf_move_name = "plain"


class GfServeDefaultTest(unittest.TestCase):
    def test_default_gf_serve_raises_naming_the_move(self):
        move = _PlainMove(name="plain")
        with self.assertRaises(NotImplementedError) as cm:
            move.gf_serve("op", None, {}, None)
        self.assertIn("plain", str(cm.exception))
        self.assertIn("op", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t7.log python -m unittest tests.test_move_build_context_layout -v; tail -n 20 .wtenv/t7.log
```
Expected: FAIL with `TypeError: MoveBuildContext.__init__() got an unexpected keyword argument 'layout'` and `AttributeError: '_PlainMove' object has no attribute 'gf_serve'`.

- [ ] **Step 3: Add the fields and the default hook**

In `src/lisatools/globalfit/moves/globalfitmove.py`, extend the `MoveBuildContext` dataclass (after `nwalkers: typing.Optional[int] = None`, line 89):

```python
    #: multi-rank context (Plan 2 populates ``curr.rank_layout`` / ``curr.fanout``
    #: / ``curr.rank``; ``state_local`` is the head-side walker slice a
    #: builder reads when it must act on this rank's block only)
    layout: typing.Any = None
    fanout: typing.Any = None
    rank: typing.Optional[int] = None
    state_local: typing.Any = None

    def __post_init__(self):
        curr = self.curr
        if self.layout is None:
            self.layout = getattr(curr, "rank_layout", None)
        if self.fanout is None:
            self.fanout = getattr(curr, "fanout", None)
        if self.rank is None:
            self.rank = getattr(curr, "rank", None)
```

Also extend the class docstring's list with one line: "the multi-rank ``layout`` / ``fanout`` / ``rank`` and the head-side ``state_local`` slice (all ``None`` in a single-process run)".

In `GlobalFitMove`, add right after `_check_substate_consistency` (before `__init__` at line 287):

```python
    def gf_serve(self, op, payload, clock, model):
        """Serve one fan-out command on a computation rank.

        The multi-rank moves override this (GB: ``gb_run_proposal`` /
        ``gb_run_tempering`` / ``gb_finish``; addremove and PSD: ``propose``).
        ``model`` is this rank's ``GlobalFitInfo`` (local ACA, map, rank RNG).
        """
        raise NotImplementedError(
            f"move {getattr(self, 'gf_move_name', getattr(self, 'name', type(self).__name__))!r} "
            f"does not serve fan-out command {op!r}"
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run the Step 2 command. Expected: `Ran 4 tests ... OK`.

- [ ] **Step 5: Run the move-layer suites to prove nothing regressed**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t7b.log python -m unittest tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one -v; tail -n 15 .wtenv/t7b.log
```
Expected: all `OK` (these two suites construct `MoveBuildContext` and `GFCombineMove` through the recipe layer, so a broken dataclass or a bad `__post_init__` shows up here).

- [ ] **Step 6: Report ready to commit (do not commit)**

"ready to commit: src/lisatools/globalfit/moves/globalfitmove.py, tests/test_move_build_context_layout.py".

---

### Task 8: Whole-plan verification and handoff

**Files:** none new.

- [ ] **Step 1: Run every new suite together plus the existing state/move suites**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/plan1_all.log python -m unittest \
  tests.test_walkerslice_roundtrip tests.test_fakecomm tests.test_rank_layout \
  tests.test_fanout_passthrough tests.test_fanout_fakecomm tests.test_move_build_context_layout \
  tests.test_gf_substate_roundtrip tests.test_module_substate_reseed -v; tail -n 20 .wtenv/plan1_all.log
```
Expected: all `OK`, no skipped tests in the new modules.

- [ ] **Step 2: Import smoke without mpi4py**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/plan1_import.log python -c "import sys; sys.modules['mpi4py'] = None; import lisatools.globalfit.communication as c; print(sorted(c.__all__))"; tail -n 5 .wtenv/plan1_import.log
```
Expected: the sorted export list printed, no `ImportError` (proves the lazy-import rule).

- [ ] **Step 3: Format check**

Run:
```sh
.wtenv/wt_run.sh $PWD/src .wtenv/plan1_fmt.log python -m black --check --line-length 100 src/lisatools/globalfit/communication tests/test_walkerslice_roundtrip.py tests/test_fakecomm.py tests/test_rank_layout.py tests/test_fanout_passthrough.py tests/test_fanout_fakecomm.py tests/test_move_build_context_layout.py; tail -n 12 .wtenv/plan1_fmt.log
```
Expected: "would reformat" for nothing; if black reports files, run it without `--check` on those files and re-run the Step 1 suite.

- [ ] **Step 4: Final report (do not commit)**

`git status --short` and report the complete ready-to-commit list (two modified files, five new source files, six new test files), plus the interface summary Plan 2 consumes: `prepare_rank(fit, comm)` before build; `WalkerBlockLayout` accessors; `WalkerFanout.run/allgather_walker_vector/ping/stop`; `ComputeService.serve`; `slice_state/merge_state`; `MoveBuildContext.layout/fanout/rank/state_local`; `GlobalFitMove.gf_serve`.

---

## Self-review notes

- Spec coverage (WP0): `walker_axes`, `_bare_like`, `slice_walkers`, `merge_walkers`, `PerLeafLadderState` overrides, `slice_state`/`merge_state` with the `walker_inds` remap, `GBState.band_info` never sliced, the roundtrip test → Tasks 1-2.
- Spec coverage (WP1): `ranks.py` (roles incl. saver aliasing, blocked device assignment for both generalisations, over-subscription and divisibility hard errors, `describe`/digest, `make_fanout_comm`, `derive_rank_seed`, device pinning with the visible/setdevice modes, `prepare_rank`, abort hook, tags/prefixed stdout, legacy switch) → Tasks 4-5; `fanout.py` (`run` single/multi semantics, error propagation with remote traceback, seq check, `ping`, `stop`, `allgather_walker_vector`, `ComputeService` builtins + dispatch + stage stamp) → Task 6; `fakecomm.py` (thread per rank, pickle-copy sends, `Split`/`Split_type`, `Abort` poisoning pending receives) → Task 3; `MoveBuildContext` fields + `gf_serve` default → Task 7. The `GF_FANOUT_WAIT_WARN_S` hang diagnostics from the spec are deferred to Plan 2 (they need the real-MPI `iprobe` polling path and a logger wired by `run.py`).
- Type consistency: `per_rank_payload(rank, w0, w1)`, `local_body(payload, model)`, `merge(results: dict[world_rank -> result])`, `gf_serve(op, payload, clock, model)`, `ComputeService(comm, layout, rank, *, registry, model, builtins, logger)`, `build_layout(comm, nwalkers, gpu_pool, *, gpus_per_rank, ranks_per_gpu, main_rank, legacy)` are used with these exact signatures in every task and test.
