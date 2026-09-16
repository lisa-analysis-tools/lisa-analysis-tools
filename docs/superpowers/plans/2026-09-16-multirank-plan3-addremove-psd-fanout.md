# Multi-rank Plan 3: whole-`propose` fan-out for the addremove and PSD move families (WP4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ResidualAddOneRemoveOneMove` (MBH, EMRI, SOBBH) and `PSDMove` (psd, galfor, sgwb) run their ENTIRE `propose` per compute rank on that rank's walker block, the head merges the blocks and adapts each tempering ladder once from pooled swap tallies; with one compute rank nothing changes (bit-identical).

**Architecture:** A mixin (`moves/walkerfanout.py`) placed first in each move's MRO owns `propose`: single mode calls the renamed, unchanged body `propose_local`; multi mode slices the head's full state into per-rank blocks (`communication.walkerslice.slice_state`), ships them through `WalkerFanout.run("propose", move=<name>)`, every compute rank (head included) runs `propose_local` on its block against its own B-row ACA inside `gf_serve`, and the head merges with `merge_state`, concatenates `accepted`, SUM-reduces the per-ladder swap tallies and applies Eryn's `_get_ladder_adjustment` once per ladder per propose. Ranks never adapt (`tc.adaptive = False`); the head writes adapted ladders straight into the state's sub-state ladder arrays (which `merge_walkers` deliberately does not touch). A pre-existing PSD fancy-swap defect (`walker_inds` permuted with the coords) is fixed first because the port would turn it into an out-of-range ACA index.

**Tech Stack:** Python 3.12, numpy, Eryn `TemperatureControl`, the Plan 1/2 `communication` package (`WalkerFanout`, `ComputeService`, `slice_state`/`merge_state`, `FakeWorld`), unittest. CPU only on the laptop.

**Spec:** `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` (Decisions 5, 6, 7, 8, 9, 10; WP4; "Semantics that change only when n_compute > 1"). Code site map with anchors used below: the controller's `plan3-sitemap.md` (scratchpad; anchors are also repeated inline here as `file:line`, valid at branch HEAD 7a65cade — locate by content if they drift).

## Global Constraints

- Single-process / single-compute-rank runs are BIT-IDENTICAL: when `move.fanout is None or move.fanout.single`, `propose` calls the unchanged body directly — no slicing, no pickling, no copies, no new RNG draws, same statement order.
- Two notions of nwalkers: the engine/state/backend/`GFCombineMove.accepted` are GLOBAL (N); the ACA, move bodies and every `TemperatureControl` are LOCAL (B). On every rank — the head included — the moves are ALREADY built at `nwalkers = B` (`recipe.py:4059`, `recipe.py:2182` via `_local_nwalkers`), so in multi mode the head must slice its own block too.
- Ladders never adapt on ranks; the head adapts once per propose from pooled tallies (accepted semantic change: `tc.time += 1` per propose instead of per repeat; documented in `moves/walkerfanout.py`'s module docstring).
- Eigen-table sidecar is single-writer: only the head keeps `eigen_store_path`.
- Mid-iteration checkpoint (`midit_checkpoint.maybe_write`) is head-only and, in multi mode, per propose instead of per leaf.
- Payloads and replies are host numpy / plain Python only (pickled by mpi4py or copied by `FakeComm`).
- Never delete in-process multi-GPU code (`dcga`, `MultiGPU*` shims stay).
- Line length <= 100 on every added line (black is NOT installed: gate with `awk 'length > 100'`).
- Tests: CPU, tiny fixtures, one python process at a time (`.wtenv/wt_run.sh` after `conda activate deving`); never run `tests/test_gbspecial_flow.py`; smokes are gated by `RUN_GF_SMOKE=1`.
- Commit per task with the given message and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never commit `.superpowers/`, `.wtenv/`, or scratch files.

---

### Task 1: Pin `walker_inds` out of the PSD fancy swap (pre-existing defect, load-bearing for the port)

**Files:**
- Modify: `src/lisatools/globalfit/recipe.py` (`build_noise_moves`, the `TemperatureControl(...)` construction at ~2230-2237)
- Test: `tests/test_psd_fancy_swap_walker_inds.py` (create)

**Interfaces:**
- Produces: module-level `_noise_temperature_control(effective_ndim, nwalkers, *, betas, ntemps, Tmax) -> TemperatureControl` with `skip_swap_supp_names=["walker_inds"]`; `build_noise_moves` uses it.

Why: `PSDMove.run_move` passes the live `supps` (holding `walker_inds`) into `temperature_swaps` (`psdmove.py:2209`); Eryn scores a fancy (walker-permuting) swap against the UNPERMUTED destination rows (`Eryn/src/eryn/moves/tempering.py:778-780, 817`) and then, on acceptance, swaps every supplemental key not listed in `skip_swap_supp_names` along with the coords (`tempering.py:607-614`) — `walker_inds` included, because nothing sets the skip list (default `[]`, `tempering.py:266`). `compute_log_like` reads `walker_inds` as the ACA row (`psdmove.py:679, 1881`), so for the rest of that propose the swapped slots score against another walker's residual. With `do_fancy = (move_i % permute_every == 0)` and `move_i` starting at 0 (`psdmove.py:2195`) this fires on the FIRST repeat of every PSD propose. Under the port a permuted `walker_inds` would index an ACA row that does not exist on the rank.

- [ ] **Step 1: Write the failing test**

```python
"""The PSD ladder never permutes ``walker_inds`` on a fancy swap (pre-port defect)."""

import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl
from eryn.state import BranchSupplemental

from lisatools.globalfit.recipe import _noise_temperature_control

NT, NW, ND = 3, 6, 2
IDENTITY = np.tile(np.arange(NW), (NT, 1))


def _like(x_here, inds=None, supps=None, branch_supps=None, **kwargs):
    """Every candidate is much better than the incumbent (logl 0) -> every pair accepted."""
    shape = next(iter(x_here.values())).shape[:2]
    return np.full(shape, 50.0), None


def _fancy_swap_once(tc, seed=3):
    np.random.seed(seed)  # temperature_swaps draws iperm/i1perm/raccept from np.random
    x = {"psd": np.random.uniform(size=(NT, NW, 1, ND))}
    zeros = np.zeros((NT, NW))
    supps = BranchSupplemental(
        {"walker_inds": IDENTITY.copy()}, base_shape=(NT, NW), copy=True
    )
    out = tc.temperature_swaps(
        x, zeros.copy(), zeros.copy(), zeros.copy(),
        supps=supps, branch_supps={"psd": None}, compute_log_like=_like,
        fancy_swap=True, permute_here=True,
    )
    # returns (x, logP, logl, logp, inds, blobs, supps, branch_supps)
    return np.asarray(out[6]["walker_inds"])


class PSDFancySwapWalkerIndsTest(unittest.TestCase):
    def test_noise_control_pins_walker_inds(self):
        tc = _noise_temperature_control(ND, NW, betas=None, ntemps=NT, Tmax=None)
        self.assertEqual(list(tc.skip_swap_supp_names), ["walker_inds"])
        self.assertEqual(int(tc.nwalkers), NW)
        self.assertFalse(tc.permute)
        np.testing.assert_array_equal(_fancy_swap_once(tc), IDENTITY)

    def test_default_control_would_permute_walker_inds(self):
        # negative control: the same swap on an unpinned ladder moves the labels,
        # so the positive test above is not vacuous
        loose = TemperatureControl(ND, NW, ntemps=NT, permute=False)
        self.assertFalse(np.array_equal(_fancy_swap_once(loose), IDENTITY))


if __name__ == "__main__":
    unittest.main()
```

If `test_default_control_would_permute_walker_inds` happens to draw identity permutations with `seed=3`, change the seed until it permutes (the positive test must stay green for every seed). If `TemperatureControl` stores the list under a different attribute name than `skip_swap_supp_names`, read `tempering.py:254-320` and adapt the attribute read (the constructor kwarg name is `skip_swap_supp_names`).

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t1.log python -m unittest tests.test_psd_fancy_swap_walker_inds -v; tail -n 20 .wtenv/p3t1.log`
Expected: `ImportError: cannot import name '_noise_temperature_control'`.

- [ ] **Step 3: Implement**

In `recipe.py`, next to `_local_nwalkers` / `_local_walker_block`:

```python
def _noise_temperature_control(effective_ndim, nwalkers, *, betas, ntemps, Tmax):
    """The ladder shared by the psd/galfor/sgwb search and PE moves.

    ``walker_inds`` is pinned OUT of the fancy (walker-permuting) swap. It is the
    ACA row ``PSDMove.compute_log_like`` scores against (psdmove.py ~679/1881);
    Eryn scores a fancy swap against the UNPERMUTED destination rows
    (tempering.py ~778-817) and would then swap the label together with the
    coords (tempering.py ~607-614), leaving slot ``w`` pointing at another
    walker's residual for the rest of the propose -- and, under several compute
    ranks, at an ACA row that does not exist on the rank.
    """
    return TemperatureControl(
        effective_ndim,
        nwalkers,
        betas=betas,
        ntemps=ntemps,
        Tmax=Tmax,
        permute=False,
        skip_swap_supp_names=["walker_inds"],
    )
```

Replace the construction in `build_noise_moves`:

```python
    temperature_control = _noise_temperature_control(
        effective_ndim, nwalkers, betas=lead_info.betas, ntemps=ntemps, Tmax=Tmax
    )
```

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t1b.log python -m unittest tests.test_psd_fancy_swap_walker_inds tests.test_noise_split_moves tests.test_psd_delayed_acceptance tests.test_recipe_local_block -v; tail -n 25 .wtenv/p3t1b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/recipe.py tests/test_psd_fancy_swap_walker_inds.py
git commit -m "fix(recipe): pin walker_inds out of the PSD fancy swap (slot label followed the coords)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `WalkerFanoutMixin` — whole-`propose` fan-out, tested with a stub move

**Files:**
- Create: `src/lisatools/globalfit/moves/walkerfanout.py`
- Test: `tests/test_walkerfanout_mixin.py` (create)

**Interfaces:**
- Consumes: `WalkerFanout` (`.single`, `.is_head`, `.layout.compute_ranks`, `.layout.block_of(rank)`, `.clock`, `.run(op, *, move, per_rank_payload, local_body, merge)`), `slice_state(full, w0, w1, *, sub_states=[...])`, `merge_state(full, part, w0, w1)`, `GFState(state, copy=True)`, `ModuleSubState.delta_counter_names`.
- Produces: `WalkerFanoutMixin` with class attrs `fanout`, `gf_rank`, `fanout_branches`, `fanout_assigns_counters`, `_fanout_body`; methods `install_walker_fanout(curr)`, `fanout_active` (property), `propose`, `fanout_propose`, `gf_serve`; subclass hooks `fanout_temperature_controls()`, `fanout_payload_extra()`, `fanout_apply_extra(extra)`, `fanout_reply_extra(part)`, `fanout_merge_extra(replies, new_state)`; helper `pooled_ladder_step(tc, betas0, acc, prop)`; constant `PROPOSE_OP = "propose"`. Every `TemperatureControl` it disables gets `gf_configured_adaptive` (its configured value) stamped once.

- [ ] **Step 1: Write the failing test**

```python
"""WalkerFanoutMixin: direct call with one compute rank; slice/run/merge with several."""

import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin, pooled_ladder_step
from lisatools.globalfit.state import GFState
from tests.test_gf_substate_roundtrip import NTEMPS, NWALKERS, make_state


class _Curr:
    def __init__(self, fanout, rank):
        self.fanout = fanout
        self.rank = rank


class _StubMove(WalkerFanoutMixin):
    """A move whose body is trivial but leaves fingerprints the merge must preserve."""

    gf_move_name = "stub"
    fanout_branches = ["mbh"]
    fanout_assigns_counters = True

    def __init__(self, tag):
        self.tag = float(tag)
        self.tc = TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.eigen_store_path = "store.h5"
        self.applied = []
        self.merged = None

    def fanout_temperature_controls(self):
        return [self.tc]

    def fanout_payload_extra(self):
        return {"tick": 7}

    def fanout_apply_extra(self, extra):
        self.applied.append(extra["tick"])

    def fanout_reply_extra(self, part):
        return {"nw": int(part.branches["mbh"].coords.shape[1])}

    def fanout_merge_extra(self, replies, new_state):
        self.merged = replies

    def propose_local(self, model, state):
        new = GFState(state, copy=True)
        new.branches["mbh"].coords[...] += 1.0
        nw = new.log_like.shape[1]
        new.log_like[...] = np.arange(nw)[None, :] + self.tag  # tag = which rank scored it
        sub = new.sub_states["mbh"]
        sub.in_model_accepted[...] = 1
        return new, np.ones(new.log_like.shape, dtype=bool)


class MixinSingleTest(unittest.TestCase):
    def test_no_fanout_is_a_direct_call(self):
        move = _StubMove(0.0)
        state = make_state(np.random.default_rng(1))
        ref = GFState(state, copy=True)
        new, acc = move.propose("model", state)
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertEqual(move.applied, [])  # no payload round trip
        self.assertFalse(move.fanout_active)

    def test_install_single_leaves_ladders_and_sidecar_alone(self):
        layout = FakeWorld(1).run(lambda r, c: build_layout(c, NWALKERS, [0], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0)
        move = _StubMove(0.0)
        move.install_walker_fanout(_Curr(fo, 0))
        self.assertTrue(move.tc.adaptive)
        self.assertEqual(move.eigen_store_path, "store.h5")
        self.assertFalse(move.fanout_active)


class MixinFakeWorldTest(unittest.TestCase):
    def test_two_compute_ranks_slice_run_merge(self):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, NWALKERS, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _StubMove(10.0 * rank)
            move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if role == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                fo.model = "head-model"
                state = make_state(np.random.default_rng(1))
                ref = GFState(state, copy=True)
                try:
                    new, acc = move.propose("head-model", state)
                finally:
                    fo.stop()
                return ref, new, acc
            return ComputeService(
                fcomm, layout, rank, registry={("pe", "stub"): move}, model="worker-model"
            ).serve()

        out = world.run(fn)
        ref, new, acc = out[0]
        self.assertEqual(out[1], 1)  # one propose served
        w0, w1 = 0, NWALKERS // 2
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        # the head scored walkers [0, N/2) (tag 0) and the worker [N/2, N) (tag 10)
        head_cols = np.arange(w1 - w0) + 0.0
        worker_cols = np.arange(NWALKERS - w1) + 10.0
        np.testing.assert_array_equal(new.log_like[0], np.concatenate([head_cols, worker_cols]))
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertTrue(acc.all())
        # delta counters: the head copy is zeroed then the rank values are summed
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == 2))
        # untouched sub-states are left alone
        for name, sub in new.sub_states.items():
            if name != "mbh" and sub is not None:
                np.testing.assert_array_equal(sub.coords, ref.sub_states[name].coords)
        # clock round trip on both ranks; extras merged per rank
        self.assertEqual(moves[0].applied, [7])
        self.assertEqual(moves[1].applied, [7])
        self.assertEqual(moves[0].merged, {0: {"nw": w1 - w0}, 1: {"nw": NWALKERS - w1}})
        # ranks never adapt; the configured value is remembered on the control
        for rank in (0, 1):
            self.assertFalse(moves[rank].tc.adaptive)
            self.assertTrue(moves[rank].tc.gf_configured_adaptive)
        self.assertEqual(moves[0].eigen_store_path, "store.h5")  # head keeps the sidecar
        self.assertIsNone(moves[1].eigen_store_path)  # worker never writes it

    def test_gf_serve_rejects_other_ops(self):
        move = _StubMove(0.0)
        with self.assertRaises(ValueError):
            move.gf_serve("score", {}, {}, None)


class PooledLadderStepTest(unittest.TestCase):
    def test_matches_one_eryn_adjustment_and_advances_time(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = True
        betas0 = np.array(tc.betas, copy=True)
        acc = np.array([3.0, 1.0, 2.0])
        prop = np.array([8.0, 8.0, 8.0])
        expect = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), acc / prop)
        out = pooled_ladder_step(tc, betas0, acc, prop)
        np.testing.assert_allclose(out, expect)
        self.assertEqual(tc.time, 1)
        # a sequential per-rank adaptation (two half-size steps) differs from the pooled one
        seq = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), np.array([2.0, 0.0, 1.0]) / 4)
        seq = seq + tc._get_ladder_adjustment(1, seq.copy(), np.array([1.0, 1.0, 1.0]) / 4)
        self.assertFalse(np.allclose(out, seq))

    def test_not_adaptive_or_single_rung_is_identity(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = False
        betas0 = np.array(tc.betas, copy=True)
        np.testing.assert_array_equal(pooled_ladder_step(tc, betas0, np.ones(3), np.ones(3)), betas0)
        self.assertEqual(tc.time, 1)
        one = TemperatureControl(2, 4, ntemps=1, permute=False)
        one.gf_configured_adaptive = True
        np.testing.assert_array_equal(
            pooled_ladder_step(one, np.array(one.betas), np.zeros(0), np.zeros(0)), one.betas
        )


if __name__ == "__main__":
    unittest.main()
```

`make_state` (from `tests/test_gf_substate_roundtrip.py`) builds a `GFState` with tempered sub-states for the roundtrip branch set (`mbh`, `psd`, `gb`; see `BRANCH_SHAPES` there), `NWALKERS = 4` walkers wide, so the two compute blocks are `(0, 2)` and `(2, 4)`; `tests/test_walkerslice_roundtrip.py` shows the same fixture sliced and merged.

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t2.log python -m unittest tests.test_walkerfanout_mixin -v; tail -n 20 .wtenv/p3t2.log`
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.moves.walkerfanout'`.

- [ ] **Step 3: Implement**

`src/lisatools/globalfit/moves/walkerfanout.py`:

```python
"""Whole-``propose`` fan-out over walker blocks (addremove family, PSD family).

Head: ``propose`` slices the FULL state into per-rank blocks, every compute rank
(the head included) runs the UNCHANGED body ``propose_local`` on its block
against its own ACA, the head merges the blocks back, SUM-reduces the per-ladder
swap tallies and adapts each ladder ONCE. With one compute rank ``propose`` is a
direct call of ``propose_local`` -- bit-identical to the pre-port code.

Semantics that change only when several compute ranks exist (design spec,
"Semantics that change only when n_compute > 1"):

* Eryn ladders adapt once per propose from the pooled swap ratio
  (sum accepted / sum proposed over ranks and repeats); ``tc.time`` advances once
  per propose instead of once per repeat. Ranks never adapt.
* The mid-iteration checkpoint granularity drops from per leaf to per propose.
* Eigen / info-matrix tables are rank-local proposal shapes; the sidecar is
  written by the head only.
* Stretch complements (PSD RedBlue splits) are the rank's local block.
* Cross-rank tempering swaps: none (within-rank); see the spec's WP8.
* Sub-state delta counters of the fanned-out branches equal the SUM over ranks
  of what the body wrote (the head's copy is zeroed first when the body assigns
  them, ``fanout_assigns_counters``).
"""

from __future__ import annotations

import numpy as np

from ..communication.walkerslice import merge_state, slice_state
from ..state import GFState

__all__ = ["PROPOSE_OP", "WalkerFanoutMixin", "pooled_ladder_step"]

PROPOSE_OP = "propose"


def pooled_ladder_step(tc, betas0, acc, prop):
    """One head-side ladder adaptation from pooled swap tallies; advances ``tc.time``.

    Returns the new ladder (``betas0`` when the control is not configured
    adaptive, has one rung, is past ``stop_adaptation``, or has no proposals).
    Mirrors ``TemperatureControl.adapt_temps`` (Eryn tempering.py ~867-897) with
    ``ratios = acc / prop`` supplied by the caller.
    """
    betas0 = np.array(betas0, dtype=float, copy=True)
    acc = np.asarray(acc, dtype=float).ravel()
    prop = np.asarray(prop, dtype=float).ravel()
    adaptive = bool(getattr(tc, "gf_configured_adaptive", getattr(tc, "adaptive", False)))
    stop = int(getattr(tc, "stop_adaptation", -1))
    running = stop < 0 or int(tc.time) < stop
    new = betas0
    if adaptive and int(tc.ntemps) > 1 and acc.size and np.all(prop > 0) and running:
        new = betas0 + tc._get_ladder_adjustment(int(tc.time), betas0.copy(), acc / prop)
    tc.time += 1
    return new


class WalkerFanoutMixin:
    """First in the MRO of a move whose whole ``propose`` runs per walker block.

    Subclass contract: rename the body to ``propose_local(model, state)``; the
    recipe stamps ``fanout_branches`` and calls ``install_walker_fanout(curr)``
    after construction; override the hooks below.
    """

    fanout = None  # the run's WalkerFanout (None single-process)
    gf_rank = None
    fanout_branches = None  # sub-states shipped and merged
    fanout_assigns_counters = False  # body ASSIGNS sub-state delta counters -> zero, then sum
    _fanout_body = False  # True while propose_local runs as a rank body

    # ---- subclass hooks --------------------------------------------------
    def fanout_temperature_controls(self):
        """Every ``TemperatureControl`` this move adapts (ranks never adapt them)."""
        raise NotImplementedError

    def fanout_payload_extra(self):
        """Head -> rank clock values (ladders, propose counters). Host numpy only."""
        return {}

    def fanout_apply_extra(self, extra):
        """Rank side: install the shipped clock values before the body runs."""

    def fanout_reply_extra(self, part):
        """Rank -> head: swap tallies and anything the body left rank-local."""
        return {}

    def fanout_merge_extra(self, replies, new_state):
        """Head: pooled ladder adaptation etc. ``replies`` = {rank: extra}."""

    # ---- wiring ------------------------------------------------------------
    @property
    def fanout_active(self):
        return self.fanout is not None and not self.fanout.single

    def install_walker_fanout(self, curr):
        """Bind the run's fan-out (``curr.fanout``) and apply the multi-rank rules."""
        self.fanout = getattr(curr, "fanout", None)
        self.gf_rank = getattr(curr, "rank", None)
        if not self.fanout_active:
            return
        for tc in self.fanout_temperature_controls():
            if tc is None:
                continue
            if not hasattr(tc, "gf_configured_adaptive"):  # shared controls: stamp once
                tc.gf_configured_adaptive = bool(getattr(tc, "adaptive", False))
            tc.adaptive = False  # ranks never adapt; the head adapts once per propose
        if not self.fanout.is_head and hasattr(self, "eigen_store_path"):
            self.eigen_store_path = None  # single-writer sidecar (head only)

    # ---- propose -------------------------------------------------------------
    def propose(self, model, state):
        if not self.fanout_active:
            return self.propose_local(model, state)
        return self.fanout_propose(model, state)

    def fanout_propose(self, model, state):
        fanout = self.fanout
        layout = fanout.layout
        branches = list(self.fanout_branches or [])
        extra = self.fanout_payload_extra()

        def payload(rank, w0, w1):
            return {"state": slice_state(state, w0, w1, sub_states=branches), "extra": extra}

        def body(p, model_local):
            return self.gf_serve(PROPOSE_OP, p, fanout.clock, model_local)

        def merge(replies):
            new_state = GFState(state, copy=True)
            if self.fanout_assigns_counters:
                for name in branches:
                    sub = (getattr(new_state, "sub_states", None) or {}).get(name)
                    if sub is None or not getattr(sub, "tempered_initialized", False):
                        continue
                    for cname in sub.delta_counter_names:
                        arr = getattr(sub, cname, None)
                        if arr is not None:
                            arr[...] = 0
            for rank in layout.compute_ranks:
                w0, w1 = layout.block_of(rank)
                merge_state(new_state, replies[rank]["state"], w0, w1)
            accepted = np.concatenate(
                [np.asarray(replies[r]["accepted"]) for r in layout.compute_ranks], axis=1
            )
            self.fanout_merge_extra({r: replies[r]["extra"] for r in layout.compute_ranks},
                                    new_state)
            return new_state, accepted

        return fanout.run(
            PROPOSE_OP,
            move=getattr(self, "gf_move_name", None),
            per_rank_payload=payload,
            local_body=body,
            merge=merge,
        )

    def gf_serve(self, op, payload, clock, model):
        if op != PROPOSE_OP:
            raise ValueError(f"{type(self).__name__} serves only {PROPOSE_OP!r}, got {op!r}")
        self.fanout_apply_extra(payload.get("extra") or {})
        self._fanout_body = True
        try:
            part, accepted = self.propose_local(model, payload["state"])
        finally:
            self._fanout_body = False
        return {
            "state": part,
            "accepted": np.asarray(accepted),
            "extra": self.fanout_reply_extra(part),
        }
```

Check `WalkerFanout.run`'s multi path (`communication/fanout.py` ~110-235) invokes `local_body(per_rank_payload(head, w0, w1), self.model)` for the head after the isends; if the head's body is invoked with a different argument order, adapt `body` — do not change `fanout.py`.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t2b.log python -m unittest tests.test_walkerfanout_mixin tests.test_walkerslice_roundtrip tests.test_fanout_fakecomm -v; tail -n 25 .wtenv/p3t2b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/walkerfanout.py tests/test_walkerfanout_mixin.py
git commit -m "feat(moves): WalkerFanoutMixin -- whole-propose fan-out over walker blocks with pooled ladder steps

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `ResidualAddOneRemoveOneMove` (MBH, EMRI, SOBBH) through the mixin

**Files:**
- Modify: `src/lisatools/globalfit/moves/addremovemove.py` (class line 87; `propose` 1543; the clock tick ~1563; the `temperature_swaps` call ~1926-1944; the per-leaf `midit_checkpoint.maybe_write` ~2062-2066)
- Test: `tests/test_addremove_fanout_hooks.py` (create)

**Interfaces:**
- Consumes: `WalkerFanoutMixin`, `pooled_ladder_step`, `midit_checkpoint.maybe_write(state, tag=..., prepare=...)`.
- Produces: `ResidualAddOneRemoveOneMove.propose_local` (the old body), `_fanout_note_swaps(leaf, tc)`, `_fanout_swap_tally` (dict leaf -> (acc, prop) summed over repeats), the five hooks. `SOBBHChunkedLikeMove` and `MultiGPUResidualAddRemoveMove` inherit everything (neither overrides `propose`).

- [ ] **Step 1: Write the failing test**

```python
"""addremove fan-out hooks: swap tallies per leaf, pooled once-per-propose ladder step."""

import types
import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

NT, B = 3, 2


def _skeleton(nleaves=2):
    """A move object with only the attributes the hooks touch (no waveform build)."""
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "mbh"
    move.gf_move_name = "mbh_pe"
    move.nwalkers = B
    move.ntemps = NT
    move.temperature_controls = [
        TemperatureControl(2, B, ntemps=NT, permute=False) for _ in range(nleaves)
    ]
    for tc in move.temperature_controls:
        tc.gf_configured_adaptive = True
        tc.adaptive = False
    move._fanout_swap_tally = {}
    move._fanout_body = False
    move._fancy_swap_clock = 4
    move._dbg_step = 9
    return move


def _state(betas_all):
    sub = types.SimpleNamespace(betas_all=[np.array(b, dtype=float) for b in betas_all])
    return types.SimpleNamespace(sub_states={"mbh": sub})


class AddRemoveFanoutHooksTest(unittest.TestCase):
    def test_mro_and_body_rename(self):
        self.assertIs(ResidualAddOneRemoveOneMove.__mro__[1], WalkerFanoutMixin)
        self.assertTrue(callable(ResidualAddOneRemoveOneMove.propose_local))
        self.assertIs(
            ResidualAddOneRemoveOneMove.propose, WalkerFanoutMixin.propose
        )

    def test_note_swaps_accumulates_over_repeats(self):
        move = _skeleton()
        tc = move.temperature_controls[1]
        tc.swaps_accepted = np.array([1, 0])
        move._fanout_note_swaps(1, tc)
        tc.swaps_accepted = np.array([0, 2])
        move._fanout_note_swaps(1, tc)
        acc, prop = move._fanout_swap_tally[1]
        np.testing.assert_array_equal(acc, [1.0, 2.0])
        np.testing.assert_array_equal(prop, [2.0 * B, 2.0 * B])  # swaps_proposed = nwalkers per call
        self.assertEqual(move.fanout_reply_extra(None)["swap_tally"].keys(), {1})

    def test_clock_round_trip(self):
        move = _skeleton()
        extra = move.fanout_payload_extra()
        self.assertEqual(extra, {"fancy_swap_clock": 4, "dbg_step": 9})
        other = _skeleton()
        other.fanout_apply_extra(extra)
        self.assertEqual((other._fancy_swap_clock, other._dbg_step), (4, 9))

    def test_merge_pools_tallies_and_adapts_each_visited_leaf_once(self):
        move = _skeleton()
        betas0 = [np.array([1.0, 0.5, 0.25]), np.array([1.0, 0.4, 0.1])]
        state = _state(betas0)
        r0 = {"swap_tally": {0: (np.array([2.0, 1.0]), np.array([4.0, 4.0]))}}
        r1 = {"swap_tally": {0: (np.array([1.0, 1.0]), np.array([4.0, 4.0]))}}
        move.fanout_merge_extra({0: r0, 1: r1}, state)
        tc0 = move.temperature_controls[0]
        ref = TemperatureControl(2, B, ntemps=NT, permute=False)
        expect = betas0[0] + ref._get_ladder_adjustment(
            0, betas0[0].copy(), np.array([3.0, 2.0]) / 8.0
        )
        np.testing.assert_allclose(state.sub_states["mbh"].betas_all[0], expect)
        np.testing.assert_allclose(tc0.betas, expect)
        self.assertEqual(tc0.time, 1)
        # leaf 1 was visited on no rank: ladder and clock untouched
        np.testing.assert_array_equal(state.sub_states["mbh"].betas_all[1], betas0[1])
        self.assertEqual(move.temperature_controls[1].time, 0)

    def test_hooks_lists(self):
        move = _skeleton(3)
        self.assertEqual(len(move.fanout_temperature_controls()), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t3.log python -m unittest tests.test_addremove_fanout_hooks -v; tail -n 20 .wtenv/p3t3.log`
Expected: failures/errors (no `WalkerFanoutMixin` in the MRO, no `_fanout_note_swaps`).

- [ ] **Step 3: Implement**

1. Import and MRO:
   ```python
   from .walkerfanout import WalkerFanoutMixin, pooled_ladder_step
   ...
   class ResidualAddOneRemoveOneMove(WalkerFanoutMixin, GlobalFitMove, StretchMove, Move):
   ```
2. Rename the body: `def propose(self, model, state):` (~1543) -> `def propose_local(self, model, state):`; add one docstring line: "Runs on the FULL state single-process and on this rank's walker slice under several compute ranks (``WalkerFanoutMixin.propose`` dispatches)."
3. Right after the clock tick (`self._fancy_swap_clock = ... + 1`, ~1563) add `self._fanout_swap_tally = {}`.
4. Directly after the `temperature_swaps(...)` call's closing parenthesis (~1944; before `if self.swap_debug:`) add:
   ```python
                self._fanout_note_swaps(leaf, temperature_control_here)
   ```
5. Guard the per-leaf checkpoint (~2062): `if not self._fanout_body:` around the `midit_checkpoint.maybe_write(...)` call (indent the call one level).
6. Add the methods (place them right before `propose_local`):
   ```python
    # ---- multi-rank fan-out (WalkerFanoutMixin hooks) ------------------------
    def _fanout_note_swaps(self, leaf, tc):
        """Accumulate this repeat's swap counts for ``leaf`` (Eryn re-zeroes them per call)."""
        acc = np.asarray(tc.swaps_accepted, dtype=float).ravel()
        prop = np.asarray(tc.swaps_proposed, dtype=float).ravel()
        a, p = self._fanout_swap_tally.get(int(leaf), (0.0, 0.0))
        self._fanout_swap_tally[int(leaf)] = (a + acc, p + prop)

    def fanout_temperature_controls(self):
        return list(self.temperature_controls)

    def fanout_payload_extra(self):
        return {
            "fancy_swap_clock": int(getattr(self, "_fancy_swap_clock", 0)),
            "dbg_step": int(getattr(self, "_dbg_step", 0)),
        }

    def fanout_apply_extra(self, extra):
        if "fancy_swap_clock" in extra:
            self._fancy_swap_clock = int(extra["fancy_swap_clock"])
        if "dbg_step" in extra:
            self._dbg_step = int(extra["dbg_step"])

    def fanout_reply_extra(self, part):
        return {
            "swap_tally": {
                int(leaf): (np.asarray(a, dtype=float), np.asarray(p, dtype=float))
                for leaf, (a, p) in getattr(self, "_fanout_swap_tally", {}).items()
            }
        }

    def fanout_merge_extra(self, replies, new_state):
        """Pool the per-leaf swap tallies over ranks; adapt each visited ladder ONCE.

        The authoritative per-leaf ladder lives on the sub-state
        (``betas_all[leaf]``, read at the top of the leaf loop, written back at
        its end); ``merge_walkers`` never touches ladders, so the head writes
        the adapted ladder there itself and mirrors it on its own control.
        """
        pooled = {}
        for extra in replies.values():
            for leaf, (acc, prop) in extra.get("swap_tally", {}).items():
                a, p = pooled.get(int(leaf), (0.0, 0.0))
                pooled[int(leaf)] = (a + np.asarray(acc, dtype=float), p + np.asarray(prop, dtype=float))
        sub = new_state.sub_states[self.branch_name]
        for leaf in sorted(pooled):
            acc, prop = pooled[leaf]
            tc = self.temperature_controls[leaf]
            ntemps = int(tc.ntemps)
            betas0 = np.array(sub.betas_all[leaf][:ntemps], copy=True)
            new = pooled_ladder_step(tc, betas0, acc, prop)
            sub.betas_all[leaf][:ntemps] = new
            tc.betas[:] = new
        if not self._fanout_body:
            midit_checkpoint.maybe_write(
                new_state, tag=f"{self.branch_name} propose", prepare=self._sync_cold_row
            )
   ```
   Wrap the long `pooled[int(leaf)] = ...` line to stay <= 100 characters. `midit_checkpoint` is already imported in this module (grep `midit_checkpoint` at the top); if not, `from .. import midit_checkpoint`.
7. Leave the return statement (`return new_state, np.asarray(accepted)[:engine_ntemps]`, ~2124) as is — it returns the last leaf's last repeat's array today; the mixin concatenates whatever the body returns (out of scope to change).

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t3b.log python -m unittest tests.test_addremove_fanout_hooks tests.test_addremove_multi_shard tests.test_addremove_signal_gen tests.test_addremove_verify_convention tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill tests.test_sobbh_ll_timing tests.test_eigen_refresh tests.test_eigen_table_persist tests.test_maxlogl_plateau -v; tail -n 30 .wtenv/p3t3b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/addremovemove.py tests/test_addremove_fanout_hooks.py
git commit -m "feat(addremove): whole-propose fan-out over walker blocks; pooled per-leaf ladder step on the head

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `PSDMove` (psd, galfor, sgwb) through the mixin

**Files:**
- Modify: `src/lisatools/globalfit/moves/psdmove.py` (class line 78; `propose` 2279)
- Test: `tests/test_psd_fanout_hooks.py` (create)

**Interfaces:**
- Consumes: `WalkerFanoutMixin`, `pooled_ladder_step`.
- Produces: `PSDMove.propose_local` (old body), `fanout_assigns_counters = True`, the hooks. `MultiGPUPSDMove` inherits (no `propose` override). The search and PE moves SHARE one `TemperatureControl` (recipe.py ~2267-2272) — the mixin stamps `gf_configured_adaptive` once, so a second install is harmless.

- [ ] **Step 1: Write the failing test**

```python
"""PSD fan-out hooks: shipped ladder, pooled swap tallies, once-per-propose adaptation."""

import types
import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

NT, B = 3, 2


def _skeleton():
    move = PSDMove.__new__(PSDMove)
    move.gf_move_name = "psd_pe"
    move.sampled_branches = ["psd", "galfor"]
    move.temperature_control = TemperatureControl(4, B, ntemps=NT, permute=False)
    move.temperature_control.gf_configured_adaptive = True
    move.temperature_control.adaptive = False
    move._tally_swaps_accepted = np.array([1, 2])
    move._tally_swaps_proposed = np.array([2 * B, 2 * B])
    move._fanout_body = False
    return move


def _state():
    subs = {k: types.SimpleNamespace(betas=np.zeros(NT)) for k in ("psd", "galfor")}
    subs["sgwb"] = None
    return types.SimpleNamespace(sub_states=subs)


class PSDFanoutHooksTest(unittest.TestCase):
    def test_mro_flags_and_body_rename(self):
        self.assertIs(PSDMove.__mro__[1], WalkerFanoutMixin)
        self.assertTrue(PSDMove.fanout_assigns_counters)
        self.assertTrue(callable(PSDMove.propose_local))
        self.assertIs(PSDMove.propose, WalkerFanoutMixin.propose)

    def test_ladder_ships_and_applies(self):
        move = _skeleton()
        move.temperature_control.betas[:] = [1.0, 0.3, 0.05]
        extra = move.fanout_payload_extra()
        other = _skeleton()
        other.fanout_apply_extra(extra)
        np.testing.assert_array_equal(other.temperature_control.betas, [1.0, 0.3, 0.05])
        self.assertEqual(move.fanout_temperature_controls(), [move.temperature_control])

    def test_reply_carries_tallies_and_none_is_zero(self):
        move = _skeleton()
        r = move.fanout_reply_extra(None)
        np.testing.assert_array_equal(r["swaps_accepted"], [1.0, 2.0])
        np.testing.assert_array_equal(r["swaps_proposed"], [4.0, 4.0])
        move._tally_swaps_accepted = None
        move._tally_swaps_proposed = None
        r = move.fanout_reply_extra(None)
        self.assertEqual(r["swaps_accepted"].size, 0)

    def test_merge_pools_adapts_once_and_publishes_the_ladder(self):
        move = _skeleton()
        tc = move.temperature_control
        betas0 = np.array(tc.betas, copy=True)
        replies = {
            0: {"swaps_accepted": np.array([1.0, 2.0]), "swaps_proposed": np.array([4.0, 4.0])},
            1: {"swaps_accepted": np.array([2.0, 0.0]), "swaps_proposed": np.array([4.0, 4.0])},
        }
        state = _state()
        move.fanout_merge_extra(replies, state)
        ref = TemperatureControl(4, B, ntemps=NT, permute=False)
        expect = betas0 + ref._get_ladder_adjustment(0, betas0.copy(), np.array([3.0, 2.0]) / 8.0)
        np.testing.assert_allclose(tc.betas, expect)
        self.assertEqual(tc.time, 1)
        for key in ("psd", "galfor"):
            np.testing.assert_allclose(state.sub_states[key].betas, expect)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t4.log python -m unittest tests.test_psd_fanout_hooks -v; tail -n 20 .wtenv/p3t4.log`
Expected: failures (MRO, missing hooks).

- [ ] **Step 3: Implement**

1. `from .walkerfanout import WalkerFanoutMixin, pooled_ladder_step`; `class PSDMove(WalkerFanoutMixin, GlobalFitMove, StretchMove):`; class attr `fanout_assigns_counters = True` next to the `_tally_*` class attrs (~152-155).
2. Rename `def propose(self, model, state):` (~2279) -> `def propose_local(self, model, state):` with the same one-line docstring note as Task 3.
3. Hooks (place them right before `propose_local`):
   ```python
    # ---- multi-rank fan-out (WalkerFanoutMixin hooks) ------------------------
    def fanout_temperature_controls(self):
        return [self.temperature_control]

    def fanout_payload_extra(self):
        return {"betas": np.array(self.temperature_control.betas, dtype=float, copy=True)}

    def fanout_apply_extra(self, extra):
        if "betas" in extra:
            self.temperature_control.betas[:] = np.asarray(extra["betas"], dtype=float)

    @staticmethod
    def _tally_or_empty(arr):
        return np.zeros(0) if arr is None else np.asarray(arr, dtype=float).ravel()

    def fanout_reply_extra(self, part):
        return {
            "swaps_accepted": self._tally_or_empty(self._tally_swaps_accepted),
            "swaps_proposed": self._tally_or_empty(self._tally_swaps_proposed),
        }

    def fanout_merge_extra(self, replies, new_state):
        """Pool swap tallies over ranks, adapt the shared ladder ONCE, publish it.

        The body copies ``tc.betas`` into every sampled sub-state
        (``sub.betas[:] = ...``) but ``merge_walkers`` never writes ladders back,
        so the head republishes the adapted ladder here.
        """
        tc = self.temperature_control
        acc = sum(np.asarray(e["swaps_accepted"], dtype=float) for e in replies.values())
        prop = sum(np.asarray(e["swaps_proposed"], dtype=float) for e in replies.values())
        tc.betas[:] = pooled_ladder_step(tc, np.array(tc.betas, copy=True), acc, prop)
        for key in self.sampled_branches:
            sub = (getattr(new_state, "sub_states", None) or {}).get(key)
            if sub is not None and getattr(sub, "betas", None) is not None:
                sub.betas[:] = tc.betas
   ```
   `sum(...)` over numpy arrays starts from `0`, which broadcasts; when every reply carries an empty array (`ntemps == 1`) `acc`/`prop` are empty and `pooled_ladder_step` is the identity. If `replies` is empty, `sum` returns `0` — guard with `if not replies: return`.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t4b.log python -m unittest tests.test_psd_fanout_hooks tests.test_psd_move_batched tests.test_psd_move_multi_shard tests.test_psd_delayed_acceptance tests.test_psd_mirror_multishard tests.test_psd_mirror_buffer tests.test_psd_mirror_routed_hooks tests.test_noise_split_moves tests.test_psd_fancy_swap_walker_inds -v; tail -n 30 .wtenv/p3t4b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/psdmove.py tests/test_psd_fanout_hooks.py
git commit -m "feat(psd): whole-propose fan-out over walker blocks; shared ladder adapted once on the head

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Recipe wiring, readiness, and the two-rank noise smoke

**Files:**
- Modify: `src/lisatools/globalfit/recipe.py` (`SingleSourcePEBuilder.build` after `move.eigen_store_path = ...` ~4170; `build_noise_moves` after the `fanout_branches` stamps ~2276-2277)
- Modify: `src/lisatools/globalfit/run.py` (`_fanout_unready_moves` docstring only: name the ported families)
- Test: `tests/test_recipe_fanout_install.py` (create), `tests/test_multirank_noise_smoke.py` (create, gated `RUN_GF_SMOKE=1`)

**Interfaces:**
- Consumes: `WalkerFanoutMixin.install_walker_fanout(curr)`; `curr.fanout` / `curr.rank` (set on every rank by `run.py` before recipe setup: `self.curr.fanout = self.fanout` ~2251).
- Produces: every addremove/PSD move built by the recipe has its fan-out installed; `_fanout_unready_moves` returns `[]` for these families.

- [ ] **Step 1: Write the failing tests**

`tests/test_recipe_fanout_install.py`:

```python
"""The recipe installs the fan-out on every addremove/PSD move it builds."""

import unittest
from unittest import mock

from lisatools.globalfit import recipe as recipe_mod
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.run import _fanout_unready_moves


class FanoutInstallTest(unittest.TestCase):
    def test_ported_families_are_not_unready(self):
        a = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
        a.gf_move_name = "mbh_pe"
        p = PSDMove.__new__(PSDMove)
        p.gf_move_name = "psd_pe"
        unready, head_only = _fanout_unready_moves([a, p])
        self.assertEqual((unready, head_only), ([], []))

    def test_builders_call_install(self):
        # the builders are exercised end to end by the gated smokes; here we pin
        # the call sites so a refactor cannot silently drop them
        src = open(recipe_mod.__file__).read()
        self.assertEqual(src.count("install_walker_fanout(curr)"), 3)


if __name__ == "__main__":
    unittest.main()
```

`tests/test_multirank_noise_smoke.py`:

```python
"""PSD fan-out end to end: noise_only synthetic fit, 1 vs 2 compute ranks (fake comm)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank noise smoke")
class MultiRankNoiseSmokeTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_noise_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_world(self, size, iterations=2):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor

        def fn(rank, comm):
            fit = erebor.noise_only(
                nwalkers=4, ntemps=2, data_mode="synthetic",
                file_store_dir=os.path.join(self.tmpdir, f"n{size}"),
                make_diagnostic_plots=False,
            )
            fit.general.num_iterations = iterations
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {"role": layout.role_of(rank).value, "acs_rows": int(gf.acs.acs_total_entries)}
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            else:
                out["log_like"] = np.array(gf.state.log_like[0], copy=True)
                out["betas"] = {
                    k: np.array(s.betas, copy=True)
                    for k, s in gf.state.sub_states.items()
                    if s is not None and getattr(s, "betas", None) is not None
                }
            return out

        return FakeWorld(size, timeout=1800.0).run(fn)

    def test_two_compute_ranks_run_the_noise_moves(self):
        out = self._run_world(2)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # ping + at least one propose per stored iteration per noise move
        self.assertGreaterEqual(out[1]["served"], 1 + 2)
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (4,))
        for betas in out[0]["betas"].values():
            self.assertTrue(np.all(np.diff(betas) <= 0))  # a valid ladder after the head's step


if __name__ == "__main__":
    unittest.main()
```

Adapt the constructor kwargs to what `erebor.noise_only(...)` accepts (read `tests/test_noise_globalfit.py:50` and `tests/test_globalfit_sample.py:47-53`: `nwalkers`, `ntemps`, `data_mode`, `file_store_dir`, `make_diagnostic_plots` are all accepted by the stock constructors). If the `noise_only` recipe's move count per iteration differs, lower the served bound to `1 + iterations`.

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t5.log python -m unittest tests.test_recipe_fanout_install -v; tail -n 20 .wtenv/p3t5.log`
Expected: `test_builders_call_install` FAILS (count 0).

- [ ] **Step 3: Implement**

`recipe.py`:
- `SingleSourcePEBuilder.build`, after `move.eigen_store_path = getattr(gi, "main_file_path", None)`:
  ```python
        # multi-rank: bind the run's fan-out (no-op single-process); clears the
        # sidecar path on non-head ranks (single writer) and freezes the ladders
        # on every rank (the head adapts once per propose from pooled tallies)
        move.install_walker_fanout(curr)
  ```
- `build_noise_moves`, after the two `fanout_branches` stamps:
  ```python
    search_move.install_walker_fanout(curr)
    pe_move.install_walker_fanout(curr)
  ```
`run.py` `_fanout_unready_moves` docstring: add "The addremove (MBH/EMRI/SOBBH) and PSD (psd/galfor/sgwb) families are served through ``moves.walkerfanout.WalkerFanoutMixin`` since Plan 3; GB/VGB remain unready until Plan 4."

- [ ] **Step 4: Run to verify pass, then the gated smokes**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3t5b.log python -m unittest tests.test_recipe_fanout_install tests.test_run_multirank_helpers tests.test_recipe_local_block -v; tail -n 20 .wtenv/p3t5b.log`
Expected: all OK.

Run: `RUN_GF_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p3t5c.log python -m unittest tests.test_multirank_noise_smoke tests.test_multirank_blank_smoke tests.test_globalfit_sample -v; tail -n 60 .wtenv/p3t5c.log`
Expected: all OK (allow up to 20 minutes; the noise smoke builds a synthetic noise fit twice). If the two-rank noise smoke fails inside `propose_local` on a rank, the log names the rank and the traceback; typical causes: a full-width `state` read the site map missed (fix in Task 3/4's file), or a sub-state the body needs that is not in `fanout_branches`.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/recipe.py src/lisatools/globalfit/run.py tests/test_recipe_fanout_install.py tests/test_multirank_noise_smoke.py
git commit -m "feat(recipe): install the walker fan-out on addremove/PSD moves; two-rank noise smoke

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Whole-plan verification

- [ ] **Step 1: Every touched suite, one process**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p3_all.log python -m unittest tests.test_psd_fancy_swap_walker_inds tests.test_walkerfanout_mixin tests.test_addremove_fanout_hooks tests.test_psd_fanout_hooks tests.test_recipe_fanout_install tests.test_run_multirank_helpers tests.test_recipe_local_block tests.test_walkerslice_roundtrip tests.test_fanout_fakecomm tests.test_fanout_passthrough tests.test_addremove_multi_shard tests.test_addremove_signal_gen tests.test_addremove_verify_convention tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill tests.test_sobbh_ll_timing tests.test_eigen_refresh tests.test_eigen_table_persist tests.test_maxlogl_plateau tests.test_psd_move_batched tests.test_psd_move_multi_shard tests.test_psd_delayed_acceptance tests.test_psd_mirror_multishard tests.test_psd_mirror_buffer tests.test_psd_mirror_routed_hooks tests.test_noise_split_moves tests.test_coarse_wdm tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one tests.test_stock_globalfit -v; tail -n 25 .wtenv/p3_all.log`
Expected: all OK.

- [ ] **Step 2: Gated smokes**

Run: `RUN_GF_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p3_smokes.log python -m unittest tests.test_multirank_noise_smoke tests.test_multirank_blank_smoke tests.test_globalfit_sample -v; tail -n 30 .wtenv/p3_smokes.log`
Expected: all OK.

- [ ] **Step 3: Line-length gate**

`git diff 7a65cade..HEAD -- src tests | grep '^+' | grep -v '^+++' | awk 'length > 101'` -> empty.

- [ ] **Step 4: Report**

`git log --oneline 7a65cade..HEAD`, `git status --short`. Cluster gates now warranted (WP7, after Plan 4 lifts the GB guard): the `RANKS_PER_GPU=2` shared-GPU parity run on a psd-only or nogb-null recipe can run BEFORE Plan 4 because every move in those recipes is now served.

---

## Self-review notes

- Spec coverage WP4: mixin first in the MRO with `propose` -> `propose_local` (T2/T3/T4); payload = `slice_state(..., sub_states=fanout_branches)` + clock (T2 + per-family extras T3/T4); rank-side `gf_serve("propose")` (T2); head merge with `merge_state`, `accepted` concatenation, pooled once-per-propose ladder adaptation writing the sub-state ladders (T2/T3/T4); midit guard + per-propose head hook (T3); `tc.adaptive = False` on every rank with the configured value remembered on the control (T2); `eigen_store_path = None` on non-head ranks (T2); PSD stretch complement = local block (documented, T2 docstring); the pre-port `skip_swap_supp_names`/`walker_inds` fix (T1); `fanout_branches` consumed (T2). Deviation from the spec's test list: `test_fanout_single_rank_identity.py` is covered by `MixinSingleTest` + the `gf_serve`-dispatch identity in `test_mro_and_body_rename`; `test_eigen_persist_fanout.py` is covered by the `eigen_store_path` assertions in `test_two_compute_ranks_slice_run_merge`; `test_rank_seeds.py` was delivered in Plan 2 (`_seed_rank_streams`).
- Placeholder scan: every step carries its code; the only conditional instructions are attribute-name checks against Eryn (`skip_swap_supp_names` storage) and `WalkerFanout.run`'s head-body call, both resolved by reading the named lines.
- Type consistency: hooks return/consume plain dicts of numpy arrays; `pooled_ladder_step(tc, betas0, acc, prop) -> ndarray` used identically in T3 and T4; `PROPOSE_OP = "propose"` is the op the head sends and the rank checks; registry keys `(stage, gf_move_name)` come from Plan 2's `_serve_registry` and the mixin sends `move=gf_move_name`.
- Known accepted semantics changes are listed in the `walkerfanout.py` module docstring (the spec's "Semantics that change only when n_compute > 1" section).
