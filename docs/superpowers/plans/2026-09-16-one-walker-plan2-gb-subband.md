# One-walker replica mode, Plan 2: GB/VGB sub-band dispersal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In one-walker replica mode, every compute rank owns a static contiguous band range of the GB (and VGB) grid: it proposes, births, tempers and caps only its bands, replicas reconcile their residual copies after every unit through a small delta ledger, the head merges per touched `(rung, slot)` and a new `gb_sync` command rebuilds every replica's residual from the merged branch at the end of each propose.

**Architecture:** The existing three-command orchestrator (`gb_run_proposal`, `gb_run_tempering`, `gb_finish`) is kept; replica mode adds a `band_range` + replica identity to the common payload, an owned-rows mask ANDed into the sorter `extra_bool` chain in `run_proposal`/`run_tempering` (dead rows further restricted to the rank's slot partition `leaf % R == r`), an allgather of changed cold-chain rows at every unit close (`self.fanout.comm`, lockstep because all ranks run the head-drawn schedule), touched-slot merges on the head, and a fourth op `gb_sync` (replica mode only) that ships the merged branch to every rank for the authoritative `check_ll_inject` rebuild and reports residual hashes. Ladders and counters need no change: swap counters are zero outside a rank's bands, so the existing SUM merge is the union and the head's single `_adapt_band_temps` stays correct. Nothing changes when `layout.replica_mode` is False.

**Tech Stack:** Python 3.12, numpy/cupy (`xp` pattern), mpi4py pickle transport via `WalkerFanout.run` and `comm.allgather`, `FakeWorld`/`FakeComm` test doubles, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md` (decisions 6-10, "GB and VGB" architecture section, gate 14).

## Global Constraints

- Worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker`, branch `one-walker-replicas`, base `dev` `aec22834`; Plan 1 is in the working tree, uncommitted. **No `git commit`, no `git push`** (standing rule). Ledger: `.superpowers/sdd/2026-09-16-one-walker-plan2-gb-subband/progress.md`.
- Test runner (one python process, CPU): `cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate deving && .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest <modules> -v; tail -8 .wtenv/<name>.log`. Never run `tests/test_gbspecial_flow.py`. The GB smoke (`RUN_GF_GB_SMOKE=1`) is the ONLY heavy test here: run it once, alone, at the end of Task 6 (≤5 GB RSS, ≤30 min).
- Every new behaviour is gated on `self._replica_active()` (defined in Task 1: `fanout_active and layout.replica_mode`). With it False, `run_proposal`, `run_tempering`, `_gb_serve_*`, `_propose_orchestrated` and `_write_back_state` must behave byte-identically to Plan 1's tree (the multirank suites and the GB smoke at 4 walkers prove it).
- All anchors are at the Plan-1 working tree (`gbspecialstretch.py` 21,775 lines at HEAD `aec22834` + Plan 1's edits elsewhere; `gbspecialstretch.py` and `gbbands.py` were NOT touched by Plan 1). Re-anchor with grep if a line moved.
- Replica identity on the move during a command: `self._owned_band_range = (b0, b1)` (global band indices, half-open), `self._replica_index = r`, `self._n_replicas = R`; all three `None`/`1` outside replica mode.
- Sign convention (`gbspecialstretch.py:3232-3246`): `remove_cold_chain_sources_from_residual` = factor `+1` (adds the template back into the residual, i.e. removes the source from the fit); `add_cold_chain_sources_to_residual` = factor `-1`.

---

### Task 1: Replica identity: band ranges, payload, rank block

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` — `_RANK_BLOCK_SAVED` (:17245-17249), `_enter_rank_block` (:17251-17337), `_exit_rank_block` (:17339-17349), `_propose_orchestrated`'s `_common` (:18655-18679); new helpers near `_unit_residue_mask` (:735-749)
- Test: `tests/test_gb_replica_bands.py` (new)

**Interfaces:**
- Produces (module level): `replica_band_ranges(num_bands, n_replicas, weights=None) -> list[tuple[int, int]]` — contiguous half-open ranges covering `[0, num_bands)`; with `weights` (per-band non-negative array) split by cumulative weight, else by count (`np.array_split` sizes). `n_replicas == 1` → `[(0, num_bands)]`.
- Produces on `GBSpecialBase`: `_replica_active(self) -> bool`; attributes `_owned_band_range = None`, `_replica_index = 0`, `_n_replicas = 1` (class defaults), saved/restored by the rank block; `_replica_band_weights(self, work) -> np.ndarray | None` (GB: `None`; VGB overrides in Task 5); the common payload keys `"band_range": (b0, b1) | None`, `"replica": (index, count)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_gb_replica_bands.py`

```python
"""GB replica identity: band ranges, rank-block attributes."""

import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase, replica_band_ranges


class BandRangesTest(unittest.TestCase):
    def test_count_split_is_contiguous_and_covers(self):
        r = replica_band_ranges(10, 3)
        self.assertEqual(r, [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(replica_band_ranges(10, 1), [(0, 10)])
        self.assertEqual(replica_band_ranges(2, 3), [(0, 1), (1, 2), (2, 2)])  # empty tail range allowed

    def test_weight_split_balances_cumulative_weight(self):
        w = np.array([0, 0, 5, 5, 0, 0, 5, 5, 0, 0])
        r = replica_band_ranges(10, 2, weights=w)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[-1][1], 10)
        self.assertEqual(r[0][1], r[1][0])
        self.assertEqual(int(w[r[0][0]:r[0][1]].sum()), 10)  # half the weight on each side

    def test_rank_block_carries_replica_identity(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        self.assertIsNone(m._owned_band_range)
        self.assertEqual((m._replica_index, m._n_replicas), (0, 1))
        m._apply_replica_payload({"band_range": (3, 7), "replica": (1, 2)})
        self.assertEqual(m._owned_band_range, (3, 7))
        self.assertEqual((m._replica_index, m._n_replicas), (1, 2))
        m._apply_replica_payload({})
        self.assertIsNone(m._owned_band_range)
        self.assertEqual((m._replica_index, m._n_replicas), (0, 1))
```

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_gb_replica_bands -v` → ImportError (`replica_band_ranges`).

- [ ] **Step 3: Implement**

Module-level, after `_unit_residue_mask` (:749):
```python
def replica_band_ranges(num_bands, n_replicas, weights=None):
    """Static contiguous band ranges, one per replica, covering ``[0, num_bands)``.

    By count (``np.array_split`` sizes) unless ``weights`` (per band, >= 0) is
    given, in which case the cut points balance the cumulative weight
    (VGB: catalogue sources per band). Ranges are half-open; a tail range may
    be empty when there are more replicas than bands.
    """
    num_bands = int(num_bands)
    n_replicas = max(1, int(n_replicas))
    if weights is None:
        sizes = [len(c) for c in np.array_split(np.arange(num_bands), n_replicas)]
        bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != num_bands:
            raise ValueError(f"weights has {w.shape[0]} entries for {num_bands} bands")
        cum = np.concatenate([[0.0], np.cumsum(w)])
        targets = cum[-1] * np.arange(1, n_replicas) / n_replicas
        cuts = np.searchsorted(cum, targets, side="left")
        bounds = np.concatenate([[0], np.clip(cuts, 0, num_bands), [num_bands]]).astype(int)
        bounds = np.maximum.accumulate(bounds)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_replicas)]
```
On `GBSpecialBase` (class attributes near `infomat_per_block = False`, :10653):
```python
    #: one-walker replica mode: this rank's static band range (global band
    #: indices, half-open) and its replica identity; None/0/1 otherwise
    _owned_band_range = None
    _replica_index = 0
    _n_replicas = 1

    def _replica_active(self) -> bool:
        fanout = getattr(self, "fanout", None)
        return (
            fanout is not None
            and not fanout.single
            and bool(getattr(fanout.layout, "replica_mode", False))
        )

    def _apply_replica_payload(self, payload):
        """Install (or clear) the replica identity a command payload carries."""
        rng = payload.get("band_range")
        self._owned_band_range = None if rng is None else (int(rng[0]), int(rng[1]))
        rep = payload.get("replica")
        if rep is None:
            self._replica_index, self._n_replicas = 0, 1
        else:
            self._replica_index, self._n_replicas = int(rep[0]), max(1, int(rep[1]))

    def _replica_band_weights(self, work):
        """Per-band weights for the static split; None = split by band count (GB)."""
        return None
```
`_RANK_BLOCK_SAVED` (:17245): append `"_owned_band_range", "_replica_index", "_n_replicas"`. In `_enter_rank_block`, right after the `tables` application (:17321-17323): `self._apply_replica_payload(payload)`. `_exit_rank_block` restores through the saved tuple (no change beyond the list).

`_propose_orchestrated`: before `_common` is defined (~:18655), compute once per propose
```python
        _replica = self._replica_active()
        _band_ranges = None
        if _replica:
            _band_ranges = replica_band_ranges(
                self.num_bands, layout.n_replicas, weights=self._replica_band_weights(work)
            )
```
and inside `_common(rank, w0, w1)`'s dict add
```python
                "band_range": (None if _band_ranges is None else _band_ranges[layout.replica_index(rank)]),
                "replica": (layout.replica_index(rank), layout.n_replicas) if _replica else None,
```
(`layout` is the fan-out layout already in scope at :18648.) The head's own local body goes through the same `_common` → `_enter_rank_block` path, so it installs its own range too.

- [ ] **Step 4: Run** — `python -m unittest tests.test_gb_replica_bands tests.test_multirank_gb_smoke -v` (the smoke module imports; its heavy tests skip without `RUN_GF_GB_SMOKE`) → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 2: Owned-rows mask in `run_proposal` and `run_tempering`

**Files:**
- Modify: `gbspecialstretch.py` — `run_proposal` (`_unit_kw` :4413-4430, `extra_bool` start :4462-4464, OPEN :4446-4449, CLOSE :4681-4684), `run_tempering` (`_tempering_swap_grid` :14434-14496 rows; the shut-off row mask site :14698-14724)
- Test: `tests/test_gb_replica_bands.py` (append)

**Interfaces:**
- Produces on `GBSpecialBase`: `_owned_rows_mask(self, band_sorter) -> xp.ndarray[bool] | None` — `None` when `_owned_band_range is None`; else `(b0 <= band_inds) & (band_inds < b1) & (inds | (leaf_inds % R == r))` (alive rows of owned bands, plus dead rows of owned bands in this rank's slot partition); `_owned_band_row_mask(self, band_index_arr) -> xp.ndarray[bool] | None` for the tempering grid (bands in range).

- [ ] **Step 1: Write the failing tests** (append)

```python
class OwnedRowsMaskTest(unittest.TestCase):
    def _sorter(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            band_inds=np.array([0, 1, 2, 3, 4, 5, 2, 3]),
            leaf_inds=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            inds=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            xp=np,
        )

    def test_none_outside_replica_mode(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        self.assertIsNone(m._owned_rows_mask(self._sorter()))
        self.assertIsNone(m._owned_band_row_mask(np.arange(6)))

    def test_owned_alive_rows_and_partitioned_dead_rows(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        m._apply_replica_payload({"band_range": (2, 4), "replica": (1, 2)})
        mask = m._owned_rows_mask(self._sorter())
        # bands 2,3: alive rows (leaf 2,3) kept; dead rows leaf 6 (band 2, 6%2==0 -> rank 0) dropped,
        # leaf 7 (band 3, 7%2==1 -> rank 1) kept; everything outside bands 2,3 dropped
        np.testing.assert_array_equal(mask, [0, 0, 1, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(m._owned_band_row_mask(np.arange(6)), [0, 0, 1, 1, 0, 0])
```

- [ ] **Step 2: Run to verify failure** → AttributeError.

- [ ] **Step 3: Implement**

On `GBSpecialBase` next to `_apply_replica_payload`:
```python
    def _owned_rows_mask(self, band_sorter):
        """Rows this replica may propose on: alive rows of its owned bands plus
        dead rows of its owned bands in its slot partition (``leaf % R == r``),
        so two replicas never birth into the same dead slot. ``None`` = no
        restriction (not in replica mode)."""
        rng = self._owned_band_range
        if rng is None:
            return None
        xp = band_sorter.xp
        b = band_sorter.band_inds
        owned = (b >= rng[0]) & (b < rng[1])
        if self._n_replicas > 1:
            mine = (band_sorter.leaf_inds % self._n_replicas) == self._replica_index
            owned = owned & (band_sorter.inds | mine)
        return xp.asarray(owned)

    def _owned_band_row_mask(self, band_index_arr):
        rng = self._owned_band_range
        if rng is None:
            return None
        b = band_index_arr
        return (b >= rng[0]) & (b < rng[1])
```
`run_proposal`:
- At :4462-4464 (start of the `extra_bool` chain) AND the owned mask:
```python
            _owned = self._owned_rows_mask(band_sorter)
            if _owned is not None:
                extra_bool = _owned if extra_bool is None else (extra_bool & _owned)
```
- `_unit_kw` (:4413-4430): after it is built, in replica mode add the owned mask so OPEN/CLOSE touch only owned bands:
```python
            if _owned is not None:   # (compute _owned once before the unit loop: it does not change per unit)
                _unit_kw = dict(_unit_kw)
                _unit_kw["extra_bool"] = (_owned if _unit_kw.get("extra_bool") is None
                                          else (_unit_kw["extra_bool"] & _owned))
```
(`get_subset_bool` ANDs `units/remainder` with `extra_bool`, gbbands.py:6060-6096, so both paths compose.)
`run_tempering`: where the grid's per-row band is known and the shut-off row mask is applied (:14698-14724, `_row_band`), add
```python
                _owned_rows = self._owned_band_row_mask(_row_band)
                if _owned_rows is not None:
                    <AND it into the same row-keep mask the shut-off gate uses>
```
so unowned bands drop out of the swap grid exactly as shut-off bands do (that path is proven not to corrupt the counters — exploration §1).

- [ ] **Step 4: Run** — `python -m unittest tests.test_gb_replica_bands tests.test_gb_band_units tests.test_gb_temper_units -v` (if those two module names do not exist, run `python -m unittest discover -s tests -p "test_gb_*unit*.py" -v`; report which ran) → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 3: Per-unit delta ledger (residual reconciliation)

**Files:**
- Modify: `gbspecialstretch.py` — `run_proposal` around OPEN (:4446-4449) and CLOSE (:4681-4684); new helpers near `adjust_sources_in_residual_buffer` (:3190-3230)
- Test: `tests/test_gb_replica_ledger.py` (new)

**Interfaces:**
- Produces on `GBSpecialBase`: `_ledger_snapshot(self, band_sorter, sel) -> dict` (`{"rows": idx, "coords_in": (n, ndim_phys), "alive": (n,), "N": (n,)}` for the selected rows, host numpy); `_ledger_delta(self, before, band_sorter) -> dict` (rows whose alive flag or physical coords changed: `{"old_coords_in", "old_alive", "new_coords_in", "new_alive", "N"}`); `_ledger_exchange(self, delta) -> list[dict]` (`self.fanout.comm.allgather(delta)`; returns the OTHER ranks' deltas); `_ledger_apply(self, model, deltas)` — for each remote delta: `fill_template(acs, old_coords_in[old_alive], walker0, N[old_alive], factor=+1)` then `fill_template(acs, new_coords_in[new_alive], walker0, N[new_alive], factor=-1)` through `self._likelihood_engine`, exactly as `adjust_sources_in_residual_buffer` does at :3223-3230.

- [ ] **Step 1: Write the failing tests** — `tests/test_gb_replica_ledger.py`

```python
"""GB per-unit ledger: snapshot/delta/exchange/apply over a FakeWorld fan-out comm."""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


def _sorter(coords_in, alive, N):
    return SimpleNamespace(
        coords_in=np.asarray(coords_in, dtype=float), inds=np.asarray(alive, dtype=bool),
        N_vals=np.asarray(N, dtype=int), xp=np,
    )


def _move(comm=None):
    m = GBSpecialBase.__new__(GBSpecialBase)
    m.fanout = None if comm is None else SimpleNamespace(comm=comm, single=False)
    m.fills = []
    m._likelihood_engine = SimpleNamespace(
        fill_template=lambda acs, params, walkers, N, factor, **kw: m.fills.append(
            (int(factor), np.asarray(params).copy(), np.asarray(N).copy())
        )
    )
    return m


class LedgerLocalTest(unittest.TestCase):
    def test_delta_lists_only_changed_rows(self):
        m = _move()
        s0 = _sorter([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1, 1, 0], [8, 8, 8])
        before = m._ledger_snapshot(s0, np.array([0, 1, 2]))
        s1 = _sorter([[1.0, 2.0], [3.5, 4.0], [7.0, 8.0]], [1, 1, 1], [8, 8, 8])  # row 1 moved, row 2 born
        d = m._ledger_delta(before, s1)
        np.testing.assert_array_equal(d["rows"], [1, 2])
        np.testing.assert_array_equal(d["old_alive"], [True, False])
        np.testing.assert_array_equal(d["new_alive"], [True, True])
        np.testing.assert_array_equal(d["new_coords_in"], [[3.5, 4.0], [7.0, 8.0]])

    def test_apply_removes_old_then_subtracts_new(self):
        m = _move()
        d = {"rows": np.array([1, 2]), "old_coords_in": np.array([[3.0, 4.0], [5.0, 6.0]]),
             "old_alive": np.array([True, False]), "new_coords_in": np.array([[3.5, 4.0], [7.0, 8.0]]),
             "new_alive": np.array([True, True]), "N": np.array([8, 8])}
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [d])
        self.assertEqual([f[0] for f in m.fills], [+1, -1])
        np.testing.assert_array_equal(m.fills[0][1], [[3.0, 4.0]])          # only the alive old row
        np.testing.assert_array_equal(m.fills[1][1], [[3.5, 4.0], [7.0, 8.0]])
        m.fills.clear()
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [{"rows": np.zeros(0, int),
            "old_coords_in": np.zeros((0, 2)), "old_alive": np.zeros(0, bool),
            "new_coords_in": np.zeros((0, 2)), "new_alive": np.zeros(0, bool), "N": np.zeros(0, int)}])
        self.assertEqual(m.fills, [])  # empty deltas make no engine calls


class LedgerExchangeTest(unittest.TestCase):
    def test_allgather_returns_the_other_ranks_deltas(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            m = _move(comm)
            mine = {"rows": np.array([rank]), "tag": rank}
            others = m._ledger_exchange(mine)
            return [o["tag"] for o in others]

        out = world.run(fn)
        self.assertEqual(out[0], [1])
        self.assertEqual(out[1], [0])
```

- [ ] **Step 2: Run to verify failure** → AttributeError.

- [ ] **Step 3: Implement** (on `GBSpecialBase`, after `add_cold_chain_sources_to_residual` :3246)

```python
    # ---- one-walker replica mode: per-unit residual reconciliation ----------
    def _ledger_snapshot(self, band_sorter, sel):
        """Host copy of the selected sorter rows (physical coords, alive flag, N)."""
        sel = np.asarray(asnumpy(sel)).reshape(-1)
        return {
            "rows": sel.astype(int),
            "coords_in": np.asarray(asnumpy(band_sorter.coords_in[sel]), dtype=np.float64),
            "alive": np.asarray(asnumpy(band_sorter.inds[sel]), dtype=bool),
            "N": np.asarray(asnumpy(band_sorter.N_vals[sel])).astype(int),
        }

    def _ledger_delta(self, before, band_sorter):
        """Rows of ``before`` whose alive flag or physical coords changed since the snapshot."""
        rows = before["rows"]
        now_c = np.asarray(asnumpy(band_sorter.coords_in[rows]), dtype=np.float64)
        now_a = np.asarray(asnumpy(band_sorter.inds[rows]), dtype=bool)
        changed = (now_a != before["alive"]) | (
            (now_a | before["alive"]) & np.any(now_c != before["coords_in"], axis=1)
        )
        return {
            "rows": rows[changed],
            "old_coords_in": before["coords_in"][changed],
            "old_alive": before["alive"][changed],
            "new_coords_in": now_c[changed],
            "new_alive": now_a[changed],
            "N": before["N"][changed],
        }

    def _ledger_exchange(self, delta):
        """Allgather this rank's delta on the fan-out comm; return the OTHER ranks' deltas."""
        comm = self.fanout.comm
        parts = comm.allgather(delta)
        me = int(comm.Get_rank())
        return [p for i, p in enumerate(parts) if i != me]

    def _ledger_apply(self, model, deltas):
        """Apply remote deltas to this replica's residual: add back the old alive
        templates (factor +1), subtract the new alive ones (factor -1)."""
        acs = model.analysis_container_arr
        for d in deltas:
            if int(np.shape(d["rows"])[0]) == 0:
                continue
            for factor, key, alive_key in ((+1, "old_coords_in", "old_alive"), (-1, "new_coords_in", "new_alive")):
                alive = np.asarray(d[alive_key], dtype=bool)
                if not alive.any():
                    continue
                params = np.asarray(d[key])[alive]
                N = np.asarray(d["N"])[alive]
                walkers = np.zeros(params.shape[0], dtype=np.int32)
                self._likelihood_engine.fill_template(
                    acs, params, walkers, N, factor=factor,
                    waveform_kwargs=getattr(self, "waveform_kwargs", {}),
                )
```
Check `adjust_sources_in_residual_buffer` (:3207-3230) for the exact `fill_template` argument order and the `waveform_kwargs`/device conventions it uses (e.g. `xp.asarray` of params, `N_vals` dtype) and mirror them verbatim in `_ledger_apply`; the test's fake engine only records the call.

`run_proposal` wiring (replica mode only, `_owned is not None`): before OPEN (:4446) compute `sel = xp.where(_owned & (band_sorter.temp_inds == 0) & <this unit's band residue mask>)[0]` — the unit residue mask is `_res_mask` on the per-walker path or `band_sorter.band_inds % units == remainder` on the scalar path — and `before = self._ledger_snapshot(band_sorter, sel)`; after CLOSE (:4684): `delta = self._ledger_delta(before, band_sorter); others = self._ledger_exchange(delta); self._ledger_apply(model, others)`. Every rank runs the same number of units (head-drawn schedule), so the allgather count matches; the head participates through its local body.

- [ ] **Step 4: Run** — `python -m unittest tests.test_gb_replica_ledger tests.test_gb_replica_bands -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 4: Touched-slot merges and the `gb_sync` command

> **SUPERSEDED (2026-09-16 review):** the slot-keyed `touched` merge below was replaced by the merge by PHYSICAL SOURCE — see spec decision 7 (amended) and `merge_owned_sources`/`_replica_merge_finish` in gbspecialstretch.py.

**Files:**
- Modify: `gbspecialstretch.py` — `GB_OPS` (:1931), `gf_serve` (:17580-17590), `_gb_serve_finish` (:17952-18098: skip `check_ll_inject` in replica mode, ship `touched`), new `_gb_serve_sync`, `_propose_orchestrated` merge loop 3 (:18862-18883) + a fourth `_fanout_cmd("gb_sync", ...)` after it, `cap_stats` merge (:18866-18872, :18917-18925: head's own in replica mode)
- Test: `tests/test_gb_replica_merge.py` (new)

**Interfaces:**
- Produces: `GB_OPS = ("gb_run_proposal", "gb_run_tempering", "gb_finish", "gb_sync")`; finish reply gains `"touched": (ntemps, nleaves_max) bool` (replica mode; `None` otherwise); module helper `merge_touched_blocks(work_coords, work_inds, replies, ranks) -> None` (raises `RuntimeError` naming the `(rung, slot)` if two ranks touched the same slot; writes each rank's touched rows into `work_coords[:, 0]`/`work_inds[:, 0]`); `_gb_serve_sync(payload, clock, model)` → `{"log_like_final", "band_counts", "residual_hash"}` after `check_ll_inject` over a `BandSorter` built from the shipped merged branch (uses `keep_all_inds=False`, alive only).

- [ ] **Step 1: Write the failing tests** — `tests/test_gb_replica_merge.py`

```python
"""Touched-slot merge for one-walker GB replicas."""

import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import merge_touched_blocks


def _reply(coords, inds, touched):
    return {"block_coords": np.asarray(coords, float), "block_inds": np.asarray(inds, bool),
            "touched": np.asarray(touched, bool)}


class MergeTouchedTest(unittest.TestCase):
    def test_each_rank_writes_only_its_touched_slots(self):
        nt, nl, nd = 2, 4, 2
        work_c = np.zeros((nt, 1, nl, nd)); work_i = np.zeros((nt, 1, nl), bool)
        r0 = _reply(np.full((nt, 1, nl, nd), 1.0), np.ones((nt, 1, nl), bool),
                    [[1, 0, 0, 0], [1, 0, 0, 0]])
        r1 = _reply(np.full((nt, 1, nl, nd), 2.0), np.ones((nt, 1, nl), bool),
                    [[0, 0, 1, 0], [0, 0, 0, 1]])
        merge_touched_blocks(work_c, work_i, {0: r0, 1: r1}, (0, 1))
        np.testing.assert_array_equal(work_c[:, 0, 0, 0], [1.0, 1.0])
        np.testing.assert_array_equal(work_c[0, 0, 2, 0], 2.0)
        np.testing.assert_array_equal(work_c[1, 0, 3, 0], 2.0)
        self.assertEqual(work_c[0, 0, 1, 0], 0.0)  # untouched stays
        self.assertTrue(work_i[0, 0, 0] and work_i[0, 0, 2] and not work_i[0, 0, 1])

    def test_conflict_raises_naming_the_slot(self):
        work_c = np.zeros((1, 1, 2, 1)); work_i = np.zeros((1, 1, 2), bool)
        r0 = _reply(np.ones((1, 1, 2, 1)), np.ones((1, 1, 2), bool), [[1, 0]])
        r1 = _reply(np.ones((1, 1, 2, 1)), np.ones((1, 1, 2), bool), [[1, 0]])
        with self.assertRaisesRegex(RuntimeError, r"\(0, 0\)"):
            merge_touched_blocks(work_c, work_i, {0: r0, 1: r1}, (0, 1))
```

- [ ] **Step 2: Run to verify failure** → ImportError.

- [ ] **Step 3: Implement**

Module level:
```python
def merge_touched_blocks(work_coords, work_inds, replies, ranks):
    """Replica-mode finish merge: every rank owns disjoint (rung, slot) cells.

    ``replies[r]["touched"]`` is ``(ntemps, nleaves_max)`` (one walker). Two
    ranks touching the same cell is a partition defect and raises.
    """
    seen = None
    for r in ranks:
        t = np.asarray(replies[r]["touched"], dtype=bool)
        if seen is None:
            seen = np.zeros_like(t)
        clash = seen & t
        if clash.any():
            rung, slot = [int(v[0]) for v in np.nonzero(clash)]
            raise RuntimeError(f"GB replica merge: cell (rung, slot) = ({rung}, {slot}) touched by two ranks")
        seen |= t
        bc = np.asarray(replies[r]["block_coords"])
        bi = np.asarray(replies[r]["block_inds"], dtype=bool)
        work_coords[:, 0][t] = bc[:, 0][t]
        work_inds[:, 0][t] = bi[:, 0][t]
```
Rank side: in `_gb_serve_finish`, after `_write_back_state` (:17980) and before the `block_coords` copy (:18005), when `self._replica_active()`: `touched = (np.any(work.coords[:, 0] != in_coords[:, 0], axis=-1) | (work.inds[:, 0] != in_inds[:, 0]))` where `in_coords/in_inds` are the shipped `part`'s branch at session open (stash them on the session in `_gb_serve_run_proposal`: `sess.in_coords = np.array(part_work.coords, copy=True)`, `sess.in_inds = ...`); skip the `check_ll_inject` at :18046-18047 in replica mode (`log_like_final = None`); add `"touched": touched` to the reply dict (:18068).

```python
    def _gb_serve_sync(self, payload, clock, model):
        """Replica mode, 4th command: rebuild this replica's residual from the merged branch."""
        from ..communication.fanout import residual_hash
        saved = self._enter_rank_block(payload, model)   # same preamble as the other commands
        try:
            merged = payload["merged"]   # {"coords": (ntemps, 1, nleaves, ndim), "inds": (ntemps, 1, nleaves)}
            branch = self._branch_from_arrays(merged["coords"], merged["inds"])   # helper: a Branch/GFState work view
            sorter = BandSorter(branch, self.band_edges, self.band_N_vals, force_backend=self.force_backend,
                                transform_fn=..., gb=..., keep_all_inds=False, ...)   # mirror the finish-time BandSorter(...) call at :18021 minus rj_prop
            log_like_final = self.check_ll_inject(model, sorter)
            band_counts = sorter.get_band_info(self.num_bands)   # (ntemps, 1, num_bands)
            return {
                "log_like_final": np.asarray(asnumpy(log_like_final)),
                "band_counts": np.asarray(band_counts),
                "residual_hash": residual_hash(model.analysis_container_arr),
            }
        finally:
            self._exit_rank_block(saved)
```
(Mirror the finish-time `BandSorter(...)` construction at :18021-18040 for the exact kwargs; `self._branch_from_arrays` wraps the arrays the way `_gb_serve_finish` obtains `work` from `new_part`.) Register the op in `GB_OPS` and `gf_serve`.

Head side, in `_propose_orchestrated` after merge loop 3: when `_replica`, replace the column writes for coords/inds/`band_counts`/`log_like_final` with `merge_touched_blocks(work.coords, work.inds, replies_f, layout.compute_ranks)` (d_h/h_h: write `sub.d_h[0][t]`/`sub.h_h[0][t]` per rank's touched cold slots the same way), then
```python
            def _payload_sync(rank, w0, w1):
                payload = _common(rank, w0, w1)
                payload.update({"merged": {"coords": np.asarray(work.coords), "inds": np.asarray(work.inds)}})
                return payload
            replies_s, _ = self._fanout_cmd("gb_sync", _payload_sync, model)
            head_rep = replies_s[layout.head_rank]
            log_like_final[:] = np.asarray(head_rep["log_like_final"])
            band_counts[:] = np.asarray(head_rep["band_counts"])
            hashes = {r: rep["residual_hash"] for r, rep in replies_s.items()}
            if len(set(hashes.values())) != 1:
                logger.warning("[GB_REPLICA] residual hashes disagree after sync: %s", hashes)
```
`cap_stats` (:18866-18872, :18917-18925): in replica mode use the head's own block only (representative).

- [ ] **Step 4: Run** — `python -m unittest tests.test_gb_replica_merge tests.test_gb_replica_bands tests.test_gb_replica_ledger tests.test_multirank_gb_smoke -v` → OK (smoke skips).

- [ ] **Step 5: Record** in the ledger.

---

### Task 5: VGB weights and parity of the inherited path

**Files:**
- Modify: `gbspecialstretch.py` — `VGBSpecialStretchMove` (:19911-20185): override `_replica_band_weights`
- Test: `tests/test_gb_replica_bands.py` (append)

**Interfaces:**
- Produces: `VGBSpecialStretchMove._replica_band_weights(work)` = per-band count of alive cold-row sources (`np.bincount(searchsorted(band_edges, f0_hz) - 1, minlength=num_bands)` using the per-leaf f0 fill through the branch's transform; mirror `BandSorter._source_freqs_hz`, gbbands.py:5643-5666).

- [ ] **Step 1: Write the failing test** (append)

```python
class VGBWeightsTest(unittest.TestCase):
    def test_weights_count_cold_sources_per_band(self):
        from types import SimpleNamespace
        from lisatools.globalfit.moves.gbspecialstretch import VGBSpecialStretchMove
        m = VGBSpecialStretchMove.__new__(VGBSpecialStretchMove)
        m.band_edges = np.array([0.0, 1.0, 2.0, 3.0])
        m.num_bands = 3
        m._cold_source_freqs_hz = lambda work: np.array([0.5, 1.5, 1.6, 2.5, 2.6, 2.7])
        w = m._replica_band_weights(SimpleNamespace())
        np.testing.assert_array_equal(w, [1, 2, 3])
```

- [ ] **Step 2: Run to verify failure** → AttributeError.

- [ ] **Step 3: Implement** on `VGBSpecialStretchMove`:
```python
    def _cold_source_freqs_hz(self, work):
        """f0 (Hz) of every alive cold-row leaf, from the per-leaf transform fill."""
        tf = self.transform_fn   # the branch transform container with n_leaf_fills
        alive = np.asarray(work.inds[0, 0])
        leaves = np.nonzero(alive)[0]
        fill_keys = list(tf.fill_dict["fill_names"]) if "fill_names" in tf.fill_dict else list(tf.fill_dict["fill_keys"])
        f0_mhz = np.asarray(tf.fill_dict["fill_values"])[:, fill_keys.index("f0")]
        return f0_mhz[leaves] / 1e3

    def _replica_band_weights(self, work):
        f0 = self._cold_source_freqs_hz(work)
        b = np.searchsorted(np.asarray(self.band_edges), f0, side="right") - 1
        b = np.clip(b, 0, self.num_bands - 1)
        return np.bincount(b, minlength=self.num_bands).astype(float)
```
(Read `BandSorter._source_freqs_hz` at gbbands.py:5643-5666 for the exact `fill_dict` key names and use them.)

- [ ] **Step 4: Run** — `python -m unittest tests.test_gb_replica_bands -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 6: One-walker GB smoke arm

**Files:**
- Modify: `tests/test_multirank_gb_smoke.py` (`NWALKERS` :186 → parametrized; a new test `test_one_walker_two_replicas` gated on `RUN_GF_GB_SMOKE`)
- Test: itself

**Interfaces:** consumes everything above.

- [ ] **Step 1: Add the test** — in `test_multirank_gb_smoke.py`, make `_build_fit(store_dir, nwalkers=NWALKERS)` take the walker count, and add
```python
    def test_one_walker_two_replicas(self):
        """1 walker on 2 compute ranks (replica mode): runs, replicas agree, no slot conflicts."""
        if not RUN:
            self.skipTest("set RUN_GF_GB_SMOKE=1")
        out = self._run_world(2, inject=True, nwalkers=1)
        # every propose logged agreeing residual hashes (grep the captured log for "[GB_REPLICA]")
        self.assertNotIn("residual hashes disagree", out["log"])
        self.assertTrue(np.all(np.isfinite(out["state"].sub_states["gb"].band_info["band_temps"])))
        self.assertGreaterEqual(int(out["state"].branches["gb"].inds[0].sum()), 1)  # the injection survives
```
(adapt `_run_world`/`_world_probe` (:345-385) to pass `nwalkers` through to `_build_fit` and to return the captured log + final state; keep the 4-walker tests unchanged).

- [ ] **Step 2: Run the light modules** — `python -m unittest tests.test_multirank_gb_smoke -v` (skips) → OK.

- [ ] **Step 3: Run the smoke ONCE, alone**: `RUN_GF_GB_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/gbsmoke.log python -m unittest tests.test_multirank_gb_smoke -v; tail -20 .wtenv/gbsmoke.log` (≤30 min, ≤5 GB; the pre-existing 4-walker two-rank and parity tests must stay green too).

- [ ] **Step 4: Record** the smoke result (runtime, RSS, verdict) in the ledger.

---

### Task 7: Regression sweep + docs

- [ ] **Step 1:** one-process sweep of Plan 1's Task-8 module list PLUS `tests.test_gb_replica_bands tests.test_gb_replica_ledger tests.test_gb_replica_merge tests.test_multirank_gb_smoke` (heavy tests skip) → OK.
- [ ] **Step 2:** docs — `docs/codebase-map.md` (one line on the replica helpers in `gbspecialstretch.py`), the spec status line (Plan 2 landed), and a short "One-walker replica mode" paragraph in `docs/multirank-cluster-gates.md` pointing at Plan 3 for the launch recipe.
- [ ] **Step 3:** `git status --short` / `git diff --stat HEAD` audit; no commits.

---

## Self-review

- **Spec coverage.** Decision 6 (static ranges, per-unit reconciliation, lockstep) → Tasks 1-3. Decision 7 (slot partition, per-slot merge, conflict assert) → Tasks 2, 4. Decision 8 (counters SUM, ladders on the head, tempering in-band, censuses after reconciliation) → Task 2 (grid mask) + existing merges (no change needed, stated in the architecture note) + Task 4 (`cap_stats` representative). Decision 9 (global tables, elastic residency, whole-grid sorter accepted) → no task (documented). Decision 10 (authoritative rebuild at propose end) → Task 4 `gb_sync`. VGB balance by source count → Task 5. Gate 14 laptop part → Task 6.
- **Placeholders.** Task 4's `_gb_serve_sync` cites the finish-time `BandSorter(...)` call to mirror (:18021-18040) rather than retyping ~20 kwargs; Task 3's `_ledger_apply` cites `adjust_sources_in_residual_buffer` (:3207-3230) for the exact `fill_template` conventions. Both are existing code to copy, not inventions. Task 2's tempering hunk names the exact site (:14698-14724) and the composition rule.
- **Type consistency.** `_owned_band_range` is a `(b0, b1)` tuple everywhere; `_replica_index`/`_n_replicas` ints; `touched` is `(ntemps, nleaves_max)` bool in both the rank reply (Task 4) and `merge_touched_blocks`; `replica_band_ranges` returns a list of tuples consumed by index `layout.replica_index(rank)` in Task 1.
