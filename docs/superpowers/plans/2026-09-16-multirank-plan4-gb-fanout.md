# Multi-rank Plan 4: GB / VGB move fan-out (WP5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GBSpecialBase.propose` (GB in-model, GB RJ prior/F-stat, VGB) runs under several compute ranks as a head orchestrator that fans out THREE commands per propose — `gb_run_proposal`, `gb_run_tempering`, `gb_finish` — each executed by every compute rank (head included) on its own walker block against its own ACA, with a per-propose rank SESSION carrying the `BandSorter`, the non-GB residual snapshot and the per-propose caches across the three commands; the head merges, SUM-reduces the band swap counters, adapts `band_temps` once, updates shut-off and caps from pooled statistics. Single-process runs execute today's body verbatim (bit-identical by construction).

**Architecture:** `GBSpecialBase` gains `install_walker_fanout(curr)` / `fanout_active` (duck-typed like the Plan 3 mixin, WITHOUT changing its MRO — GB's three-command shape does not fit `WalkerFanoutMixin.propose`), a `GBRankSession` dataclass held on `self._gb_session` between commands (token = the command clock's `seq` of the opening `gb_run_proposal`), `gf_serve` dispatching the three ops, `_enter_rank_block` / `_exit_rank_block` (device pin, `nwalkers = B`, clock sync, read-only tables) and `_setup_from_directive` (F-stat epoch tables from the epoch dir's npz; never `setup()` on a rank). Today's `propose` body (`gbspecialstretch.py:16849-17610`) is renamed `_propose_legacy` untouched; the new `propose` dispatches to it unless `fanout_active` (or `GB_PROPOSE_ORCHESTRATE=1` for the parity test) and otherwise runs `_propose_orchestrated`. Small default-preserving signature extensions let the head ship RNG draws (`scan_schedule`, `tmp_start`) and skip the rank-side ladder adaptation (`adapt_band_temps=False`). The recipe's GB/VGB builders are sized at the local block and install the fan-out on every GB/VGB move.

**Tech Stack:** Python 3.12, numpy / cupy (`xp`), Eryn, the Plan 1-3 `communication` package (`WalkerFanout.run`, `ComputeService`, `slice_state`, `FakeWorld`) and `moves/walkerfanout.py` (`pooled_ladder_step` is NOT used — GB has its own `_adapt_band_temps`). CPU-only unit tests; a memory-guarded gated smoke.

**Spec:** `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` — WP5 (three commands, per-propose session, head orchestrator, per-rank tables, VGB, legacy setups, empty-block guard, RNG, tests), Decisions 5-10, "Cross-rank swaps: assessment" (GB/VGB cell tempering stays within-rank; `# TODO(multi-rank)`). Site map with current anchors: `scratchpad/plan4-gb-sitemap.md` (controller session; every anchor below was re-read at branch HEAD 9e19a75a; the spec's `gbbands.py` anchors drifted by 37-57 lines — use the ones here).

## Global Constraints

- **Single-process / single-compute-rank runs are BIT-IDENTICAL**: `propose` calls `_propose_legacy` (today's body, byte-for-byte apart from the rename and the three default-preserving signature extensions whose defaults reproduce today's draws and calls exactly). No new RNG draw, no reordering, no extra copy on that path.
- Under several ranks: every rank runs the SAME three commands in the SAME order per propose; the head decides "run at all", the tempering gate, and every shipped RNG draw ONCE; ranks never call `_adapt_band_temps`, `_update_band_shutoff`, `_update_band_leaf_caps`' gate, `_temper_cadence_fire`, `setup()`/`_run_fstat_fit`, or write `DONE.json`/the eigen sidecar.
- `self.nwalkers` / `self.ntemps` are B inside a rank block and N on the head outside one; every head-side fan-out is wrapped so N is restored in a `finally` (many flat-index helpers read `self.nwalkers`).
- A rank's ACA row i is global walker `w0 + i`; the `BandSorter` derives `nwalkers` from the branch shape and labels walkers `0..B-1`, so the invariant holds without code (site map §4) — never index the ACA by a GLOBAL walker id on a rank.
- Payloads and replies are host numpy / plain Python only (`_to_numpy` everything device-side before replying).
- Never delete in-process multi-GPU code (router, `BandView`, `gpu_splits`, the 09-15 stream-edge fix in `gbbands.py:1426-1449`).
- Line length <= 100 on every added line (black is NOT installed: `git diff -U0 -- <file> | grep '^+' | awk 'length > 101'`).
- Tests: CPU, tiny fixtures, one python process at a time (`.wtenv/wt_run.sh` after `conda activate deving`); NEVER run `tests/test_gbspecial_flow.py`; the GB smoke is gated by `RUN_GF_GB_SMOKE=1` AND memory-guarded (Task 6). Known pre-existing failures on dev, not ours: `tests/test_maxlogl_plateau.py` (4/5), and `tests/test_stock_globalfit.py::LiteVariantTest::test_lite_kwarg_matches_twin` when run after `tests/test_psd_move_batched.py` in the same process (USE_GPU env leak; fixed in Plan 3's wave).
- Commit per task with the given message and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never commit `.superpowers/`, `.wtenv/`, scratch files; no `git push`.
- Every task's implementer reads the site-map sections named in the task before editing; anchors are `file:line` at HEAD 9e19a75a and may drift by a few lines — locate by content.

---

### Task 1: Size the GB/VGB builders' moves at the local block

**Files:**
- Modify: `src/lisatools/globalfit/recipe.py` (`build_gb_moves` ~2454-3689; `build_vgb_moves` ~3690-4001)
- Test: `tests/test_recipe_gb_local_sizing.py` (create)

**Interfaces:**
- Consumes: `_local_nwalkers(acs)` (recipe.py ~2066), `_local_walker_block` (~2077).
- Produces: in both builders a second local `nwalkers_local = _local_nwalkers(acs)` used for every MOVE/ACA-level size; the state-level `initialize_band_information(nwalkers, ...)` keeps the GLOBAL `nwalkers`.

Site map §8 lists every site. Move-level (→ `nwalkers_local`): `TemperatureControl(effective_ndim, nwalkers, ...)` at ~2773-2775 and ~3886-3887; `nfriends=nwalkers` at ~2783 and ~3916; every `<move>.accepted = np.zeros((ntemps, nwalkers))` in `build_gb_moves` (~3059, 3079, 3103, 3161, 3184, 3213, 3317, 3417, 3486, 3572, 3595, 3609, 3648) and `_ridge.accepted = np.zeros((1, nwalkers))` (~3680); `vgb_move.accepted` (~3963). State-level (STAYS `nwalkers`): `state.sub_states["gb"].initialize_band_information(nwalkers, ntemps, band_edges, band_temps, cap_edges=...)` (~2712-2715) and the vgb twin (~3858-3860); the `band_temps` writes; `Stage.setup`'s combined `accepted` (untouched, other file).

- [ ] **Step 1: Write the failing test**

```python
"""GB/VGB builders size MOVES at the local walker block; the band tables stay global."""

import re
import unittest

from lisatools.globalfit import recipe as recipe_mod


def _function_source(name):
    src = open(recipe_mod.__file__).read()
    start = src.index(f"\ndef {name}(")
    nxt = re.search(r"\n(def |class )", src[start + 1:])
    return src[start: start + 1 + (nxt.start() if nxt else len(src))]


class GBLocalSizingTest(unittest.TestCase):
    def test_gb_builder_moves_are_local_and_band_tables_global(self):
        body = _function_source("build_gb_moves")
        self.assertIn("nwalkers_local = _local_nwalkers(acs)", body)
        # every move-level size reads the local count ...
        self.assertEqual(body.count("np.zeros((ntemps, nwalkers))"), 0)
        self.assertEqual(body.count("np.zeros((1, nwalkers))"), 0)
        self.assertGreaterEqual(body.count("np.zeros((ntemps, nwalkers_local))"), 13)
        self.assertIn("nfriends=nwalkers_local", body)
        self.assertRegex(body, r"TemperatureControl\(\s*effective_ndim,\s*nwalkers_local")
        # ... while the state-level band tables keep the GLOBAL count
        self.assertRegex(body, r"initialize_band_information\(\s*nwalkers,")

    def test_vgb_builder_moves_are_local_and_band_tables_global(self):
        body = _function_source("build_vgb_moves")
        self.assertIn("nwalkers_local = _local_nwalkers(acs)", body)
        self.assertEqual(body.count("np.zeros((ntemps, nwalkers))"), 0)
        self.assertIn("np.zeros((ntemps, nwalkers_local))", body)
        self.assertIn("nfriends=nwalkers_local", body)
        self.assertRegex(body, r"TemperatureControl\(\s*effective_ndim,\s*nwalkers_local")
        self.assertRegex(body, r"initialize_band_information\(\s*nwalkers,")


if __name__ == "__main__":
    unittest.main()
```

(A source-text tripwire, like Plan 3's `test_recipe_fanout_install.py`: the builders need a full GB build to exercise — the gated smoke in Task 6 does that. Adjust the `13` to the number of `.accepted` sites you actually convert in `build_gb_moves`; it must equal the count of pre-change `np.zeros((ntemps, nwalkers))` occurrences in that function.)

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t1.log python -m unittest tests.test_recipe_gb_local_sizing -v; tail -n 20 .wtenv/p4t1.log`
Expected: both tests FAIL (`nwalkers_local` absent).

- [ ] **Step 3: Implement**

In `build_gb_moves`, right after `nwalkers: int = general_info.nwalkers` (~2508):
```python
    # MOVE/ACA-level sizes (ladders, friend windows, per-move accepted arrays)
    # follow this rank's walker block; the band tables on the sub-state stay
    # engine-wide (see initialize_band_information below). Equal single-process.
    nwalkers_local = _local_nwalkers(acs)
```
Then replace the move-level uses listed above with `nwalkers_local` (and ONLY those). Same in `build_vgb_moves` after ~3714. Update the rung-reconciliation comment at ~3880 if it mentions the sizes.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t1b.log python -m unittest tests.test_recipe_gb_local_sizing tests.test_recipe_local_block tests.test_recipe_fanout_install tests.test_band_ntemps_reconcile tests.test_stock_globalfit -v; tail -n 25 .wtenv/p4t1b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/recipe.py tests/test_recipe_gb_local_sizing.py
git commit -m "feat(recipe): GB/VGB moves sized at the local walker block; band tables stay engine-wide

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Default-preserving plumbing in `gbspecialstretch.py`

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` — `run_proposal` (4124; draw site 4241-4244), `run_tempering` (14358; `tmp_start` 14445; `_adapt_band_temps` call 15091), `_write_back_state` (15098-15176), `_temper_rng` creation (13341-13342), `_update_band_leaf_caps` (16509-16600+), `GBSpecialRJFStatGridMove._install` (18471-18559; `_band_shutoff_epoch_sync` at 18543)
- Test: `tests/test_gb_rank_plumbing.py` (create)

**Interfaces (Produces):**
- `run_proposal(self, model, state, band_sorter, band_temps, *, scan_schedule=None)`: when `scan_schedule` is a `(unit_starts, unit_dirs)` pair it REPLACES the `_draw_unit_scan_schedule(model.random, ...)` draw (no `model.random` consumption); `None` = today's draw. `_assert_unit_scan_partition` still runs on whatever is used.
- `run_tempering(self, model, state, band_sorter, band_temps, *, tmp_start=None, adapt_band_temps=True)`: `tmp_start=None` → today's `np.random.randint(units)`; an int is used as is (no draw). `adapt_band_temps=False` skips the `_adapt_band_temps` call (ranks). Returns unchanged.
- `_write_back_state(self, new_state, band_sorter) -> (inds_new, alive)` where `inds_new` is the `(temp, walker, leaf)` numpy triple written and `alive` the sorter mask (both branches: identity and dense repack).
- `_temper_rng`: created as `np.random.default_rng(self._rank_rng_seed)` when `getattr(self, "_rank_rng_seed", None) is not None`, else today's unseeded `default_rng()`. Class attr `_rank_rng_seed = None`.
- `_cap_stats_local(self, model, new_state) -> dict(band_lls=(nwalkers, num_bands), cell_lls=(nwalkers, n_cells) | None, dof=..., band_dof=...)`: the residual-dependent half of `_update_band_leaf_caps` (`_band_residual_lls(model.analysis_container_arr)`, `_track_band_best_ll` input, the `is_cells` branch calling `_cap_cell_lls(model, new_state, band_lls)`), factored out; `_update_band_leaf_caps(self, model, new_state, band_counts, *, precomputed=None)` calls it when `precomputed is None` and otherwise uses the supplied dict (the head passes walker-CONCATENATED arrays). The gate logic after `cur_max = lls.max(axis=0)` is unchanged; `bi["band_cold_ll"][:] = ...` / `bi["cap_cell_cold_ll"][:] = ...` writes happen on the head from the (possibly concatenated) arrays exactly as today.
- `GBSpecialRJFStatGridMove._install(self, k, ..., sync_shutoff=True)`: `sync_shutoff=False` skips `_band_shutoff_epoch_sync()` (ranks receive the valve state from the head instead).

- [ ] **Step 1: Write the failing test**

```python
"""Default-preserving plumbing GB needs for the rank commands (no GPU, no build)."""

import inspect
import unittest

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs


class SignatureTest(unittest.TestCase):
    def test_run_proposal_accepts_a_shipped_scan_schedule(self):
        sig = inspect.signature(gbs.GBSpecialBase.run_proposal)
        self.assertIn("scan_schedule", sig.parameters)
        self.assertIs(sig.parameters["scan_schedule"].default, None)
        self.assertEqual(sig.parameters["scan_schedule"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_run_tempering_accepts_tmp_start_and_adapt_flag(self):
        sig = inspect.signature(gbs.GBSpecialBase.run_tempering)
        self.assertIs(sig.parameters["tmp_start"].default, None)
        self.assertIs(sig.parameters["adapt_band_temps"].default, True)

    def test_update_band_leaf_caps_accepts_precomputed(self):
        sig = inspect.signature(gbs.GBSpecialBase._update_band_leaf_caps)
        self.assertIs(sig.parameters["precomputed"].default, None)
        self.assertTrue(callable(gbs.GBSpecialBase._cap_stats_local))

    def test_install_accepts_sync_shutoff(self):
        sig = inspect.signature(gbs.GBSpecialRJFStatGridMove._install)
        self.assertIs(sig.parameters["sync_shutoff"].default, True)

    def test_write_back_state_returns_inds_and_alive(self):
        src = inspect.getsource(gbs.GBSpecialBase._write_back_state)
        # both branches return the written (temp, walker, leaf) triple and the alive mask
        self.assertEqual(src.count("return inds_new, alive"), 2)


class TemperRngSeedTest(unittest.TestCase):
    def test_rank_seed_makes_the_vertical_swap_rng_deterministic(self):
        # _temper_rng is created lazily in the in-model block; exercise the factory only
        make = gbs.GBSpecialBase._make_temper_rng
        a = make(type("M", (), {"_rank_rng_seed": 123})())
        b = make(type("M", (), {"_rank_rng_seed": 123})())
        c = make(type("M", (), {"_rank_rng_seed": None})())
        self.assertEqual(a.random(), b.random())
        self.assertIsInstance(c, np.random.Generator)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t2.log python -m unittest tests.test_gb_rank_plumbing -v; tail -n 20 .wtenv/p4t2.log`
Expected: failures (missing parameters / `_make_temper_rng`).

- [ ] **Step 3: Implement** (each change default-preserving; single mode calls with the defaults)

1. `run_proposal` (4124): add `*, scan_schedule=None`; at 4241-4244:
   ```python
        if scan_schedule is None:
            _unit_starts, _unit_dirs = _draw_unit_scan_schedule(
                model.random, self.nwalkers, units,
                _per_walker_start, _per_walker_dir,
            )
        else:  # head-drawn (multi-rank): this rank's block of the N-walker schedule
            _unit_starts, _unit_dirs = scan_schedule
   ```
   (`_draw_unit_scan_schedule`'s own signature/body must not change — `tests/test_band_unit_scan_order.py` pins it.)
2. `run_tempering` (14358): add `*, tmp_start=None, adapt_band_temps=True`; at 14445: `tmp_start = np.random.randint(units) if tmp_start is None else int(tmp_start)`; at 15091: `if adapt_band_temps: self._adapt_band_temps(...)`.
3. `_write_back_state`: return `inds_new, alive` at the end of BOTH branches (identity branch after `self._sync_cold_row(new_state)`; dense branch likewise). Update the `-> None` annotation and the first docstring line.
4. `_temper_rng`: add
   ```python
    _rank_rng_seed = None  # set by _enter_rank_block (multi-rank); None = today's entropy

    @staticmethod
    def _make_temper_rng(move):
        seed = getattr(move, "_rank_rng_seed", None)
        return np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
   ```
   and at 13341-13342: `self._temper_rng = self._make_temper_rng(self)`.
5. `_cap_stats_local` + `precomputed`: move the block from `band_lls = self._band_residual_lls(...)` through the `else: lls, dof = band_lls, self._band_dof` into `_cap_stats_local(self, model, new_state)` returning `{"band_lls": band_lls, "lls": lls, "dof": dof, "band_dof": self._band_dof, "is_cells": is_cells}`; `_update_band_leaf_caps` computes `bi`, `cap, iters, best`, then `stats = precomputed if precomputed is not None else self._cap_stats_local(model, new_state)`, sets `self._band_dof = stats["band_dof"]`, performs the `bi["band_cold_ll"]`/`bi["cap_cell_cold_ll"]` writes and `_track_band_best_ll(bi, stats["band_lls"])` exactly as today (same order), then continues with `lls, dof = stats["lls"], stats["dof"]` into the unchanged gate. Keep every log line.
6. `_install(k, ..., sync_shutoff=True)`: wrap the `_band_shutoff_epoch_sync()` call at 18543 in `if sync_shutoff:`.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t2b.log python -m unittest tests.test_gb_rank_plumbing tests.test_band_unit_scan_order tests.test_temper_batch_perms tests.test_temper_census_hoist tests.test_temper_compact_rows tests.test_temper_shutoff_bands tests.test_temper_skip_empty tests.test_tempering_swap_cap tests.test_vertical_swap tests.test_cell_label_deferred tests.test_gb_cap_cell_grid tests.test_cap_stagger tests.test_band_shutoff_designation tests.test_band_shutoff_revival tests.test_fstat_ctr_epoch -v; tail -n 30 .wtenv/p4t2b.log`
(Drop any module name that does not exist; `ls tests/test_fstat_*.py` for the ctr-epoch name.) Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_gb_rank_plumbing.py
git commit -m "feat(gb): default-preserving plumbing for rank commands -- shipped scan schedule/tmp_start, adapt flag, write-back return, seeded temper rng, cap stats split

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The rank session and the three `gf_serve` commands

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` (new methods on `GBSpecialBase`; place them directly above `propose` ~16849)
- Test: `tests/test_gb_rank_session.py` (create)

**Interfaces (Produces):**
- `GBSpecialBase.install_walker_fanout(self, curr)`: `self.fanout = getattr(curr, "fanout", None)`, `self.gf_rank = getattr(curr, "rank", None)`; when active and not head: `self.eigen_store_path = None` if the attribute exists. Class attrs `fanout = None`, `gf_rank = None`. Property `fanout_active` (same semantics as the mixin's).
- `GB_OPS = ("gb_run_proposal", "gb_run_tempering", "gb_finish")` module constant.
- `@dataclasses.dataclass class GBRankSession`: `token`, `band_sorter`, `keep_all_inds`, `start_diffs`, `snapshot` (the value of `self.reset_non_gb_linear_data_arr`), `part` (the rank's slice `GFState`), `new_part` (its working copy), `ntemps`, `nwalkers`, `timer`, `engine_ntemps`, plus `saved` (a dict of the move attributes saved by `_enter_rank_block` for `_exit_rank_block` to restore on the head: `nwalkers`, `ntemps`, `time`, `num_proposals`, `_reseed_firing`, `temper_vertical`, `_cap_leaf_cap`, `_band_leaf_cap`, `_rj_band_shutoff`, `_rank_rng_seed`).
- `_enter_rank_block(self, payload, clock, model)`: `pin_main_device(self.xp, acs.gpus)`; `_configure_domain(acs)`; `_bind_parent_acs(acs)` (the same three calls the body does at 16910-16917); set `self.nwalkers = B`, `self.ntemps = payload["ntemps"]`, `self.time = clock_vals["time"]`, `self.num_proposals = clock_vals["num_proposals"]`, `self._reseed_firing = clock_vals["reseed_firing"]`, `self.temper_vertical = clock_vals["temper_vertical"]`, `type(self)._branch_propose_counts[branch] = clock_vals["branch_propose_count"]` (class dict; read by `_fstat_clock`), `self._cap_leaf_cap` / `self._band_leaf_cap` = the shipped read-only copies (or None), `self._rj_band_shutoff` = shipped copy (or None), `self._rank_rng_seed = payload["rank_seed"]`, `self._temper_rng = None` (re-created seeded on first use). Returns the saved dict.
- `_exit_rank_block(self, saved)`: restores every saved attribute (the head needs N back; on a pure rank the restore is harmless).
- `_setup_from_directive(self, directive)`: for `GBSpecialRJFStatGridMove` instances (`hasattr(self, "_install")`): if `directive["epoch"] is not None`: `self._install(directive["epoch"], sync_shutoff=False)` unless the `_FSTAT_GRID_REGISTRY` already holds it (the registry makes repeats cheap — check the registry key the way `setup()` does at 18570), then `self._install_ctr_table(directive["epoch"], model=None)` when `directive["ctr_table"]`; base class: no-op. NEVER calls `setup()` / `_run_fstat_fit`.
- `_make_slice_state(self, payload) -> GFState`: `part = payload["state"]` (a `slice_state(..., sub_states=[branch])` product built on the head); returns it (already a GFState). The working copy is `GFState(part, copy=True)`; bare resurrected sub-states of OTHER branches are irrelevant (GB never touches them; nothing is merged from the reply state).
- `gf_serve(self, op, payload, clock, model)` (overrides `GlobalFitMove.gf_serve`):
  - `"gb_run_proposal"`: `saved = _enter_rank_block(...)`; `_setup_from_directive(payload["directive"])`; `part = _make_slice_state(payload)`; if `payload["neutral"]` (head decided this block has nothing to do, see Task 4) → open a session with `band_sorter=None` and reply zeros of the right shapes; else `new_part = GFState(part, copy=True)`, `work = self._work_branch(new_part)`, `band_temps = self.xp.asarray(payload["band_temps"])`, BandSorter construction exactly as 17146-17169 (with `keep_all_inds=payload["keep_all_inds"]`, resetting `_sorter_dh/_hh`), `_infomat_wdm_logged = False`, `_tables_indexed = False`, snapshot round trip 17185-17193 (stores `self.reset_non_gb_linear_data_arr`), start check 17197-17211 (`start_diffs` kept), `_replace_accept_forensics = []`, then the `run_proposal` passes 17233-17250 with `scan_schedule=payload["scan_schedule"]` (a `(starts, dirs)` pair for THIS block, or None when the head drew a global scalar schedule — see Task 4), `new_part.log_like[0] += ll_change_sum[0]`, drift check/rebuild 17263-17312. Session stored on `self._gb_session` with `token=clock["seq"]`. Reply: `{"log_like_cold": (B,), "prop_counts": (2, ntemps, num_bands) walker-sum, "acc_counts": same, "cold_counts": (2, 2, B, num_bands) = [prop, acc][:, 0] cold rows, "alive_per_temp": list, "n_alive": int, "start_diffs": ..., "drift": float, "rj_at_cap": ..., "timing": tm.snapshot() or None}`. `_exit_rank_block(saved)` in a `finally` (the session keeps its own copies of what it needs; `nwalkers` is re-set to B by `_enter_rank_block` on the next command).
  - `"gb_run_tempering"`: check `payload["session"] == self._gb_session.token` (else `RuntimeError("stale GB rank session")`); `_enter_rank_block`; `band_temps = xp.asarray(payload["band_temps"])`; zero `band_swaps_*`; `run_tempering(model, new_part, band_sorter, band_temps, tmp_start=payload["tmp_start"], adapt_band_temps=False)`; `new_part.log_like[0] += ll_change_sum_temp[0]`; post-tempering drift check/rebuild 17375-17397. Reply: `{"log_like_cold", "band_swaps_accepted": (num_bands, ntemps-1) host ints, "band_swaps_proposed", "ll_change_sum_temp_cold": (B,), "drift", "census": the `_sc` dict if available, "timing"}`. Neutral session → zeros.
  - `"gb_finish"`: token check; `_enter_rank_block`; `inds_new, alive = _write_back_state(new_part, band_sorter)` (skip when neutral); teardown of the first sorter + mempool free (17445-17447); the second alive-only sorter 17448-17465; `band_counts = new_band_sorter.get_band_info()["band_counts"]` (ntemps, B, num_bands); `log_like_final = check_ll_inject(model, new_band_sorter)` (B,); `cap_stats = self._cap_stats_local(model, new_part) if payload["want_cap_stats"] else None` (NOTE: `_cap_stats_local` reads `new_state.sub_states[branch].band_info` ONLY through `_cap_cell_source_lls`/`_cold_occupancy` — confirm on the slice, which has NO `band_info`; if it does read `band_info`, pass the head's cap-relevant arrays in the payload under `"cap_tables"` and have `_cap_stats_local` accept them — the implementer verifies and documents); teardown: `_buffer_cache_teardown()`, `self._fstat_nm_lanes = None`, `self._gb_session = None`. Reply: `{"block_coords": (ntemps, B, nleaves_max, ndim), "block_inds": (ntemps, B, nleaves_max) bool, "d_h": (B, nleaves), "h_h": (B, nleaves), "band_counts": (ntemps, B, num_bands), "log_like_final": (B,), "cap_stats": {...} | None, "fstat_ctr_fallback_rows": int, "band_dof": ..., "rj_split": ..., "replace_census": ..., "timing": tm.report string or dict}` — **changed in fix round 4 (2026-09-16)**: the reply ships the whole BLOCK branch, not `alive_coords`/`alive_twl`. A `keep_all_inds` `BandSorter` writes its rejected-birth fill into `work.coords`' dead slots (its coords are a view of the branch array on CPU) and `_propose_legacy` returns that, so an alive-only export left the head's dead slots at their pre-propose values. A NEUTRAL block replies `None`/`None` for both keys.
  - unknown op → `ValueError`.
- Empty-block rule: the HEAD decides (Task 4) whether a block runs; a rank never returns early on its own.

- [ ] **Step 1: Write the failing test**

`tests/test_gb_rank_session.py` — a skeleton instance (`GBSpecialBase.__new__(GBSpecialBase)`) with the attributes the plumbing reads (`branch_name="gb"`, `name="gb_test"`, `xp=np`, `nwalkers=4`, `ntemps=2`, `time=3`, `num_proposals=7`, `_reseed_firing=False`, `temper_vertical=False`, `_cap_leaf_cap=None`, `_band_leaf_cap=None`, `_rj_band_shutoff=None`, `_gb_session=None`, `fanout=None`, `gf_rank=None`, `eigen_store_path="x.h5"`) and monkeypatched heavy methods (`pin_main_device` → no-op via `unittest.mock.patch.object(gbs, "pin_main_device")`, `_configure_domain`, `_bind_parent_acs` → no-ops), testing:
1. `_enter_rank_block` sets `nwalkers=B`, `ntemps`, `time`, `num_proposals`, `_reseed_firing`, `temper_vertical`, `_rank_rng_seed`, the class census; `_exit_rank_block(saved)` restores `nwalkers=4`, `time=3`, etc.
2. `install_walker_fanout` with a single-rank fanout → `fanout_active` False, `eigen_store_path` kept; with a stub fanout (`single=False`, `is_head=False`) → cleared; head → kept.
3. `gf_serve("gb_run_tempering", {"session": 99, ...})` with no session → `RuntimeError` mentioning "stale"; `gf_serve("nope", ...)` → `ValueError`.
4. Neutral `gb_run_proposal` → opens a session (`token == clock["seq"]`) and replies zero arrays of the right shapes: `log_like_cold (B,)`, `prop_counts (2, ntemps, num_bands)`, `cold_counts (2, 2, B, num_bands)`; then neutral `gb_run_tempering` and `gb_finish` reply zeros / empty alive arrays and close the session (`_gb_session is None`).
5. `_setup_from_directive({"epoch": None, "ctr_table": False})` is a no-op on the base skeleton.

Write the assertions concretely (shapes from `num_bands = len(band_edges) - 1` with `band_edges = np.linspace(1e-3, 2e-3, 5)` on the skeleton).

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t3.log python -m unittest tests.test_gb_rank_session -v; tail -n 20 .wtenv/p4t3.log`
Expected: `AttributeError` (no `_enter_rank_block` / `gf_serve` override).

- [ ] **Step 3: Implement** the interfaces above. Copy the body fragments from `propose` (line ranges given) into the commands VERBATIM (they become the rank body); do not edit `propose` in this task. `GBRankSession` is a module-level dataclass. The neutral path allocates zeros with `num_bands = len(self.band_edges) - 1`.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t3b.log python -m unittest tests.test_gb_rank_session tests.test_gb_rank_plumbing tests.test_run_multirank_helpers tests.test_move_build_context_layout -v; tail -n 25 .wtenv/p4t3b.log`
Expected: all OK. Also `_fanout_unready_moves` now sees `gf_serve` overridden on `GBSpecialBase` → GB moves are no longer "unready"; confirm `tests.test_run_multirank_helpers` still passes (it uses stub moves).

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_gb_rank_session.py
git commit -m "feat(gb): per-propose rank session and the three gf_serve commands (run_proposal / run_tempering / finish)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The head orchestrator (`_propose_orchestrated`) and `_propose_legacy`

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` (`propose` 16849-17610)
- Test: `tests/test_gb_orchestrator_merge.py` (create)

**Interfaces (Produces):**
- `_propose_legacy(self, model, state)`: today's `propose` body, renamed, byte-identical.
- `propose(self, model, state)`: `if self.fanout_active or (self.fanout is not None and os.environ.get("GB_PROPOSE_ORCHESTRATE", "0") == "1"): return self._propose_orchestrated(model, state)`; else `return self._propose_legacy(model, state)`.
- `_propose_orchestrated(self, model, state)` (head): prologue 16861-17108 VERBATIM (census tick, timer, device pin/bind, consistency check, the three early returns — they return `(state, zeros((engine_ntemps, N)))`; `self.nwalkers = N; self.ntemps = ntemps`; reseed on the FULL sub-state; cap arming; `self.setup(model, state.branches)` (the REAL setup — F-stat fit + DONE.json on the head only); `num_proposals += 1`), then:
  1. `new_state = GFState(state, copy=True)`; `work = self._work_branch(new_state)`; `band_temps_host = np.array(band_info["band_temps"], copy=True)`; periodic wrap 17131-17133 on the full branch; `keep_all_inds` 17141-17144.
  2. Directive: `directive = {"epoch": <the epoch k that setup() installed, from the same helper setup() uses (_latest_epoch / the registry key), or None>, "ctr_table": bool(self._fstat_ctr_table_active())}` — read `setup()` 18561-18600 to derive it; base-class moves → `{"epoch": None, "ctr_table": False}`.
  3. Scan schedule: `units = self.band_units if self.num_bands > 1 else 1`; `_per_walker_start = bool(self.band_unit_start_per_walker) and units > 1`; `_per_walker_dir = bool(self.band_unit_dir_per_walker) and units > 2`; `starts, dirs = _draw_unit_scan_schedule(model.random, N, units, _per_walker_start, _per_walker_dir)`; per block `sched_block(w0, w1) = (starts if np.ndim(starts) == 0 else starts[w0:w1], dirs if np.ndim(dirs) == 0 else dirs[w0:w1])` (read `_draw_unit_scan_schedule` 631-675 for the scalar-vs-array return contract and `_unit_pass_remainder` for how a scalar is consumed; ship exactly what `run_proposal` would have drawn locally for that block).
  4. Clock values: `{"time": self.time, "num_proposals": self.num_proposals, "reseed_firing": bool(self._reseed_firing), "temper_vertical": bool(self.temper_vertical), "branch_propose_count": type(self)._branch_propose_counts.get(self.branch_name, 0)}`; tables: `_to_numpy` copies of `_cap_leaf_cap`, `_band_leaf_cap`, `_rj_band_shutoff` (or None); `rank_seed`: `derive_rank_seed(self.fanout.clock.get("seed_base") or 0, self.fanout.layout, rank)` per rank (import from `..communication.ranks`) — the head's own is derived the same way.
  5. Neutral decision per block: `neutral = (not keep_all_inds) and not np.any(work.inds[:, w0:w1])`.
  6. **Fan-out 1**: `self._fanout_cmd("gb_run_proposal", per_rank_payload=lambda r, w0, w1: {...}, ...)` where `_fanout_cmd` wraps `self.fanout.run(op, move=self.gf_move_name, per_rank_payload=..., local_body=lambda p, m: self.gf_serve(op, p, self.fanout.clock, m), merge=lambda r: r)` in `try: ... finally: self.nwalkers = N; self.ntemps = ntemps` and returns `{rank: reply}`. Payload per rank: `{"state": slice_state(state, w0, w1, sub_states=[branch]), "ntemps": ntemps, "band_temps": band_temps_host, "keep_all_inds": keep_all_inds, "scan_schedule": sched_block(w0, w1), "clock": clock_vals, "tables": {...}, "directive": directive, "rank_seed": seed_r, "neutral": neutral_r}`. Merge: `new_state.log_like[0, w0:w1] = rep["log_like_cold"]`; `prop_counts_sum += rep["prop_counts"]`, `acc_counts_sum += ...`; `cold_prop[:, w0:w1] = rep["cold_counts"][0]`, `cold_acc[:, w0:w1] = rep["cold_counts"][1]`; logs (alive census summed, drift max).
  7. Tempering gate 17316-17363 VERBATIM on the head (it calls `_temper_cadence_fire` once); if it fires: `units_t = self.band_units if self.num_bands > 1 else 2`; `tmp_start = int(np.random.randint(units_t))`; **Fan-out 2** with payload `{"session": seq_of_fanout_1, "band_temps": band_temps_host, "tmp_start": tmp_start, "clock": clock_vals}` (the session token = `self.fanout.clock["seq"]` captured right after fan-out 1 — read `WalkerFanout.run` to confirm `clock["seq"]` is the value the workers saw; in single mode `seq` does not advance — then use `call_index`; make `_fanout_cmd` return `(replies, token)` with `token = dict(self.fanout.clock)`-derived `(seq, call_index)` tuple and have the rank store the same tuple from ITS `clock`). Merge: `new_state.log_like[0, w0:w1] = rep["log_like_cold"]`; `swaps_acc += rep["band_swaps_accepted"]`, `swaps_prop += ...`; then ONCE on the head: `bt = self.xp.asarray(band_temps_host); self._adapt_band_temps(bt, self.xp.asarray(swaps_acc), self.xp.asarray(swaps_prop)); band_temps_host = _to_numpy(bt)` (reads `self.time` BEFORE the increment, as today).
  8. **Fan-out 3**: payload `{"session": token, "want_cap_stats": self._band_leaf_cap is not None and self.leaf_cap_update, "clock": clock_vals}`. Merge, in this order (**changed in fix round 4 (2026-09-16)**: no global `work.inds[:] = False` wipe, and the block arrays replace `alive_coords`/`alive_twl`): for each NON-NEUTRAL rank: `work.coords[:, w0:w1] = rep["block_coords"]; work.inds[:, w0:w1] = rep["block_inds"]` — a neutral rank replies `None`/`None` and the head simply keeps the slice it shipped, which is byte-equal to the old wipe-and-skip because a neutral block's `inds` are all-False over the whole block; `sub.d_h[w0:w1] = rep["d_h"]; sub.h_h[w0:w1] = rep["h_h"]`; `band_counts[:, w0:w1] = rep["band_counts"]`; `log_like_final[w0:w1] = rep["log_like_final"]`; cap stats concatenated on the walker axis (`band_lls`, `lls`; `dof`/`band_dof`/`is_cells` from any rank); `self._fstat_ctr_fallback_rows = sum(...)`; `self._band_dof = rep["band_dof"]`. Then `self._sync_cold_row(new_state)`.
  9. `_update_band_shutoff(self._band_occupancy_cold_max(new_state), new_state)` under `_band_shutoff_enabled()` with the same try/except as 17429-17437 (now AFTER the write-back merge — the legacy order writes back first too).
  10. `self.time += 1` (17490's position relative to `_adapt_band_temps` preserved: adaptation happened in step 7).
  11. Sub-state writes 17499-17514: `band_info["band_temps"][:] = band_temps_host`; `band_num_binaries[:] = band_counts`; `accumulate_proposals(prop_counts_sum[0].T ...)` — note the legacy passes `prop_counts[0].sum(axis=1).T` (walker-summed): the reply's `prop_counts` is ALREADY walker-summed `(2, ntemps, num_bands)`, so pass `prop_counts_sum[0].T` / `acc_counts_sum[0].T` (RJ) and `[1]` (in-model) — check `accumulate_proposals`' expected shape at state.py:1233-1241; `accumulate_swaps(swaps_prop, swaps_acc)` (zeros when tempering did not fire, exactly as today).
  12. `new_state.log_like[:] = log_like_final[None, :]` (17519-17522 semantics).
  13. `if self._band_leaf_cap is not None and self.leaf_cap_update: self._update_band_leaf_caps(model, new_state, band_counts, precomputed=cap_stats_concat)`.
  14. `accepted = np.zeros((engine_ntemps, N), dtype=bool)`; the propose-end logs 17537-17602 from the merged counters (`[GB_ACCEPT]` per-walker decomposition from `cold_acc[1] / cold_prop[1]`), `[FSTAT_CTR]` from the summed fallback rows, `[GB_TIMING]` from the head's own timer plus one line per rank from `rep["timing"]` (DEBUG level), and the `[FANOUT]` lines come from `WalkerFanout.run` itself.
  15. `return new_state, accepted`.
- `_fanout_cmd(self, op, per_rank_payload, payload_common=None) -> (replies, token)` as described.

- [ ] **Step 1: Write the failing test**

`tests/test_gb_orchestrator_merge.py`: a `_StubGB(GBSpecialBase)`-free approach — build a `GBSpecialBase.__new__` skeleton (as in Task 3) whose `gf_serve` is REPLACED by a deterministic stub (`self.gf_serve = types.MethodType(stub, move)`) that records the op sequence and returns synthetic replies (log_like_cold = `w0 + arange(B) + 100*op_index`, `band_counts` = rank-tagged ones, `block_inds`/`block_coords` = one alive source per (temp 0, local walker 0) at leaf 0 with coords `[rank, 0, ...]` (**changed in fix round 4 (2026-09-16)** from `alive_twl`/`alive_coords`), swap counters = rank-dependent ints, `band_swaps_*` such that the pooled ratio is known, cap stats = walker-tagged rows). Then drive `_propose_orchestrated` over `FakeWorld(3)` (two compute ranks, nwalkers 4) with `_propose_legacy` prologue parts monkeypatched to no-ops where they need real machinery (`pin_main_device`, `_configure_domain`, `_bind_parent_acs`, `setup`, `_check_substate_consistency`, `periodic.wrap`, `_temper_cadence_fire` → True, `_update_band_shutoff` → record call, `_update_band_leaf_caps` → record `precomputed`), a real `GBState`-bearing `GFState` from `tests/test_gf_substate_roundtrip.make_state` (has a `gb` sub-state; call `initialize_band_information(N, ntemps, band_edges, band_temps)` on it first — read state.py ~930-1010 for the signature and the required `cap_edges`/`branch_name` kwargs), and a real `_adapt_band_temps` (numpy `xp`). Assert:
   - the op sequence on the worker is exactly `["gb_run_proposal", "gb_run_tempering", "gb_finish"]` and the head served the same;
   - `new_state.log_like[0]` equals the concatenation of the two ranks' `log_like_final` blocks in walker order;
   - `work.inds` has exactly one True per block at `(0, w0, 0)` and the coords carry the rank tag;
   - `band_num_binaries[:, w0:w1]` equal each rank's `band_counts`;
   - `band_info["band_temps"]` equals `_adapt_band_temps` applied ONCE to the initial ladder with the SUMMED counters (compute the expectation with a fresh numpy call) and differs from applying it per rank sequentially;
   - `self.time` advanced by exactly 1; `self.nwalkers == N` after the propose; the recorded `precomputed["lls"]` has N rows in walker order;
   - the neutral flag was False for both blocks; a second scenario with `work.inds[:, 2:4] = False` and `keep_all_inds=False` sends `neutral=True` to rank 1 only.
   The stub must ALSO verify the payload contract it receives: `payload["state"].branches["gb"].nwalkers == B`, `payload["scan_schedule"]` is a pair, `payload["session"]` equals the token it stored on the first command.

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t4.log python -m unittest tests.test_gb_orchestrator_merge -v; tail -n 20 .wtenv/p4t4.log`
Expected: `AttributeError: _propose_orchestrated`.

- [ ] **Step 3: Implement** as specified. Keep `_propose_legacy` byte-identical (verify with `git diff` that its body is a pure move). Delete NOTHING from it (dead variables included) — the parity test needs it unchanged.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t4b.log python -m unittest tests.test_gb_orchestrator_merge tests.test_gb_rank_session tests.test_gb_rank_plumbing tests.test_module_substate_reseed tests.test_gf_substate_roundtrip tests.test_walkerslice_roundtrip -v; tail -n 25 .wtenv/p4t4b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_gb_orchestrator_merge.py
git commit -m "feat(gb): head orchestrator over three rank commands; today's propose kept verbatim as _propose_legacy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: VGB, legacy setups, recipe wiring, readiness

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` (VGB `in_model_proposal` assert ~17898-17901; `GBSpecialRJSerialSearchMCMC.setup` ~18938; `GBSpecialRJRefitMove.setup` ~19236)
- Modify: `src/lisatools/globalfit/recipe.py` (`build_gb_moves`: after each GB move is constructed and stamped; `build_vgb_moves`: the vgb move)
- Modify: `src/lisatools/globalfit/run.py` (`_fanout_unready_moves` docstring: GB/VGB now served)
- Test: `tests/test_recipe_fanout_install.py` (update the count), `tests/test_gb_rank_session.py` (append the legacy-setup test)

Steps:
1. VGB: at the assert add the comment `# TODO(multi-rank): the red/blue complement is this rank's walker block (design spec, "Cross-rank swaps: assessment")` and extend the message with the block hint.
2. Legacy setups: at the top of both `setup` methods: `if getattr(self, "fanout_active", False): raise NotImplementedError(f"{type(self).__name__} (FD dev-search path, scalar-walker ParaEnsembleSampler) is not ported to several compute ranks; run with one compute rank or GF_LEGACY_RANK_LAYOUT=1")`.
3. Recipe wiring: every GB/VGB move object `build_gb_moves`/`build_vgb_moves` return (the ~14 moves whose `.accepted` Task 1 resized, plus the ridge move and the vgb move) gets `move.install_walker_fanout(curr)` right after its `.accepted` assignment. Count the sites; update `tests/test_recipe_fanout_install.py::test_builders_call_install` to the new total (`3 + <GB/VGB sites>`), with a comment listing the three groups.
4. `run.py` `_fanout_unready_moves` docstring: replace "GB/VGB remain unready until Plan 4" with "GB/VGB (GBSpecialBase) serve the three-command protocol since Plan 4; the FD dev-search setups raise under several ranks".
5. Test (append to `tests/test_gb_rank_session.py`): a skeleton `GBSpecialRJSerialSearchMCMC.__new__` with `fanout` = stub(`single=False`) → `setup(None, None)` raises `NotImplementedError`; with `fanout=None` the method proceeds past the guard (patch the next call it makes to raise a sentinel and assert the sentinel).
6. Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4t5.log python -m unittest tests.test_recipe_fanout_install tests.test_gb_rank_session tests.test_run_multirank_helpers tests.test_recipe_gb_local_sizing tests.test_vgb_eigen_inmodel -v; tail -n 25 .wtenv/p4t5.log` (drop a vgb module name that does not exist). Expected: all OK.
7. Commit:
```bash
git add src/lisatools/globalfit/moves/gbspecialstretch.py src/lisatools/globalfit/recipe.py src/lisatools/globalfit/run.py tests/test_recipe_fanout_install.py tests/test_gb_rank_session.py
git commit -m "feat(gb): install the fan-out on every GB/VGB move; VGB block-complement TODO; legacy FD setups refuse several ranks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Gated GB smoke + seeded parity + whole-plan verification

**Files:**
- Create: `tests/test_multirank_gb_smoke.py` (gated `RUN_GF_GB_SMOKE=1`)

- [ ] **Step 1: Write the smoke**

```python
"""GB fan-out end to end on the debug-preset gb_no_fg synthetic fit (fake communicator).

Gated by RUN_GF_GB_SMOKE=1 (heavier than the noise/blank smokes). Three scenarios:
single rank (today's body), two compute ranks (orchestrator), and the seeded PARITY
of the orchestrator at ONE compute rank against the legacy body.
"""

import os
import resource
import shutil
import tempfile
import unittest

import numpy as np

RUN = os.environ.get("RUN_GF_GB_SMOKE", "") not in ("", "0")


def _rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 ** 3) if ru > 1e7 else ru / (1024.0 ** 2)  # macOS bytes / linux KB


@unittest.skipUnless(RUN, "set RUN_GF_GB_SMOKE=1 to run the multi-rank GB smoke")
class MultiRankGBSmokeTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        os.environ.setdefault("GB_DEBUG", "1")  # debug preset: 3-day Tobs, tiny grids
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_gb_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.assertLess(_rss_gb(), 5.0, "GB smoke exceeded the 5 GB laptop budget")

    def _fit(self, subdir, nwalkers):
        from lisatools.globalfit.stock import erebor

        fit = erebor.gb_no_fg(
            nwalkers=nwalkers, ntemps=2, data_mode="synthetic",
            file_store_dir=os.path.join(self.tmpdir, subdir), make_diagnostic_plots=False,
        )
        fit.general.num_iterations = 2
        fit.general.random_seed = 4242
        return fit

    def _run_world(self, size, nwalkers=4, env=None):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit

        def fn(rank, comm):
            for k, v in (env or {}).items():
                os.environ[k] = v
            fit = self._fit(f"n{size}", nwalkers)
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {"role": layout.role_of(rank).value, "acs_rows": int(gf.acs.acs_total_entries)}
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            else:
                sub = gf.state.sub_states["gb"]
                out["log_like"] = np.array(gf.state.log_like[0], copy=True)
                out["coords"] = np.array(gf.state.branches["gb"].coords, copy=True)
                out["inds"] = np.array(gf.state.branches["gb"].inds, copy=True)
                out["band_temps"] = np.array(sub.band_info["band_temps"], copy=True)
                out["band_num_binaries"] = np.array(sub.band_info["band_num_binaries"], copy=True)
            return out

        return FakeWorld(size, timeout=3600.0).run(fn)

    def test_two_compute_ranks_run_the_gb_moves(self):
        out = self._run_world(2)
        self.assertEqual((out[0]["role"], out[1]["role"]), ("head", "compute"))
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        self.assertGreaterEqual(out[1]["served"], 1 + 3)  # ping + 3 commands per GB propose
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (4,))
        self.assertTrue(np.all(np.diff(out[0]["band_temps"], axis=-1) <= 0))

    def test_orchestrator_at_one_rank_matches_the_legacy_body(self):
        legacy = self._run_world(1, env={"GB_PROPOSE_ORCHESTRATE": "0"})[0]
        orch = self._run_world(1, env={"GB_PROPOSE_ORCHESTRATE": "1"})[0]
        for key in ("log_like", "coords", "inds", "band_temps", "band_num_binaries"):
            np.testing.assert_array_equal(orch[key], legacy[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()
```

Notes for the implementer: (1) `gb_no_fg` stock kwargs — mirror `tests/test_globalfit_sample.py`; `general.random_seed` seeds eryn's stream (the parity test needs both runs seeded identically; `_seed_rank_streams` is single-mode-inert, and in single mode `GB_PROPOSE_ORCHESTRATE=1` requires `self.fanout` to exist: `install_walker_fanout(curr)` sets it only when `curr.fanout` is not None — in single mode `_make_fanout` returns None. So for the parity scenario the orchestrator must run through a SINGLE-rank `WalkerFanout`: make `GlobalFit._make_fanout` return a `WalkerFanout(None, layout, rank)` (direct-call mode, no comm) when `GB_PROPOSE_ORCHESTRATE=1` even for a single layout — a 3-line, env-gated change in run.py `_make_fanout`, documented as test-only; verify `install_walker_fanout` then binds it and `fanout_active` stays False so addremove/PSD still take their direct path). (2) The neutral/empty case and the F-stat directive are exercised only if the debug recipe includes RJ stages — check `erebor.gb_no_fg`'s recipe; if the debug preset only has in-model moves, add a second gated scenario with the RJ stage enabled when cheap. (3) Memory: run the two-rank scenario FIRST on its own with `/usr/bin/time -l` and watch `maximum resident set size`; if it exceeds 5 GB, reduce `GB_N_SUBBANDS`/grid knobs via env in `setUp` or mark the scenario `skip` with the measured number and move it to the cluster gate (WP7) — record the decision in the report.

- [ ] **Step 2: Run the smoke** (one process, generous timeout; NOT in the background)

Run: `RUN_GF_GB_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p4t6.log python -m unittest tests.test_multirank_gb_smoke -v; tail -n 80 .wtenv/p4t6.log`
Expected: `Ran 2 tests ... OK`. A parity failure names the first differing array: investigate the RNG order (the head draws the scan schedule before the sorter build; `rj_prop.rvs` uses its own generator — check whether it consumes `np.random`, in which case the legacy order draws `tmp_start` AFTER the sorter and the orchestrator must draw it at the same point in the global stream, i.e. after fan-out 1 — it already does) before touching any body.

- [ ] **Step 3: Whole-plan verification**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p4_all.log python -m unittest tests.test_recipe_gb_local_sizing tests.test_gb_rank_plumbing tests.test_gb_rank_session tests.test_gb_orchestrator_merge tests.test_recipe_fanout_install tests.test_recipe_local_block tests.test_run_multirank_helpers tests.test_move_build_context_layout tests.test_walkerslice_roundtrip tests.test_walkerfanout_mixin tests.test_fanout_fakecomm tests.test_band_unit_scan_order tests.test_temper_batch_perms tests.test_temper_census_hoist tests.test_temper_compact_rows tests.test_temper_shutoff_bands tests.test_temper_skip_empty tests.test_tempering_swap_cap tests.test_vertical_swap tests.test_cell_label_deferred tests.test_band_view_multi_shard tests.test_gb_shard_router tests.test_router_stream_edges tests.test_band_ntemps_reconcile tests.test_band_shutoff_designation tests.test_band_shutoff_revival tests.test_gb_cap_cell_grid tests.test_cap_stagger tests.test_gb_cold_reseed_gate tests.test_module_substate_reseed tests.test_gf_substate_roundtrip tests.test_stock_globalfit -v; tail -n 25 .wtenv/p4_all.log`
Expected: all OK. Then the gated smokes from Plans 2-3: `RUN_GF_SMOKE=1 ... tests.test_multirank_blank_smoke tests.test_multirank_noise_smoke tests.test_globalfit_sample` → OK.

- [ ] **Step 4: Line-length gate + report**

`git diff 9e19a75a..HEAD -- src tests | grep '^+' | grep -v '^+++' | awk 'length > 101'` → empty. `git log --oneline 9e19a75a..HEAD`. The cluster gates (WP7) now unblocked: `RANKS_PER_GPU=2` shared-GPU parity on the full 3mo recipe, the 1-node vs 2-node transport parity, the statistical gate vs the in-process 2-GPU run.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multirank_gb_smoke.py src/lisatools/globalfit/run.py
git commit -m "test(gb): gated multi-rank GB smoke with seeded orchestrator-vs-legacy parity

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage WP5: three commands with a per-propose session (T3); head orchestrator with the head-only decisions (early returns, reseed, caps arming, `setup()`, tempering gate, `_temper_cadence_fire`, shipped `scan_schedule`/`tmp_start`, `_adapt_band_temps` once, `_update_band_shutoff` MAX over N via the merged state, `self.time += 1` order, `_update_band_leaf_caps(precomputed)`, `accepted` zeros (engine_ntemps, N)) (T4); signature extensions (T2); `_write_back_state` return (T2); rank-local tables (`_install`/`_install_ctr_table(model=None)` via `_setup_from_directive`, `_fstat_reference_walker` local, `num_band_preload_total` per rank via `gb.gpus`, `fstat_nm_lanes` self-disables, `_tempering_walker_groups` None) (T3); VGB inherits + TODO (T5); legacy setups raise (T5); empty-block neutral reply decided by the head (T3/T4); RNG dispositions (T2/T3/T4: head draws shipped, rank streams seeded via `rank_seed` and `_rank_rng_seed`); tests (i)/(ii) by `test_gb_orchestrator_merge` with stubbed commands, (iii) seeded parity by the gated smoke (T6). Deviation from the spec, ruled by the controller: single mode runs `_propose_legacy` (bit-identical by construction) instead of the orchestrator in direct-call mode; the orchestrator's single-rank parity is proven by the gated test and `_propose_legacy` is deleted only after the WP7 cluster gates.
- Anchors: taken from the 09-16 site map (spec `gbbands.py` anchors corrected; `BandSorter` 5383, `get_band_info` 6419, `_get_fill_buffer_ind_map` 5176-5280).
- Type consistency: reply keys are spelled identically in T3 (producer) and T4 (consumer): `log_like_cold`, `prop_counts`, `acc_counts`, `cold_counts`, `band_swaps_accepted`, `band_swaps_proposed`, `block_coords`, `block_inds` (**changed in fix round 4 (2026-09-16)** from `alive_coords`/`alive_twl`), `d_h`, `h_h`, `band_counts`, `log_like_final`, `cap_stats`, `fstat_ctr_fallback_rows`, `band_dof`, `timing`; payload keys `state`, `ntemps`, `band_temps`, `keep_all_inds`, `scan_schedule`, `clock`, `tables`, `directive`, `rank_seed`, `neutral`, `session`, `tmp_start`, `want_cap_stats`.
- Known accepted semantics changes (add to the mixin docstring's list or a GB docstring): friend windows / infomat cold tables / F-stat reference walker are rank-local; `[GB_TEMPER_CHECK]` etc. per rank; no cross-rank cell tempering.
