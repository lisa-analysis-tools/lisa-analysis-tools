# Multi-rank global fit: one MPI rank per GPU, walker-block sharding

Repo: `LISAanalysistools`, base `dev` at `d65e02d1`. All work happens in a **separate worktree on its own branch** (see Workspace below), never in the main `dev` checkout, so other dev updates continue independently. No commits or pushes without the user's say-so.

## Workspace (first step, before any code)

- Worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank` on new branch `multirank-walker-blocks`, following the sibling pattern already in use (`-cold-chain`, `-noise-merge`, `-psd-batch`, `-temper`):
  ```sh
  git -C /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools worktree add ../LISAanalysistools-multirank -b multirank-walker-blocks dev
  ```
  The main checkout carries uncommitted edits (`scripts/diagnostics/*`, `stock/erebor/vgb.py`) that belong to other work; the new worktree starts from committed `dev` HEAD and must never sweep those in.
- Tests must run against the worktree, not the editable-installed main tree: the scikit-build-core `_lisaanalysistools_editable` meta-path finder hard-maps `lisatools.*` to the main tree and beats `PYTHONPATH`. Copy the working recipe from `LISAanalysistools-noise-merge/.wtenv/` (`sitecustomize.py` strips that finder when `LAT_WORKTREE_SRC` is set; `wt_run.sh` is the load-gated, nice-10, single-thread runner) into `LISAanalysistools-multirank/.wtenv/`, pointing its `PYTHONPATH` line at the new `.wtenv`. Pure-Python changes need no rebuild (`lisatools_backend_cpu` stays importable from site-packages). `.wtenv/` is untracked scratch and is never committed.
  ```sh
  .wtenv/wt_run.sh /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank/src .wtenv/<name>.log python -m unittest tests.test_<name>
  ```
- Laptop rules apply inside the worktree: one guarded python process at a time, CPU only, tiny fixtures. Never run `tests/test_gbspecial_flow.py` itself locally (10 to 26 GB); importing it for its fixture is fine.
- Eryn is not expected to change (`TemperatureControl` is used as is); if that changes, it gets its own worktree the same way.

## Context

The cluster's 4-GPU allocations on `gpu-80-spot` are two nodes of 2 GPUs. Today's engine is single-process multi-GPU: one MPI rank drives every local device by switching cupy contexts, the other ranks are a saver and stopped spares (`run.py:8`, `run.py:386-411`, `run.py:2256-2262`). It cannot span nodes, and a 09-15 audit found that its in-process cross-device paths rest on ambiguous `cp.asarray` semantics, an unfixed `BandView._scatter` bug, and an unguarded SOBBH comp. The handoff doc recommended band-range sharding; the user ruled instead for **walker-block sharding with one rank per GPU**, the head rank also computing, and the fan-out living inside the proposals. That makes each rank's ACA single-device, so the whole in-process router collapses to its exonerated single-shard passthrough and the cross-device sites leave the critical path.

Outcome for the current pipeline (one rank per GPU, one GPU per rank): a 2-node × 2-GPU job runs one head, three computation ranks and one saver; a 1-node × 2-GPU job is head + one computation rank + saver; a 1-GPU job is bit-identical to today with zero MPI traffic. The rank↔GPU mapping is general in both directions and adjustable by users later: several computation ranks may share one device, and one rank may own several devices (in which case today's in-process multi-device sharding runs inside that rank).

## Decisions (fixed)

1. Roles: **HEAD** (rank 0: sequences the recipe via eryn `run_mcmc`, owns the full host `GFState`, owns a device and walker block 0), **COMPUTE** (one device, one walker block, runs a command loop), **SAVER** (highest rank, unchanged). No spares.
2. **A rank owns a device list; a device may host several ranks.** Two settings on `general` (adjustable by users later, both 1 for the current pipeline): `gpus_per_rank` (a rank owns that many devices and shards its walker block across them with today's in-process `gpu_splits` machinery) and `ranks_per_gpu` (that many computation ranks share each device, each with its own context and memory pool). At most one of the two exceeds 1. `n_compute = total GPUs × ranks_per_gpu / gpus_per_rank`. `GB_N_SUBBANDS` keeps its meaning "per run device" (`num_band_preload_total = GB_N_SUBBANDS × len(gb.gpus)`, `gbspecialstretch.py:4901-4918`), so sharing ranks must be sized so their combined residency fits the device.
3. Equal static walker blocks: `nwalkers % n_compute == 0` is a hard error. Rank r owns global walkers `[r*B, (r+1)*B)`.
4. Every rank runs `fit.build()` and the full recipe setup (identical move objects). Only the ACA residual is per-rank: B rows over the rank's device list (one device now), row i = global walker `w0+i`.
5. Fan-out lives inside proposals. GB: head runs `propose` as orchestrator on the full state and fans out at `run_proposal` and `run_tempering`; every rank (head included) runs the unchanged body on "only my walkers". addremove family (MBH/EMRI/SOBBH) and the noise moves (PSD, galfor, sgwb, all `PSDMove` instances): the entire `propose` runs per rank on its slice; the head merges.
6. Tempering and fancy swaps stay within a rank in the base port (WP0 to WP7). **WP8 then extends the fancy swaps across all walkers for the non-GB, non-VGB moves only**: the addremove family (MBH, EMRI, SOBBH) and the noise moves (PSD, galfor, sgwb). GB and VGB cell tempering stays within a rank as a `# TODO(multi-rank)` for later (assessed as very hard; see the "Cross-rank swaps" section).
7. Stretch proposals are out of scope. Where a stretch path exists (PSD RedBlue, VGB `stretch` kind, `MBH_INNER_MOVE_KIND=stretch`) the complement is the rank's local walkers; comment out if it blocks.
8. Ladders and counters have no walker axis. Ranks never adapt a ladder. The head SUM-reduces swap counters and adapts once (`band_temps` via `_adapt_band_temps`, eryn ladders via `TemperatureControl._get_ladder_adjustment`).
9. `n_compute == 1` is a direct call: no MPI, no pickling, no copies. Existing 1-GPU runs are bit-identical.
10. Two notions of nwalkers: engine, state, backend and `GFCombineMove.accepted` stay GLOBAL; the ACA, move bodies and every `TemperatureControl` are LOCAL (`B == len(acs)`).
11. Setup phase uses symmetric collectives (bcast the initial state, allgather likelihoods). Sampling phase uses head-directed commands. Compute ranks never open the HDF store.
12. Keep the in-process multi-GPU code (router, `BandView`, `gpu_splits > 1`) in place: it is the intra-rank mechanism when `gpus_per_rank > 1`. The current pipeline hands each rank one device, so that code runs only in its exonerated single-shard form. `gpus_per_rank > 1` is layout-supported but not validated by this work; the 09-15 audit items (`cp.asarray` cross-device semantics, `BandView._scatter`, the SOBBH comp guard) are prerequisites before anyone runs it. No deletions. A rollback switch `GF_LEGACY_RANK_LAYOUT=1` restores today's roles for the running campaign's resubmits.
13. Gate: FakeComm unit tests on the laptop (CPU, one python process, tiny synthetic data), transport parity on the cluster (2 ranks on 1 node vs 2 ranks on 2 nodes, same seeds, bit-identical), a shared-device parity run (2 ranks on 1 GPU via `ranks_per_gpu=2`, bit-identical to the 2-node run), and a statistical check against today's in-process 2-GPU run.

## Architecture

### Phases per rank

```
all ranks : ranks.prepare_rank(fit, comm)   # layout + device pin BEFORE fit.build()
all ranks : fit.build()
HEAD      : load_info -> full GFState -> fanout_comm.bcast(state)
COMPUTE   : state = fanout_comm.bcast(None)
all ranks : setup_acs(state, walker_block=(w0, w1))  -> B-row ACA on this rank's device
all ranks : state.log_like[:] = allgather(acs.likelihood())   # 3 sites: run.py 1813 / 2044 / 2160
all ranks : recipe setup (moves built at B; GB/VGB setup-time subtraction block-gated)
HEAD      : run_mcmc  (moves call fanout.run(...) inside propose)  -> fanout.stop() -> finish_run -> saver
COMPUTE   : ComputeService.serve()  (blocking recv is the clock; {"op": "stop"} exits)
SAVER     : unchanged
```

Two communicators: `COMM_WORLD` keeps head↔saver traffic exactly as today (`hdfbackend.py:838`, `run.py:2236`); a split `fanout_comm` (head + compute) carries commands so the saver's `iprobe` never sees them.

### The one fan-out API (all three slices code against this)

All new communication code lives in a new subpackage **`src/lisatools/globalfit/communication/`** (`__init__.py` re-exports the public names): `ranks.py`, `fanout.py`, `fakecomm.py`, and `walkerslice.py` (the state↔payload slice/merge layer). Short names like `ranks.prepare_rank` below mean `lisatools.globalfit.communication.ranks.prepare_rank`. The move-side mixin stays with the moves (`moves/walkerfanout.py`).

```python
# src/lisatools/globalfit/communication/ranks.py
class RankRole(Enum): HEAD, COMPUTE, SAVER
@dataclass(frozen=True) class RankPlacement: rank, role, node, local_index, devices: tuple[int, ...], device_slot, w0, w1
    # devices = the rank's device list (length gpus_per_rank; 1 now); a device may appear in several ranks'
    # lists when ranks_per_gpu > 1; device_slot = this rank's index among the ranks sharing its device
@dataclass(frozen=True) class WalkerBlockLayout:
    size, head_rank, saver_rank, compute_ranks, nwalkers, block (B), placements, device_mode,
    gpus_per_rank, ranks_per_gpu
    n_compute; is_single(); block_of(rank) -> (w0, w1); local_gpus(rank) -> list[int] | None
    ranks_on_device(node, g) -> tuple[int, ...]; make_fanout_comm(comm)
def resolve_roles(size, main_rank=0) -> (head, saver, compute_ranks)   # saver aliased to head when size < 3
def build_layout(comm, nwalkers, gpu_pool, gpus_per_rank=1, ranks_per_gpu=1) -> WalkerBlockLayout
    # node via comm.Split_type(COMM_TYPE_SHARED); on each node the compute ranks are ordered by world rank
    # and assigned blocked: compute index i on that node ->
    #   gpus_per_rank = k > 1 : devices = gpu_pool[i*k:(i+1)*k]
    #   ranks_per_gpu = m > 1 : devices = (gpu_pool[i // m],), device_slot = i % m
    # hard errors: both settings > 1; compute ranks on a node > len(gpu_pool) * ranks_per_gpu / gpus_per_rank
def select_rank_device(layout, rank) -> ([g] | None, mode)             # CUDA_VISIBLE_DEVICES, fallback cudaSetDevice
def prepare_rank(fit, comm) -> WalkerBlockLayout                       # THE driver hook, idempotent, pre-build
def derive_rank_seed(base_seed, layout, rank) -> int                   # SeedSequence(base).spawn(n_compute)[i]

# src/lisatools/globalfit/communication/fanout.py
class WalkerFanout:
    def run(self, op, *, move, per_rank_payload, local_body, merge, shared=None):
        # per_rank_payload(rank, w0, w1) -> dict (host numpy)     [head only]
        # local_body(payload, model) -> reply dict                 [every compute rank incl. head, its own block]
        # merge(replies: dict[rank -> reply]) -> Any               [head]
        # single: return merge({head: local_body(per_rank_payload(head, 0, N), head_model)})  -- no pickling
        # multi : isend payloads to workers, run head block locally, recv replies in rank order, raise
        #         RemoteWorkerError (with remote traceback) on any ok=False, then merge
    def allgather_walker_vector(self, local_1d) -> np.ndarray          # setup-phase collective, rank order
    def stop(self)
class ComputeService:
    def serve(self) -> int      # recv -> handle -> reply; builtins: ping, likelihood, digest, reseed, gc_pool
    # dispatch: registry[cmd["move"]].gf_serve(op, payload, clock, model)  where model = GlobalFitInfo(local_acs, map, rank_rng)
cmd   = {seq, op, move, clock: {iteration, stage, stage_kind, move, call_index, seq, seed_base}, payload, shared}
reply = {seq, rank, ok, result, wall_s, error: {type, msg, traceback} | None}

# src/lisatools/globalfit/communication/fakecomm.py
class FakeWorld(size, nodes=None): comm(rank) -> FakeComm; run(fn) -> dict   # thread per rank, pickle-copy sends
```

`GlobalFitMove.gf_serve(op, payload, clock, model)` is the single rank-side override point (default raises). `MoveBuildContext` gains `layout`, `fanout`, `rank`, `state_local` (auto-filled from `ctx.curr`).

## Work packages (dependency order)

### WP0. Walker slicing of the state (no deps)

Files: `src/lisatools/globalfit/state.py`, new `src/lisatools/globalfit/communication/walkerslice.py`.

- `ModuleSubState.walker_axes = {"coords": 1, "inds": 1, "log_like": 1, "log_prior": 1, "d_h": 0, "h_h": 0}`; `PerLeafLadderState` overrides `log_like/log_prior` to axis 2 (`state.py:1342`). Add `_bare_like()`, `slice_walkers(w0, w1)` (fresh instance via `initialize_tempered` at B, `state.py:483-542`; ladder copied by value; counters zeroed) and `merge_walkers(part, w0, w1)` (walker columns written back; counters SUM-added; ladders untouched). `GBState` inherits the tempered-block version; its `band_info` is never sliced by this helper (GB ships `band_temps` and the read-only tables explicitly in WP5).
- `walkerslice.slice_state(full, w0, w1, sub_states=...) -> GFState` and `merge_state(full, part, w0, w1)`: main `coords/inds/log_like/log_prior/blobs` `[:, w0:w1]`, `branch_supplemental` via `BranchSupplemental.__getitem__` (Eryn `state.py:179`), `supplemental["walker_inds"]` remapped to `tile(arange(B))` (consumers `psdmove.py:679/1881/2535` index the local ACA). Never calls `initialize_band_information` (walker guard at `state.py:1000-1010` stays head-only).
- Test: `tests/test_walkerslice_roundtrip.py` (base, PerLeafLadder and GBState sub-states; slice two halves, merge into a zeroed copy, `array_equal` everywhere; ladders unchanged; counters summed).

### WP1. Control plane (no deps)

Files: new `communication/__init__.py`, `communication/ranks.py`, `communication/fanout.py`, `communication/fakecomm.py`; `moves/globalfitmove.py` (`MoveBuildContext` fields at 69-89; `GlobalFitMove.gf_serve` default at ~230).

- `ranks.py` as specified above. Device selection: `CUDA_VISIBLE_DEVICES=<comma list of the rank's devices>` before any CUDA call, verified via ctypes `cudaGetDeviceCount() == len(devices)`; the rank then sees its devices as `0..k-1` and `general_info.gpus = list(range(k))` (k = 1 now), which is exactly what `pin_main_device` and the ACA expect. If a CUDA-aware MPI already initialised the driver, fall back to `cudaSetDevice(devices[0])` with `general_info.gpus = devices` and export `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `SLURM_LOCALID` is a cross-check only. Device sharing (`ranks_per_gpu > 1`) needs nothing extra: several processes may name the same device and each gets its own context and memory pool. `layout.describe()` prints one line per rank with its devices and slot so the mapping is visible at startup. `general_info.gpus` becomes `[g]` on every rank so every `len(gpus) > 1` branch collapses (`run.py:1458/1542/1556/1662`, `recipe.py:2026/2193/2433/2538/3630/3699`, `globalfitmove.py:383-400`, `gbspecialstretch.py:4907`).
- `fanout.py`: `WalkerFanout`, `ComputeService`, `RemoteWorkerError`, schemas. Worker exceptions travel back as `ok=False` with the traceback; exceptions outside `handle` call `comm.Abort`. The head's `RemoteWorkerError` reaches the existing abort hook (port `run_combined_staged.py:988-1050` into `ranks.install_mpi_abort_on_error`). Optional `GF_FANOUT_WAIT_WARN_S` hang diagnostics.
- `fakecomm.py`: in-process N-rank simulator (`Get_rank/Get_size/Get_processor_name/send/recv/isend/iprobe/bcast/allgather/barrier/Split/Split_type/Abort`), thread per rank, `send` does `pickle.loads(pickle.dumps(obj))`. One python process, laptop-safe.
- Tests: `tests/test_rank_layout.py` (sizes 1/2/3/5; two nodes; `ranks_per_gpu=2` on one GPU gives two compute ranks with the same device and slots 0/1; `gpus_per_rank=2` on a 2-GPU node gives one compute rank with devices (0, 1); both > 1 is an error; the over-subscription hard error), `tests/test_fanout_fakecomm.py` (round trip in walker order, error propagation, stop, saver isolation), `tests/test_fanout_passthrough.py` (`is_single()` → `local_body` called with the same object, no comm access), `tests/test_fakecomm.py`.

### WP2. `run.py` integration (deps WP0, WP1)

File: `src/lisatools/globalfit/run.py`.

- `GlobalFit.__init__` (413-467): `self.layout` from `curr.rank_layout` (or `build_layout` late path on CPU); `self.role`, `self.compute_ranks`, `self.used_ranks = compute + [saver]`, `self.fanout_comm`. Keep `resolve_rank_roles` as a compat wrapper returning `(head, saver, [])`. Unify the raw `settings_dict.rank_info.main_rank` read at 2220 onto `self.main_rank`. Per-rank log files (`loginfo.setup_root_file_handler` gains `filename`); head keeps `globalfit_run.log` so parsers are untouched; non-head stdout is line-prefixed `[r<k>/c<k>]`.
- `prepare_main` (1678-2199) split: `_collect_priors_periodic()`, `_attach_walker_supplemental()`, shared `_setup_acs_and_recipe(state)`; `prepare_main()` = head path (`load_info` → `fanout_comm.bcast(state)` → shared → midit arm → HDF engine → recipe first step → `fanout.run("ping")` layout-hash handshake); `prepare_compute()` = compute path (`bcast(None)` → shared → in-memory eryn `backend=None` engine → `setup_run` on every stage to stamp periodic/temperature_control → `ComputeService`).
- `setup_acs(state, rebuild_residuals, walker_block=None)` (1394-1672): loop `for i, w in enumerate(range(w0, w1))`, `_build_walker_ac(w)` unchanged (global `w` for the `walker_{w}` registry name and the psd/galfor/sgwb rows), an ACA of B rows over `general_info.gpus` (the rank's device list; with one device it is single-shard, with more the existing `np.array_split` walker→device map at 1456-1463 applies within the rank); `rebuild_residuals` and bulk `get_templates` paths read a `slice_state(state, w0, w1, sub_states=[])`. `walker_block=None` is byte-identical to today.
- `_global_likelihood(acs)` replaces the three `acs.likelihood()` sites (1813, 2044, 2160) with `fanout.allgather_walker_vector` when `n_compute > 1`.
- `run_global_fit` (2201-2262): HDF backend construction moves into the HEAD and SAVER branches; COMPUTE branch = `prepare_compute()` → `serve()` → release own device pools. `_release_helper_gpu_pool` → `_release_rank_gpu_pool` (iterates the rank's own device list, no `getDeviceCount` loop). Head: after `run_mcmc`, `fanout.stop()` then `finish_run`. Make the legacy-group HDF probe in `GFHDFBackend.__init__` (`hdfbackend.py:531-548`) lazy so compute ranks never open the store.
- Seeds: after recipe materialization and only when `n_compute > 1`: `seed = derive_rank_seed(general_info.random_seed, layout, rank)`; `np.random.seed(seed)`, `cp.random.seed(seed)`, per-rank `RandomState` as `model.random` for bodies. The head's eryn `_random` (decision RNG, `ensemble.py:718`) is never handed to a body.
- Test: `tests/test_setup_acs_walker_block.py` (gated `RUN_GF_SMOKE`, `erebor.blank`-style synthetic, `nwalkers=4`, block (2,4) → `len(acs)==2`, likelihood equals the ungated run's `[2:4]`).

### WP3. `recipe.py` (deps WP0, WP2)

File: `src/lisatools/globalfit/recipe.py`.

- Move construction at B: `SingleSourcePEBuilder.build` (3990, 4035, 4101) and `build_noise_moves` (2119, 2167-2174, 2211-2212) use `nwalkers_local = len(acs)` for `coords_shape`, `TemperatureControl(nwalkers=...)` and the move's own `accepted`; stamp `move.fanout_branches`. `Stage.setup`'s `combined.accepted` (1051) stays global.
- **Block-gate the GB/VGB setup-time residual subtraction** (gap the GB plan missed): 1917-1930 (neighbour subtraction), 2508-2519 (GB init `data_index_1 = walker_vals`), 2572-2579 and 3729-3736 (WDM init gathers), 3688-3711 (VGB twin). Read `ctx.state_local` (a `slice_state(state, w0, w1)`) and `arange(B)`; `num_per_gpu_walker` (2538/3699) is already `None` on one device.
- `MoveBuildContext.state_local` populated at `run.py:2177-2187`.

### WP4. addremove + PSD/galfor whole-proposal fan-out (deps WP0, WP1, WP3)

Covers `ResidualAddOneRemoveOneMove` (MBH, EMRI, SOBBH via `SOBBHChunkedLikeMove`) and every `PSDMove` instance the recipe builds (`psd_pe`, `galfor_pe`, `sgwb_pe` from `build_noise_moves`, `recipe.py:2119-2212`).

Files: new `moves/walkerfanout.py`; `moves/addremovemove.py`; `moves/psdmove.py`.

- `WalkerFanoutMixin` first in the MRO of `ResidualAddOneRemoveOneMove` (87) and `PSDMove` (78); rename their `propose` to `propose_local` (bodies untouched). Mixin `propose`: `if fanout is None or n_compute == 1: return self.propose_local(model, state)` (bit-identical); else `fanout_propose`. This covers every invocation path (`_run_sequence` 824, bypasses 762/791/799, `MaxLogLCombineMove`).
- Payload: `slice_state(state, w0, w1, sub_states=fanout_branches)` + clock (`_fancy_swap_clock`, `_dbg_step`, per-leaf `tc.time`, eigen visit counts; PSD: `tc.betas`, `tc.time`). Reply: the B-walker `new_state`, `accepted (engine_ntemps, B)`, extras (addremove: per-leaf `swap_tally = (Σ tc.swaps_accepted, Σ tc.swaps_proposed)`; PSD: the existing `_tally_*` arrays).
- Rank side `gf_serve("propose", ...)`: stamp `gf_stage_kind`, apply clock, `propose_local(GlobalFitInfo(local_acs, map, rank_rng), part)`.
- Head merge: `GFState(state, copy=True)`; `merge_state` per rank; `accepted = concatenate(axis=1)`; `fanout_adapt_ladders` (sum swap counters, `_get_ladder_adjustment` once per visited leaf, write `betas_all`/`betas`, `tc.time += 1`); eigen persistence; `_sync_cold_row`; one `midit_checkpoint.maybe_write`.
- Body edits: guard the per-leaf `midit_checkpoint.maybe_write` (addremovemove 2060-2066) with `if not self._fanout_body`; add the per-leaf swap tally after `temperature_swaps` (1944). `install_walker_fanout` sets `tc.adaptive = False` on every rank (head included) when `n_compute > 1` so `adapt_temps` (`tempering.py:867-897`) is a no-op in bodies; sets `eigen_store_path = None` on compute ranks (single-writer sidecar `eigen_table_persist.py:252-281`; guard `nwalkers` written at B). `walker_max` eigen tables become rank-local expansion points (valid MH).
- PSD: stretch complement is the local block; `live_dangerously` already true (recipe 2180).
- **Pre-port check (PSD fancy swap semantics).** `TemperatureControl.skip_swap_supp_names` defaults to `[]` (`Eryn tempering.py:266`) and nothing in LAT sets it, so `_apply_swaps_vectorized` (`tempering.py:607-614`) swaps `supps` wholesale, including `walker_inds`, while `perform_fancy_swap_acceptance_fraction` scores with `supps` unpermuted (`tempering.py:780`) and PSD's `compute_log_like` reads `walker_inds` as the ACA index (`psdmove.py:1881`). Those three cannot all be consistent. Settle it with a small CPU check before WP4 (expected fix: add `walker_inds` to `skip_swap_supp_names` when PSD builds its `TemperatureControl`). addremove passes `supps=None`, so only PSD/galfor is affected. Under the port a swapped `walker_inds` would point at an ACA row that does not exist on the rank, so this is load-bearing.
- Tests: `tests/test_addremove_fanout.py`, `tests/test_fanout_ladder_adapt.py` (pooled ratio equals a 1-rank ladder step), `tests/test_fanout_single_rank_identity.py`, `tests/test_eigen_persist_fanout.py`, `tests/test_rank_seeds.py`; existing `test_addremove_multi_shard.py`, `test_psd_move_multi_shard.py`, `test_eigen_*` stay green.

### WP5. GB move fan-out (deps WP1, WP2, WP3)

Files: `moves/gbspecialstretch.py` (`propose` 16849-17610, `run_proposal` 4124, `run_tempering` 14358, `_write_back_state` 15098, `_update_band_leaf_caps` 16509, `setup`/`_install_ctr_table` 18390-18600, VGB 17677); `moves/gbbands.py` (`BandSorter` 5346, `get_band_info` 6362).

Design change vs the "stateless commands" wording: a rank keeps a **per-propose session** (its `BandSorter`, the non-GB residual snapshot, buffer caches, capture arrays) across the three commands of one propose. The head runs the three commands strictly sequentially with nothing in between that touches the rank's residual, so this is safe, and it removes the frozen-label sorter override and the `d_h/h_h` capture round trip that statelessness would force (and their parity corner case).

- Small signature extensions, bodies untouched: `run_proposal(..., scan_schedule=None)` (head draws `_draw_unit_scan_schedule` with N and ships each block's slice; the draw site moves earlier with no other `model.random` consumer in between), `run_tempering(..., tmp_start=None, adapt_band_temps=True)` (head draws `tmp_start`; ranks skip `_adapt_band_temps` at 15091), `_write_back_state` returns `(inds_new, alive)`, `_update_band_leaf_caps(..., precomputed=None)` with the residual-window half factored into `_cap_stats_for_rank`. `_temper_rng` (13342) seeded from `self._rank_rng_seed` when set.
- Rank-local slice state: `_make_slice_state(payload)` builds a `GFState` with a `GBState` sub-state at B via `initialize_tempered` (bodies never read `state` except through `_work_branch` and `sub.d_h/h_h`).
- Commands (`gf_serve` dispatch), each entered via `_enter_rank_block` (pin device, `self.ntemps/self.nwalkers = ntemps/B`, clock sync of `time`/`num_proposals`/`_branch_propose_counts`, read-only tables `_cap_leaf_cap`/`_band_leaf_cap`/`_rj_band_shutoff`, `_reseed_firing`, `temper_vertical` override, per-rank `_ProposeTimer`) and left via `_exit_rank_block`; the head restores `self.nwalkers = N` after every fan-out:
  1. `gb_run_proposal`: `_setup_from_directive` (F-stat epoch tables loaded from the epoch dir's npz, never `setup()` on a rank), slice state, ONE RJ `BandSorter` build (the only `rj_prop.rvs` draw), snapshot round trip (17186-17193), start check (17198-17211), `run_proposal` passes (17233-17250), ll add, drift check/rebuild (17253-17312). Reply: `log_like_cold (B,)`, `ll_after`, `start_diffs`, `prop/acc_counts` walker-summed `(2, ntemps, num_bands)`, per-walker cold counts, alive-per-temp, drift, timing. Session kept.
  2. `gb_run_tempering`: `run_tempering(..., tmp_start, adapt_band_temps=False)` on the session sorter, ll add, drift check/rebuild (17373-17397). Reply: `log_like_cold`, `band_swaps_accepted/proposed (num_bands, ntemps-1)`, `ll_change_sum_temp`, census, timing.
  3. `gb_finish`: `_write_back_state` on the session sorter → export the alive slice (compact `alive_coords (n, ndim)`, `alive_twl (n, 3)` with local walker ids) + `d_h/h_h (B, nleaves_max)`; second alive-only sorter → `get_band_info` → `band_counts (ntemps, B, num_bands)`; `check_ll_inject` → `log_like_final (B,)`; `_cap_stats_for_rank` if caps are on; teardown (`_buffer_cache_teardown`, `_fstat_nm_lanes = None`, session cleared).
- Head orchestrator (rewrite of 17109-17610, prologue 16849-17108 unchanged incl. early returns, reseed with `np.random.permutation(N)` on the full sub-state, cap arming, `setup()`): `new_state = GFState(state, copy=True)`; periodic wrap on the full branch; draw scan schedule; fan-out 1 → merge `log_like[0]`, counters; tempering gate 17316-17363 unchanged (head decides, ships the decision); if it fires: draw `tmp_start`, fan-out 2 → merge `log_like[0]`, SUM swap counters → `_adapt_band_temps` once; `_update_band_shutoff(_band_occupancy_cold_max)` (MAX over N) ; `self.time += 1` (17490 order preserved); fan-out 3 → write alive slices into `work` (global walker = `w0 + local`), `sub.d_h/h_h`, `band_num_binaries` concat, `band_temps` write-back, `accumulate_proposals/accumulate_swaps` from summed counters, `new_state.log_like[:] = concat(log_like_final)`, `_update_band_leaf_caps(precomputed=concat)`; `accepted = zeros((engine_ntemps, N))`; propose-end logs from merged replies.
- Keep today's 17109-17610 verbatim as `_propose_legacy` behind `GB_PROPOSE_LEGACY=1` for the parity test only; delete once green.
- Per-rank tables: infomat table rank-local (valid MH; `GB_INFOMAT_PER_BLOCK=1` retires the borrow); F-stat epoch tables loaded per rank via `_install`/`_install_ctr_table(model=None)` (registries make it once per process; head runs `setup()` first so `DONE.json` exists); `_fstat_reference_walker` rank-local (proposal centre only, same centre on forward 8721 and reverse 8760+); `gb.gpus = [g]` so `num_band_preload_total` (4901-4918) = `GB_N_SUBBANDS` per rank; `fstat_nm_lanes` (5705-5745) self-disables; `_tempering_walker_groups` (14366) is `None` → one permutation over B, documented correct at 14205-14211.
- VGB inherits the three commands (no RJ, `preserve_leaf_identity`, no caps, `infomat_per_block`). Under `VGB_INMODEL_PROPOSAL=stretch` the red/blue pairing is within the local block: assert `B >= 2 and B % 2 == 0`, `# TODO(multi-rank)` comment.
- Legacy `GBSpecialRJSerialSearchMCMC.setup` / `GBSpecialRJRefitMove.setup` (scalar-walker `ParaEnsembleSampler`, FD only): raise `NotImplementedError` when `n_compute > 1`.
- Per-block empty guard: a block with zero alive sources and `keep_all_inds=False` returns a neutral reply (unchanged slice, zero counters). The head owns the "run at all" decision, so ranks can never desync on `_temper_cadence_fire`.
- RNG: head streams untouched; `model.random` at 4242 → head draw shipped; `np.random` at 14445 → head draw shipped; 16985 head-only; all `cp.random`/`np.random`/`_temper_rng` draws inside bodies → per-rank seeded streams.
- Tests: `tests/test_gb_rank_commands.py` on the `tests/test_gbspecial_flow.py` fixture with a `FakeWalkerFanout(n_compute)` holding one move instance and one B-row ACA per simulated rank: (i) `band_num_binaries`/counters/`log_like` equal concat/sum of the per-rank replies; (ii) `band_temps` after a propose equals `_adapt_band_temps` applied once to the summed swap counters and differs from per-rank sequential adaptation; (iii) seeded parity: `_propose_legacy` vs new orchestrator with `n_compute=1`, bit-identical `coords/inds/log_like/band_info/d_h/h_h` for RJ prior, in-model, and VGB moves. Existing `test_band_unit_scan_order.py`, `test_temper_*`, `test_gbspecial_flow.py`, `test_band_view_multi_shard.py` stay green.

### WP6. Drivers, submit scripts, docs (deps WP2)

- `EreborGeneralSettings.gpus_per_rank: int = 1` and `ranks_per_gpu: int = 1` (`stock/erebor/fit.py`; env names follow the repo's capitalized-attribute rule: `GPUS_PER_RANK`, `RANKS_PER_GPU`); `prepare_rank` passes both to `build_layout`.
- `scripts/fstat_proposal/run_combined_staged.py:949-952`: `ranks.prepare_rank(fit, COMM_WORLD)` before `fit.build()`; rank guards use `layout.head_rank`; drop the local abort hook in favour of `ranks.install_mpi_abort_on_error`. `scripts/run_global.py:131-141`: invert the early exit (every rank builds). `stock/base.py:837-851` `StockGlobalFit.run(comm)`: call `prepare_rank` if not already done; add `rank_layout`/`fanout` to `_BUILT_ONLY_ATTRS`.
- `submit_gf_6mo_v8.sh` / `_nogb_null.sh`: `GPUS_PER_RANK=${GPUS_PER_RANK:-1}`, `RANKS_PER_GPU=${RANKS_PER_GPU:-1}`; `N_COMPUTE = NGPUS * RANKS_PER_GPU / GPUS_PER_RANK`; `--ntasks = N_COMPUTE + 1` (the saver). NGPUS=2 at the defaults: header and `mpiexec -n 3` unchanged (rank 1 is now the compute rank on GPU 1; `GPUS` from `SLURM_GPUS_ON_NODE` is now the per-node pool; `NWALKERS=10 % 2 == 0`). NGPUS=4 → same partition, `--partition=gpu-80-spot --gres=gpu:2 --nodes=2 --ntasks=$((N_COMPUTE+1)) --distribution=cyclic` with `srun`, so the saver lands with the head and each node carries `N_COMPUTE/2` compute ranks, replacing the pend warning (the NGPUS→partition self-dispatch at `submit_gf_6mo_v8.sh:396-416` collapses to one partition with `--nodes` set from NGPUS). Add a `GF_LAYOUT_DRY_RUN=1` preflight (prints `layout.describe()` from every rank, exits non-zero on layout errors, no build). NWALKERS must be a multiple of `n_compute` (10 is not divisible by 4: the script defaults to 12 under NGPUS=4 with a loud `[SUBMIT]` line; user decision at first 4-GPU launch).
- Diagnostics parsers (`scripts/diagnostics/gf_run_log_digest.py` `TIM_RE`, `gf_monitor_gen.py`): negative filter on `^\[r\d+/` so worker-prefixed lines are ignored; `[GF_TIMING]` stays head-only via `_run_sequence`.
- Docs: `docs/global-fit-launch.md` (np table, `-n 2` semantics change), `run.py` module docstring, `docs/multigpu-cluster-validation.md` addendum, `multinode_gpu_handoff.md` closing note pointing at this design.

### WP7. Cluster validation (deps WP0 to WP6)

The cluster items of the Verification section: layout dry runs, the three-layout transport parity, the statistical gate against today's in-process 2-GPU run.

### WP8. Cross-rank fancy swaps for the non-GB, non-VGB moves (deps WP7 green)

Scope: `ResidualAddOneRemoveOneMove` (MBH, EMRI, SOBBH via `SOBBHChunkedLikeMove`) and every `PSDMove` (psd, galfor, sgwb). GB and VGB are explicitly excluded and keep `# TODO(multi-rank)` pointing at the assessment section. Setting `general.cross_rank_swaps: bool = False` (env `CROSS_RANK_SWAPS`), default off until the parity gate below is green. Eryn stays untouched: the synced control is a LAT-side subclass.

Files: new `communication/tempering.py` (`RankSyncedTemperatureControl(TemperatureControl)`), `moves/addremovemove.py`, `moves/psdmove.py`, `moves/walkerfanout.py`, `recipe.py` (build the synced control in `SingleSourcePEBuilder.build` and `build_noise_moves` when the setting is on and `n_compute > 1`).

1. `RankSyncedTemperatureControl.temperature_swaps(...)`, same signature as Eryn's (`tempering.py:618-757`), reusing `_apply_swaps_vectorized` (`:543`) and `_get_ladder_adjustment` (`:840`). Per fancy round: allgather the full `(ntemps, N, ...)` coords plus `logl/logp/logP` on the fan-out comm once at entry (about 65 KB); per rung: draw `iperm`, `i1perm`, `raccept` from a dedicated `np.random.Generator` seeded from `(seed_base, iteration, move, leaf, rung)` out of the clock so every rank draws identical values; each rank scores only its own B destination columns (the existing `log_like_for_fancy_swaping` at `addremovemove.py:1493-1533` and PSD `compute_log_like` with local `data_index`); allgather `new_like` for the two rows (384 B); every rank computes the identical accept mask and applies the swap on its replicated arrays; after the rung loop the full-ladder re-score at `tempering.py:755` is local columns plus one allgather. `swaps_accepted` is identical on every rank by construction, so WP4's once-per-propose head-side ladder adaptation is unchanged.
2. Symmetry requirements in the bodies, all mandatory: the head ships the leaf order in the clock (replaces `np.random.permutation` at `addremovemove.py:1601`); the leaf-alive `continue` at `:1607` becomes an allreduced OR so a leaf alive anywhere is visited everywhere (ranks with it dead locally still join the collectives with empty scoring); `permute_every` and `num_repeats` asserted equal across ranks at `install_walker_fanout` (they are env-overridable per process, `:765-776`); the `_fanout_body` guard already keeps the per-leaf midit hook out; any rank-local exception inside a leaf visit routes to `comm.Abort` through the abort hook so a hang is impossible. PSD `run_move`'s `do_fancy` phase (`psdmove.py:2195`) follows the same rule. The pre-port `skip_swap_supp_names`/`walker_inds` fix from WP4 is a prerequisite: the synced control never swaps `walker_inds`.
3. Nothing else changes: the addremove fold-back (`:1979-1992`) and PSD's end-of-propose publish (`psdmove.py:2454-2489`) act on the post-swap local cold row and are rank-local; the head merge from WP4 concatenates the post-swap columns as before.
4. Tests: FakeComm 2-rank test proving swap-kernel parity, the synced control with `n_compute=2` produces exactly the swap decisions and final coords of Eryn's `TemperatureControl` over all N walkers given the same seeded permutations and acceptance draws; a deadlock test where a leaf is alive on one rank only; `GF_FANOUT_CHECK` comparing merged `log_like` with gathered `acs.likelihood()`; the three-layout transport-parity gate rerun with `CROSS_RANK_SWAPS=1`.

Effort about 10 to 20 percent of the base port; transport is about 180 KB and 1 ms per swap round.

## Cross-rank swaps: assessment (user question) and optional WP8

**Eryn fancy swaps (addremove MBH/EMRI/SOBBH, PSD/galfor): easy to moderate.** The swap moves coordinates only and re-scores them against the destination walker's residual by column index (`addremovemove.py:1520-1529`, `tempering.py:761-843`); residuals never move, and the leaf fold-back that does touch the residual (`addremovemove.py:1979-1992`) is rank-local after the swap. PSD needs no `sens_mat` republish at swap time because its scoring restores every `sens_mat` in a `finally` (`psdmove.py:1970-2010`) and the authoritative publish is once per propose from the cold row (`psdmove.py:2454-2489`). Volume per swap round at production scale is about 180 KB over roughly 20 to 50 tiny allgathers, about 1 ms against leaf visits of many minutes. The real work is symmetry, not transport: broadcast the leaf order (`addremovemove.py:1601`), allreduce the leaf-alive mask (`:1607`), give the swap its own seeded generator for `iperm/i1perm/raccept` (`tempering.py:693-702`) instead of the process-global stream, assert `permute_every`/`num_repeats` equal across ranks at startup (env-overridable per process, `addremovemove.py:765-776`), and forbid rank-local raises or throttled branches inside the leaf visit (`:1656`, `:1719`, `:1995`, midit hook `:2060+`). Changes: `temperature_swaps` and `perform_fancy_swap_acceptance_fraction` take an injected RNG and an optional allgather callback for `new_like`/`logl`; `addremovemove.propose` around 1601/1607/1937; PSD's `run_move` swap at 2203-2220. Effort about 10 to 20 percent on top of the base port, and much of the RNG and control-flow discipline is needed by the base port anyway.

**GB and VGB in-band cell tempering: very hard. Recommendation: do not.** Three structural problems compound. The swap unit is device-resident slab state (`swap_template_slots`, `gbbands.py:5231-5247`), so a cross-rank exchange is either tens of GB per iteration of slab traffic or about 40k latency-bound round trips per iteration inside the loop that was just de-synced to about 2 host syncs per rung pair (`gbspecialstretch.py:14681-14695`). Acceptance would require source rows to migrate between ranks' flat `BandSorter` tables, which have no insert or delete primitive and freeze their row maps into the buffer binding (`gbbands.py:6055-6078`). The deferred relabel window's correctness model is "sources never change cell" (`gbbands.py:5654-5657`), which a cross-rank swap violates by definition. On top of that every loop bound in `run_tempering` depends on local occupancy (`GB_TEMPER_SKIP_EMPTY`, `GB_TEMPER_COMPACT_ROWS`, `_rows_live`, chunk counts at 14503-14520 and 14620-14712), so no collective can be placed inside the rung, chunk or unit loops without undoing those optimizations. Effort comparable to or larger than the whole base port with a real regression risk. The device-local restriction is already the parallel-resources P1 ruling (`gbspecialstretch.py:14205-14211`, `docs/multigpu-cluster-validation.md:18-22, 56-59`), whose "no acceptance-rate change" expectation was written as a validation gate that appears never to have been run. Vertical same-walker swaps (`_vertical_swap_sweep`, `:12236`) are rank-local, cost nothing, and give per-walker ladder mixing; the per-band ladder stays global via the swap-counter reduce already in WP5. VGB inherits the identical path (`:17677`, no override).

**Honest caveat on value.** Correctness is not at stake either way (walkers are exchangeable). No diagnostic in the tree measures autocorrelation or round-trip rate, so whether within-rank swaps at B = 5 to 12 mix as well as all-walker swaps cannot be read off the code. The existing swap counters (`tempering.py:671-673`, `gbspecialstretch.py:14377-14378`, `[GB_TEMPER_CHECK]`) can measure acceptance rates within-rank vs all-walker on today's single-process code by forcing `_tempering_walker_groups` on and off. What demonstrably degrades at small B is not the swap but the ensemble complement: the addremove red/blue split leaves 2 to 6 complement points for an 11-dimensional stretch (`addremovemove.py:1781-1800`), and GB's friend table pools cold sources across walkers (`gbbands.py:6270-6278`). Both are out of scope by the stretch ruling; if mixing at small B ever matters, an allgather of cold-row coords for the complement is the cheap lever, not cross-rank swaps.

This assessment is the rationale for WP8's scope: cross-rank fancy swaps for the non-GB, non-VGB moves only; GB and VGB keep `# TODO(multi-rank)` with a pointer to this section.

## Verification

Laptop (CPU, one python process, FakeComm, tiny synthetic fixtures):
1. Unit suites per WP listed above. Run with `python -m unittest tests.test_<name>`.
2. Parity (iii) in WP5 and `test_fanout_single_rank_identity.py` in WP4 prove `n_compute == 1` is bit-identical to today.
3. Full existing suite: `python -m lisatools.tests`.

Cluster (gpu-80-spot throughout: `--nodes=1 --gres=gpu:2` for the 1-node runs, `--nodes=2 --gres=gpu:2` for the 2-node runs):
4. `GF_LAYOUT_DRY_RUN=1` for `-n 3` on 1 node and `-n 3` across 2 nodes (head on A, compute on B); both print identical layouts.
5. Transport parity: same env (`DATA_MODE=synthetic NWALKERS=8 NUM_ITERATIONS=N MIDIT_CHECKPOINT=0 MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1`, same seed) in three layouts that all have `n_compute=2`: 2 ranks on 1 GPU (`RANKS_PER_GPU=2`, a 1-GPU allocation, the cheapest and first to run), 2 ranks on 2 GPUs of 1 node, 2 ranks across 2 nodes. Compare per-iteration `[FANOUT_DIGEST]` residual hashes and `gf_state_digest.py` over `backend.get_last_sample()` (coords, inds, log_like, betas, every sub-state array): all three bit-identical.
6. Statistical gate vs today's in-process 2-GPU run on the 3mo recipe (`GF_LEGACY_RANK_LAYOUT=1` vs default): acceptance rates, cold-chain leaf counts, per-band tempering acceptance via the existing snapshot tooling (`processing-gf-snapshots` flow).
7. `[FANOUT]` per-iteration line (head time, max worker time, wait time) confirms load balance; `route_dispatch` in `[GB_TIMING]` should drop to ~0 per rank.

## Semantics that change only when `n_compute > 1` (accepted, must be visible in docs)

- Eryn ladders (addremove, PSD) adapt once per propose from pooled swap ratios, `tc.time += 1` per propose instead of per repeat. GB `band_temps` cadence is unchanged (already once per propose).
- Mid-iteration checkpoint granularity for addremove drops from per-leaf to per-propose.
- Infomat, eigen `walker_max` and F-stat reference tables are rank-local proposal shapes.
- Stretch complements (PSD, optional VGB/MBH stretch kinds) are the local block.
- Cross-rank tempering swaps: within-rank in the base port; WP8 extends them across all walkers for MBH/EMRI/SOBBH and PSD/galfor/sgwb behind `cross_rank_swaps`; GB/VGB cell tempering stays within-rank as `# TODO(multi-rank)` (see the assessment section).

## Risks

- CUDA-aware MPI initialising the driver before `CUDA_VISIBLE_DEVICES` is set → fallback path; verify on the first dry run.
- `gpus_per_rank > 1` re-enables the in-process cross-device paths inside a rank. It is supported by the layout but unvalidated here; the audit's `cp.asarray` question, `BandView._scatter` fix and SOBBH `_assert_comp_device` guard must land before it is used.
- Launcher placement of 5 ranks over 2 nodes (`mpiexec` vs `srun`, block vs cyclic) → the dry run is the arbiter; nothing builds until the layout validates.
- The head's ACA is B rows: any head-side "all walkers" read of the ACA must go through the fan-out. Add `assert acs.acs_total_entries == B` at the top of every orchestrator.
- Initial-state identity across ranks is a hard requirement (fresh-start coords are drawn with entropy in `load_info`); the bcast covers it.
- One-time bcast is ~450 MB at 3mo; mpi4py ≥ 3.1 for >2 GB pickles at longer Tobs.
- Campaign resubmits on spot partitions pick up whatever code is on `dev`: do this on a feature branch, and keep `GF_LEGACY_RANK_LAYOUT=1` as the rollback knob after merge.
