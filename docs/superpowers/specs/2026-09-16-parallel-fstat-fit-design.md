# Parallel F-stat epoch fit across compute ranks — design

Status: approved for implementation 2026-09-16 (user: "build the parallel f-stat fit"). Base: `dev` at 3d7f3cb9 (the multi-rank walker-block port is merged; the runbook is `docs/multirank-cluster-gates.md`). Evidence with file:line anchors: `.superpowers/sdd/fstat-parallel-fit-exploration.md` in the multirank worktree (git-ignored scratch; copy the facts you need into the plan).

## Problem

Under the walker-block layout the GB F-stat epoch grid fit runs inside the head's `setup()` (`GBSpecialBase._propose_orchestrated` prologue, gbspecialstretch.py ~18525) while every other compute rank is parked in `ComputeService.serve()`. Measured on the 6mo campaign (epoch 0, 2026-09-16): **1 h 45 min, all of it stage B** (six groups of stacked peak-box grids; group 1 = 5378 boxes × 79×3×8×8 nodes); the comb/peak stage was seconds (cached comb) and the centre table 25 s. With `GB_FSTAT_REFIT_EVERY=50` that is one serial 1.75 h section per ~50 iterations, on 1 of 2 (soon 1 of 4) GPUs.

A second, correctness-flavoured defect of the port: the fit's reference walker is `argmax(acs.likelihood())` over the HEAD'S LOCAL BLOCK (`_fstat_reference_walker`, ~4192), not over all walkers as in the single-process run (which chose walker 4 of 10 on the live run). And `walker_ref` is used as an ACA ROW index (`_fstat_call` `data_index=noise_index=walker_ref`, ~20524; `_fstat_NM` ~6701; `_gb_free_residual` ~20592) — a global index ≥ B would be out of range on a B-row rank.

## Decisions (fixed)

1. **Global reference walker.** The head gathers every walker's likelihood through the existing `WalkerFanout.gather_likelihood(acs)` (`LIKELIHOOD_OP`; same numbers as today's argmax — `inner_product` complex default False) and takes the global argmax. `DONE.json`'s `walker_ref` becomes the GLOBAL index; the log line names the global index and the owning rank.
2. **The reference row is replicated, once, by a symmetric fan-out op.** New GB op `gb_fstat_ref_row`: every compute rank (head included) enters the body; the OWNER rank (the one whose block holds the reference walker) opens the GB-free window on its LOCAL index (today's `_gb_free_residual` semantics: `remove_sources_from_residual(model, sorter, temp=0, walker=<local>, apply_inds=True)`), snapshots the walker's residual row and inverse-PSD row to host numpy (the pattern at gbbands.py ~2351-2364: slice the rows ON the owning device, host only the rows), then restores its residual (add the sources back) — the live residuals are NOT left mutated during the fit (today's serial fit mutates the head's live residual for the whole sweep; the shipped-row design removes that). Then `comm.Bcast` (uppercase, contiguous float64 buffers, root = the owner's fan-out rank) on the fan-out communicator — legal because `WalkerFanout.comm` is `ComputeService.comm`, includes the head, and every compute rank reaches the body exactly once per command (precedent: `allgather_walker_vector`). Every rank wraps the pair in a `_FStatRefRowHolder` (gbbands.py ~926, made public: constructor from a host row pair on the rank's device) scored with `data_index=noise_index=0`. Sizes at 6mo: residual row ≈ 18 MB, invC row ≈ 54 MB (confirm `Nf_active` from a live log). At one compute rank (single mode) the op is a direct call with no MPI and no copies beyond today's.
3. **Stage A (comb reuse or comb scan + `select_comb_peaks`) stays on the head**, scored through the replicated row holder. Its epoch-1 wall time is logged and is the trigger for a later split; not in scope.
4. **Stage B is split per group by CONTIGUOUS box range, equal box counts per compute rank, the head taking an equal share.** Contiguity is load-bearing: the sig-het reference blocks are built on f0 boundary crossings (one block resident at a time), so an f0-contiguous range divides that build proportionally, while interleaving would rebuild every block on every rank (the 4,600-rebuild pathology recorded at fstat_gridfit.py ~1033-1038). The rank→range map is a pure function of `(g_edges, n_compute)` so a resume reproduces it. One command per group (new GB op `gb_fstat_stage_b`), payload = the group's sliced host inputs (`f0_los[a:b]`, `f0_dxs[a:b]`, `mc_ax`, `alpha_ax`, `sd_ax`, `node_shape=(b-a, n_f0, n_Mc, n_alpha, n_sd)`, `fingerprint_extra`, the checkpoint name `stageb_g{gi}_r{r}`, the partial-result path). Each rank calls the EXISTING `run_stacked_peak_sweep` on its sliced inputs with `call_fstat` from its row holder; checkpoints go to the SHARED epoch `_parts` directory under the per-rank name (the existing `ckpt_*` layer resumes a per-rank sweep whose fingerprint — which hashes the sliced inputs and `node_shape` — matches; a changed `n_compute` restarts that group cleanly; `ckpt_clear(_parts, "stageb")` already removes `stageb_g*_r*` files by prefix). The finished partial is written as raw float64 (`np.save`) to `<epoch_dir>/fstat_grid_parts/stageb_g{gi}_r{r}.npy`; the reply carries only `(gi, a, b, n_rows, sha1, wall_s)`. No grid ever travels through a pickled MPI reply.
5. **The head assembles** each group as `np.concatenate(parts in box order, axis=0)` (box is the slowest axis of the group grid), builds the `GroupedStackedFStatProposal` / single-group `StackedFStatProposal4D` exactly as `run_stacked_stage_b` does today, writes the `.npz` with the SAME keys (writer factored out of `run_stacked_stage_b`; `group_sizes`, `logp_grids_g{gi}`, `mc_ax_g{gi}`, global `f0_los`/`f0_dxs`/`alpha_ax`/`sin_delta_ax`, `grid_basis`, `grid_c_t`, `peak_*`, `band_idx`, `band_f0_lo/hi`, `band_edges`; single-group legacy keys unchanged), deletes the partials, writes `DONE.json` LAST (now with the global `walker_ref` and an `n_compute` field), flushes (`_flush_epoch_artifacts`); the ranks `_setup_from_directive` as today.
6. **The centre table (`_install_ctr_table` → `run_center_sweep`) uses the SAME global reference and the SAME replicated row holder** (today it re-derives `walker_ref` and re-opens its own GB-free window, ~20738-20739 — under a global reference that would silently score the centres against a different residual, the exact bug the 2026-08-24 comment there guards). Head-only (25 s).
7. **Bit-identity is the gate.** The sweep has no RNG and no cross-row reduction; a split changes only `FSTAT_BATCH` grouping and the single-block/grouped fast-path choice, both row-independent. The assembled grid must be bit-identical to the serial fit.
8. **Unchanged:** the refit trigger (`_fstat_fit_decision`), epoch numbering, the ranks' completeness check (`_epoch_missing_for_ranks`), `_install`, the birth container, the ctr-table directive, and every single-process path (`_propose_legacy`, `n_compute == 1`: the serial fit runs as today except that it scores through the row holder built from its own walker's rows — verify byte-identical grids on the laptop).

## Architecture

```
head, inside setup() (ranks parked in serve()):
  lls  = fanout.gather_likelihood(acs)                      # (nwalkers,) global order
  w    = argmax(lls); owner, local = layout.owner_of(w)      # new helper on the layout
  row  = fanout.run("gb_fstat_ref_row", ...)                 # symmetric: owner snapshots (GB-free window), Bcast, all build holders
  comb/peaks (stage A) on the head via the holder            # unchanged code, holder-scored
  prep = stage-B host prep (boxes sorted, groups, n_Mc)      # unchanged code, factored to return the plan
  for gi in groups:
      ranges = split_boxes(g_edges[gi], g_edges[gi+1], n_compute)   # contiguous, equal counts
      fanout.run("gb_fstat_stage_b", per_rank_payload=..., local_body=<head's range>, merge=<collect (gi,a,b,n,sha)>)
  grids = [concat(np.load(part) for part in rank order) per group]
  write npz (factored writer); centre table via the holder; DONE.json; flush
ranks: gf_serve("gb_fstat_ref_row") / gf_serve("gb_fstat_stage_b") → run_stacked_peak_sweep(sliced, ckpt=stageb_g{gi}_r{r}) → np.save part → reply
```

New GB ops go into `GB_OPS` (~1931) and `gf_serve` (~17580); the head issues them with `move=self.gf_move_name` through the `_fanout_cmd` pattern (closing over `propose`'s `model`, restoring `nwalkers`/`ntemps`/`_prop_timer` in a `finally`). The GB session token is captured from the opening `gb_run_proposal`, so the extra commands before it are harmless. `WalkerFanout.run` is head-only and runs `local_body` after the isends complete, so head and workers overlap.

## Verification

Laptop (CPU, FakeWorld, one python process, tiny synthetic band — the fixtures of tests/test_multirank_gb_smoke.py and tests/test_gb_rank_session.py):
1. `tests/test_fstat_parallel_fit.py`: (a) 2-rank FakeWorld epoch fit vs the 1-rank serial fit on the same tiny data: the stacked `.npz` arrays bit-identical (every key), `DONE.json` `walker_ref` = the global argmax, the reference row shipped equals the owner's row (both the GB-free window applied and the live residual restored afterwards); (b) resume: kill the stage-B loop after the first group's checkpoint on rank 1, rerun with the same `n_compute` → the sweep resumes from the checkpoint (`ckpt_resume` log) and the result is unchanged; rerun with a different `n_compute` → clean restart, same result; (c) single rank: the serial path's `.npz` byte-identical to the pre-change serial fit (golden captured at the start of the plan); (d) the centre table uses the replicated row (assert the holder object identity / the `walker_ref` recorded).
2. Existing gated smokes stay green (`RUN_GF_GB_SMOKE=1`, `RUN_GF_SMOKE=1`), plus the GB regression batches.
Cluster (WP7 addendum): the continuation run's epoch 1 (`GB_FSTAT_REFIT_EVERY=50`): `[FSTAT_EPOCH]`/stage-B wall per rank, the head log's `[FANOUT] op=gb_fstat_stage_b` balance, and `gf_state_digest`-style comparison of the epoch-1 `.npz` against a serial refit of the same state if one is affordable.

## Risks

- Payload/collective sizes: the row pair (~72 MB) goes by `Bcast`, never pickled per rank; partials never enter replies.
- Device memory: each rank allocates only its slice of `F_flat` (group 1 = 653 MB serial); one resident sig-het block (~GB) plus the fold budget per rank, as today on the head.
- `FSTAT_SIGHET_MULTIDEV=1` (in the 6mo script) is silently off at one GPU per rank; the rank split replaces that parallelism — note it in the runbook.
- Load: per-box cost is uniform within a group; the reference-block build scales with f0 span, slightly uneven for equal box counts. Accept; log per-rank wall so the WP7 read can weight by span later if needed.
- A rank dying mid-sweep: the head's `RemoteWorkerError` aborts the run as for any op; the per-rank checkpoints make the retry cheap.

## Out of scope

Stage A parallelisation; centre-table parallelisation; VGB (no F-stat fit); the sub-band dispersal design (parked); WP8.
