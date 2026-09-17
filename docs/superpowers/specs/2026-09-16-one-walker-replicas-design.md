# One-walker replica mode: replicated residual, sub-band GB dispersal, head-controlled likelihood row scatter

Status: Plans 1-3 landed in the working tree on 2026-09-16, uncommitted (Plan 1 = layout/row scatter/addremove+PSD seams/PSD eigen inner/knobs; Plan 2 = GB/VGB sub-band dispersal incl. the merge-by-physical-source amendment of decision 7 and the `gb_sync` command; Plan 3 = submit-script NWALKERS=1 exemption, launch/runbook docs with the one-walker cluster gate, log-digest replica summarizers). Cluster gates not yet run.

Repo: `LISAanalysistools`, base `dev` at `aec22834` (which contains the whole
multirank walker-block port, branch `multirank-walker-blocks` @ `92cd7747`, as an
ancestor). All work happens in the worktree
`/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker` on branch
`one-walker-replicas`, never in the main `dev` checkout. Nothing is committed
or pushed unless the user says so in the session (standing rule); the work
accumulates in the worktree's working tree, reviewed with `git diff`, and the
final merge into `dev` is the user's call.

Evidence: the read-only exploration `.superpowers/sdd/subband-dispersal-exploration.md`
(file:line at `92cd7747`; copied into this worktree, git-ignored) and the
2026-09-16 design conversation. Line numbers below are at `92cd7747` unless
noted; `dev` adds one multi-GPU commit on top, so plans must re-anchor.

## Scope ruling (user, 2026-09-16)

**This mode exists ONLY for `nwalkers == 1`.** With more than one walker the
run is exactly the existing multirank walker-block port: one block per compute
rank, the divisibility rule, `WalkerFanoutMixin` whole-propose fan-out for the
addremove and PSD families, GB's three-command orchestrator, the WP7 cluster
gates. None of that code changes behaviour for `nwalkers > 1`, and
`n_compute == 1` stays a direct call (bit-identical to single-process) in both
regimes.

Today a one-walker run on several compute ranks is impossible: `build_layout`
raises on `1 % n_compute` (`communication/ranks.py:249-256`). This mode makes
every compute rank a **replica** of the single walker's residual and gives each
family its own way of using the replicas:

| family | at one walker | control | what the replicas do |
|---|---|---|---|
| GB, VGB | **sub-band dispersal** | head orchestrates (existing three commands) | each rank owns a static band range: proposals, RJ, tempering, caps for its bands |
| addremove (MBH, EMRI, SOBBH) | **likelihood row scatter** | head runs the unchanged body | score the rows they are sent; replay expose/fold |
| PSD family (psd, galfor, sgwb) | **likelihood row scatter** | head runs the unchanged body | score the rows they are sent; replay begin/publish |

## Workspace (done before any code)

- Worktree created: `git -C LISAanalysistools worktree add ../LISAanalysistools-onewalker -b one-walker-replicas dev`.
- Tests run against the worktree through the `.wtenv/` shim (copied from the
  multirank worktree, `PYTHONPATH` line repointed): `.wtenv/wt_run.sh
  /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker/src
  .wtenv/<name>.log python -m unittest tests.test_<name>` (needs `conda
  activate deving`). `.wtenv/` and `.superpowers/` are excluded via
  `.git/info/exclude`.
- Laptop rules: one guarded python process at a time, CPU only, tiny
  fixtures, never run `tests/test_gbspecial_flow.py` itself.
- Eryn changes are out of scope for this spec (two savings are listed under
  "Deferred" and get their own worktree if pursued).

## Decisions (fixed)

1. **Trigger.** `nwalkers == 1 and n_compute > 1` selects replica mode in
   `build_layout`; anything else is unchanged. `GF_ONE_WALKER_REPLICAS=0`
   refuses (today's hard error) as the escape hatch. No group-size knob: there
   is exactly one group and `n_replicas == n_compute`.
2. **Layout.** Every compute rank gets `RankPlacement(w0=0, w1=1)`;
   `WalkerBlockLayout` gains `replica_mode: bool`, `n_replicas`,
   `replica_index(rank)` (= position in `compute_ranks`, head = 0),
   `band_range_of(rank)` filled at GB setup (see 6). `block_of` returns
   `(0, 1)` for every compute rank; `fanout_rank`, roles, device selection,
   seeds (`derive_rank_seed`), the fan-out communicator and the
   `WalkerFanout` comm-size assertion (`fanout.py:103-113`) are unchanged.
3. **Residual.** Every rank builds the full one-row ACA with
   `setup_acs(state, rebuild_residuals=True, walker_block=(0, 1))` (it takes
   the block as a plain argument and has no uniqueness assumption). Replicas
   start identical and are kept aligned by (a) the addremove/PSD replays,
   (b) GB's per-unit delta ledger, (c) an authoritative rebuild at every GB
   propose end. Nothing relies on bitwise agreement: the template FILL uses
   `atomicAdd` over overlapping binaries (`lat_chunked_het_kernels.hh:~1897-1917`),
   so replicas agree to ~1e-12. A residual hash joins the `GF_FANOUT_DIGEST`
   line (`fanout.py:68-88` hashes state only today); the existing drift check
   and `check_ll_inject` rebuild remain the safety net.
4. **Non-GB families: head runs the body, replicas score rows.** In replica
   mode `WalkerFanoutMixin.propose` does NOT slice/merge; it calls
   `propose_local(model, state)` on the head with the full (one-walker) state,
   exactly the single-process body. Every scoring call inside the body goes
   through ONE seam per family, which scatters its rows over all compute ranks
   (head included, computing its own share locally), gathers, and returns.
   Replicas draw no random numbers and reply nothing to the merge. Gathers of
   "all walkers" (`gather_likelihood`, `_global_likelihood`,
   `allgather_walker_vector`, `concat_blocks`) return the head's own values
   in replica mode (one representative; every replica holds the same walker).
5. **Knob.** Per-branch attribute `likelihood_fanout` on the move, seeded by
   `{BRANCH}_LIKELIHOOD_FANOUT` (`MBH_`, `EMRI_`, `SOBBH_`, `PSD_`,
   `GALFOR_`, `SGWB_`; the PSD move is keyed by its primary sampled branch,
   the same convention as its `{P}_DEBUG` knobs), default `1`. `0` = the head
   scores every row itself and the replicas idle for that family (they still
   serve GB). At `n_compute == 1` the scatter is a direct local call. This is
   the on/off the user asked for; there is no rows-versus-blocks mode because
   the walker count selects the regime.
6. **GB and VGB: static sub-band ownership, per-unit reconciliation.** Rank r
   owns a contiguous band range `[b0_r, b1_r)` (GB: balanced by band count;
   VGB: balanced by catalogue source count since its bands are per-source).
   Ownership is STATIC for the run (user ruling: adjacent bands carry
   different ladders, so a hot-rung source must not migrate owners
   mid-chain). Band labels are already frozen per propose; a source whose
   f0 drifts past a range edge stays with its label's owner until the next
   propose's re-sort, then belongs to the new band's owner. Reconciliation is
   PER UNIT: at every unit close each rank allgathers its changed cold-chain
   sources (old rows, new rows, slot ids) on the fan-out communicator and
   every other rank applies them through the batched `fill_template`
   primitive (`adjust_sources_in_residual_buffer`, `gbspecialstretch.py:~3190-3246`).
   All ranks run the head-drawn unit schedule in lockstep (same `unit_i`),
   which the global `band % stride` rule already makes adjacency-safe.
7. **GB RJ slot partition and the merge by physical source (amended
   2026-09-16, Plan 2 Task 4 review).** Births on rank r use only dead slots
   with `slot % n_replicas == replica_index` (bounds each rank's birth
   attempts to its share). Leaf SLOTS are not stable across ranks: GB's
   `_write_back_state` densely renumbers leaves per (rung, walker) in
   frequency order over the whole alive set, so a birth or death in a low
   band shifts every higher leaf index, and forcing `preserve_leaf_identity`
   is unsafe under band swaps between rungs (two sources could share a leaf
   id at a rung). The head therefore merges by PHYSICAL SOURCE: each rank
   ships, per rung, the band id of the alive source at every written-back
   slot; the head takes the union of each rank's alive sources whose band
   lies in that rank's owned range, sorts by f0 exactly as the write-back
   does, dense-repacks `inds`, and carries the cold-row `d_h`/`h_h` along the
   sort. Band ownership makes the union conflict-free by construction.
8. **GB counters, ladders, caps, shutoff.** `band_temps`, `band_swaps_*`,
   `band_num_binaries`, caps and shutoff are per band: the head takes the
   owner's row for values and SUMs counters (zeros elsewhere, so SUM = union,
   as the walker-block merge already does). Ladder adaptation stays on the
   head, once per propose. Caps/shutoff censuses that read the residual run
   AFTER reconciliation. Tempering stays in-band on the owner rank
   (`_tempering_swap_grid` rows are per band; nothing crosses ranks).
9. **GB tables and memory.** Friend table, info-matrix borrow table and the
   F-stat birth container/centre table stay global (loaded on every rank as
   today). Each rank binds `SubBandBuffer` residency only for its owned bands
   (already elastic via `special_band_inds`). The whole-grid `BandSorter`
   build on every rank is accepted for the first version and listed under
   "Deferred" as the first optimisation (band-restricted sorter
   construction).
10. **Authoritative rebuild.** At `gb_finish` the head ships the merged
    (coords, inds) to every rank and each runs the existing `check_ll_inject`
    rebuild (zero, restore the non-GB snapshot, subtract the full cold chain,
    `likelihood()`), so replicas re-align at every GB propose end.
11. **PSD inner proposal at one walker: eigen, not stretch.** The PSD move's
    inner step is the vanilla Eryn stretch (`psdmove.py:2187`, also `:2102`
    on the delayed-acceptance route); with one walker it has no complement.
    New PSD settings field `inner_move_kind` (env `PSD_INNER_MOVE_KIND`,
    values `eigen` | `stretch`; default `eigen` when `nwalkers == 1`, else
    `stretch`, so multi-walker runs are untouched). `stretch` at one walker
    is a hard error naming the knob. The eigen path reuses the addremove
    machinery unchanged: `eigen_refresh.eigen_tables_from_ll_batch` builds
    per-rung tables from likelihood second differences, `EigenAxisMove`
    (`eryn.moves.eigenaxis`, an `MHMove`) proposes along them, and
    `MHMove.propose(tmp_model, tmp_state)` replaces `StretchMove.propose` at
    both call sites. Tables are refreshed every `PSD_EIGEN_REFRESH` proposes
    (default 10) at the propose top with the fixed-component covariances
    already prepared; one table per sampled branch (block-diagonal across
    branches: the other branches are held at their rung values inside
    `call_ll`); sidecar persistence is NOT part of this spec. No `1/sqrt(beta)`
    scaling of hot-rung steps, the same convention addremove uses today.
12. **Fancy swaps at one walker.** With one walker the walker permutation is
    the identity, so the fancy swap is provably the plain swap while still
    paying `(ntemps-1)*(2 + ntemps)` likelihood rows per firing (Eryn
    `tempering.py:~686-757`). Both families skip the fancy re-evaluation when
    `nwalkers == 1` and run the plain swap (no likelihood evaluations). Under
    the walker-block regime the gate keys off the per-rank block width, so a
    one-walker block also skips the fancy re-evaluation (a pure saving,
    identical chain); the PSD inner-proposal default keys off the RUN's
    walker count (see decision 11).
13. **Parallelism per scoring call is `min(n_compute, ntemps)`.** A one-walker
    batch has `ntemps` rows. The runbook says so; the info-matrix batches
    (`(2*ndim^2+1) * ntemps` rows under per_walker scope) and GB dispersal use
    the rest of the machine.
14. **Gate.** FakeComm CPU unit tests on the laptop; parity at one rank
    (the scatter collapses to a direct call: bit-identical by construction);
    replica parity on the cluster: one walker on 1 rank vs 2 ranks on 1 GPU
    vs 2 ranks on 2 nodes (addremove/PSD values identical to fp precision;
    GB to ~1e-12 with the digest line reporting the residual hash); a
    statistical check of a short one-walker fit against the single-rank
    run.

## Architecture

### Phases per rank (replica mode)

```
all ranks : prepare_rank -> layout.replica_mode = True, every compute rank block (0, 1)
all ranks : fit.build(); recipe setup at nwalkers = 1 (moves built at N = 1 on every rank)
HEAD      : full GFState -> fanout_comm.bcast(state, seed_base)
all ranks : setup_acs(state, rebuild_residuals=True, walker_block=(0, 1))   # identical one-row ACA
all ranks : state.log_like[:] = head's likelihood (representative; no allgather concat)
GB setup  : head computes band ranges -> ships band_range_of(rank) in the setup payload
HEAD      : run_mcmc; addremove/PSD bodies run on the head, scattering rows; GB orchestrates
COMPUTE   : ComputeService.serve(): ops ll_rows / ll_rows_check / ar_replay / psd_replay / gb_* / stop
SAVER     : unchanged
```

### The row-scatter primitive (new, `communication/rowfanout.py`)

```python
class RowFanout:
    """Scatter independent rows over all compute ranks, gather in row order."""
    def __init__(self, fanout: WalkerFanout, move_name: str): ...
    def run(self, op: str, rows: dict[str, np.ndarray], *, extra=None) -> dict[str, np.ndarray]:
        # rows: {"coords": (N, D), "data_index": (N,), ...}; contiguous even split
        # over layout.compute_ranks (head included, evaluated locally via local_body);
        # each rank's reply is {"ll": (n_r,), ...side outputs...}; results are
        # concatenated back in row order. n_compute == 1 -> direct local call.
    def replay(self, op: str, payload: dict) -> None:
        # broadcast a state-mutating step to every rank (head runs it locally too)
```

It rides `WalkerFanout.run` unchanged (pickle `isend`/`recv`, `seq` and
`RemoteWorkerError` handling, head local body first). Ops are registry-routed
to `move.gf_serve(op, payload, clock, model)` like `propose` and the `gb_*`
ops. The split is contiguous and deterministic: in-model rows are ordered
rung-major, so when `n_compute` divides `ntemps` each rung stays on one
replica for the whole leaf and a row's `prev_logl` and `logl` come from the
same residual copy (exact MH ratio against that copy). `pad_out_of_prior`
keeps the row count constant; without it the split still works, the pinning
is just lost.

### addremove family

- Seam: `ResidualAddOneRemoveOneMove.compute_like(coords_in, data_index)`
  (`addremovemove.py:1284`) becomes the scatter point; the existing body
  moves to `compute_like_local`. `SOBBHChunkedLikeMove.compute_like`
  (`sobbhspecialmove.py:276`) is renamed `compute_like_local` (its
  `setup_likelihood_here` override stays; each replica arms its own
  `_exposed_offset` from its own residual). `MBHSpecialMove` is dead code
  and untouched. Every evaluation already funnels through the seam: entry
  `prev_logl` (`:1821`), in-model scoring (`:1935`, `:1946`), fancy-swap
  re-scoring via `log_like_for_fancy_swaping` (`:1494-1534`), the eigen
  table builds via `call_ll` (`:501-506`, `:530-536`) and the max-lnL walker
  pick (`:489`).
- `_verify_prev_logl` (`:1392-1475`, `ADDREMOVE_CHECK_LL` default on) scores a
  full batch through `compute_acs_like` with the move's own generator; it gets
  its own op `ll_rows_check` so it scatters too. `_record_leaf_inner_products`
  stays on the head (it stashes d_h/h_h into the head's containers; off by
  default on the container path).
- Side outputs: the SOBBH scorer sets `_last_d_h`/`_last_h_h` per row
  (`sobbhspecialmove.py:311-322`); the `ll_rows` reply carries them and the
  head reassembles the arrays in row order.
- Replays per leaf, in this order, each broadcast to every rank and run on
  the head locally: `ar_replay {kind: "expose", coords}` before the leaf's
  first evaluation (`remove_cold_chain_sources`, `:1790`); `ar_replay {kind:
  "setup", coords}` (`setup_likelihood_here`, `:1802`); ... scoring ...;
  `ar_replay {kind: "fold", coords}` (`add_back_in_cold_chain_sources`,
  `:2109`). Payload = the cold-row waveform-basis coords (`(1, ndim)`);
  `_apply_cold_chain_sources` (`:1121-1152`) is a pure function of them with
  a deterministic domain-error skip. `_current_leaf` is shipped with the
  expose so per-leaf transform fills resolve identically.
- Fancy swap: `fancy_swap = ... and self.nwalkers > 1` (`:2022-2026`).
- Eigen tables, sidecar, ladders, counters, checkpoints: head only,
  unchanged (the per-leaf checkpoint hook is back on since the head runs the
  body in single mode).
- `gf_serve` gains the four ops; the readiness guard keeps treating the move
  as served.

### PSD family

- Seam: extract the tier dispatch of `compute_log_like`
  (`psdmove.py:1895-2012`: kernel fast path, galfor sub-band coarse, coarse
  batch, `PSD_BATCH`, container fallback) into
  `_score_rows(walker_inds_keep, psd_coords, galfor_coords, sgwb_coords) ->
  logl`. `compute_log_like` keeps the prior cut and `_merged_noise_rows`
  (`:1865-1893`) on the head, calls the scatter, and sets `prev_logl`. Rows
  are independent given the residual; the container fallback's sens-mat
  snapshot/restore is rank-local.
- Replays: `psd_replay {kind: "begin", fixed_noise_coords}` at the propose
  top (`:2389-2403`: freeze the fixed branches, `_prepare_fixed_component_covariances`,
  coarse `refresh_P` + coarse fixed covariances); `psd_replay {kind:
  "publish", psd, galfor, sgwb}` at the end (`_publish_one(0)` +
  `reset_linear_psd_arr`, `:2527-2562`). `before_vals`/`after_vals` are the
  head's own `acs.likelihood()`.
- Fancy swap in `run_move` (`:2208`, `:2231-2232`): `do_fancy = ... and
  nwalkers > 1`.
- Eigen inner move (decision 11): a small `PSDEigenInner` helper owned by the
  move: `refresh(tmp_state)` builds per-branch, per-rung tables with
  `eigen_tables_from_ll_batch(call_ll, pts, widths, eps_rel)` where `pts` is
  `(ntemps, ndim_b)` and `call_ll` merges the branch rows with the other
  sampled branches' rung values (tiled by `rows // ntemps`) and the fixed
  branches, then scores through the same `_score_rows` scatter with
  `walker_inds = 0`; installs `set_axes(branch, axes[:, None, None], sigmas[:, None, None])`
  in the full `(ntemps, nwalkers, nleaves_max, ...)` form. `run_move` calls
  `self._inner.propose(model, state)` (an `EigenAxisMove` carrying the
  move's `temperature_control`, `periodic`, and the model's prior/like fns)
  instead of `StretchMove.propose`, on both call sites.
- Search mode (`run_move_max_likelihood`) and delayed acceptance keep their
  structure; only the proposal call swaps.

### GB and VGB

- Layout → setup: the head computes `band_range_of(rank)` from the band grid
  (GB: `np.array_split` of band indices; VGB: split by cumulative source
  count) once at setup and ships it in `_common` (`gbspecialstretch.py:~18655-18679`)
  with every command. Ranks apply it as one more AND on the `extra_bool`
  sorter mask (`:~4462`, same mask in tempering `:~14652/14711`), and bind
  `SubBandBuffer` residency to the owned subset.
- `_enter_rank_block` sets `nwalkers = 1` (as today) plus the band mask and
  the slot partition; `_exit_rank_block` restores.
- Per-unit ledger inside `run_proposal`: at each unit CLOSE, after the
  rank's own `add_cold_chain_sources_to_residual`, allgather
  `{slots, old_rows, new_rows, rung 0 only}` for the sources of the unit's
  owned bands whose cold-row coords or alive flag changed, and apply the
  other ranks' entries with `fill_template(+1, old)` then `fill_template(-1,
  new)`. Every rank runs the same number of units (head-drawn schedule),
  so the collective count matches. The head participates through its local
  body (the `isend`/`wait` ordering in `fanout.py:187-195` guarantees the
  workers hold the command before the head enters the body).
- Merge at each command: coords/inds per touched `(rung, slot)` from the
  toucher; counters SUM; per-band values from the owner. `gb_finish` ships
  the merged branch to every rank and each runs `check_ll_inject`.
- VGB inherits everything (no RJ, no caps; `preserve_leaf_identity`).
- Cross-band couplings that stay global and unchanged: friend table, F-stat
  centre table, info-matrix borrow table (retired per block anyway),
  `_fstat_reference_walker` (one walker), prior ranges, cap-cell stagger
  (a cap cell straddling a range edge belongs to the owner of its lower
  band; the census runs after reconciliation).

### Knobs (rule 0: env = capitalized attribute)

| attribute | env | default | meaning |
|---|---|---|---|
| layout replica mode | `GF_ONE_WALKER_REPLICAS` | `1` | `0` refuses one-walker multi-rank runs (today's error) |
| `likelihood_fanout` (addremove, PSD moves) | `{BRANCH}_LIKELIHOOD_FANOUT` | `1` | `0` = head scores every row itself |
| `<noise>.inner_move_kind` (psd, galfor, sgwb settings blocks) | `{P}_INNER_MOVE_KIND` (`PSD_`, `GALFOR_`, `SGWB_`) | `eigen` at one walker, `stretch` otherwise | inner proposal of the PSDMove instance sampling that block |
| `<noise>.eigen_refresh_every` | `{P}_EIGEN_REFRESH` | `10` | proposes between table refreshes |
| `<noise>.eigen_eps_rel` | `{P}_EIGEN_EPS_REL` | `1e-4` | finite-difference step, fraction of the prior box |

## Testing

Laptop (FakeComm, CPU, one process):
- `build_layout(nwalkers=1, size=3)` → replica mode, every compute rank block
  `(0, 1)`, `replica_index`, escape hatch; `nwalkers=2` unchanged.
- `RowFanout.run` over 3 fake ranks: split/gather in row order for N not
  divisible by n_compute, N < n_compute, N == 0; side outputs; remote error
  propagation; `n_compute == 1` direct call; knob off → local.
- addremove replay ordering: a fake service records the op sequence per leaf
  (`expose, setup, ll_rows..., fold`) and the coords shipped; the head's own
  residual after the propose equals the single-process propose's residual
  (same seed) to fp precision.
- PSD `_score_rows` extraction: `compute_log_like` output unchanged
  (bit-identical) on the existing `test_psd_move_*` fixtures; the scatter
  over fake ranks equals the local call.
- PSD eigen inner: tables from a quadratic synthetic likelihood recover the
  known axes/sigmas; `MHMove.propose` path runs with the move's temperature
  control; `stretch` at one walker raises.
- GB one-walker arm: 1 walker on 2 fake replicas runs, both replicas answer
  every `gb_sync`, the per-rank `log_like_final` agree (rtol 1e-10) and the
  residual hashes agree on CPU, no merge conflict, the injection survives
  (`test_one_walker_two_replicas`). A bit-identical match to the one-rank
  propose is NOT expected: each rank draws its own RJ proposals.
- Regression: the existing multirank suites (`test_fanout_fakecomm`,
  `test_walkerfanout_mixin`, `test_addremove_fanout_hooks`,
  `test_psd_fanout_hooks`, `test_recipe_fanout_install`,
  `test_run_multirank_helpers`, `test_walkerslice_roundtrip`) stay green,
  proving `nwalkers > 1` is untouched.

Cluster (runbook addition to `docs/multirank-cluster-gates.md`): the parity
and statistical gates of decision 14, plus a timing readout of the
`min(n_compute, ntemps)` bound.

## Plans (in order; no pauses between them)

1. **Layout + row scatter + non-GB families + PSD eigen + knobs + tests.**
   Self-contained and laptop-verifiable. Deliverables: `ranks.py` replica
   mode, `communication/rowfanout.py`, `WalkerFanoutMixin` replica branch and
   the four addremove/PSD ops, `compute_like`/`compute_like_local` split,
   `_score_rows` extraction, `PSDEigenInner`, fancy-swap gating, knobs,
   representative gathers, residual hash in the digest line.
2. **GB/VGB sub-band dispersal.** Band ranges in the layout/setup payload,
   `extra_bool` band gate in proposals and tempering, buffer residency
   subset, slot partition, per-unit ledger, merges, `gb_finish` rebuild,
   caps/shutoff after reconciliation, VGB balance by source count, the
   one-walker GB smoke/parity arm.
3. **Scripts, docs, cluster gates.** Submit-script support for
   `NWALKERS=1`, the runbook section, the `GF_LAYOUT_DRY_RUN` line for
   replica mode, the timing readout.

## Deferred (recorded, not in scope)

- Band-restricted `BandSorter` construction (every replica builds the
  whole grid today; dominant wasted cost at 3mo+).
- Eryn `tempering.py:749-757`: the full re-evaluation after every rung pair
  is redundant given `new_like` (~5x fancy-swap saving for multi-walker
  runs). Separate Eryn worktree.
- `1/sqrt(beta)` scaling of eigen steps at hot rungs (shared with
  addremove).
- PSD eigen sidecar persistence.
- Replica placement across nodes (co-location) has no effect at one group.

## Risks

- Replica residual mismatch (~1e-12) puts ~1e-7 absolute noise into an MH
  ratio when a row's old and new likelihoods come from different replicas;
  the rung pinning of the contiguous split removes it for in-model steps;
  swaps mix rungs at the same level as today's run-to-run atomicAdd noise.
- Replay determinism of the single-source template add/subtract on the
  addremove path needs a check per waveform (plan 1 test: hash the residual
  on two fake ranks after a leaf).
- The whole-grid sorter per replica costs memory and time at long
  observations (deferred item 1).
- GB ledger correctness rests on band ownership plus the slot partition;
  plan 2 must assert no `(rung, slot)` is touched by two ranks in one
  command.
