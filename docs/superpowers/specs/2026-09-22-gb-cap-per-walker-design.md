# Per-walker GB leaf caps and cap-cell logL tracking

Date: 2026-09-22
Branch: `gb-cap-per-walker` (worktree `LISAanalysistools-cap-per-walker`, base
`dev` @ `6d641643`)
Status: IMPLEMENTED. See "As built" at the end for the places the
implementation went beyond or differs from this design.

## Intent

The GB progressive leaf cap is stored and advanced **per cap cell**, with the
walker axis collapsed the moment the gate runs
(`cur_max = lls.max(axis=0)` in `GBSpecialBase._update_band_leaf_caps`). One
walker's evidence therefore ramps the allowance for every walker: the leading
walker's lnL improvement and its occupancy-at-cap raise a cap that laggard
walkers have not filled and have not paid for. The 6-month production run shows
the symptom directly — a 400-1000 leaf spread across four cold walkers, and a
snapshot-19 increment census in which the occupancy-at-cap condition held for
only 2 of 54 observed increments under every reduction tried.

**Goal:** each walker earns, holds and is enforced against its own per-cell
leaf cap, driven by its own lnL plateau and its own occupancy. GB only.

**Hard constraint:** the live 6-month run (`gf_prod_6mo_v8_4gpu`) may still be
relaunched from a build containing this change. With the feature off the code
must allocate nothing new, write nothing new, and take no new branch — it must
be bit-identical to `dev` @ `6d641643`.

**Success criteria**

1. `GB_LEAF_CAP_PER_WALKER=0` (default): no new `band_info` key, no new HDF5
   dataset, no changed gate arithmetic. Covered by an explicit test, not by
   inspection.
2. `GB_LEAF_CAP_PER_WALKER=1`: `cap_cell_leaf_cap_w[w, c]` diverges between
   walkers on real evidence, and every enforcement site gates row `(t, w, ...)`
   against `cap[w, cell]`.
3. Enabling the flag on a store written without it seeds every walker from the
   stored shared cap — no migration script.
4. Increment decisions are auditable from the run log at the moment they are
   taken (see "Diagnostics").

## Non-goals

- The band RJ shutoff valve (`band_rj_shutoff` / `band_occ_streak` /
  `band_occ_last`) stays per band, max over walkers. Per-walker shutoff is a
  full RJ freeze and could lock a walker out of a band the others populated
  until the next F-stat epoch revives it. Considered and deliberately deferred.
- The per-band temperature ladder (`band_temps`), the F-stat refit reference
  walker, and the cap ceiling (`GB_CAP_CELL_MAX`) stay global.
- PE is untouched: caps are disarmed (`-1`) in PE exactly as today, and the
  cap remains a search-only proposal veto, not a prior term.

## Semantics

A cap is indexed by **walker index only** and applies at **every temperature**,
exactly as the shared cap does today. This matches the existing per-walker
array `cap_cell_cold_ll`, shape `(nwalkers, ncells)`, and the flat census index
`_cap_flat_index(temp, walker, cell) = (temp * nwalkers + walker) * ncells + cell`.

Tempering: a swap is gated against **each side's own walker's cap**. The
vertical sweep shares a walker index, so `cap_a == cap_b` there and that path is
unchanged; only `_permute_walkers_for_swaps` (cross-walker) can now see
asymmetric caps.

## 1. Flag and settings

`GBSettings.leaf_cap_per_walker: bool`, env `GB_LEAF_CAP_PER_WALKER`, default
`False`, following the existing `env_default(...)` field pattern in
`stock/erebor/gb.py`. Threaded to the move constructor as
`leaf_cap_per_walker=` alongside `leaf_cap_start` / `cap_divisor` /
`cap_stagger` / `cap_overlap_frac`.

On the move it becomes `self.leaf_cap_per_walker`, and the derived predicate

```python
@property
def _cap_per_walker(self) -> bool:
    return bool(self.leaf_cap_per_walker) and self._leaf_cap_enabled
```

is the single switch every new branch tests.

## 2. Storage

Allocated by `ensure_cap_cell_fields(band_info, num_cells, staggered=,
per_walker=)` only when `per_walker` is true:

| key | shape | fill | dtype |
|---|---|---|---|
| `cap_cell_leaf_cap_w` | `(nwalkers, ncells)` | `-1` | int |
| `cap_cell_iters_w` | `(nwalkers, ncells)` | `0` | int |
| `cap_cell_best_ll_w` | `(nwalkers, ncells)` | `-inf` | float |
| `band_best_ll_w` | `(nwalkers, nbands)` | `-inf` | float |

`cap_cell_cold_ll` is already `(nwalkers, ncells)` and is unchanged.
`band_best_ll_w` is a monitor mirror only (running max per walker per band, no
reset) — without it a snapshot cannot show whether the per-walker caps actually
diverged, which is the whole thing we need to observe.

`ensure_leaf_cap_fields` gains the same `per_walker` seam for `band_best_ll_w`.

### Registration (`globalfit/state.py`)

- `GBState._bare_ndim`: all four new keys at bare ndim `2` (strip the backend's
  leading step axis on reload).
- `GBState.legacy_dtype_names`: add all four, matching their 1-D twins, so the
  HDF5 round trip keeps the backend float dtype convention the cap family
  already uses.
- `initialize_band_information` `_expected_shapes`: add the per-walker shapes so
  a store whose walker count changed is refused loudly rather than resumed onto
  a grid it was not measured on.
- Cap-free branch (`leaf_caps=False`, VGB) drop list: add
  `cap_cell_leaf_cap_w`, `cap_cell_iters_w`, `cap_cell_best_ll_w`.

No change is needed to `GBState.from_stored` (the `cap_cell_` and `band_`
prefixes already catch them) or to `storage_arrays()` (it persists every
ndarray in `band_info`).

### Mirrors

The 1-D arrays stay live as **max-over-walkers** summaries so the monitor and
every diagnostic script keep working untouched:

- `cap_cell_leaf_cap[c] = max_w cap_cell_leaf_cap_w[w, c]`
- `cap_cell_iters[c] = max_w cap_cell_iters_w[w, c]`
- `cap_cell_best_ll[c] = max_w cap_cell_best_ll_w[w, c]`
- `band_leaf_cap[b]` keeps its existing definition (max over the band's cells)
  applied to the mirrored cell caps, i.e. max over walkers *and* cells.

When the flag is on these are a **summary view, not the gate's state**. That is
documented at the allocation site and in `_mirror_band_leaf_cap`.

## 3. The gate (`_update_band_leaf_caps`)

`_cap_state_arrays(bi)` returns the per-walker triple when
`self._cap_per_walker`, otherwise today's 1-D triple. Everything downstream in
the gate is already elementwise, so the change is to stop collapsing:

- `cur = lls` (shape `(nwalkers, ncells)`) replaces `cur_max = lls.max(axis=0)`.
- `improved = cur > best + thresh`, `best = maximum(best, cur)` — unchanged
  arithmetic on a 2-D array.
- Engagement latch `_cap_ll_improved_once` and baseline `_cap_ll_prev_stat`
  gain the walker axis. Both stay **in-memory only**, matching today (a restart
  re-earns engagement, which holds caps longer — the conservative direction).
- Occupancy: `converged &= (cap >= 1) & (_occ_w >= cap)`, dropping the
  `_occ_max = _occ_w.max(axis=0)` reduction. `_cold_occupancy` already returns
  `(nwalkers, ncells)`.

  This does **not** reverse the 2026-09-19 ruling that `max` is the correct
  reduction for the occupancy demand signal. That ruling was about a *shared*
  allowance, where a single walker pressed against its cap is evidence the cell
  needs room for everyone. With per-walker allowances the same principle is
  relocated, not abandoned: each walker's own demand drives its own cap.
- Ceiling: `converged &= cap < _ceiling` — unchanged, now elementwise on 2-D.
- Increment: `cap[converged] += 1; iters[converged] = 0; best[converged] = -inf`
  — unchanged, now over `(walker, cell)` pairs.
- `leaf_cap_iter_only` and the legacy nsigma gate go per-walker too; both are
  already elementwise. The nsigma gate's `lls.min(axis=0)` becomes `lls` and its
  `cold_counts.max(axis=0)` becomes `cold_counts`.
- `GB_LEAF_CAP_ALL_WALKERS`: subsumed. When `_cap_per_walker` is on it is
  honored as a **no-op with a one-time warning**, so an existing runbook does
  not silently change meaning. `_cap_all_walkers_converged` keeps working for
  the flag-off path.
- `_track_band_best_ll` additionally maintains `band_best_ll_w` when present.

## 4. Enforcement sites

A single accessor is the only place the two layouts are distinguished:

```python
def _cap_for_rows(self, cap, walker_inds, cells):
    """Per-row cap: ``cap[cells]`` (shared) or ``cap[walker_inds, cells]``."""
    if cap.ndim == 1:
        return cap[cells]
    return cap[walker_inds, cells]
```

Keyed on `cap.ndim`, not on the flag, so a device-side snapshot carries its own
answer and no call site needs the move's configuration.

Routed through it (all four already have `walker_inds` in scope):

| site | today | change |
|---|---|---|
| `_row_at_cap` | `cap[cells]`, `cap[nb_cells]` | `_cap_for_rows(cap, walker_inds, ...)` |
| `_cap_at_cap_mask` | `cap[cap_inds]`, `cap[nb_inds]` | same, with `band_sorter.walker_inds` |
| `_cap_new_entry_veto` | `cap[_cell]` (twice) | same, with `walker_inds` |
| `_cap_budget_transitions` | `cap_acc` gathered per row | gather site supplies the per-walker value |

Two need more than a re-index:

- **`_swap_cap_ok` / `tempering_swap_cap_ok`.** `_swap_cap_ok` already passes
  `(t_a, w_a)` and `(t_b, w_b)` separately. It gathers `cap[w_a[:, None], cells]`
  and `cap[w_b[:, None], cells]`; `tempering_swap_cap_ok` gains a `cap_b`
  parameter defaulting to `cap_a`, so the existing signature and its tests keep
  working. The disarmed-sentinel rule (`cap < 0` -> unconstrained) is evaluated
  per side.
- **`_band_saturated_flat`.** The flat leading axis is `temp * nwalkers`, so the
  cap broadcasts as `(nwalkers, ncells)` -> `(ntemps, nwalkers, ncells)` ->
  `(ntemps * nwalkers, nb, k)` before the existing reshape/`all(axis=2)`. The
  staggered boundary-cell term uses the same broadcast array.

`_cap_drift_gate_setup` and `_replace_cap_state` ship the cap to device
unchanged (2-D instead of 1-D); the indexing happens inside
`_cap_new_entry_veto` / `_row_at_cap`.

## 5. Multi-rank fan-out

`tables["cap_leaf_cap"]` is currently shared across ranks. With a walker axis it
must be **sliced to the rank's block**, `self._cap_leaf_cap[w0:w1]`, because a
rank runs with `self.nwalkers = B` and computes block-local walker indices. This
is exactly the pattern `_sched_block(w0, w1)` already uses for the per-walker
scan schedule.

Consequence: the cap table moves out of the shared `tables` dict into the
per-rank payload built by `_payload_proposal(rank, w0, w1)`. `band_leaf_cap` and
`rj_band_shutoff` stay shared (both remain per band).

`_update_band_leaf_caps` runs head-only on stats already concatenated back to N
walkers, so the gate needs no fan-out change. `_RANK_BLOCK_SAVED` already
saves/restores `_cap_leaf_cap`.

## 6. Enabling on an existing store

`ensure_cap_cell_fields(..., per_walker=True)` seeds from the stored shared
arrays when they exist:

```python
cap_cell_leaf_cap_w[:] = cap_cell_leaf_cap[None, :]
cap_cell_iters_w[:]    = cap_cell_iters[None, :]
cap_cell_best_ll_w[:]  = cap_cell_best_ll[None, :]
```

Every walker inherits the shared cap and its clock; divergence starts from the
first post-enable increment. No migration script.

The reverse direction (a store containing `*_w` keys read by a build with the
flag off) degrades safely: `_cap_state_arrays` ignores the `*_w` arrays and the
1-D mirrors are maintained live, so the run continues on the max-over-walkers
cap. Not a supported workflow; documented so it cannot surprise.

`scripts/fstat_proposal/rescale_store_walkers.py` gains the four new
walker-axis arrays in its rescale list.

## 7. Diagnostics (explicitly required)

The existing cap machinery has a history of gates that could not be judged from
the store after the fact (snapshot 19's increment anomaly; `band_cap_iters`
reading zero because it is an unmirrored legacy array). This change adds
increment-time logging so the same question cannot recur.

- **Arming.** One line when the per-walker caps arm, naming the mode, the
  walker count, the cell count, and whether the arrays were freshly allocated,
  restored, or seeded by broadcast from a shared cap.
- **`[GB_CAP_PW]` increment line**, once per iteration when anything
  incremented: number of `(walker, cell)` pairs, the per-walker breakdown, and
  the new cap range. Enough to see divergence appear without a store dump.
- **`GB_CAP_PW_DIAG=1`** (default off): per increment, the tuple the decision
  was actually taken on — `walker, cell, cell frequency edges, occupancy, cap
  before, iters, best_ll, cur_ll`. This is the direct answer to "was
  occupancy-at-cap really met", logged from the gate's own arrays at the moment
  of decision rather than reconstructed from a snapshot.
- **Divergence summary**, once per iteration at INFO: spread of
  `cap_cell_leaf_cap_w` across walkers (count of cells where walkers disagree,
  max disagreement). If this stays 0 the feature is not doing anything and the
  log says so.
- **Mirror consistency assert** under `GB_CAP_PW_DIAG=1`: the 1-D mirrors equal
  the max over walkers, warned (not raised) on mismatch.

## 8. Testing

New `tests/test_gb_cap_per_walker.py` (unittest, CPU-only, no GPU required):

1. **Compat gate.** Flag off: `ensure_cap_cell_fields` adds no `*_w` key and
   `_cap_state_arrays` returns the 1-D triple.
2. Per-walker allocation shapes/fills; broadcast seeding from an existing
   shared cap.
3. Gate: two walkers, one plateaued and at cap, one still improving — only the
   first increments.
4. Gate: occupancy-at-cap is per walker — a walker below its own cap does not
   increment even when another walker is at cap.
5. `_row_at_cap` and `_cap_new_entry_veto` veto on the row's **own** walker cap.
6. `tempering_swap_cap_ok` with asymmetric `cap_a`/`cap_b` and with one side
   disarmed (`-1`).
7. `_band_saturated_flat` per-walker broadcast over the temp axis.
8. Mirrors equal the max over walkers; `band_leaf_cap` is the max over walkers
   and cells.
9. Rank payload: the shipped cap table equals `cap[w0:w1]`.

Regression: `tests/test_gb_cap_cell_grid.py`, `tests/test_cap_stagger.py`,
`tests/test_gbspecial_flow.py`, `tests/test_vgb_no_cap_cells.py` must pass
unchanged with the flag off.

## Risks

- **Mixing.** Per-walker caps plus a per-side swap gate can suppress
  cross-walker (permuted) tempering swaps between walkers at different cap
  levels. The vertical sweep — which carries most dimension change into the cold
  chain (T0-T1 35-60%) — shares a walker index and is unaffected. Watch the
  permuted-swap acceptance in the first run.
- **Fan-out slicing.** A cap table shipped unsliced to a rank would gate
  block-local walker 0 against global walker 0's cap. Covered by test 9 and by
  the arming log line, which prints the table width a rank received.
- **Divergence never appearing.** If every walker ramps together the change is
  inert; the divergence summary line makes that visible in one iteration rather
  than after a snapshot analysis.

## Launch runbook — verifying this on the 1-year run

**Turning it on.** `submit_gf_1yr_v8_4gpu.sh` does NOT export it today (it
has `GB_LEAF_CAP_REQUIRE_IMPROVEMENT=1` and `GB_LEAF_CAP_MIN_ITERS=3`, but
no per-walker knob), so the run would use the shared cap. Add:

```sh
export GB_LEAF_CAP_PER_WALKER=1
# optional, for the first launch only -- the per-increment decision tuple
export GB_CAP_PW_DIAG=1
```

`GB_LEAF_CAP_ALL_WALKERS` is not exported by any submit script, so nothing
will trip the subsumed-flag warning.

**The four checks, in the order you can make them.**

1. *Did it arm at all?* — first GB propose, once per move per process:

   ```
   grep "GB_CAP_PW.*leaf caps are" <log>
   ```

   Expect `leaf caps are PER WALKER: 4 walkers x 9856 cap cells, caps
   1-1, fresh (all disarmed)`. **If it says `SHARED`, the flag did not
   take** — this line fires in both modes precisely so a silent
   fall-back is impossible to miss. On a resume of a per-walker store it
   says `resumed (walkers already differ)`.

2. *Did every rank get its own block?* — multi-rank only, once per rank:

   ```
   grep "GB_CAP_PW.*rank cap table" <log>
   ```

   Expect one line per compute rank reading `PER WALKER, 1 walkers x 9856
   cap cells` at 4 walkers over 4 ranks. A wrong slice **raises** rather
   than logging, so silence here with the run still alive means the ranks
   never entered a block.

3. *Is it doing anything?* — every iteration:

   ```
   grep "GB_CAP_PW.*cap spread" <log> | tail -20
   ```

   Early on expect `every walker holds the SAME cap in all 9856 cells
   (per-walker caps are currently inert)`. The finding is when that turns
   into `N/9856 cells differ between walkers, max spread M`. **If it
   never turns over, the feature is running and buying nothing** — that
   is a real result, and this line is how you learn it in one iteration
   instead of from a snapshot.

4. *Was each increment justified?* — the snapshot-19 question, answered at
   decision time:

   ```
   grep "GB_CAP_PW.*incremented for" <log>            # per-walker breakdown
   grep -A3 "GB_CAP_PW.*incremented for" <log>        # with GB_CAP_PW_DIAG=1
   ```

   The detail line carries `w<N> cell <c> [lo, hi] mHz: occ=<n> cap <a> ->
   <b>, iters=<i>, best=<x>, cur=<y>`. `occ` and `cap <a>` are read from
   the gate's own arrays at the moment it decided, which is exactly what
   the store could not tell us for snapshot 19's 54 increments. Capped at
   `GB_CAP_PW_DIAG_MAX` (default 20) lines per iteration.

**In the store**, for post-hoc work: `cap_cell_leaf_cap_w`,
`cap_cell_iters_w`, `cap_cell_best_ll_w` `(nsteps, nwalkers, ncells)` and
`band_best_ll_w` `(nsteps, nwalkers, nbands)`. The monitor page is
unchanged and still reads the 1-D max-over-walkers mirrors, so it will
show the ensemble envelope, not the spread — the spread is only in the
`_w` arrays and in check 3's log line.

## As built

Where the implementation went beyond this design, and why.

1. **Row-alignment guard in the gate (new, and load-bearing).** Raised by the
   `gpu-count-routing` session, 2026-09-22. The shared cap reduced the cell
   statistic with `max(axis=0)`, which is blind to how many walker rows arrive
   and in what order. Per-walker caps index `cap[w, c]` against `lls[w, c]`
   **positionally**, so row count and row order became load-bearing — and
   `_propose_orchestrated` builds that statistic by concatenating each compute
   rank's `cap_stats` on the walker axis. That is exactly N rows only while
   `layout.block_of` PARTITIONS the walker axis. Under a layout where several
   compute ranks share a walker block (GPUs > walkers, in development on
   `gpu-count-routing`) the merge would deliver N x R rows and the gate would
   score walker `w` on another walker's series, in range and without raising.
   `_update_band_leaf_caps` now refuses when `lls.shape[0] != cap.shape[0]`,
   with a message naming the shared-block layout and the fix (reduce over one
   rank per block). Covered by `PerWalkerRowAlignmentTest`.

   The ship direction needs nothing: `cap_leaf_cap` is a read-only head->rank
   slice, and several ranks holding the same walkers receiving the same rows
   is correct.

2. **`_cap_table_for_block` is a method, not a closure.** The fan-out slice was
   originally written inline in `_propose_orchestrated`. It is the one place a
   mistake is silent (every index stays in range on every rank), so it was
   extracted to be directly testable.

3. **The fused in-model accept kernel stands down.** `GB_INMODEL_ACCEPT_KERNEL`
   (default OFF) passes `dg_cap` to a raw-pointer ABI as a FLAT per-cell array.
   A per-walker cap would arrive flattened and the kernel would read walker 0's
   cell for every walker. Rather than teach a default-off path a second layout,
   the kernel declines with a warning and the (always-correct) python chain
   runs.

4. **`band_best_ll_w` is maintained at every divisor**, unlike its shared twin,
   which the band-grid gate owns. On the band grid the per-walker gate reads
   the cap-cell arrays, so nothing else would write it.

5. **The cap-headroom stop gate keeps the walker axis.** `GB_SEARCH_CAP_HEADROOM`
   (default 0) grants +1 where a cell is short of free slots; its fallback
   occupancy path reduced with `max(axis=0)`, which would have granted one
   walker's slot on another walker's crowding.

6. **`migrate_gb_cap_grid.py` re-seeds the `_w` family** (only when already
   present) from the freshly migrated shared arrays. Without it a cap-grid
   migration would leave the per-walker tables on the old cell count — caught
   by the resume guard, but only as a refusal the operator then has to
   diagnose. `band_best_ll_w` is per band and is deliberately not migrated.

7. **`GB_CAP_PW_DIAG_MAX`** (default 20) caps the per-increment detail lines so
   a lockstep iteration cannot flood the run log.
