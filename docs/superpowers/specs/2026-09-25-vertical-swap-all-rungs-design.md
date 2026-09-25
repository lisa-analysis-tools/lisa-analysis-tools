# Vertical swap across ALL rungs of a (walker, band) column

**Date:** 2026-09-25
**Status:** designed + audited, NOT yet implemented
**Requested by the user:** "We want to be swapping across all rungs at each
25 repeat batch regardless of how many sources are in each sub-band and
regardless if a source/band is picked or not." And: "even if it does not
have a source in that band it has a total likelihood ... It still holds a
total likelihood and should get swapped accordingly."

Residency ruling (user): "unpicked rungs should not be resident they should
just be accounted for in swapping"; "Maybe you can load it, calculate the
likelihood and unload it and then just use bookkeeping so you can keep
trying to add more to the in-model batch."

Independently audited by a second Opus agent, 2026-09-25. Findings below
are that audit's, verified against code; line numbers are as of `a8fab0a4`.

---

## The gap

`_vertical_pairs` (`gbspecialstretch.py:14612`) keys on `t_i, w_i, b_i` --
the block's PICKED rows -- and returns row indices into them. So only
picked-vs-picked pairs are ever emitted. A rung holding an unpicked source,
or no source, can never be a swap partner.

The accept path does NOT have this limitation.
`band_sorter.exchange_cell_labels_batch` (`gbbands.py:6157`) works on cell
IDs via `xp.isin(self.special_band_inds, src)` -- an empty partner simply
matches zero rows. The per-cell ledgers (`ll_change_log`,
`prop_counts[1]`, `acc_counts[1]`) are indexed `[t, w, b]` over the full
grid. The cap gate (`_swap_cap_ok` / `_cap_swap_apply`) reads a
sorter-built census and already knows about unpicked and empty cells.

So this is a pair-discovery + scoring change, not an accept-path rewrite.

## Why an unpicked/empty rung's likelihood is obtainable and stable

At unit open the move calls `remove_cold_chain_sources_from_residual`
(`:6368`) and at unit close `add_cold_chain_sources_to_residual` (`:6626`);
nothing writes the parent plane in between. Every slab write is
slot-addressed (`_adjust_via_engine -> fill_template(..., params_index=slot)`,
`gbbands.py:5521`), and the only two in a block are the removal at `:16129`
and the write-back at `:17437`, both over the block's own slots.

Therefore a non-resident cell's total likelihood is a function of (frozen
parent plane, its own unmoving sources) -- a **unit** constant, a fortiori
a block constant. Do NOT justify this with the orthogonality premise
(`_run_ortho_premise_check`, `:7700`); that is about additivity of
concurrently-sampled deltas and is not what protects this cache.

All rungs of one `(walker, band)` read a bit-identical slab, so an EMPTY
rung's total is **one number per column** -- the bare parent-walker plane
over the band window.

## No per-cell TOTAL exists today

| quantity | indexed by | total or delta | kept for cells with no picked row |
|---|---|---|---|
| `cell_ll_state["ll0"]` | slot (`spec[slot]` names the cell) | TOTAL | bound slots only |
| `ll_change_log` | `[t,w,b]` full grid | DELTA from propose start | value exists, but is a delta |
| `_vert_base` | slot | total, but `L_free` not `L_with` | picked slots only |
| `band_info["band_cold_ll"]` | band | total | cold rung only, no `(t,w,b)` axis |
| `band_likelihoods(slots=)` | slot | TOTAL | resident slots only |

`ll0` is the right quantity but is slot-scoped and, on the production
path, only ever taken over the pooled cells. A measurement is required.

`_vert_base` (`:16370`) is `band_likelihoods(source_only=True, slots=slots)`
taken after the picked source was removed, so it equals the bare-slab value
**only when the picked source was that cell's sole leaf**. For a multi-leaf
cell the other leaves are still subtracted.

## Chosen design

Pre-block transient measurement, cached, permuted on accept.

1. **Finalize any open `cell_ll_state` brackets FIRST**, then run the
   measurement passes by calling `_cached_get_buffer` **directly -- never
   `_rebind`** -- batched up to capacity, then bind the block and open
   brackets. (See hazard 1.)
2. Measure with `band_likelihoods(source_only=True)`, a pure reduction.
   Do **not** call `setup_in_model_likelihood` on these cells: the 10.2 s
   `inmodel_sighet_setup` is per picked row and must stay untouched.
3. Per column this needs **1 bare measurement + 1 per unpicked-but-occupied
   rung**, not `ntemps`. The bare value is FREE whenever any rung of the
   column is a sole-occupant picked cell -- `_vert_base[slot]` already is it.
4. Cache into a per-column table:

   ```
   carrier[rung] -> block row index, or -1
   cached[rung]  -> measured total (authoritative only where carrier == -1)

   L(rung) = cell_ll_base[carrier] + ll_ref[carrier]   if carrier >= 0
           = cached[rung]                               otherwise
   ```
5. On an accepted swap, **permute** `carrier` and `cached` between the two
   rungs, exactly as `ll_change_log` is swapped at `:15081`. An accepted
   swap is a pure relabel (`:14853`), so nothing is recomputed. This is what
   makes the design cost one measurement per block rather than one per
   accept.
6. Scope: once per BLOCK, same scope as `_vert_base`, i.e. after the
   preceding RJ round and before the removal at `:16129`. Three call sites
   each constitute a block: every `_polish` on the direct path (`:8368`,
   including each `_converge_refill_loop` iteration), every `_cls` flush on
   the grouped path (`:8568`), and every `GB_INMODEL_SETUP_BATCH` sub-block
   (`:16040`). Per-propose caching would be WRONG -- a later RJ round can
   birth or kill in an unpicked cell.

Cost: rebinds are "construction minus allocation" since the fixed-capacity
fix (`:7089`), so the transient passes add zero GPU memory. Against a real
900.6 s `rj_warm_search` propose (buffer_build 1.6 s, inmodel_vertical_base
0.56 s), the extra is ~6-17 s, i.e. **0.7-2%**.

## ☠ Hazard 1 -- the `_rebind` bracket trap

`GB_CELL_LL_CREDIT` defaults to 1 (`:7949`). `_rebind` (`:8168`) finalizes
the outgoing binding's open cell-ll brackets **against the current
`buffer_obj`** before rebinding. Inserting transient binds between the RJ
batch and `_polish` makes `_polish`'s `_rebind` finalize the previous
batch's brackets against the TRANSIENT buffer -- silent corruption of
`ll_change_log` and of `state.log_like[0]` (credited at `:15719`).

Order must be: finalize open brackets -> transient measurement passes via
`_cached_get_buffer` -> bind block -> open brackets. Also, `get_buffer`
calls `_assert_cell_labels_flushed` (`gbbands.py:6335`), so the passes must
run BEFORE `begin_cell_label_window` at `:16462`.

## ☠ Hazard 2 -- empty↔empty pairs are ALWAYS ACCEPTED

Two empty rungs of one column have identical `L`, so `paccept = 0.0`, and
`acc = paccept >= xp.log(u)` (`:14992`) with `log u <= 0` is
**unconditionally true**. Those accepts land in `prop_by_bandrung_dev` /
`acc_by_bandrung_dev` (`:14980`, `:15061`), which feed
`_vertical_ladder_bank -> _vertical_adapt_ladder -> _adapt_band_temps`
(`:14682`). At `ntemps=24` a sparse column contributes ~20 always-accept
pairs against 1-3 real ones: **the band's measured acceptance goes to ~1
and the ladder collapses.**

This is the same failure already documented for `GB_TEMPER_COMPACT_ROWS`
(`:2022`). Mitigation: skip empty↔empty pairs entirely (they are exact
no-ops -- `exchange_cell_labels_batch` would match zero rows) and, if
bit-comparable ladder statistics are wanted, add their deterministic +1/+1
back analytically the way `_temper_compact_rows_on` does.

## ☠ Hazard 3 -- `cell_ll_state` must be re-pointed ONE-SIDED

`:15091-15100` swaps `spec/ll0/led0/rep0` between `slots[h]` and
`slots[c]`. With one partner non-resident there is no second slot. The
correct rule is **`st["spec"][s] = <the other cell's packed special>`, and
leave `ll0/led0/rep0` alone.** Verified against `_cell_ll_finalize`
(`:7625`), which computes `ll_change_log[spec] = led0[s] + (lls[s] - ll0[s])`
-- ledger-at-open plus realized slab delta, credited to whatever label the
slot now claims. Since the sweep also swaps `ll_change_log[t_h] <-> [t_c]`,
re-pointing `spec` alone gives the correct per-label delta. A two-sided
swap is wrong here; a no-op silently credits the wrong cell.

## Other required changes

- **Widen `begin_cell_label_window`** (`:16462`) to all rungs of every
  column in the block. `GB_CELL_LABEL_DEFERRED` defaults ON
  (`gbbands.py:416`) and `gbbands.py:5955` states a cell outside the window
  is a programming error raised at the flush -- without this, every run
  raises.
- **Row-indexed writes must be guarded:** `t_h, t_c = t_i[h].copy(), ...`
  and `w_hc, b_hc = w_i[h], b_i[h]` (`:15048`), and
  `t_i[h], t_i[c] = t_c, t_h` / `beta[...]` (`:15103`). Pairs must become
  cell-level `(t_hot, t_cold, w, b)` + a carrier row (or -1).
- **Disjointness guard.** Parity on `t_cold` (`:14939`) keeps pairs
  disjoint even on a full ladder, so admitting empty rungs does not break
  the contract *by itself*. But if the pair generator enumerates rungs
  independently of `carrier`, a rung that is both picked and separately
  enumerated appears twice, `src` gets a duplicate, and
  `searchsorted(..., side="left")` (`gbbands.py:6198`) silently maps that
  cell's rows to one arbitrary destination -- a half-applied relabel.
  `GB_INDEX_ASSERTS` is OFF in production (`gbbands.py:6202`) and the
  block-end `special_index_check` (`:17204`) can pass anyway. Make
  `carrier[rung]` the single source of truth and add a host-side
  `len(unique(src)) == len(src)` guard once per sweep.

## Correctness notes

For a sole-occupant cold cell vs an empty hot rung:
`paccept = -(b_cold - b_hot) * ll_ref_cold`, so a harmful leaf
(`ll_ref < 0`) is pushed hot and a good one stays cold. For a multi-leaf
cell it generalizes to minus the cell's whole add-log-likelihood against
the bare slab -- the cell is evicted iff its entire model is net-harmful.
That is the intended physics, and it is the eviction route recorded as
missing in `project_sole_occupant_swap_gap`.

Detailed balance holds: this is symmetric-proposal MH on
`prod_t exp(beta_t L_t)`, and swapping a whole cell configuration between
rungs is the same move whether or not one side is empty. Condition: the
nleaves / model-dimension prior must be temperature-independent (eryn
tempers the likelihood only). The leaf cap is a proposal veto, not a prior
(`:15003`), and is disarmed in PE, so it cancels.

One genuine new systematic: a picked row's `L_with` drifts with sig-het
error over a block, and against a FIXED exact cached constant that drift no
longer partly cancels as it does between two picked rows. Est. <=1 nat at
the cold rung over a 25-repeat mature block, <=10 over a 200-repeat newborn
block. Worth a diagnostic, not a blocker.

## Build order

1. Finalize brackets -> transient measurement passes -> bind -> open.
2. Widen `begin_cell_label_window`.
3. `_vertical_pairs` -> cell-level `(t_hot, t_cold, w, b)` + carrier rows,
   parity selection unchanged.
4. Sweep: build `L` from `carrier`; guard the row-indexed writes; one-sided
   `spec` re-point when a partner is non-resident.
5. Skip empty↔empty; restore the ladder contribution analytically.
6. Permute `cached[]` alongside `ll_change_log` on accept.

**Gate the whole thing behind an env knob, default OFF.** Both silent
corruption routes (hazard 1, hazard 3) are invisible in aggregate logs. The
instrument that catches them already exists: `[GB_CELL_LL]`'s
sampled-vs-realized reconciliation (`_cell_ll_report`, `:7678`). Require a
clean `[GB_CELL_LL]` line on a real run before making it default.

## Corrections to earlier claims in this work

- "`gbbands.py:3756/3765/3808/3818`" in the `_vertical_pairs` docstring was
  stale; the walker-addressed fill maps are at `5429/5438/5481/5491`. Fixed.
- "Empty" and "not a scheduler cell" are NOT the same set: an RJ subset
  includes DEAD rows whose band comes from a drawn `f0`, so many cells with
  no alive source ARE scheduler cells.
- The non-direct grouped path (`:8417`) does NOT rebind for the flush --
  it runs against the scheduler's staged residency, so unpicked cells ARE
  resident there. Any fix must not assume non-residency.
- `gbbands.py:6522` (`add_sources_to_band_buffer`) is not guarded for zero
  rows the way the fill at `:6493` is; `run_tempering` always routes empties
  through `fill_slots`, so the zero-row path may be untested. Check before
  binding a truly empty cell.
