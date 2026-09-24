# Convergence-driven in-model polish for freshly-born GB sources

**Status:** implemented, untested on hardware · **Date:** 2026-09-24 ·
**Branch:** `gb-inmodel-converge` (worktree
`LISAanalysistools-inmodel-converge`, uncommitted) · **Reviewers:** two
independent adversarial passes, both incorporated (see *What the reviews
changed*).

## The ask (user, 2026-09-24)

> During "extreme_search_mode", when an rj is birthed (we already separate
> this group for 100 steps rather than 50), can we run them through in-model
> moves until their subbands' logL (that they are tracking as the final added
> source to the residual) converges. Similar to the other logL convergence
> checks. Within some logL difference over some iterations. … We should be
> able to slot in and out sources that finish so we do not have to wait for
> the whole group to finish to change them out. … Make sure you do this by
> band and walker individually, but remember all temperatures have to travel
> together for the vertical swaps within a band.

Follow-ups:

> it should be like it does not increase for maybe at least 50 iterations.
> Something larger since these are non-thinned at all and they may only step
> a little bit.

> convergence is determined for each individual temperatures set per (band,
> walker). Those move on and off from the search together. … You can turn
> them off after they converged but cannot be removed from the active
> computing area until all temperatures have converged to a logL (this is
> all within one in-model repeat block within one propose)

> do the rj --> run inmodel until each converge in logL (just like we do now
> but more than 100 steps and the steps are unique to each walking source)

> Let's make sure the upper half of temperatures do not hold us up, so once
> the lower have converged we can move on.

> "Your swap-in/swap-out is implemented as the freeze, not as a refill" —
> when a full band is done, we should swap it out for a new band, not wait
> for the other bands to converge.

> Make sure it is just an option to be added.

## What exists today

"extreme_search_mode" = the `GB_MODE=search` campaign. The 100-vs-50 split is
`GB_INMODEL_REPEATS_NEWBORN=100` / `GB_INMODEL_REPEATS_SURVIVOR=50`
(`submit_gf_6mo_v8.sh:1036,1052`) → `self.inmodel_repeats_newborn` /
`_survivor` (user ruling 2026-08-15). The consumer is the **direct-batch**
grouped path in `GBSpecialBase._run_band_unit`: every RJ batch completes,
the survivor pool splits by pick-time provenance (`_split_by_newborn`), and
each class runs fixed-width chunks through `_run_in_model_repeats` with a
**fixed** budget.

## The design, in one paragraph

The structure is unchanged — RJ, then one in-model block per chunk, one
setup per block. The only change is that the block's budget becomes a
**ceiling** and each ROW stops when its own likelihood stops climbing. A
converged row is dropped from the per-half row sets, so it costs nothing for
the rest of the block; it stays resident, so its column keeps every rung
co-located for the vertical swaps; and the block ends when every row is
frozen. Steps are therefore unique per source and may exceed 100.

### The statistic: per-row accumulated gain

Per row (one `(temp, walker, band)` cell):

```
G_row    = running sum of ACCEPTED delta_ll     # lnL gained in this block
best     = max(best, G_row)                     # monotone
converged when  seen >= W  and  best - best_at(seen - W) <= thresh
                                                # thresh = D/2 = 4.0, W = 50
```

`best_at(seen - W)` comes from a `(W, n_rows)` ring buffer: one gather and
one scatter per repeat.

`G_row` equals `L_with(i) - L_with(0)` exactly — every accepted move changes
the cell's whole-model likelihood by precisely its `delta_ll`, a rejected one
changes nothing — so it **is** "the subband's logL tracking the newly added
source", up to a per-row constant. Using the gain rather than the likelihood
itself is what makes it safe:

* `ll_ref` is a **sig-het** value whose absolute offset from exact is
  1–17 lnL on cold rungs — *above* a 4.0 threshold. That offset cancels in a
  delta and only in a delta (`_anchor_err` exists to report exactly this).
* Production re-anchors the reference **mid-block** (`GB_SIGHET_REFRESH_EVERY=25`,
  `..._DPHASE=0` ⇒ every source, every rung) and re-bases `ll_ref`. An
  absolute statistic takes a discontinuity there every 25 repeats — the same
  cadence as the patience window.

The gain is immune to both, is unaffected by a vertical swap (a pure relabel:
`_vertical_swap_sweep` mutates `t_i`/`beta` only, never `ll_ref`, `slots` or
`curr`), and is **free** — it is the same `where(accept, delta_ll, 0)`
product `ll_change_log` already accumulates per cell, kept per row because a
swap exchanges the per-cell ledgers while rows keep their own coordinates.

### ⚠ The cap gate's literal form does NOT port to this clock

```
improved = L > best + thresh ; best = max(best, L)
patience = 0 if improved else patience + 1 ; converged at patience >= W
```

`_update_band_leaf_caps` resets patience whenever a single tick beats
`best + D/2`. Its tick is a whole sampler iteration, so a steadily improving
cell posts > D/2 per tick. A tick here is one unthinned MH step, which turns
the same code into a **per-step** test that retires a row mid-climb. Measured
on synthetic traces (300 trials, 30 lnL of climb over `climb` repeats then
σ=1.5 noise, D/2 = 4.0; stop repeat / lnL left unclaimed):

| climb | cap-gate form, W=25 | ring-buffer form, W=25 |
|---|---|---|
| 5 | 30 / 0.00 | 30 / 0.00 |
| 25 | 11 / **17.50** | 34 / **0.00** |
| 60 | 26 / **17.29** | 81 / **0.00** |

`tests/test_inmodel_converge.py::test_the_cap_gate_form_would_retire_mid_climb`
pins the disagreement so a future refactor cannot quietly lose the ring.

### Why W = 50

Read `(thresh, W)` as a **rate**: a row is retired once its best improves by
less than `thresh / W` lnL per repeat. D/2 over 50 is 0.08 lnL/repeat.

| climb | W = 25 | W = 50 | W = 100 |
|---|---|---|---|
| 5 | 30 / 0.00 | 55 / 0.00 | 105 / 0.00 |
| 25 | 49 / 0.00 | 75 / 0.00 | 125 / 0.00 |
| 60 | 81 / 0.00 | 107 / 0.00 | 158 / 0.00 |
| 150 | 38 / **22.55** | 188 / 0.00 | 241 / 0.00 |
| 300 | 27 / **27.39** | 76 / **22.47** | 375 / 0.00 |

W = 50 resolves climbs out to ~150 repeats with nothing left on the table
(W = 25 gives up at 38), and its **floor** — what a row that converged
instantly still pays — is `W + 1` ≈ 51, under the fixed newborn budget of
100, so the mode still pays on easy rows.

**The shipped default is W = 100** (user ruling, raised from 50 later the
same day). The rate is then 0.04 lnL/repeat, which tracks climbs out to
~300 repeats where W = 50 gave up at 76. ⚠ **At W = 100 the mode is a SPEND,
not a saving**: the floor is 101, already above the stock newborn budget of
100, so every row costs at least what it costs today and the slow tail costs
up to the ceiling (4×). That is the deliberate trade — sources finish
climbing instead of being cut off at a fixed 100 — and the `[GB_IMCONV]`
work ratio prices it per phase. W = 50 is the setting that makes it a saving
instead.

### The ladder gate: the hot half does not get a vote

Only the coldest `ceil(ntemps * gate_frac)` rungs — half by default — are
tested, frozen and allowed to hold the block open. Hotter rungs keep
sampling for the life of the block and are recorded **released**, never
converged or capped.

Two reasons, and the first is why the gate is not optional in practice. A hot
chain accepts freely, so its gain **random-walks** instead of climbing, and
the running max of a random walk keeps creeping (record statistics) — over a
50-repeat window that creep clears D/2 easily. The hot rungs would therefore
be the last to "converge" while being the rungs whose convergence means
least, and every cold source that was genuinely done would wait on them.
Second, they are not frozen either, because they are the transport that
feeds good states down to the cold rungs still working — and that is nearly
free, since the gate is what ends the block early in the first place.

The mask is evaluated on the **live `t_i` at each poll**, not once at block
start: a vertical swap moves rows between rungs, so the gate is a property of
the rung, not of the row. This is the one place where evaluating after the
swap sweep is load-bearing. A pool whose columns kept no cold rung falls back
to gating every row, so the gate can only ever shorten a block that has a
cold rung to shorten it on.

### Column refill: a finished band leaves, a queued one takes its slots

`_converge_refill_loop` retires whole `(walker, band)` columns and refills
the freed width from the pool. A generation ends when its block does; retired
columns drop out, unretired ones **carry with their full state** (gain,
running best, ring buffer, clock), new columns come off the queue. Whole
columns are the unit, so every rung of a retiring band leaves together and
every rung of an arriving one arrives together — the vertical swaps keep
their partners on both sides.

**The trigger is columns, not rows, and the default is 0.5, not 1.0.** Both
follow from what the swap-out is for. At `stop_frac = 1.0` the block runs
until every column is done, so they all retire together, nothing carries, and
this loop degenerates into exactly the fixed chunk loop it replaced — 1.0
silently disables the refill. And counting *rows* rather than columns can hit
the threshold with the frozen rows spread thinly and no band actually
complete, at which point the progress rule below would cap columns that only
needed more time. Counting finished columns is both what "a full band is
done" means and what freeing buffer capacity is measured in.

**Progress is mandatory.** If a generation retires no column, those columns
are capped rather than carried: carrying the identical active set would
re-pay the setup for the same work and, with the width already full, starve
the queue forever. That bounds the loop at one generation per column plus
one. (A block can legitimately retire nothing — it exits on `stop_frac` or on
the repeat ceiling, and with `stop_frac < 1` the frozen rows can be spread so
that no single column completes.)

**The cost is real and is reported.** Each generation re-pays one block setup
≈ 73 in-model repeats. `GB_INMODEL_CONVERGE_REFILL=0` turns the loop off and
keeps every column in its block for that block's life, which is strictly
cheaper in setups and strictly worse in slot occupancy; the `[GB_IMCONV]`
line prints the generation count and the work ratio so the two can be A/B'd
on a real run. The optimization that would cut the price is
`_cached_get_buffer`'s existing `fill_slots` argument — refill only the slots
that actually changed — but it gates the residual/PSD copy **without** gating
the source re-injection, so a carried slot would be injected twice. Not
attempted.

Cell labels move under the vertical swaps, so the pool's cached `specials` /
`temp_inds` are re-read from the sorter before every generation. Both the
rebind and `get_index` resolve through those labels, and a stale one binds a
row to another cell's slab with no error anywhere.

### Freeze, and "all temperatures travel together"

A converged row is dropped from `_build_half_pre`'s row sets at the next
poll. It then makes no proposal, no prior evaluation, no `get_add_ll` and no
accept. It is **not** removed from the buffer: its slot stays bound, its
column keeps every rung co-resident, and `_vertical_swap_sweep` still pairs
it — which is exactly the user's "you can turn them off after they converged
but cannot be removed from the active computing area". Because nothing is
removed mid-block, the column-level rule is satisfied structurally: the whole
block, and therefore every column in it, leaves together when the last row
freezes.

Detection is exact per repeat (device-side latch); only *acting* on it costs
a host sync, so the poll is amortized over `_CONVERGE_POLL_EVERY = 5`
repeats. A row therefore proposes for at most 4 repeats past its own
convergence — asserted in
`test_frozen_rows_stop_proposing_entirely`.

### Two behaviour-preserving guard changes the freeze required

Both were `⟺` equivalences that the freeze breaks, and both would have failed
**silently**:

* `if sub is not None:` (sync the sorter coords for the stretch complement)
  → `if self.sequential_parity_repeats:`. The freeze makes `sub` an index
  array on the single-sweep path too; syncing there would change what the
  group-stretch friend table reads, i.e. change the proposal.
* `and sub is halves[-1]` (sig-het refresh once per repeat) → `and
  _h_i == len(_half_pre) - 1`. The rebuilt `_half_pre` holds new arrays, so
  the identity test would never fire and the reference would stop refreshing.

### Scope gates

* **Stage.** A convergence-plateau stop is *optional stopping* on the chain's
  own lnL trajectory: allowed in search
  (`feedback_search_no_detailed_balance`), banned in PE
  (`feedback_no_pe_maximization`). `is_rj_prop` is not a sufficient gate —
  `rj_fstat_pe` / `rj_prior_pe` / `rj_replace_pe` are all `is_rj_prop` on the
  same path. `_converge_stage_allows` is a **deny-list** on `*_pe` names
  rather than an allow-list on `"search"`, because campaigns run
  `GB_MODE=search` through moves that are not search-named; every move that
  arms is named in the `[GB_IMCONV … armed]` log line.
* **Class.** Newborns only by default (`…_CLASSES`).
* **Path.** The state is created per class per unit and referenced only by
  that local: nothing on `self`, so nothing leaks across proposes or units,
  nothing is live at deepcopy/pickle time, and nothing survives a resume
  (GB has no mid-iteration checkpoint hooks).
* **Column-atomic staging** is forced whenever the mode is on, not just under
  `GB_TEMPER_VERTICAL=1`: otherwise `GB_INMODEL_SETUP_BATCH=2048` splits a
  column across sub-blocks and its rungs stop at different repeats.

## Knobs (every one defaults to today's behaviour)

`{BRANCH}` is the move's branch (`GB_…`, `VGB_…`). Precedence is the house
rule: explicit kwarg > env > default.

| Env | default | meaning |
|---|---|---|
| `{BRANCH}_INMODEL_CONVERGE` | `off` | `off` / `observe` / `on` (`0`/`1` accepted) |
| `{BRANCH}_INMODEL_CONVERGE_ITERS` | `100` | W, the window (user ruling; see the floor caveat) |
| `{BRANCH}_INMODEL_CONVERGE_DLL` | `0.5 * leaf_cap_ndim` = 4.0 | the threshold |
| `{BRANCH}_INMODEL_CONVERGE_MAX` | `0` → 4× the class budget | per-row ceiling |
| `{BRANCH}_INMODEL_CONVERGE_STOP_FRAC` | `0.5` | end the block once this fraction of its COLUMNS has fully retired — the swap trigger |
| `{BRANCH}_INMODEL_CONVERGE_GATE_FRAC` | `0.5` | the coldest fraction of the ladder that gets a vote; `1.0` = every rung must converge |
| `{BRANCH}_INMODEL_CONVERGE_REFILL` | `1` | retire finished columns and refill their slots from the pool |
| `{BRANCH}_INMODEL_CONVERGE_CLASSES` | `newborn` | `newborn`, `mature`, or both |

**`observe` is the intended first step.** It runs the entire rule under the
unchanged fixed budget — no freezing, no early exit, not one proposal
different — and logs the repeat distribution the rule *would* have produced.
It is the paired negative control (`feedback_paired_negative_controls`) and
it answers the question that decides whether arming is worth anything: if
every row would retire at exactly W, the rule is a constant budget of W and
`GB_INMODEL_REPEATS_NEWBORN=50` says the same thing for free.

## Diagnostics

Three `[GB_IMCONV]` lines per class per in-model phase, plus one `armed` line
per move:

```
[GB_IMCONV rj_fstat_search] newborn: 2418 row(s) in 412 column(s) over 3
generation(s) -- converged 2390 (98.8%), at the 400-repeat ceiling 28
(1.2%), released unjudged (hot rungs) 1204 (49.8%); columns fully retired
401/412; work = 1.43x a fixed 100-repeat budget, plus 2 extra block
setup(s) from the refill (window 100, dll 4.00 = 0.040 lnL/repeat,
floor 101).

[GB_IMCONV rj_fstat_search] newborn calibration: repeats
p10/p50/p90/max = 101/137/284/400, AT FLOOR (<=101) 312 (12.9%); lnL
gained per row p10/p50/p90 = 0.41/6.80/41.2 (max 310.5), rows gaining
< dll(4.00) = 705 (29.1%); hist 1-100:0 101-150:1440 151-200:512 ...

[GB_IMCONV rj_fstat_search] newborn by rung: GATED 1214 row(s) repeats
p50 137, lnL p50 9.10, converged 98.8% | RELEASED 1204 row(s) repeats
p50 137, lnL p50 2.31 (never judged).
```

**Line 2 is the one that decides whether the rule means anything**, and it
exists because the repeat percentiles alone cannot tell two very different
outcomes apart:

* everything at the floor with ~zero gain = the rule never fired on
  evidence, it just ran the window out. That is a constant budget of W, and
  `GB_INMODEL_REPEATS_NEWBORN=W` buys it for free. A newborn born at an
  F-stat peak under phase maximization is *already* at its ML point, which is
  exactly how this arises — the same shape as the leaf cap's
  ghost-increment defect.
* a broad spread with gains of order D/2 and up = rows genuinely climbing at
  their own rates, which is the point.

So read `AT FLOOR` and `rows gaining < dll` first. If `AT FLOOR` is near
100%, stop and re-derive the threshold from a measured `delta_ll`
distribution rather than the cap gate's per-*iteration* D/2.

Line 3 splits gated (cold) from released (hot), because the ladder gate
judges them by different rules and a pooled number hides which is paying.

## What the reviews changed

Two independent adversarial reviews ran against the first draft. The first
draft proposed a *streaming driver*: converged columns swapped out at
generation boundaries, freed slots refilled from the pending pool by
rebinding the buffer. Both reviews rejected it, and the second one settled it
with a measurement:

> **One block setup costs ~73 in-model repeats.** From
> `gf3mo_v7_349.log` `[GB_TIMING rj_fstat_search]`, accounting closing to
> 132.06 s: setup/block = (cholesky 39.328 + sighet_setup 6.195 + removal
> 3.279 + addback 3.903 + ll_ref 0.522)/32 = 1.663 s, against 22.74 ms per
> repeat. Corroborated at 33 repeats on the friendliest 3mo v8 reading, and
> structurally unavoidable: the info matrix is ~160 lnL evaluations per
> source. Plus `buffer_build` 18.3 s/propose, which sits *outside* the
> `inmodel_repeats` span.

Break-even for a `g`-generation scheme is `(g−1)·s ≤ B − m`; at the draft's
own numbers that is `g ≤ 1.53` against a worked example of 6 generations —
i.e. **net slower**. Freezing gets the same intent for nothing: the block
keeps its single setup, and total work becomes `Σ_rows (repeats until
frozen)` instead of `rows × B`, which beats the fixed budget whenever the
mean is under the budget, with no break-even condition at all.

**The refill was then reinstated on user ruling** (2026-09-24, after this
concern was put to them): *"when a full band is done, we should swap it out
for a new band, not wait for the other bands to converge."* It ships behind
`..._REFILL`, default on with the mode, with the 73-repeat price documented
at the call site and reported per run. Note that the arithmetic above is
kinder now than it was for the draft: the ladder gate plus the freeze make
each block much shorter, so the carried fraction — and therefore the extra
generation count — is smaller than the 6 the draft assumed. It is still the
one part of this feature that can plausibly be net-negative, and it is the
one part with a single-knob off switch.

Other blockers, all fixed by the redesign or explicitly:

| Finding | Resolution |
|---|---|
| Carried-over `specials` / `temp_inds` go stale after a swap → wrong slot on rebind | dissolved: no carry-over, no extra rebind |
| `L_with` is an absolute sig-het value; offset 1–17 lnL cold, re-anchored every 25 repeats | statistic switched to accumulated accepted `delta_ll` |
| `GB_INMODEL_SETUP_BATCH` splits columns when `GB_TEMPER_VERTICAL=0` | column-atomic staging forced by the mode |
| A column whose rows all die can never retire → driver hangs | dissolved with the driver |
| `max` over 24 tempering rungs is not "the subband's logL" | per-row test (also the user's own ruling) |
| `direct and is_rj_prop` arms the mode in PE | `_converge_stage_allows` deny-list + warn-once |
| The freeze was deferred to "v2" on a bit-exactness burden that does not apply | the freeze IS v1 |
| Hot rungs would be the stragglers that never retire | the ladder gate (user ruling; reviewer 2 raised the same point about max-over-24-rungs) |
| No measurement step | `observe` mode |
| `n_rep` log lines wrong on early exit | all four use the executed count |

## Testing

`tests/test_inmodel_converge.py`, 35 tests, fake-based, no GPU (~0.1 s):
knob resolution and validation; the rule (flat / above-rate / below-rate /
late spike / latching / independence) plus the cap-gate-form regression
guard; state persistence and the ring-buffer carry-over; the column-AND
retirement rule; and **end-to-end through the real `_run_in_model_repeats`**
on the numpy harness — off is bit-identical and leaves no state on the move,
the freeze shrinks the per-repeat work, frozen rows stop proposing within one
poll, `observe` is bit-identical to the fixed budget while still filling in
the bookkeeping, the ceiling bounds a never-converging block, `stop_frac`
trims the tail, and the whole thing survives `GB_TEMPER_VERTICAL=1` against
`test_vertical_swap`'s real-semantics sorter fake.

Regression: every other test module was run on this branch and on clean
`dev`. The counts are identical (`test_buffer_fixed_capacity`,
`test_fstat_gridfit`, `test_fstat_parallel_fit`: 16F+10E both sides;
`test_gb_fdot_astro`, `test_gb_observable_basis_wiring`,
`test_global_fit_signal_gen_mojito`, `test_maxlogl_plateau`,
`test_nogb_null_composition`: 14F+23E both sides;
`test_submit_scripts_layout`, `test_vgb_observable_basis`: 2F+9E both
sides) — **all pre-existing, none introduced.** `test_gbspecial_flow` is
excluded: it balloons to 10–26 GB and SIGKILLs an 8 GB laptop
(`project_test_gbspecial_flow_balloon`); it needs a cluster run.

## Not done / known risks

* **Never run on hardware.** Every number above is synthetic or from an
  existing production log. Run `observe` first.
* **Refill cost.** Each generation re-pays a full block setup (~73 repeats);
  `fill_slots` would cut it but double-injects carried slots as written. A/B
  `..._REFILL=0` against the default on a real run before trusting either.
* **Released hot rungs.** With the gate on, roughly half the ladder is
  released unjudged when the block ends. Those rungs got however many repeats
  the cold half needed — fewer than the fixed budget when convergence is
  fast. If the hot ladder turns out to need that sampling, raise
  `..._GATE_FRAC` (1.0 restores "every rung must converge").
* **The tail is not free.** A repeat is ~99% host launch overhead, so a
  100-repeat tail with 5 live rows still costs most of 100 repeats.
  `stop_frac` is the control; the report's p90/max columns are how to see it.
* **No engagement latch.** The cap gate has one (`_cap_ll_improved_once`)
  because an empty cell can never improve. Every row here has a source by
  construction, so the analogous failure is different: a newborn born at an
  F-stat peak under phase maximization may already be at its ML point and
  retire at exactly W. That is arguably correct (it *has* converged), but if
  `observe` shows the whole population retiring at W, the rule is degenerate
  and the threshold needs re-deriving from a measured per-repeat `delta_ll`
  distribution rather than borrowed from the cap gate's per-iteration D/2.
* **Multi-rank load imbalance.** No deadlock risk (GB's fan-out is
  point-to-point; there are no collectives inside `_run_band_unit`), but
  `_ledger_exchange`'s per-unit `allgather` is a barrier, and data-dependent
  block lengths mean a rank holding one hard band makes its peers wait.
  Watch per-rank propose times before trusting a 4-GPU speedup number.
