# Per-(walker, band) GB search-stage schedule

**Status:** implemented 2026-09-24 on `gb-cap-per-walker` (to be rebased on
`dev` and merged together with the per-walker leaf-cap work).

**Related:** [`2026-09-22-gb-cap-per-walker-design.md`](2026-09-22-gb-cap-per-walker-design.md)
— this builds directly on the `(nwalkers, ncells)` band-state apparatus that
spec introduced, and reuses its accessor, mirror, resume and multi-rank
patterns verbatim.

## Intent

User request 2026-09-23:

> Recently we did walk on a per walker cap cell limit tracking and
> adjustment. Can we do the same for adjusting the search? Basically
> adjusting snr limit and phase maximization per band per walker. Start with
> snr limit 8 on the fstat search and rejection sampled opt snr with phase
> maximization. When that band on each walker reaches an equilibrium in terms
> of the number of sources, then you adjust it to snr limits of 5 for opt snr
> rejection sample, 6.25 for fstatistic snr limit in that band, and phase
> maximization off.

Follow-up 2026-09-24: **"I will adjust the limits and settings later, but
just make sure the capabilities are in place."** So this spec ships the
MECHANISM with every threshold as a named knob; the production values are the
user's to set.

Second follow-up 2026-09-24:

> Can you also allow for band shutoff per walker that is in addition to the
> band shutoff we have now (so that one stays). When a band's number of
> sources within its current recipe step converges [...] I want to shutoff
> that band/walker until the next recipe step begins. This is only during
> SEARCH.

and then, superseding the quantity:

> Rather than an nleaves limit to cause RJ shutoff, let's do the logL of the
> (band, walker). That logL has to converge to shut it off.

So there are TWO features here, separately switchable, sharing one
per-iteration statistic: the STAGE SCHEDULE (§1-§6) and the PER-WALKER RJ
VALVE (§6b).

The search currently runs one global SNR floor for every walker and every
band for the whole run. That is the wrong shape: a band where a walker has
already assembled its sources wants a LOWER floor (dig into the faint tail),
while a band still finding bright sources wants the HIGH floor (do not
poison the model with noise births). Those two states coexist at the same
iteration, in different bands, and in the SAME band on different walkers.

## Non-goals

- **Phase maximization is DEFERRED** (user instruction 2026-09-24). See
  "Why phase-max is not here" below; the latch this spec builds is the hard
  half, and a later change can consume it.
- No change to the F-stat comb, the sweep, the epoch cache layout or the
  reference-walker selection.
- No change to PE. Every knob here is scoped to the SEARCH cycle.

## Semantics

A **stage** is indexed by `(walker, band)` and applies at **every
temperature**, exactly as the per-walker leaf cap does. Stage `0` = COARSE
(the run's starting floors), stage `1` = FINE (the relaxed floors). The
latch is **one-way**: a band that has settled and then loses a source does
not climb back to the coarse floor, because the relaxed floor is a strictly
larger prior support and bouncing the support is how a sampler ends up with
states outside its own prior.

Tempering: the vertical sweep shares a walker index, so both sides carry the
same floor and that path is unchanged. Only `_permute_walkers_for_swaps`
(cross-walker) can see asymmetric floors, and it is gated — the same
treatment and the same call site as `tempering_swap_cap_ok`.

## The equilibrium signal

"Reaches an equilibrium in terms of the number of sources" is taken
literally: the cold-chain SOURCE COUNT in that band, for that walker, has
stopped changing.

This is deliberately NOT the cap gate's `converged` mask. That mask means
"the lnL has plateaued AND the walker is pressed against its allowance",
i.e. *the band wants more room* — close to the opposite of settled. The
count-plateau signal already has a proven implementation in this file: the
band-shutoff valve's `band_occ_streak` / `band_occ_last` pair
(`_update_band_shutoff`), which counts consecutive iterations of UNCHANGED
occupancy. This spec is that mechanism with a walker axis.

The statistic is `band_counts[0]`, shape `(nwalkers, num_bands)` — the
cold-chain per-band census the cap gate already computes every iteration
(`_cold_occupancy`'s `_cap_is_band_grid` branch returns exactly it). No new
residual work, no new kernel call.

## 1. Flags and settings

| env | `GBSettings` field | default | meaning |
|---|---|---|---|
| `GB_SEARCH_STAGE_PER_WALKER` | `search_stage_per_walker` | `False` | master switch |
| `GB_SEARCH_STAGE_MIN_ITERS` | `search_stage_min_iters` | `20` | consecutive unchanged-count iterations required |
| `GB_OPT_SNR_LIMIT_SEARCH_COARSE` | `opt_snr_limit_search_coarse` | falls back to `GB_OPT_SNR_LIMIT_SEARCH` (8.0) | stage-0 opt-SNR floor |
| `GB_OPT_SNR_LIMIT_SEARCH_FINE` | `opt_snr_limit_search_fine` | `5.0` | stage-1 opt-SNR floor |
| `FSTAT_PEAK_MIN_SNR_COARSE` | — | falls back to `FSTAT_PEAK_MIN_SNR` (8.0) | stage-0 F-stat peak floor |
| `FSTAT_PEAK_MIN_SNR_FINE` | — | `6.25` | stage-1 F-stat peak floor |
| `GB_SEARCH_BAND_SHUTOFF_PER_WALKER` | `search_shutoff_per_walker` | `False` | the per-walker RJ valve (§6b), SEARCH only |
| `GB_SEARCH_BAND_SHUTOFF_CONV_ITER` | `search_shutoff_conv_iter` | `5` | consecutive non-improving iterations before a pair freezes |
| `GB_SEARCH_BAND_SHUTOFF_LL_TOL` | — | `leaf_cap_ndim / 2` (4.0) | lnL improvement that keeps a valve open |
| `GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC` | — | `1` | a band must hold a source before its valve can close |
| `GB_SEARCH_STAGE_DIAG` | — | `0` | per-transition decision tuple |

Derived predicate on the move, the single switch every new branch tests:

```python
@property
def _stage_per_walker(self) -> bool:
    return bool(self.search_stage_per_walker) and self.mode_is_search
```

**Default OFF is a hard guarantee, not a convention**: zero arrays
allocated, zero datasets written, zero new branches, and
`opt_snr_rej_samp_limit` stays the scalar float it is today. A 6mo or 1yr
run can relaunch from a build carrying this with no behavioral change.

## 2. Storage

`ensure_search_stage_fields(band_info, num_bands, per_walker=)` allocates
only when `per_walker` is true:

| key | shape | fill | dtype |
|---|---|---|---|
| `band_stage_w` | `(nwalkers, nbands)` | `0` | int8 |
| `band_stage_occ_last_w` | `(nwalkers, nbands)` | `-1` | int64 |
| `band_stage_streak_w` | `(nwalkers, nbands)` | `0` | int64 |

`-1` for `occ_last` is the same unreachable sentinel `_zero_band_shutoff`
uses, so the first update always starts a fresh streak at 1 rather than
crediting a spurious match against a zero-filled array.

ALL-OR-NOTHING restore, modeled on `BAND_SHUTOFF_FIELDS`: the three arrays
are one consistent record. A partial set, or one sized to a different band
grid, is discarded whole and restarted from zero rather than replayed onto a
grid it was not measured on. That is the conservative direction — every
walker restarts COARSE, which is the tighter prior.

Registration (`globalfit/state.py`), mirroring the cap family:
- `GBState._bare_ndim`: all three at bare ndim `2`.
- `_expected_shapes`: `(nwalkers, nbands)` for all three, so a store whose
  walker count changed is refused loudly.
- **NOT** in `legacy_dtype_names` — these are integer counters and a stage
  index; coercing them to the backend float dtype (what that list is for)
  would be wrong. Same reasoning as the shutoff family.
- Cap-free branch (VGB) drop list: add all three.

### Mirror

`band_stage` `(nbands,)` int8 is written as the **min over walkers** — the
monitor and any diagnostic that wants one number per band gets the
conservative (coarsest) answer, and a band only reads as FINE once every
walker agrees. Like the cap's 1-D mirrors this is a summary view, never the
latch's state.

## 3. The latch (`_update_search_stages`)

Runs once per iteration from the designated cap-updating move, immediately
after `_update_band_leaf_caps` (same call sites, same `leaf_cap_update`
ownership rule — a second updater would double-advance the streak).

```
occ      = band_counts[0]                      # (nwalkers, nbands) cold census
same     = (occ == occ_last) & (occ_last >= 0)
streak   = where(same, streak + 1, 0)
occ_last = occ
promote  = (stage == 0) & (streak >= min_iters) & (occ > 0)
stage[promote] = 1
```

Three deliberate choices:

- **`occ > 0` is required to promote.** An EMPTY band's count is trivially
  unchanged forever, so without this every empty band promotes itself on a
  fixed clock — precisely the ghost-increment failure the cap gate's
  engagement latch was written to stop (and which measured 920 of 1232 cells
  incrementing in lockstep on the 3-month run). An empty band has found
  nothing, so it has reached no equilibrium.
- **The streak resets to 0, not to 1, on a change.** A band that just gained
  a source must post `min_iters` full quiet iterations, not `min_iters - 1`.
- **In-memory nothing.** Unlike the cap gate's engagement latch, the whole
  record persists. A restart mid-run must not silently re-tighten the floor
  under an assembled model: that would start rejecting sources the model
  already holds, and the in-model SNR gate enforces on every update, not
  only on birth.

## 4. Enforcement — the opt-SNR floor

`opt_snr_rej_samp_limit` is documented as the **GB SNR PRIOR BOUNDARY**: any
proposed state (RJ birth, replace NEW side, in-model update) whose optimal
SNR falls below it is force-rejected. It is a scalar float compared
elementwise against an `opt_snr` array at four sites, every one of which
already has the row's `(walker, band)` in scope.

A single accessor is the only place the two layouts are distinguished,
copied from `_cap_for_rows` and keyed on `ndim` for the same reason (a
device-side snapshot must carry its own answer):

```python
@staticmethod
def _snr_lim_for_rows(lim, walker_inds, bands):
    """Per-row SNR floor: ``lim`` (scalar) or ``lim[walker_inds, bands]``."""
    if np.ndim(lim) == 0:
        return lim
    return lim[walker_inds, bands]
```

| site | today | change |
|---|---|---|
| `_run_rj_step` birth gate (`gbspecialstretch` ~9678) | `_lim = buffer_obj.opt_snr_rej_samp_limit` | `_snr_lim_for_rows(..., w_rows, b_rows)` |
| `_run_replace_step` NEW side (~10578) | same | same |
| in-model repeat gate (~14584) | same | same |
| `SubBandBuffer.get_swap_ll` (`gbbands` ~4874) | same | same |

The table reaches the buffer the same way the scalar does today: threaded
through `BandSorter(opt_snr_rej_samp_limit=...)` into `SubBandBuffer`, one
source of truth. Nothing about the comparison changes — it is
Python/CuPy elementwise either way, so **there is no kernel change**.

Two sites need more than a re-index:

- **`GB_INMODEL_ACCEPT_KERNEL`** (default off) takes the floor as a raw
  scalar in its pointer ABI and would read walker 0 for everyone. It
  **stands down** with a one-time warning while the stage table is live,
  exactly as it does under `GB_LEAF_CAP_PER_WALKER`.
- **`tempering_swap_snr_ok`** is new, and is the `tempering_swap_cap_ok`
  story again: the vertical sweep shares a walker so both sides see the same
  floor, but the permuted path can move a state into a walker whose floor it
  violates. Composed into the existing swap gate with `&`, so it can only
  ever REFUSE a swap, never permit one the cap gate refused.

## 5. Enforcement — the F-stat peak floor

Two separable problems, and only one of them is cheap.

**Per-band: nearly free.** `select_comb_peaks` already computes
`band_of_node` for the per-band cap. `min_F` becomes an optional
`(num_bands,)` array and the scalar comparison becomes a gather:

```python
min_F_row = min_F if np.ndim(min_F) == 0 else min_F[band_of_node]
cand = (tier1 | local3) & (F_max >= min_F_row) & interior
```

The comb is untouched. Only stage B fits more peaks in the loosened bands,
and `FSTAT_PEAKS_PER_BAND` (200) bounds the blast radius: in a band already
at the cap, loosening the floor changes nothing.

**Per-walker: structurally blocked, and worked around.** The search grid is
ONE comb swept against ONE reference walker's residual (the min-lnL cold
walker since `2995a631`), refit on a 40-iteration cadence and cached per
epoch on disk. Per-walker peak lists mean N comb sweeps, and the comb is the
expensive half.

The resolution: the catalog is shared, and a band's catalog floor is the
**min over walkers** of that band's stage floor — so a band where ANY walker
has settled gets the looser catalog, and the per-walker half of the schedule
is carried entirely by the opt-SNR boundary of §4. This is the right
direction under the standing recall-over-precision ruling (2026-09-23: "err
on the side of grabbing things not missing them"): a peak a walker is not
ready for costs some proposal mass and dies in its own SNR gate, while a
peak that is not in the catalog is invisible to every walker.

⚠ **The cache stamp.** `_check_cached_peak_threshold` refuses a stacked
cache whose `peak_min_F` stamp disagrees with the knob. With a per-band
vector the stamp becomes the vector; a scalar stamp from before this change
still compares against the scalar knob exactly as today. What must NOT
happen is stamping a per-walker value — the cache has no walker axis and a
resume would refuse its own cache every time the stage table moved.

## 6b. The per-(walker, band) RJ shutoff valve

ADDITIONAL to the existing per-band valve (`_update_band_shutoff` /
`band_rj_shutoff`), which is **untouched**. The two compose with OR: a row is
frozen if either says so, and neither writes the other's state. The names are
deliberately the `_w` twins of the existing ones.

**Criterion — the lnL, not the count** (user ruling, superseding the first
form). Per (walker, band), scoped to the current recipe step: the band's
cold-chain residual lnL must fail to beat its running best within the step by
more than `_shutoff_ll_tol` for `search_shutoff_conv_iter` CONSECUTIVE
iterations. Any qualifying improvement zeroes the counter.

The tolerance defaults to `leaf_cap_ndim / 2` — D/2 = 4.0 for GBs, the lnL a
genuinely new D-parameter source has to buy — deliberately the SAME threshold
the leaf cap's own plateau gate uses, so the two features cannot disagree
about what counts as an improvement worth waiting for.

*Why the lnL and not the leaf count:* a count can sit still for many
iterations while the sampler is still materially improving the fit of the
sources already there — refining a blend, splitting a confused pair — and
freezing RJ in that band would end the search while it was still paying.

**Occupancy guard** (`GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC`, default on): an
EMPTY band's residual lnL does not improve either, so without it every empty
band freezes itself after one patience window — and an empty band is exactly
where an undiscovered source lives. This is the ghost-increment failure the
cap gate's engagement latch exists to stop, in valve form.

**The statistic** comes from the cap gate's stash, computed once per
iteration and stamped with the propose counter, so one residual pass serves
both features and a stale stash is detectable. If the cap gate did not run,
it is recomputed; if it cannot be obtained, the valve is inert. It is NEVER
read from `band_info['band_cold_ll']` as a fallback — on a sub-layer cap grid
that array holds the CELL statistic the gate wrote over it, and on an
iteration where the gate did not run it holds last iteration's values. Either
would look exactly like a converged band.

**Scope and release.** State is `band_rj_shutoff_w` `(nwalkers, nbands)` bool
plus `band_shutoff_w_step` `(1,)` int64 recording which step earned it; both
persisted. The running best and the streak are in-memory and per-step (a
restart re-earns them, leaving bands open longer — the direction that cannot
lose a source). `recipe.begin_search_recipe_step(moves, serial)` releases the
valve AND the window at each new step; `BaseRecipeStep.setup_run` and
`RJRecipeStep.setup_run` both call it. Clearing the window and not only the
boolean is load-bearing: a step inheriting the previous step's running-best
lnL would find nothing able to beat it and re-freeze immediately, which looks
exactly like "converged" and is not.

**Enforcement.** The RJ subset gate (`extra_bool`), FULL FREEZE — alive rows
leave the subset too, so the pair takes no births AND no deaths, the per-band
valve's own 2026-08-28 rule. Plus `_swap_shutoff_ok` inside `_swap_cap_ok`,
so a tempering swap cannot transport a band into or out of a frozen pair.

**Failure modes made loud** (this valve is load-bearing for the search
schedule, so none of them may pass quietly):

- flag on for the move but the state unallocated → **raises** at arming,
  naming `initialize_band_information(..., search_shutoff_per_walker=True)`;
- valve live but no recipe step ever announced → warns once, because nothing
  would ever RELEASE it and a valve that only closes freezes the search;
- census / lnL rows not matching the valve → raises.

**Stage-convergence interface** (agreed with the v9-recipe session
2026-09-24, treated as stable):

```python
recipe.band_shutoff_w_pending_total(moves) -> int   # 0 == fully converged
recipe.band_shutoff_w_armed(moves) -> bool
```

`pending` counts OCCUPIED pairs not yet shut. OCCUPIED is load-bearing: an
empty pair can never shut, so "every pair is shut" is unreachable in any run
with an empty band, i.e. every run. And `pending_total` returns 0 when the
feature is off, which reads as "converged" on the first check — hence
`band_shutoff_w_armed` as the guard, and hence composing with the
nleaves-plateau gate rather than replacing it.

## 6. Multi-rank fan-out

`_update_search_stages` runs head-only on the same concatenated
`band_counts` the cap gate consumes, so the latch needs no fan-out change
and inherits the cap gate's row-alignment guard.

The SHIP direction follows `cap_leaf_cap` exactly: the floor table is a
read-only head→rank slice `[w0:w1]` in `_payload_proposal(rank, w0, w1)`,
because a rank runs with `self.nwalkers = B` and computes block-local walker
indices. Several ranks holding the same walkers receiving the same rows is
correct.

## 7. Why phase-max is not here

`phase_maximize` is a scalar bool that reaches the engine as a kernel
argument, and a single `get_add_ll` / `get_swap_ll` call carries rows from
many walkers and many bands. Per-row support means either splitting every
scoring call in two (doubling launches on the hottest call in `propose`, and
breaking the f0-contiguity the sig-het reference blocks want) or a per-row
flag through GBGPU and the sig-het fused path — cross-repo.

It is also the least urgent third: production has had
`GB_RJ_PHASE_MAXIMIZE=0` throughout, so the schedule's phase-max half
changes nothing against today's baseline until it is turned on globally
first. The latch built here is the reusable half; when phase-max is picked
up, it consumes `band_stage_w` and adds nothing to this design.

## 8. Diagnostics

`[GB_STAGE]` lines, following `[GB_CAP_PW]`:

1. **arming** — mode, table shape, and `fresh` / `restored` / `reset(...)`,
   fired in BOTH modes so a flag that did not take is impossible to miss.
2. **transition** — per promotion: `w<N> band <b> [lo,hi] mHz: occ=<n>
   streak=<k> -> FINE (opt_snr <a>-><b>)`.
3. **spread** — every iteration, and says **"search stages are currently
   inert"** while no band has promoted. If that never turns over, the
   feature bought nothing, and that is the result.
4. `GB_SEARCH_STAGE_DIAG=1` adds the per-band decision tuple capped at
   `GB_SEARCH_STAGE_DIAG_MAX` (20).

## 9. Testing

`tests/test_gb_search_stage_schedule.py`:

- flag OFF allocates no array and persists exactly the historical key set
  (the hard no-op guarantee, the same two assertions the cap suite makes);
- the latch promotes after exactly `min_iters` unchanged iterations, not
  `min_iters - 1`;
- a count change resets the streak to 0;
- an EMPTY band never promotes, however long it sits still;
- promotion is one-way — a post-promotion count change does not demote;
- `_snr_lim_for_rows` gathers per (walker, band) and is bit-identical to the
  scalar expression when handed a scalar;
- the swap gate refuses a cross-walker swap into a walker whose floor the
  state violates, and permits the vertical (same-walker) sweep;
- `select_comb_peaks` with a per-band vector selects per band, and with a
  scalar is bit-identical to today;
- the VALVE: a flat lnL freezes after exactly `conv_iter` iterations; an
  improvement above D/2 resets the clock and one below it does not; the best
  is a RUNNING max so a dip does not reopen; an empty band never freezes
  (and the guard can be turned off); NaN folds to -inf rather than
  propagating; a STALE stash is not replayed; **two consecutive recipe steps
  re-earn their shutoffs from scratch**;
- round trip: allocate → promote → `storage_arrays` → `from_stored` →
  the stage survives, and a walker-count mismatch is refused;
- a partial / wrong-grid record is discarded whole rather than half-honored.
