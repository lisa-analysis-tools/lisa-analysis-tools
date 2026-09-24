# v9 recipe restructure: three GB search stages

**Status:** design, not implemented · **Date:** 2026-09-24 ·
**Depends on:** `gb-cap-per-walker` (peer session, merging to `dev`) and
`gb-inmodel-converge` (this session, uncommitted).

## The ask (user, 2026-09-24)

RJ flip fraction → 1.0 for the search parts. Replace the single `gb_search`
stage with three, then `full_pe` unchanged:

| stage | noise | phase max | opt SNR | F-stat peak | ends on |
|---|---|---|---|---|---|
| `gb_search_1` | **fixed** (last run's psd + galfor) | **on** | **8** | **8** | all occupied (walker, band) shut off |
| `gb_search_2` | **fixed** | off | 5 | 6.25 | same |
| `gb_search_3` | sampled, as today | off | 5 | 6.25 | same |
| `full_pe` | sampled | off | 5 | 6.25 | unchanged from today |

Every search stage runs the same per-iteration move order:

```
1  rj_warm_search        (warm-start RJ)
2  in_model x5           (25 repeats/source each, ALL sources)
3  rj_fstat_search       (F-stat RJ)
4  in_model x5
5  rj_prior_removal      (remove only, no add)
6  in_model x5
```

`noise_search` / `noise_vgb_search` stay in the file but run **only when
there is no psd/foreground estimate from a previous run**.

### Clarifications given (2026-09-24)

* **Keep both in-model paths.** The RJ moves keep their own internal polish
  of their survivors; the standalone `in_model` passes run over the whole
  source set on top of that.
* **"Convergence only happens during the instant the source is birthed.
  Otherwise it is doing the regular in-model setup."** So the
  convergence-driven budget applies to the NEWBORN class inside the RJ
  moves' internal polish and nowhere else. The standalone passes are a plain
  fixed 25 repeats. This is exactly what
  `GB_INMODEL_CONVERGE_CLASSES=newborn` already selects — **no change to the
  convergence feature is needed**, and the standalone `in_model` move
  (`is_rj_prop=False`) is deliberately left on its plain
  `num_repeat_proposals`.
* **Stage 1 is SNR 8 everywhere**, including the F-stat peak threshold;
  stage 2 onward is opt-SNR 5 / peak 6.25.

## What already exists

* `scripts/fstat_proposal/run_combined_staged.py::build_fit` owns the stage
  list (`source_search → noise_search → noise_vgb_search → gb_search →
  full_pe`). Stages are declarative `Stage(name, kind, moves=[Move(...)])`.
* `build_gb_moves` (recipe.py) builds every GB move ONCE; `Move("name")`
  resolves by name. **Three stages referencing `rj_fstat_search` therefore
  share one instance** — this is the constraint that shapes the design.
* The standalone in-model move already exists: `name="in_model"`,
  `is_rj_prop=False`, gated on `_gb_mode_search and gb_info.search_in_model`.
* Per-stage move-name uniqueness is required (recipe.py ebd8612), which is
  what lets a move name recur across stages.
* `GB_SEARCH_RJ_FLIP_FRACTION` (0.5) / `GB_PE_RJ_FLIP_FRACTION` (0.1);
  `GB_OPT_SNR_LIMIT_SEARCH` (script: 5.0) / `_PE`; `FSTAT_PEAK_MIN_SNR`
  (script: 6.25); `GB_RJ_PHASE_MAXIMIZE` (setdefault on).

## The two traps

### TRAP 1 — the F-stat peak threshold is baked into a CACHE, and the loader refuses a mismatch

`FSTAT_PEAK_MIN_SNR` selects the peak list **when the grid is fitted**;
`fstat_gridfit`'s stage-B load path is "load and return, nothing
recomputed". Since 2026-09-23 the stacked cache stamps `peak_min_F` and the
loader **refuses** a mismatch.

So stage 1 at peak 8 followed by stage 2 at peak 6.25 does **not** work by
changing an env var at the stage boundary. It will either silently reuse
stage 1's list (pre-stamp caches) or hard-refuse (post-stamp).

**RESOLUTION (user, 2026-09-24): no file surgery.** Make the peak threshold
an *updateable value in code* and force an F-stat refit on the first call
inside a new stage. The stage profile sets the live threshold; the grid
machinery invalidates its own in-memory/registry cache for that stage and
refits, so the on-disk comb stays authoritative and nothing has to be
deleted. (The delete-the-stacked-npz route is the manual migration and stays
documented as the operator escape hatch, not the mechanism.)

### TRAP 2 — "all bands and walkers shut off" is literally unreachable

From the peer session that owns the mechanism:

> `pending` counts OCCUPIED pairs only. An EMPTY (walker, band) can NEVER
> shut off — the criterion requires a non-zero source count, because a band
> that has found nothing hasn't plateaued, it hasn't started. So "all bands
> and walkers are shut off" as literally phrased is unreachable in any run
> with an empty band, i.e. every run.

So the stage criterion is **"every OCCUPIED (walker, band) pair has shut
off"**, and the empty bands are covered by keeping the existing nleaves
plateau gate composed in rather than replaced.

## Design

### A. Per-stage profiles, applied at stage entry (not three move sets)

Because the moves are built once and shared, the per-stage differences
(`phase_maximize`, `opt_snr_rej_samp_limit`) are applied as a **stage
profile** at `setup_run`, not by building three named copies. Three copies
would trile the F-stat-carrying RJ move instances for no sampling benefit.

```python
SEARCH_STAGE_PROFILES = {
    "gb_search_1": dict(phase_maximize=True,  opt_snr=8.0, peak_min_snr=8.0,
                        sample_noise=False),
    "gb_search_2": dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25,
                        sample_noise=False),
    "gb_search_3": dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25,
                        sample_noise=True),
}
```

Applied by a `SearchStageProfileStep` (a thin `RJRecipeStep` subclass) whose
`setup_run`:

1. calls `super().setup_run(...)` — **mandatory**, see D;
2. walks the stage's move tree and sets `phase_maximize` /
   `opt_snr_rej_samp_limit` on every GB/VGB move it finds, and on the
   buffer-facing copies the moves hand down;
3. on a change of `peak_min_snr` vs the previous stage, updates the live
   threshold and arms a one-shot "refit on first F-stat call in this stage"
   flag (TRAP 1) — no file is touched.

Each mutation is logged as one `[V9-STAGE]` line naming stage, knob, old and
new value — this machinery is invisible otherwise and a silently-unapplied
profile would look like a normal run.

### B. The move list per search stage

```python
def _search_stage_moves(sample_noise):
    inm = lambda tag, n: [Move(f"in_model_{tag}_{i}", branch="gb")
                          for i in range(n)]
    return (
        (noise_vgb_gb if sample_noise else vgb_only)
        + source_pe(gb_search_cadence=True)
        + warm()            + inm("warm", 5)
        + [Move("rj_fstat_search", branch="gb")] + inm("fstat", 5)
        + replace()
        + [Move("rj_prior_removal", branch="gb")] + inm("prune", 5)
        + gb_ridge() + vgb_ridge()
    )
```

15 `in_model` instances per stage, distinct names (per-stage uniqueness
forces the tag+index). They are the same class with
`num_repeat_proposals=25`; `build_gb_moves` gains a small loop registering
them instead of the single `in_model`.

**Open point to confirm before coding:** the current stage also carries
`replace()` (`rj_replace`, the exact-MH F-stat replacement) between the
F-stat birth and the removal judge. The user's 1–6 ordering does not mention
it. The list above keeps it in its current position; if it should go, that is
a one-line deletion.

### B2. In-group convergence for the PURELY in-model passes

**User ruling 2026-09-24, and it replaces the "25 repeats x 5 passes" shape
of section B.** The standalone in-model proposal is not a fixed number of
passes: it runs until **every `(walker, band)` sub-band's COLD-CHAIN logL has
converged**, shutting off each sub-band as it does, and ends when all are
shut.

This is the THIRD convergence scope in the design and they are easy to
confuse, so, explicitly:

| # | unit | statistic | clock | scope | owner |
|---|---|---|---|---|---|
| 1 | one `(temp, walker, band)` ROW | that row's accumulated accepted `delta_ll` | one in-model REPEAT | one repeat block, **newborns only** | `gb-inmodel-converge` (built) |
| 2 | one `(walker, band)` SUB-BAND | that sub-band's **cold-chain** logL | one PASS of the in-model move | **one in-model proposal group** (e.g. the block after warmstart) | THIS section (new) |
| 3 | one `(walker, band)` SUB-BAND | that sub-band's converged **leaf count** | one sampler ITERATION | one **recipe stage** | peer session (built, unrun) |

Scope 1 is newborn-only inside the RJ moves, so it does NOT fire in the
standalone passes ("convergence only happens the instant a source is
birthed"). Scopes 2 and 3 share a unit but differ in clock, statistic and
lifetime — 2 resets at the start of every in-model group, 3 persists for the
whole stage.

**The statistic is free.** Do NOT measure the sub-band likelihood directly:
`band_lls` / `band_cold_ll` is a full residual reduction (the cap gate's
per-iteration work) and would be paid once per pass. The in-model move
already accumulates `ll_change_log[t, w, b]` — the total ACCEPTED `delta_ll`
per cell per pass. Summing the **cold rung** entry across passes gives that
sub-band's cold-chain logL trajectory up to an additive constant, at zero
extra cost, and it is reference-invariant for the same reason the per-row
gain is (the sig-het offset cancels in a delta and only in a delta; see the
in-model convergence spec).

```
per (walker, band):
    G_wb  = running sum over passes of  ll_change_log[0, w, b]
    best  = max(best, G_wb)
    converged when  passes >= W  and  best - best_at(passes - W) <= thresh
```

Same ring-buffer form as scope 1, so the same "the cap gate's per-tick form
does not port" argument applies and the same helper can be reused.

**Shut off** = the `(walker, band)`'s cells are removed from `eligible` for
the remaining passes of this group. That is a THIRD mask alongside
`band_rj_shutoff` (existing, per band) and `band_rj_shutoff_w` (peer, per
walker+band, per stage). ⚠ Mask proliferation is a real risk: they must
compose by OR at a single point, and the group mask must be cleared at group
entry or the next group starts fully shut.

**Bounded**: a `max passes` ceiling, and a pass that shuts nothing while the
set is unchanged ends the group (same "progress is mandatory" rule as the
refill loop).

**Open tuning question (raised with the user):** the threshold is a
*per-sub-band* quantity, so a band holding 50 sources each gaining 0.1 posts
5.0 and keeps running, while a band holding 2 sources cannot clear 4.0 even
when both are climbing. A flat D/2 therefore makes convergence time scale
with occupancy. Either that is intended (a dense band genuinely has more to
gain), or the threshold should scale with the band's live source count.

### C. Fixed noise in stages 1 and 2

Simply omit the psd/galfor moves from those stages — an unsampled branch
does not move, which is what "fixed" means here. The *values* they are fixed
AT come from the previous run's estimate, which is the separate
seed-the-foreground work (`2026-09-24-*-foreground` — not yet written).
Until that lands, stages 1–2 hold whatever the run starts with, which is only
correct if the run is seeded.

⚠ Stage 3 turning noise back on after two stages of GB-only work is the
moment the foreground will try to re-inflate to match the source count — the
feedback loop from `galfor_6mo_absorbs_unresolved_gb_0923`. The hold-down
mechanism belongs to that other piece of work and should land with it.

### D. Stage convergence

Compose, do not replace:

```python
if stop and band_shutoff_w_armed(getattr(sampler, "moves", None)):
    if band_shutoff_w_pending_total(getattr(sampler, "moves", None)):
        stop = False
```

* `band_shutoff_w_pending_total(moves) -> int` (0 == converged) and
  `band_shutoff_w_armed(moves) -> bool`, both public, in `recipe.py` next to
  `_cap_ramp_pending_total`, from the peer's branch.
* The `_armed` guard is **required**: `pending_total` returns 0 when the
  feature is off, which reads as "converged" on the first check.
* Knobs: `GB_SEARCH_BAND_SHUTOFF_PER_WALKER` (default 0 — v9 must export 1),
  `GB_SEARCH_BAND_SHUTOFF_CONV_ITER` (default 5), `GB_SEARCH_STAGE_DIAG=1`.
* ⚠ The flag must reach BOTH the move AND
  `initialize_band_information(..., search_shutoff_per_walker=True)`; the
  peer raises rather than silently no-opping if only one gets it.
* The valve is scoped to a recipe step and released when the next begins.
  **Superseded 2026-09-24:** the announcement moved out of
  `RecipeStep.setup_run` into `Recipe._announce_recipe_step`, which `Recipe`
  calls right after `setup_run` on both paths (first step and advance). So a
  `Stage` subclass overriding `setup_run` can no longer break it and the
  `super()` call is **not** load-bearing — it is still made, and the
  "announced itself" assertion is kept as a regression guard on `Recipe`,
  but neither is required for correctness.
* ⚠ The step serial is the **step index**, not the backend iteration (peer
  fix, 2026-09-24). Reading the iteration meant a mid-step resume looked
  like a NEW step and the valve released itself on every restart — which
  from the recipe's side would have presented as "the stage ended early
  after a restart", i.e. indistinguishable from convergence. Pinned by their
  `test_the_valve_HOLDS_across_a_mid_step_resume`.
* The stage schedule and the valve are independent flags: this design wants
  `GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1` with `GB_SEARCH_STAGE_PER_WALKER=0`
  — SNR limits move per recipe STAGE, never per band.
* The peer's mechanism is **built and wired but never run against a real
  fit**. Treat the interface as stable, the numbers as unvalidated.

### E. Conditional noise stages

`noise_search` / `noise_vgb_search` run only when no prior estimate exists.
Gate on the same thing the seeding uses — presence of a resolvable previous
psd+galfor estimate — with an explicit env override
(`STAGE_FORCE_NOISE_SEARCH=1`) so a fresh campaign can still ask for them.
Absent the seeding work, the gate is "was a seed supplied", which is false
today, so the stages keep running exactly as now until the seed lands.

### F. Flip fraction

`GB_SEARCH_RJ_FLIP_FRACTION=1.0` in the v9 script. One line; it reaches
every search-named RJ move through the existing `_search_rj_flip_default()`.

## Cost

Blunt, and now UNBOUNDED by construction: the standalone passes run until
every sub-band converges rather than for a fixed 15 x 25, on top of the RJ
moves' internal polish, which is kept. The `max passes` ceiling is the only
hard bound, so it is a cost knob, not a safety net. Against a v8 iteration where `inmodel_repeats` was already 32% of
852 s, this is the dominant new cost in the restructure — far larger than
anything the convergence feature saves. That is a deliberate choice (polish
quality over iteration rate) but it should be a priced one: budget roughly a
3–5× increase in the in-model share and confirm on the first iteration's
`[GB_TIMING]` before committing a long allocation.

## Testing

* Stage list / order / per-stage move names, at construction level
  (`test_staged_sources_wiring.py` pattern, `build_fit()`, no data).
* Profile application: each stage's resolved `phase_maximize` /
  `opt_snr_rej_samp_limit`, and that stage 2 entry drops the stacked peak
  cache and keeps the comb.
* The convergence composition: armed + pending > 0 holds the stage open;
  armed + pending == 0 with the plateau also satisfied advances; NOT armed
  falls back to the plateau alone.
* `super().setup_run` is called (guard against the pre-frozen-stage-2 bug) —
  assert `begin_search_recipe_step` fired for each stage.
* Conditional noise stages present/absent under the seed gate.
