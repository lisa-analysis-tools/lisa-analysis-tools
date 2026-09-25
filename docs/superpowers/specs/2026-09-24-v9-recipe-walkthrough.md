# v9 — how the recipe actually runs, step by step

**Written 2026-09-24, for the operator watching the v9 6mo 4-GPU launch.**
Every numbered step names the log line that proves it happened. If a step's
line is missing from the log, that step did not run.

Script: `scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh`
Driver: `scripts/fstat_proposal/run_combined_staged.py`

---

## 1. Launch preamble — before a single sample is drawn

1. **The feature preflight** runs and can abort the job (`exit 2`).
   1. Imports `gbspecialstretch` and checks the convergence symbols exist.
   2. **RESOLVES** each `GB_INMODEL_GROUP*` / `GB_INMODEL_CONVERGE` knob
      through the real resolver against the real environment, with a
      sentinel default. A knob that comes back as its sentinel was NOT SEEN,
      and the job refuses. ⚠ This is the check that a `hasattr` probe cannot
      do, and the one that would have caught the dead-knob bug found in the
      2026-09-24 audit.
   3. Checks the v9 stage-profile machinery exists
      (`SearchStageProfileStep`, `force_fstat_refit`,
      `set_peak_min_F_override`, `arm_fstat_refit`, the `gb_search` stage
      kind).
   * ✅ `[V9-PREFLIGHT] in-model convergence + group knobs RESOLVE from the
     environment; v9 stage-profile support present.`

2. **One seed store feeds two things** (`GF_SEED_STORE`).
   1. `GB_WARM_START_SOURCE_STORE` derives from it. If the refereed
      warm-start npz is absent it is built in-process at recipe build
      (fit → referee → apply).
   2. The seed defaults to the 3mo 10-walker **noise-fix** run. If the
      exact `.h5` is absent but the directory holds exactly one, the script
      resolves it and says so; if the directory is absent it lists the
      candidate `gf_prod_3mo*` directories. ⚠ It never substitutes a
      *different* run — a seed is a provenance statement.
   3. The **noise pin** is extracted from the same store:
      `python -m lisatools.globalfit.warmstart.noise_pin --store $GF_SEED_STORE --export`
      reads the **best-logL cold walker of the last valid row**, converts it
      to PHYSICAL/linear units (auto-detecting the source run's log basis —
      a log-sampled column is strictly positive under its physical prior, so
      a value ≤ 0 can only be a log), and emits
      `PSD_START_PARAMS=` / `GALFOR_START_PARAMS=`, which the shell `eval`s.
   4. A mismatch between the two sources prints a `[V9-SEED] WARNING`,
      and an existing warm-start npz is checked against a `.source` sidecar
      recording which store it was fitted from.
   * ✅ `[V9-SEED] store=…`, `[V9-SEED] PSD_START_PARAMS=…`

3. **`build_fit()` composes the recipe** and prints it.
   * ✅ `[combined] stage   gb_search_1 (gb_search): [...]` — one line per
     stage, listing its moves in cycle order.

---

## 2. The stage ladder

| # | stage | present when | ends when |
|---|---|---|---|
| 1 | `source_search` | source ids armed AND `STAGE_SKIP_SOURCE_SEARCH=0` (v9 exports 1, so **absent**) | joint max-lnL plateau |
| 2 | `noise_search` | **no** noise pin supplied | joint max-lnL plateau |
| 3 | `noise_vgb_search` | same | same |
| 4 | `gb_search_1` | always | gate 6 (§6) |
| 5 | `gb_search_2` | always | gate 6 |
| 6 | `gb_search_3` | always | gate 6 |
| 7 | `full_pe` | always | never — `NUM_ITERATIONS` bounds it |

Stages 2–3 are **skipped in v9** because the noise is pinned: there is
nothing for a convergence-gated noise burn-in to find.
⚠ A **half** pin (psd set, galfor not) keeps them — otherwise `gb_search_1`,
which does not sample noise at all, would run a whole stage against a random
foreground.
* ✅ `[combined] v9: SKIPPING noise_search / noise_vgb_search — the noise
  model is PINNED at a previous run's estimate`

The three search stages differ in **exactly four things**:

| | noise | phase max | opt SNR | F-stat peak |
|---|---|---|---|---|
| `gb_search_1` | FIXED | **on** | **8** | **8** |
| `gb_search_2` | FIXED | off | 5 | 6.25 |
| `gb_search_3` | sampled | off | 5 | 6.25 |

"Fixed noise" is the **absence** of the psd/galfor moves from the stage — an
unsampled branch does not move, and `setup_acs` rebuilds each walker's
sensitivity from the state coords every pass.

---

## 3. Fresh-start state (iteration 0)

1. `gb` starts at **zero leaves**. The search finds everything.
2. `psd` and `galfor` start **pinned** at the physical values from §1.2,
   converted into whatever basis this run samples in (psd `ln`, galfor
   `log10` on amp/fk/f_1/f_2, alpha linear). Every walker and every rung
   starts at the same point — a pin is meant to be a pin, so there is no
   `START_FACTOR` scatter.
   * ✅ `[NOISE-PIN] psd start coords PINNED at the supplied physical point …`
3. `vgb` is seeded from the catalogue (55 known sources, `VGB_START_FACTOR=0`).
4. Armed source branches (`mbh`/`emri`/`sobbh`) start at truth
   (`*_START_FACTOR=0`) and are subtracted from the residual.

---

## 4. Stage entry — what happens the moment a search stage becomes active

1. `SearchStageProfileStep.setup_run` installs the stage's moves and weights
   on the sampler, anchors `_stage_start_iter` (so the plateau window is
   stage-scoped, not run-scoped), and threads periodicity + temperature
   control onto each move.
2. `Recipe._announce_recipe_step` then fires, with the **step index** as the
   serial (deliberately not the iteration — the index is stable across a
   mid-step restart, so a resume does not look like a new step):
   1. **`note_recipe_step(serial)` → the profile is applied.**
      1. `opt_snr_rej_samp_limit` → **every** GB-branch band move, the pure
         in-model ones included. It is the SNR *prior boundary*: a birth move
         and an in-model move disagreeing about where the prior ends would
         let a source be walked into a region the move that birthed it treats
         as zero-prior. VGB moves are never touched (they carry the same
         attribute name with their own value, default off, on 55 known
         sources).
      2. `phase_maximize` → the **RJ moves only**. The pure in-model move is
         built `False` on purpose — in-model scoring is at the actual phase,
         and phase maximization is a *birth* heuristic.
      3. `peak_min_snr` → installs a process-wide in-code override of the
         F-stat peak floor, and **on a change** arms a forced fresh-epoch
         refit. ⚠ The floor is consumed when the grid is *fitted* and stamped
         into the stage-B cache, whose loader refuses a mismatch — so an env
         change at a stage boundary would either silently reuse stage 1's
         peak list or hard-refuse. A new epoch directory sidesteps both.
         **Nothing on disk is deleted**; stage 1's epoch stays, valid and
         unread.
      * ✅ `[V9-STAGE gb_search_2] rj_fstat_search.opt_snr_rej_samp_limit
        8.000 -> 5.000`
      * ✅ `[V9-STAGE gb_search_2] F-stat peak floor -> SNR 6.250 (F 19.531)`
      * ✅ `[V9-STAGE rj_fstat_search] FORCED F-stat refit: opening epoch N`
      * ✅ `[V9-STAGE rj_replace] FORCED F-stat refit: joining epoch N already
        opened for this recipe step` — the second move **shares** the epoch,
        so the stage pays for one comb scan, not three.
   2. **`begin_search_recipe_step(moves, serial)`** releases the
      per-(walker, band) RJ shutoff valve: a pair that stopped paying in lnL
      under the *previous* stage's caps and floors has said nothing about
      this stage's.

---

## 5. One iteration = one pass through the stage's move list

The stage's moves are wrapped in a single `GFCombineMove`, and eryn calls its
`propose` exactly once per global-fit iteration. The moves run **sequentially
in list order**. For a search stage:

```
 0  noise / vgb / source moves      (stage 3: joint noise search; 1-2: vgb only;
                                     mbh/emri/sobbh on a 1-in-5 cadence)
 1  rj_warm_search                  (stage 3: every 5th iteration)
 2  in_model
 3  rj_fstat_search
 4  in_model_fstat
 5  rj_replace
 6  in_model_replace
 7  rj_prior_removal                ← THE CYCLE ENDS HERE
 8  gb_ridge_gibbs / vgb_ridge_gibbs (zero-likelihood fiber moves)
```

The cycle **ends on the removal judge** so every birth and every swap has had
a full in-model refinement pass before it is judged for death. A source still
sitting at its birth or swap coordinates looks far more deletable than the
same source after it has walked onto its peak. Each in-model slot is named for
the RJ move it polishes.

### 5A. Inside one RJ move's `propose()` (warm / fstat / replace / removal)

1. **`setup()` — the F-stat grid decision.** One of:
   * `skip` — the refit clock has not elapsed (`GB_FSTAT_REFIT_EVERY=40`,
     counted in **elapsed** sampler iterations);
   * `load` — a complete epoch exists on disk;
   * `fit` — run the comb scan → peak selection → stage B into a new epoch.
   A **forced** refit (§4.2.1.3) overrides all three, including `skip`.
2. **Build the `BandSorter`** from the current state: every leaf labelled by
   `(temp, walker, band)`, sorted so a sub-band's rows are contiguous.
3. **Band units → batches of cells → RJ pick rounds.** Per round, one
   candidate per sub-band cell (serial-within-band); birth/death — or, for
   `rj_replace`, a fixed-dimension swap — is proposed, scored and
   accepted/rejected. Survivors are pooled. Rounds continue until the batch
   stops yielding picks.
   * **Flip fraction 1.0.** `GB_SEARCH_RJ_FLIP_FRACTION=1.0` reaches **all
     four** search-cycle RJ slots — `rj_warm_search`, `rj_fstat_search`,
     `rj_replace`, `rj_prior_removal` — so every round visits **every**
     eligible row rather than a 20% sample of them. PE is unaffected
     (`GB_PE_RJ_FLIP_FRACTION=0.1`). ⚠ The precedence is *per-move kwarg >
     the GLOBAL `GB_RJ_FLIP_FRACTION` > the stage knob*, so the global must
     stay unexported or it clobbers both stages at once; in this script it is
     commented out on purpose.
   * ✅ `[GB_ACCEPT rj-split …]`, and for replace
     `[GB_ACCEPT replace-split …]` (proposals, cold acceptance, SNR-gated and
     non-finite counts, cold-accepted Δ lnL mean/max).
4. **The in-model polish phase.** Once every batch's RJ is done, all the
   unit's survivors are polished in capacity-width chunks:
   * **newborn** rows get `GB_INMODEL_REPEATS_NEWBORN=100` repeats,
     **mature** survivors `GB_INMODEL_REPEATS_SURVIVOR=50`;
   * **convergence scope 1** applies to newborns only
     (`GB_INMODEL_CONVERGE_CLASSES=newborn`): a row retires when its
     accumulated **accepted** `delta_ll` has not improved on its best over the
     last `W=100` repeats (ring buffer — the leaf cap's patience form does not
     port to a fine clock, it retires mid-climb);
   * a **ladder gate** keeps the hottest half of the ladder from holding the
     block open (hot chains random-walk and never plateau);
   * **refill**: when a whole `(walker, band)` column finishes, it is swapped
     out and a new column swapped in, rather than the block waiting.
   * ✅ `[GB_IMCONV …]` — repeats used, generations, retirement distribution.
5. **Vertical band-temperature swaps** (`GB_TEMPER_VERTICAL=1`). ⚠ The
   permuted ("fancy") swaps are OFF in v9 (`GB_RUN_FANCY_TEMPERING=0`), and
   `_adapt_band_temps` had exactly one caller — `run_tempering`. So the
   ladder is now adapted from the **vertical** swap acceptance instead;
   without that it would silently freeze for the whole run.
6. Write back into the state.

### 5B. Inside one pure in-model move's `propose()`

**Fixed repeats, adaptive passes.**

1. **Pass 1**: every live source gets `GB_NUM_REPEAT_PROPOSALS=25` in-model
   repeats. This count is FIXED and never adapts.
2. **Convergence scope 2** (`GB_INMODEL_GROUP=1`), per `(walker, band)`:
   1. Accumulate that sub-band's **cold-rung** `ll_change_log[0, w, b]` — the
      total accepted `delta_ll` for the pass. This is free (already computed)
      and reference-invariant, which a measured sub-band likelihood would not
      be.
   2. A pair is **shut off** when, over a window of
      `GB_INMODEL_GROUP_ITERS=3` passes, its running best has not improved by
      more than `GB_INMODEL_GROUP_DLL=4.0` (flat D/2 — *"this will focus more
      resources on the sub-bands with more sources"*).
   3. Shut pairs drop out of the eligible set for the rest of this group.
      Unoccupied pairs shut immediately.
3. **Repeat the whole pass** until every *occupied* pair is shut, or the
   `GB_INMODEL_GROUP_MAX_PASSES=20` ceiling hits.
4. The group state is **scoped to one `propose()`** and resets at the next.
   * ✅ `[GB_IMGROUP in_model] pass N: cold dlnL +X over M occupied
     sub-band(s); shut A (+B this pass) / open C …`
   * ✅ `[GB_IMGROUP in_model] group CONVERGED after N pass(es)` — or
     `group hit the N-pass CEILING with M sub-band(s) still open`, which is a
     **warning**: the group is being truncated, not finishing.
   * ✅ `[GB_IMGROUP in_model] group done: N pass(es) x 25 repeats/source …
     Compare against a fixed 5-pass block: Fx the passes.` — F > 1 means this
     costs more than the 5-pass block it replaced.

⚠ Per-source totals are **not uniform** across the proposal: a source in a
band that keeps earning passes gets more repeats than one in a band that shuts
early. That is the intent, and it is a search-only licence — the group rule is
refused on PE-named moves and on RJ moves.

### 5C. `rj_replace` — two internal passes in one `propose()`

`rj_replace` runs §5A **twice**, with the candidate container swapped:

1. **Pass 1 — WARM.** Candidates drawn from the warm-start mixture (the same
   container object `rj_warm_search` births from).
2. **Pass 2 — FSTAT.** Candidates from the fitted F-stat grid.

Each pass prices **both sides** of its MH ratio against its own container —
the sorter's death-side `factors` are computed from whichever container the
pass installs — so the two never mix.

**Phase maximization inside the swap** (`GB_REPLACE_PHASE_MAX`, `auto` →
ON for a search-stage install): the **ADDED** source is scored
phase-maximized, the **REMOVED** one always at its **actual** phase. The
maximizing rotation is written into the accepted candidate's φ₀ before the
write-back, so the credited Δ is attainable at the parameters actually kept.
⚠ This is a *different* switch from the move's `phase_maximize` attribute,
which drives only its internal in-model repeats and stays `False`; the v9
stage profile deliberately does not touch it.

⚠ **This move was retired once.** Its recorded failure signature was a
propose-level lnL drift of **1.5–1.9e3**, three orders above every other move,
root-caused to maximized credit *without* that write-back — unattainable at
any actual φ₀. Rotation-on-accept is the fix, not renouncing maximization.
The 2026-08-24 redesign replaced that with rotation-on-accept (the scored rows
ARE the final rows). It is reinstated on the hypothesis that the rest was the
old GPU setup — **watched, not trusted**:
* ✅ `[GB_REPLACE rj_replace] pass 1/2: candidate source = WARM container.`
* ✅ `[GB_REPLACE rj_replace] pass 1/2 (warm) done: max |dlnL| over the cold
  chain = X`
* 🚨 A **WARNING** fires above 1.0e3. If it does: check
  `[GB_ACCEPT replace-split]` and `[GB_ORTHO_LL rj_replace]`, then relaunch
  with `GB_SEARCH_RJ_REPLACE=0`. Nothing else in the cycle depends on it.

---

## 6. Stage end — gate 6

Checked after every iteration. **Composed, never replaced**:

1. **The nleaves plateau.** The cold-chain max leaf count over the last
   `GB_PLATEAU_ITERS=20` iterations of *this stage* is no higher than over the
   window before it. A stage never ends at zero leaves — a search that has not
   found its first source has not plateaued, it has not started.
2. **AND the per-(walker, band) RJ valve.** `band_shutoff_w_pending_total`
   must be 0: *"every OCCUPIED (walker, band) pair has shut off."*
   * ⚠ `band_shutoff_w_armed` is **mandatory** before reading the count.
     `pending_total` returns 0 both when everything converged and when the
     feature is off, and the second reading would end every stage at its first
     check — indistinguishable from convergence in the log.
   * ⚠ OCCUPIED is load-bearing. An **empty** pair can never shut off, so the
     literal "all pairs shut" is unreachable in any run with an empty band,
     i.e. every run. The plateau gate covers the empty ones.
3. ✅ `[V9-STAGE gb_search_1] nleaves plateau reached but N occupied
   (walker, band) pair(s) have NOT shut off — holding the stage open.`
4. ✅ `[V9-STAGE gb_search_1] STAGE COMPLETE: nleaves plateau AND every
   occupied (walker, band) pair has shut off.`
5. On completion the backend stamps `status=True` **and**
   `completed_iteration`, which is what lets the HTML monitor draw the stage
   boundary on every per-iteration trace. Without it, a leaf count that steps
   up at the `gb_search_1 → _2` boundary (opt SNR 8→5, peak 8→6.25) reads as a
   discovery burst rather than a floor change.

---

## 7. `full_pe`

Unchanged from v8. A **weighted-cycle** stage (`FULL_PE_WEIGHTED_CYCLE=1`,
`FULL_PE_RANDOM_CHOICE=0`, both code defaults and neither exported): one
propose per iteration, one stored row per iteration, but the composition is
DRAWN each iteration — `len(moves)` weighted draws **with replacement**. So a
given move runs on roughly 1 − (1 − 1/N)^N ≈ 63% of iterations rather than all
of them, which is why `GB_FSTAT_REFIT_EVERY_PE=250` exists: a search-tuned
cadence here means materially more elapsed iterations than the number says.
(The legacy one-move-per-step `random_choice` mode, where the ratio is 1/N, is
still available behind those two flags.) It never stops on its own;
`NUM_ITERATIONS=2000` bounds the run.

---

## 8. First-hour checklist

1. `[V9-PREFLIGHT]` passed — else the job aborted.
2. `[V9-SEED]` shows a store and two non-empty pin lines.
3. `[combined] stage` lines show `gb_search_1/2/3` + `full_pe`, each with the
   7-slot cycle.
4. `[V9-STAGE gb_search_1] entering … profile {'phase_maximize': True,
   'opt_snr': 8.0, 'peak_min_snr': 8.0}` and one mutation line per GB move.
5. `[NOISE-PIN]` for psd and galfor.
6. `[GB_IMGROUP … armed]` then per-pass lines — if these are missing, the
   group rule is off and the preflight lied.
7. `[GB_REPLACE]` pass lines with |dlnL| well under 1e3.
8. `[GB_TIMING]` on the first iteration: budget a **3–5× increase in the
   in-model share** vs v8. The standalone passes are unbounded by
   construction and the pass ceiling is a *cost* knob, not a safety net.
   Confirm the per-iteration cost before committing a long allocation.

## 9. Kill switches, in increasing order of severity

| symptom | knob |
|---|---|
| `rj_replace` drifting | `GB_SEARCH_RJ_REPLACE=0` |
| in-model passes dominating the iteration | `GB_INMODEL_GROUP_MAX_PASSES=5`, then `GB_INMODEL_GROUP=0` |
| newborn polish too long | `GB_INMODEL_CONVERGE=off` |
| ladder misbehaving | `GB_RUN_FANCY_TEMPERING=1` (restores the permuted swaps) |
| the whole restructure | `STAGE_V9_SEARCH=0` — one legacy `gb_search`, bit-identical to v8's composition |
