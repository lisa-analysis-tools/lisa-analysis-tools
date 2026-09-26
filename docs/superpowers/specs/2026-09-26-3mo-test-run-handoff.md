# 3-month test run: what to carry over from the 6-month v9 run

**Date:** 2026-09-26
**For:** whoever sets up the 3mo test run (likely a separate session)
**Approach set by the user:** start from the CURRENT 6mo script and change
only what 3 months specifically requires. The user will name those changes.
Nothing here prescribes them; this is the context you need to make them
safely and to know what "working" looks like.

Source of truth: `scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh` at
`origin/dev` `33d5c54f`.

---

## 0. READ THIS FIRST: do NOT derive from `submit_gf_3mo_v9.sh`

It exists, and it has **drifted from the 6mo script by ten knobs**. The
guard test `ThreeMonthTwinTest` has been failing on exactly this for days
and is a known, accepted failure — not something you introduced:

```
FSTAT_GRID_MEM_MB            GB_RJ_PHASE_MAXIMIZE
FSTAT_PEAK_MIN_SNR           GB_SEARCH_NOISE_ITERS_PER_STEP
GB_FSTAT_REFIT_EVERY         GB_SEARCH_PRIOR_REMOVAL_ONLY
GB_OPT_SNR_LIMIT_SEARCH      GB_SEARCH_SOURCE_EVERY
GB_PE_RJ_FSTAT_FRACTION      MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM
```

Several of those are correctness-relevant (the phase-maximization ruling,
the SNR floors, the unequal-arm PSD reference). The 3mo v9 file predates
most of 2026-09-25's work. **Copy the 6mo file and edit it.** The test's own
docstring says why: *"The reason to derive from the 6mo file rather than the
old 3mo one."*

If you do derive from 6mo, expect to update `ThreeMonthTwinTest`'s reversion
list — that test exists to make 3mo-vs-6mo divergence DECLARED rather than
accidental, and a legitimately 3mo-specific knob belongs on the list.

---

## 1. What is inherently 6-month-specific

These are the places the observation length or the run's provenance is baked
in. Not a change list — a "these will need a decision" list.

| knob / item | current 6mo value | why it is Tobs-bound |
|---|---|---|
| `TOBS_TARGET` | `15552000` (180 d) | sets the WDM grid; comment notes Nf 1440 x Nt 4320 x dt 2.5 is an exact factor-2 of 3mo in Nt |
| `GF_SEED_STORE` | `.../gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5` | the noise pin comes from here; it is ALREADY a 3mo store |
| `GB_WARM_START_COMPONENTS` | `${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz` | warm-start mixture, also already 3mo-derived |
| `GB_WARM_START_SOURCE_STORE` | defaults to `GF_SEED_STORE` | same |
| `STORE_DIR` | `/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/` | must be a NEW dir or you resume the 6mo run |
| `PSD_START_PARAMS` | from the seed store's maxlogL cold walker | |
| `GALFOR_START_PARAMS` | `1.436180605904e-44, 2.533915614978e-03, 5.0, 1.0e-02, 1.405721657329e-03` | hand-set offline estimate, NOT from the store |
| F-stat epoch cache | `${STORE_DIR}/gb_fstat_fit/` | fitted against a 6mo residual; a 3mo run must not reuse it |
| eigen sidecar | `<store>_eigen_tables.pkl` | guarded on `df = 1/Tobs`, so a 6mo one is correctly REJECTED at 3mo — harmless, just no saving |

⚠ **The warm start and the noise pin are already 3-month products.** The 6mo
run seeds from `gf_prod_3mo_v8_10walkers`. For a 3mo test run that is either
a convenience or a circularity depending on what you are testing — worth
being deliberate about.

⚠ **`GALFOR_START_PARAMS` is the 3mo fixed point**, chosen by the user on
09-25 with the 3mo-vs-6mo comparison in front of them (the 6mo fixed point
is `1.162927e-44, 7.332070e-03, 3.371045, 2.620588e-03, 5.505988e-04`, a
1.235x smaller amplitude). Its third parameter `f_1 = 5.0` sits EXACTLY on
the prior rail — flagged at the v9 launch audit and still true. It only
matters once `gb_search_3` releases the foreground.

⚠ **The injection is `1.5e-11` (Soms_d) and `3e-15` (Sa_a), UNEQUAL ARM.**
`psd_truth_levels`' docstring names `1.496182e-11, 2.982412e-15` as the
EQUAL-arm literal it exists to retire. I got this wrong twice before the
user corrected me; do not re-derive it from a monitor page.

---

## 2. The current recipe (what a 6mo iteration does)

```
gb_search_seed → gb_search_1 → gb_search_2 → gb_search_3 → full_pe
```

`gb_search_seed` (new 2026-09-26, `GB_SEARCH_SEED_ITERS=5`): `vgb_pe` +
`rj_warm_search` + `in_model`, a FIXED 5 iterations, gb_search_1's profile
verbatim, **no `rj_fstat_search`** (so no grid fit can trigger inside it),
no sobbh/mbh/emri, and the per-walker RJ shutoff valve disarmed. `0`
disables it entirely.

The three numbered stages run a 7-slot cycle (8 before `rj_replace` was
switched off):

```
vgb_pe → rj_warm_search → in_model → rj_fstat_search → in_model_fstat
       → in_model_replace → rj_prior_removal → ridge → sources
```

Stage profiles: stage 1 is `phase_maximize=True, opt_snr=8, peak=8`; stages
2 and 3 are `False, 5, 6.25`. **Only stage 1 maximizes** — stage 2 drops the
floors and maximizing on top would add a second optimistic bias to the
population least able to afford one (user ruling 2026-09-25). Stage 3 is the
only one that samples the noise.

### Knobs as shipped, with why

| knob | value | rationale |
|---|---|---|
| `GB_INMODEL_CONVERGE` | `on` | per-ROW convergence in the RJ moves |
| `GB_INMODEL_CONVERGE_ITERS` | `250` | newborn window |
| `GB_INMODEL_CONVERGE_ITERS_SURVIVOR` | `100` | survivor window; dll auto-scaled to 1.6 |
| `GB_INMODEL_CONVERGE_CLASSES` | `newborn,mature` | both classes converge |
| `GB_INMODEL_CONVERGE_MAX` | `20000` | tripwire, loud |
| `GB_INMODEL_GROUP` / `_ITERS` / `_DLL` | `1` / `3` / `4.0` | per-SUB-BAND rule on the pure in-model move |
| `GB_INMODEL_GROUP_MAX_PASSES` | `1000` | tripwire, loud (628 reached 13-17) |
| `GB_NUM_REPEAT_PROPOSALS` | `25` | see the rate trap in §4 |
| `GB_SEARCH_RJ_REPLACE` | `0` | off on yield, not on misbehaviour |
| `GB_FSTAT_REFIT_EVERY` | `1` | every search iteration |
| `GB_SIGHET_REFRESH_EVERY` | `50` | |
| `GB_SEARCH_BAND_SHUTOFF_CONV_ITER` | `3` | consecutive flat ITERATIONS per (walker,band) |
| `GB_RJ_DIRECT_BATCH` | `0` | staged scheduler: RJ round -> converge -> refill |
| `GB_RJ_GROUPED_INMODEL` | `1` | |
| `GB_TEMPER_CELL_ORDER` | `band` | now the code default too |
| `COARSE_Q` / `COARSE_GPU_MODE` | `1` / `off` | |
| `NWALKERS` / `GB_NTEMPS` | `4` / `24` | |
| `GB_N_SUBBANDS` | `8192` per GPU | |

`NGPUS=4` means **2 nodes x 2 GPUs**, not one 4-GPU node.

---

## 3. Baselines from job 628 to compare against

Three complete `gb_search_1` cycles, 7h42m, zero crashes. These are the
numbers a 3mo run should be read against (scaled for Tobs).

| | it 1 | it 2 | it 3 |
|---|---|---|---|
| wall (GB only) | 7,877 s | 7,955 s | 8,307 s |
| leaves/walker (end) | 1,430 | 1,543 | 1,624 |
| new leaves | 1,430 | +113 | +81 |
| **s per new leaf** | **7.4** | **70.6** | **101.9** |
| ΔlogL/walker | — | +5,422 | +3,284 |

**Where the time went (it 3, before the 09-26 changes):** in-model
convergence polish **~85%**, RJ proposals **~10%**. "The RJ schedule is
expensive" was the wrong reading.

**Per-move cold RJ accepts:** `rj_warm_search` 157→72→42,
`rj_fstat_search` 135→96→54, `rj_prior_removal` ~30 flat,
`rj_replace` **0,0,0,0,0,0** (8 accepts in ~707k proposals, none cold →
switched off).

**F-stat fit cost (job 627, the only full one):** stage A 12m00s
(222.9M evals → 10,828 peaks), stage B 17m24s, centre table 33s ≈ **30 min**.
Stage B group 1 (0.57–5.14 mHz, 86% of peaks) is 46% of stage B.
4 walkers instead of 1 would be ~4x = ~2 h; the ranks already split one
walker's grid four ways, so there is no free parallelism left.

**GPU:** ~47–48% mean utilisation, 11–15% idle on the sampled node.

**Doubles status:** 3 of 4 walkers clean at the three high-f truths
(16.7163 / 19.6682 / 20.3804). Walker 0 has split 19.6682 into two leaves
0.16 bins apart and carries 2–3 sub-3-bin pairs above 9 mHz. Below ~6 mHz
the truth is genuinely sub-bin dense, so **only the >9 mHz count and the
6.0–6.5 mHz band are discriminating** — never use f0 proximity alone.

---

## 4. Traps that have actually bitten, in this codebase, this month

**The recurring shape: a knob resolves, and nothing reads it.** Six separate
defects this month. A preflight proves the NAME resolves, not that any code
path CONSUMES it. Always confirm from a production log line that the feature
ran, not from the export.

Specific instances worth knowing:

1. **Two stage assemblies.** `run_combined_staged.py` has a `GB_ONLY`
   composition and a full one, both iterating `V9_SEARCH_STAGE_PROFILES`.
   Adding a stage to one only is easy and silent.
2. **`GB_INMODEL_GROUP` was inert on the fan-out path** — its state factory
   had one caller, in `_propose_legacy`, while production takes
   `_propose_orchestrated`.
3. **The temperature ladder was frozen** — `_adapt_band_temps` was reachable
   only from the permuted-swap block, so `GB_RUN_FANCY_TEMPERING=0` froze it
   with nothing logging.
4. **CuPy refuses host operands.** `cupy.bincount` evaluates
   `int(cupy.max(x))+1` BEFORE `minlength`, so a zero-size selection raises
   where numpy returns zeros. Same class: `cp.maximum(<numpy array>, 1)`
   → `TypeError: Unsupported type <class 'numpy.ndarray'>`, which killed the
   first `rj_replace` any job reached. **A numpy-only test cannot reproduce
   either**; the suite uses a strict fake module for exactly this.
5. **(window, thresh) is a RATE, `thresh/window` lnL per repeat.** Shortening
   a window at fixed thresh LOOSENS the bar rather than letting things stop
   sooner. This bit twice: once in the survivor window (caught, thresh is now
   derived) and once in `GB_NUM_REPEAT_PROPOSALS`, where I raised 25→100
   believing it coarsened granularity when it actually made the group rule
   **4x stricter** (its window is 3 PASSES x this many repeats against a
   fixed 4.0). Reverted.
6. **Block-size is not a class label.** I used "≥60 sources = mature" as a
   proxy and concluded survivors were 96.3% of the work; job 630 then showed
   every `rj_warm_search` block sitting at the newborn floor, i.e. the proxy
   misread a birth-dominated move. `rj_prior_removal` is the move that is
   100% mature *by construction*.

**Machine rules for the laptop:** one python process at a time (8 GB);
**never** run `tests.test_gbspecial_flow` (10–26 GB, SIGKILLs the machine);
never `unittest discover`; never bare `git stash` in a shared worktree — and
note that `git checkout HEAD -- <file>` during a paired control will silently
discard uncommitted work (it ate a fix of mine today; copy files instead).

---

## 5. Open questions the 3mo run could help settle

- **Is the yield decay saturation or grid staleness?** `GB_FSTAT_REFIT_EVERY=1`
  is a +22% bet that it is staleness. Untested. A 3mo run with a cheaper fit
  is a good place to A/B it.
- **Does the survivor window actually pay?** It was measured on a proxy that
  turned out wrong. The decisive observation is `rj_prior_removal`'s block
  lengths showing a **105 floor** instead of 255.
- **Per-walker F-stat fits.** Currently one walker's residual (the MIN-lnL
  cold walker, in search and — since 2026-09-26 — in PE too) is replicated to
  all four ranks. Whether per-walker fitting is worth 4x is unmeasured.
- **The all-rungs vertical swap** is designed, audited and NOT built; the
  helpers are installed but inert. See
  `docs/superpowers/specs/2026-09-25-vertical-swap-all-rungs-design.md`,
  including two silent-corruption hazards.

---

## 6. Where the rest of it is

- `docs/superpowers/specs/2026-09-24-v9-recipe-walkthrough.md` — the recipe
- `docs/superpowers/specs/2026-09-25-vertical-swap-all-rungs-design.md`
- The launcher itself is heavily commented and is the most reliable record
  of WHY each knob is where it is; prefer it over any summary, including this
  one.
- Tests that encode the rulings: `test_v9_search_stages.py`,
  `test_inmodel_converge.py`, `test_submit_scripts_layout.py`,
  `test_vertical_swap.py`.
