# 3-month v9 run, 2 GPUs — design

**Date:** 2026-09-26
**Artifact:** `scripts/fstat_proposal/submit_gf_3mo_v9_2gpu.sh` (new), plus two
new knobs in `scripts/fstat_proposal/run_combined_staged.py` and two test files.
**Derived from:** `scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh` — per
`docs/superpowers/specs/2026-09-26-3mo-test-run-handoff.md` §0, **copy the 6mo
file and edit it**; do NOT derive from `submit_gf_3mo_v9.sh`.

Every divergence from the 6mo script is listed in §7. Anything not on that list
is meant to be byte-identical, and `ThreeMonthV9TwinTest` (§9) enforces it.

---

## 0. What this run is

A 3-month global fit that reuses the entire v9 GB search machinery and changes
four things by intent:

1. the **foreground is a fixed, deliberately conservative reference** for the
   first three stages instead of a previous run's fitted value;
2. the run **opens with a PSD-only max-logL search** against a residual that
   still contains the GBs;
3. there is **no warm start and no `gb_search_seed`** — a 3-month run is the
   *source* of warm-start components, not a consumer;
4. it runs on **2 GPUs with 4 walkers**, which is also the first production
   exposure of the walker-block layout with `NWALKERS > NGPUS`.

It is NOT a resume of anything. Fresh store, fresh F-stat epoch cache, fresh
eigen sidecar.

---

## 1. The recipe

```
noise_search   kind=search   psd_pe ONLY, under one joint max-logL criterion
                             galfor FROZEN at the reference (§2)
                             vgb frozen at exact truth, already subtracted
gb_search_1    phase_max on,  opt_snr 8, F-stat peak floor 8    noise FIXED
gb_search_2    phase_max off, opt_snr 5, F-stat peak floor 6.25 noise FIXED
gb_search_3    phase_max off, opt_snr 5, F-stat peak floor 6.25 psd+galfor SAMPLED
full_pe
```

**The three profiles are the 6mo ones unchanged** (confirmed 2026-09-26), i.e.
`V9_SEARCH_STAGE_PROFILES` in `run_combined_staged.py` is not touched — including
`gb_search_3` at phase-max OFF / opt_snr 5 / peak 6.25. Only stage 1 maximizes:
stage 2 is the stage that DROPS the floors, so maximizing on top of that would
add a second optimistic bias to exactly the population least able to afford one,
and stage 3 inherits stage 2's profile while additionally sampling the noise.

Differences from the 6mo recipe, and only these:

* **`gb_search_seed` is gone** (`GB_SEARCH_SEED_ITERS=0`). It exists to seed hard
  off a warm-start posterior before any F-stat fit; with no warm start it would
  be five iterations of `in_model` on an empty model.
* **`rj_warm_search` is absent from all three stages**, and `rj_warm_pe` from
  `full_pe` (`GB_WARM_START_COMPONENTS=` explicitly empty — the documented off
  switch, which `warm()` / `warm_pe()` in `run_combined_staged.py` read).
* **`noise_search` samples psd only**, and **`noise_vgb_search` is dropped**
  (user rulings 2026-09-26).

The 7-slot cycle inside each numbered stage is the 6mo one verbatim:

```
in_model → rj_fstat_search → in_model_fstat → in_model_replace
         → rj_prior_removal → gb_ridge_gibbs → vgb_ridge_gibbs
```

(`rj_warm_search` and `rj_replace` are both absent — the latter because
`GB_SEARCH_RJ_REPLACE=0` carries over from 6mo, switched off on yield.)

### Why stage 0 works the way it does

"Fixed galfor" is the **absence of the `galfor_pe` move**, not a pin applied at
runtime. `setup_acs` rebuilds every walker's sensitivity from the state coords
each pass, so an unsampled branch simply contributes its start coordinates —
which is the same mechanism `gb_search_1/2` already use at 6mo. The value it is
fixed at is `GALFOR_START_PARAMS`.

`JointMaxLogLSearch` wraps `psd_pe` (not `psd_search`): `max_logl_mode` is
consulted in exactly one place and the `_pe` move IS the search move minus its
private plateau loop, so wrapping the `_search` move would nest two loops.
`Stage(kind="search")` is correct because the stopping criterion lives inside
the move and has to span the whole stage.

**The handover to `gb_search_1` is automatic.** The max-logL search leaves the
chain at its maximum-likelihood point, and `gb_search_1/2` carry no psd move, so
they hold exactly that. Nothing needs to be copied.

---

## 2. The foreground reference

```
GALFOR_START_PARAMS = 1.795225757380e-44, 2.787307176476e-03, 5.0,
                      9.9e-03, 1.405721657329e-03
                      (amp, fk, alpha, f_1, f_2 — PHYSICAL/LINEAR)
```

### Provenance

The base is the **90-day add-back fixed point**,
`galfor_6mo_figs/addback_fixedpoint_90d.json`, `history[-1].prior_box_refit`,
amplitude modulation-corrected by `<M_XX> = 0.8080513` (the 90 d window sits on
a DIM stretch of the annual sweep, so the correction RAISES the amplitude; using
the raw fit would understate the foreground by 19%):

```
amp 1.436180605904e-44   fk 2.533915614978e-03   alpha 5.0
f_1 1.0e-02              f_2 1.405721657329e-03
```

This is the same vector the 6mo v9 script ships, because the 6mo run was
deliberately started from the 3-month offline estimate. Here it is the *right*
Tobs rather than a conservative substitute.

The GALFOR brick has a **two-year** resolvable set subtracted (8420 binaries at
SNR>7 / Tobs=1.999 yr), so the binaries that no longer clear SNR 7 at 90 d are
regenerated and added back; that is iterated to a fixed point because added
power demotes more binaries. Converged at iteration 6, n_drop 5227 → 6792,
fit rms 0.0755, fmax 3e-3. **`fit_galfor`'s default fmax 8e-3 RAILS at every
Tobs** — three sessions confirmed that independently; 3e-3 is the working value.

### The inflation

**amp ×1.25, fk ×1.10.** Amplitude per the user; the knee estimated from the
curve. `alpha`, `f_2` untouched.

Read as the factor every SNR threshold is effectively multiplied by,
`sqrt(S_total(inflated) / S_total(reference))` on the A channel:

| f | amp×1.25 | +fk×1.06 | +fk×1.10 | +fk×1.20 |
|---|---|---|---|---|
| 1 mHz | 1.10 | 1.11 | 1.12 | 1.13 |
| 2 mHz | 1.11 | 1.15 | 1.17 | 1.21 |
| **3.5 mHz** | **1.10** | **1.19** | **1.25** | **1.40** |
| 5 mHz | 1.04 | 1.09 | 1.13 | 1.24 |
| 6 mHz | 1.01 | 1.02 | 1.03 | 1.06 |
| 8 mHz | 1.00 | 1.00 | 1.00 | 1.00 |

Three things decide the knee factor:

* **The knee inflation is the only lever that acts where the uncertainty is.**
  Amplitude inflation lifts the whole 0.6–2 mHz plateau, which is the part of
  the fit that is solidly measured; knee inflation acts almost entirely in
  2.5–5.5 mHz, which is where the confusion edge actually moves between galaxy
  models. Above 6 mHz the foreground is 8.5% of the total and both levers
  vanish.
* **×1.06 is the self-consistent partner of ×1.25.** Binary number density goes
  as `dN/df ∝ f^(-11/3)`, so scaling the galaxy by G moves the confusion edge as
  `G^(3/11)`; at G = 1.25 that is ×1.063. ×1.10 is deliberately slightly beyond
  it, because different galaxy models differ in shape and not only in
  normalization.
* **×1.20 is the line not to cross.** It reproduces `k ≈ 2.0` at 3.5 mHz, which
  is the measured 6mo pathology (`k = 1.85–2.00` over 3–4 mHz) that made the
  stock SNR-8 F-stat floor an effective true-SNR threshold of 11.3 and hid 693
  findable binaries, 506 of them in 3–5 mHz.

At the shipped values the stage-1 SNR-8 F-stat floor acts like ~10 at 3.5 mHz
and ~9 below 2 mHz; stage 2's 6.25 acts like ~7.8 and ~7.0. That is a real cost
and it is accepted deliberately: it is bounded, it is confined to the confusion
band, and **`gb_search_3` releases the foreground**, so the inflation shapes
three stages and then stops mattering.

⚠ It cuts against the standing search policy ("err toward grabbing, not
missing"). If the stage-1/2 yield in 3–5 mHz reads badly, fk ×1.06
(`2.685950551877e-03`) is the fallback and needs no other change.

### Two details that are not cosmetic

* **`f_1` is moved off the prior rail**, 1.0e-2 → 9.9e-3. The v9 launch audit
  flagged the exact-boundary value. It is physically irrelevant here: with
  `fk = 2.79 mHz` and `f_2 = 1.41 mHz` the tanh has already cut the spectrum by
  ~1e-3 by the time the exponential starts to bite, so the in-band curve moves
  by well under a percent. `f_1` is not a measurement at 3 months — the
  unconstrained fit wants ~0.365 Hz (i.e. the exponential OFF and the tanh doing
  all the cutting), refitting inside the prior box changed the rms by nothing
  (0.07547 vs 0.07547), and that is what an unidentified parameter looks like.
  Expect the branch to wander along the exp-vs-tanh degeneracy once
  `gb_search_3` releases it. Not a pathology.
* **`alpha = 5.0` is interior only because `GALFOR_ALPHA_MAX=20.0`** is exported
  (inherited from 6mo). Do not drop that line.

### What stage 0 is actually fitting

The residual in stage 0 contains the **whole** galaxy — nothing has subtracted a
GB yet — while the reference models only the *unresolved* part. The 2-parameter
PSD will therefore sit high, absorbing the resolvable set. That is inherent to
the design and is bounded by the PSD's rigidity (two levels with fixed spectral
shapes, not a free spectrum); the inflation absorbs part of it rather than
leaving it all to `Sa_a`. `gb_search_3` is what fixes it properly.

---

## 3. The code change

`scripts/fstat_proposal/run_combined_staged.py`, two new env knobs:

**`STAGE_NOISE_PSD_ONLY=1`** — the standalone noise stage's `_noise_names` drops
`galfor_pe`. Scope is the standalone stage only: `gb_search_3`'s interleaved
`noise_joint_search_1..4` and the `noise_vgb_joint_search` rider keep both moves,
because stage 3 is the stage that releases the foreground.

It **refuses** to run with `GALFOR_START_PARAMS` unset. Without a start pin the
galfor branch begins at a prior draw and then sits frozen there through
`noise_search`, `gb_search_1` and `gb_search_2` — three stages hunting GBs
against a random foreground. This is the same failure the existing half-pin
guard on `_noise_pinned` exists to prevent, one layer down.

**`STAGE_SKIP_NOISE_VGB=1`** — drops the `noise_vgb_search` stage. Separate knob
on purpose: "the noise stage samples psd only" and "there is no vgb burn-in
stage" are two decisions, and folding them into one flag is how a knob acquires
a second, invisible job. With `VGB_START_FACTOR=0` the VGBs start at exact truth
and there is nothing for a max-logL burn-in to find; they begin sampling in
`gb_search_1`, where `vgb_pe` already rides the fixed-noise stages.

Also fixed in the same edit: the `_noise_pinned` log line, which currently
reports "PSD_START_PARAMS / GALFOR_START_PARAMS unset" when only one of them is.

⚠ **Two stage assemblies exist in this file** — the `GB_ONLY` composition and
the full one. The noise stages live only in the full one, so only it needs the
change; the handoff's §4 trap ("adding a stage to one only is easy and silent")
is noted here so the next person checks rather than assumes.

---

## 4. No warm start, no seed store

`GB_WARM_START_COMPONENTS=` (explicitly empty) is the only supported way to run
without the warm move; `warm()` and `warm_pe()` key on it being non-empty.

With it empty, the seed store has no remaining job — the warm mixture was one of
its two products and the other was `PSD_START_PARAMS`, which this run
deliberately does not pin because searching the PSD **is** stage 0. So
`GF_SEED_STORE`, `GB_WARM_START_SOURCE_STORE`, `GB_WARM_START_SOURCE_TOBS`,
`GB_WARM_START_LAST_K`, `GB_WARM_START_FLOOR_EPS`, `GB_WARM_START_CIRC_IMAGES`
and `GB_SEARCH_3_WARM_EVERY` all come out, and with them the `[V9-SEED]` and
`[WARMSTART]` preflight blocks. `GB_REPLACE_WARM_PASS` goes 1 → 0 (there is no
warm container for a warm pass to draw from; `rj_replace` is off anyway, but a
knob that would silently mean "pass 1 has nothing to propose from" should not be
left armed).

**`PSD_START_PARAMS` must stay unset.** Setting it flips `_noise_pinned` and
`run_combined_staged.py` then SKIPS the noise stages entirely — deleting stage 0.
`STAGE_FORCE_NOISE_SEARCH=1` is the escape if a start value is ever wanted.

---

## 5. Hardware shape

Intended invocation:

```
NGPUS=2 ./scripts/fstat_proposal/submit_gf_3mo_v9_2gpu.sh
```

→ `gpu-80-spot`, `--nodes=1 --gres=gpu:2 --ntasks=3`, walker-block layout
(`GF_LEGACY_RANK_LAYOUT=0`), `N_COMPUTE = 1 × 2 × 1 / 1 = 2` compute ranks plus a
saver. `NWALKERS=4` divides by 2 → **2 walkers per rank**.

This is the point of the shape: it is the first production run with
`NWALKERS > NGPUS`. The NGPUS=4 branch of the dispatch block is kept intact so
the layout machinery and its tests stay uniform.

Three consequences to expect rather than discover:

* **Per-rank rows double** against the 6mo 4-GPU shape (1 walker/rank there).
  The GB moves scale with rows, so a 3mo iteration on this shape lands in the
  same wall-clock neighbourhood as a 6mo iteration on four GPUs despite half the
  data.
* **Total resident sub-band slots halve** (2 GPUs, not 4). `GB_N_SUBBANDS` is
  per-GPU, so per-GPU memory is unchanged; what changes is how many sequential
  passes a band unit takes.
* **Spot, not on-demand.** NGPUS=2 is a single node, so a preemption is
  recoverable — which makes `MIDIT_CHECKPOINT` load-bearing rather than a
  nicety. It stays on at `MIDIT_CHECKPOINT_MIN_INTERVAL=600`.

---

## 6. Data, and the MBH forensics

`SOURCE_TYPES=COMBINED,GB,VGB`. The COMBINED stream is mojito's pre-summed L1
data — it establishes the data and the orbits and contains *everything*. The GB
and VGB entries after it are **catalogue-only** (nleaves sizing, VGB seeding,
F-stat overlays); their bricks are not read. The loader refuses COMBINED+NOISE
and COMBINED+GALFOR as double-counts; the NOISE brick must still exist on disk
because `UNEQUAL_ARM=1` reads its `/ltts` table by path.

`MBHB_IDS` / `EMRI_IDS` / `SOBHB_IDS` are **UNSET, not set-empty**. Set-empty
replaces the settings default `{"MBHB": [], "EMRI": [1], "SOBHB": []}` with three
empty lists and `import lisatools.globalfit.stock.erebor` then FAILS outright
from an unrelated variant's constructor. `STAGE_SKIP_SOURCE_SEARCH` is removed —
it *raises* when no source branch is armed.

So the MBH, EMRI and SOBBH signals are in the data and nothing models them. That
was checked rather than assumed.

### MBHB: the merger just after 3 months is clean

The question was whether MBHB id18 — `TimeCoalescencePhenomTPHMSSBFrame` =
**91.97 d**, i.e. 1.97 d after a 90-day window closes — deposits power in band.
It does not, and the reason is its mass, not its timing: SSB chirp mass
6.855e6 M☉ (M_tot 1.722e7), so

```
f_ISCO = 2.55e-4 Hz      f22 at t = 90 d = 8.8e-5 Hz      2nd harmonic 1.8e-4 Hz
```

The source merges *at* the analysis floor. Measured directly off
`MBHB_731d_2.5s_L1_source18...h5` (`tdis/A2`, 2% cosine taper, whitened with the
scirdv1 A PSD), the in-band SNR over [0, 90 d] is **1e-8 of** the same quantity
over [0, 120 d], the window that contains the merger.

**Every other MBHB is below the floor at 90 d except id5.** id5 (M_c 3.05e5,
t_c 104.74 d, catalogue SNR 227) crosses 2.5e-4 Hz at t = 82.9 d and reaches
2.9e-4 Hz at the window edge:

```
SNR deposited in [0, 90 d] above MIN_FREQ=2.5e-4  :  <= 63   (0.25-0.29 mHz)
SNR deposited above GB_MIN_FREQ=5.5e-4            :  0.0000
```

The bound is a Newtonian SPA ratio normalized on the catalogue SNR with no
merger or ringdown in the denominator, so the true number is lower — probably by
a factor of a few.

**Verdict: the GB search is untouched.** The exposure is to *stage 0*: an
unmodelled source of order SNR 20-60 sitting in WDM layers 2-3, and the lowest
layers carry disproportionate weight in the PSD fit (the `MIN_FREQ` forensics
measured layer 1 alone at 43% of the fit's entire chi² budget). Accepted,
flagged, with two escapes on the shelf:

* `MBHB_IDS=5` **plus** `MBH_MERGER_TIME_BUFFER=20` — `prepare_mbh_branch` keeps
  an MBH only if `t_merge < observation_end + buffer`, and at the 2-day default
  104.74 d against a 90 d window is dropped;
* `MIN_FREQ=4.2e-4` (layer 4) — clean, but it **drops the four lowest VGBs**
  (0.3117 / 0.3364 / 0.3365 / 0.3392 mHz), which is exactly what the
  4e-4 → 2.5e-4 move existed to avoid.

**Do this before launch:** the id5 brick exists on the cluster, so the `gf_snr`
pattern (`erebor` → `fit.build()` → `setup_acs(rebuild_residuals=False)` →
`sqrt(ac.inner_product().real)`) with `TOBS_TARGET=7776000 MERGER_FRAC=1.0` and
`MBHB_IDS=5` settles the number in minutes.

### EMRI / SOBBH

Sub-threshold at this baseline: the earlier short-window census measured EMRI
id1 at 2.47 (1 month) → 5.67 (4 months) and SOBHB id0/id1 at 2.31/2.68 → 6.96/
6.73. At 90 d they are ~4-5 each. Eight EMRIs and six SOBHBs unmodelled is the
realistic-full-data-challenge configuration the 2026-09-14 ruling accepted at
6 months; it is smaller here. Note SOBHBs live near ~10 mHz, i.e. inside the GB
band — watch for GB leaves tracking a chirping source there.

---

## 7. The complete diff against `submit_gf_6mo_v9_4gpu.sh`

### 7.1 Tobs and its derived settings

| knob | 6mo | 3mo | why |
|---|---|---|---|
| `TOBS_TARGET` | `15552000` | **`7776000`** | 90 d; grid resolves Nf 1440 × Nt 2160 × dt 2.5 (the 6mo Nt is an exact factor 2 of this) |
| `SIGHET_NT_LAYER` | `120` | **unset** | default 64 snaps to 60; 2160/60 = stride 36 = **36 h**, the same constant temporal density the 6mo 120 gives |
| `EDGE_CROP_WAVELETS` | `60` | **unset** | default 20. Taper is `ceil(alpha/2 × Nt)` = 11 layers at Nt=2160, +8 margin = 19 ≤ 20 — passes with one layer spare. At 6mo it is 22+8=30 and the build guard raises without the 60 |
| `GB_NLEAVES_MAX` | `15000` | **`10000`** | shallower confusion at 3 mo |
| `GB_N_SUBBANDS` | `8192`/GPU | **`32768`**/GPU | 3mo slot ≈0.25 MB against 6mo ≈0.5 MB; the value the 3mo 10-walker arm ran healthy through iteration 210 |
| `GB_RJ_INMODEL_CHUNK` | `32768` | **`65536`** | byte parity (6mo cells are ~2× the bytes) |
| `BASE_FILE_NAME` | `gf_prod_6mo` | **`gf_prod_3mo`** | every analysis tool takes the DIRECTORY, so this name is free |
| `STORE_DIR` | `.../gf_prod_6mo_v9_4gpu/` | **`.../gf_prod_3mo_v9_2gpu/`** | fresh store, mandatory |
| job name / `--output` / `SLURM_LOG` | `gf6mo_v9_4gpu` | **`gf3mo_v9_2gpu`** | |

### The sig-het staging knobs do NOT revert (user ruling 2026-09-26)

The v8 3mo arm ran the aggressive sizing — `GB_INMODEL_SETUP_BATCH=0`
(one-block staging), `GB_SIGHET_FOLD_MAX_BYTES=8 GiB`, both `*_MEMPOOL_FREE=0`
— and `ThreeMonthTwinTest` has that split on its reversion list. **This run does
not take it.** All four keep the 6mo post-OOM values:

```
GB_INMODEL_SETUP_BATCH=2048        GB_SIGHET_FOLD_MAX_BYTES=1073741824 (1 GiB)
GB_INFOMAT_MEMPOOL_FREE=1          GB_INMODEL_BATCH_MEMPOOL_FREE=1
```

That pairing matters, because it is what makes `GB_N_SUBBANDS=32768` safe here.
The sig-het stash goes as `CELLS × N_sparse_t` and the two knobs MULTIPLY:
60 × 32768 = 1.97e6, which is above the 23-month 1.1e6 precedent and not far
from the 2.2e6 product that OOM'd v4@270. What bounds the actual transient is
the fold chunker — `gbsignalhetcomputations.py` sizes each chunk to FILL
`GB_SIGHET_FOLD_MAX_BYTES`, so an 8 GiB cap builds an ~8 GiB transient *by
design*, and that plus one-block staging is precisely what killed the 6mo run in
`bin_fold_real`. At a 1 GiB cap and a 2048-wide setup batch the transient is
bounded whatever the slot count, so the run gets 32768 resident slots (fewer
sequential passes) without the transient that the 3mo v8 arm was gambling on.

All four are transient/scheduling knobs — no stored number depends on them, so
a mid-store resume is safe — and all four stay env-overridable. Cost is more
chunks per unit (launch overhead), not accuracy.

**If GPU memory still trends past ~70 GB in `gpu_util_*.csv`, back
`GB_N_SUBBANDS` off to 8192 before touching anything else, and never reach for
`SIGHET_NT_LAYER`** (not a mid-store knob, and it multiplies with
`GB_N_SUBBANDS` in the same byte product).

### 7.2 Hardware

| knob | 6mo | 3mo |
|---|---|---|
| intended `NGPUS` | 4 (2 nodes × 2 GPUs, on-demand) | **2 (1 node × 2 GPUs, spot)** |
| `NWALKERS` | 4 (1/rank) | 4 (**2/rank**) |

### 7.3 Data and source branches

| knob | 6mo | 3mo |
|---|---|---|
| `SOURCE_TYPES` | `COMBINED,GB,VGB,MBHB,EMRI,SOBHB` | **`COMBINED,GB,VGB`** |
| `MBHB_IDS` / `EMRI_IDS` / `SOBHB_IDS` | `2,5,16,18` / `0..7` / `0..5` | **unset** (not set-empty) |
| `STAGE_SKIP_SOURCE_SEARCH` | `1` | **removed** (raises with nothing armed) |

The mbh/emri/sobbh knob block (`*_NTEMPS`, `*_NUM_PROP_REPEATS`,
`*_PERMUTE_EVERY`, `*_EIGEN_SCOPE`, `*_EIGEN_REFRESH`, `*_INNER_MOVE_KIND`,
`*_START_FACTOR`, `SOBBH_M_BAND_HALF_WIDTH`, `SOBBH_CHECK_LL_EVERY`, `EMRI_EPS`)
**stays in the file under an "INERT AT 3 MONTHS — no source branch is armed"
banner.** Keeping it keeps the twin diff small and keeps the knobs right for a
future 3mo run that does arm them; the banner stops the file lying about what it
runs.

### 7.4 Recipe

| knob | 6mo | 3mo |
|---|---|---|
| `GB_SEARCH_SEED_ITERS` | `5` | **`0`** |
| `GB_WARM_START_COMPONENTS` | `${STORE_DIR}/warmstart/...npz` | **empty** |
| `GF_SEED_STORE`, `GB_WARM_START_SOURCE_STORE`, `_SOURCE_TOBS`, `_LAST_K`, `_FLOOR_EPS`, `_CIRC_IMAGES`, `GB_SEARCH_3_WARM_EVERY` | set | **removed** |
| `GB_REPLACE_WARM_PASS` | `1` | **`0`** |
| `PSD_START_PARAMS` | from the seed store's maxlogL cold walker | **not set** (searching it IS stage 0) |
| `GALFOR_START_PARAMS` | 3mo fixed point, uninflated | **inflated reference, §2** |
| `STAGE_NOISE_PSD_ONLY` | — | **`1`** (new knob) |
| `STAGE_SKIP_NOISE_VGB` | — | **`1`** (new knob) |

### 7.5 One correctness fix the fresh store unlocks

| knob | 6mo | 3mo |
|---|---|---|
| `MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM` | `0` | **removed** → defaults to 1 (user ruling 2026-09-26) |

That line exists solely so a v8-lineage store stays resumable across the
2026-09-23 change that made the NOISE-brick scalar fit use the brick's own link
delays; its own comment says "DELETE THIS LINE for a fresh store — the arm model
is the better answer". The unequal-arm reference fit
(`1.500004011496e-11, 3.000107254658e-15` against the equal-arm
`1.496182116469e-11, 2.982411739286e-15`) is `general.psd_injection`, i.e. the
"truth" line every monitor and whitening test compares against — and the PSD-bias
forensics concluded the estimator was fitting EQUAL-arm while the run ran
unequal. With `COARSE_Q=1` there is no coarse fiducial digest to invalidate.

---

## 8. What deliberately does NOT change

**The whole v9 GB stack**: the three-stage profiles and
`V9_SEARCH_STAGE_PROFILES`, V9-1 per-row in-model convergence (window 250,
survivor 100 with its derived dll 1.6, ceiling 20000, gate 0.5, stop 0.5, refill
on, classes newborn+mature), V9-6 per-sub-band group convergence
(`GB_NUM_REPEAT_PROPOSALS=25`, iters 3, dll 4.0, scale flat, max_passes 1000),
the staged RJ scheduler (`GB_RJ_DIRECT_BATCH=0`), caps off
(`GB_LEAF_CAP_START=` empty and its four companion knobs), fancy tempering off
with vertical on, `GB_TEMPER_CELL_ORDER=band`, the per-(walker,band) RJ shutoff
at `CONV_ITER=3`, `GB_PLATEAU_ITERS=20`, `GB_FSTAT_REFIT_EVERY=1`, the adaptive
stage-B sky grid and its 2.0 GB per-group budget, `FSTAT_PEAK_MIN_SNR=8.0` as
epoch 0's fit value, `GB_OPT_SNR_LIMIT_SEARCH=8.0` as the constructed value,
`GB_SEARCH_RJ_REPLACE=0`, exact-fine noise (`COARSE_Q=1`, `COARSE_GPU_MODE=off`),
`GALFOR_LOG_SAMPLING=1`, `GALFOR_ALPHA_MAX=20.0`, the unequal-arm noise model
and `WDM_PSD_METHOD=layer_calibrated`, `MIN_FREQ=2.5e-4` (layer 2 — keeps all 55
VGBs), `GB_SUBBAND_DIVISOR=8` and `GB_BAND_UNIT_STRIDE=9` (a Hz grid, so
Tobs-independent), and the whole VGB block.

**Sig-het accuracy knobs stay put, and that is a decision, not an omission.**
`SIGHET_N_CP=256` (verified on the 3-month production grid: 32 → 256 nodes takes
the scored anchor from max |log hh ratio| 0.31 to 1.5e-4 at +2.6% setup cost),
`SIGHET_TUKEY_ALPHA=0.01`, `GB_SIGHET_TRUST_PHASE_C=49`,
`GB_SIGHET_REFRESH_DPHASE=0`, `GB_SIGHET_REFRESH_MIN_BETA=0`,
`GB_SIGHET_REFRESH_EVERY=50`.

The last one is the only one that looks like it might want reverting (the 3mo v8
arm ran 25, and 50 was a 2026-09-25 cost cut measured on 6mo job 628). It stays
at 50: reference staleness costs phase error, phase error from a given parameter
drift scales with Tobs, so a 50-repeat cadence is **safer** at 3 months than at
6, where it is already shipped. The two instruments that would show otherwise
are `[GB_TRUST]` rejection fraction (was 4-13%) and the end-of-block
`ll AUDIT vs exact` COLD median (was 0.026-0.073, max 0.14-0.73).

---

## 9. Tests

* **`tests/test_v9_search_stages.py`** (extend `FullCompositionTest`):
  `STAGE_NOISE_PSD_ONLY=1` drops `galfor_pe` from the standalone noise stage;
  `gb_search_3` still carries both psd and galfor in all four interleaved
  slots and in the leading rider; `STAGE_SKIP_NOISE_VGB=1` removes the
  `noise_vgb_search` stage and nothing else; `STAGE_NOISE_PSD_ONLY=1` with
  `GALFOR_START_PARAMS` unset raises.
* **`tests/test_submit_scripts_layout.py`**: add `submit_gf_3mo_v9_2gpu.sh` to
  `SCRIPTS`, add a dispatch scenario for `NGPUS=2 / NWALKERS=4` asserting
  1 node, `gpu:2`, `--ntasks=3`, `GF_LEGACY_RANK_LAYOUT=0`, `N_COMPUTE=2`, and
  add **`ThreeMonthV9TwinTest`** comparing it against
  `submit_gf_6mo_v9_4gpu.sh` with §7 as the declared delta list — same shape as
  the existing `ThreeMonthTwinTest`, which guards the v8 pair.

`ThreeMonthTwinTest` (the v8 pair) is a known, accepted failure and is out of
scope here; this new class is what makes the v9 divergence declared.

**Machine rules while testing:** one python process at a time (8 GB laptop);
**never** run `tests.test_gbspecial_flow` (10-26 GB, SIGKILLs the machine);
never `unittest discover`.

---

## 10. Watch list on first launch

In rough order of what would invalidate the run:

1. `[combined] stage` lines — the composition must print exactly
   `noise_search / gb_search_1 / gb_search_2 / gb_search_3 / full_pe`, with
   `noise_search` carrying `noise_joint_search` and **no** `galfor_pe`, and no
   `rj_warm_search` anywhere.
2. `grep "sig-het engine resolved"` — wants **`nt_layer=60 (stride 36)`,
   sparse spacing 36.0 h, `tukey_alpha=0.01`, `n_cp=256`**. Nothing else echoes
   these, and the defaults are being taken rather than pinned.
3. `[V9-PREFLIGHT]` must pass (it RESOLVES the convergence knobs from the
   environment rather than probing for the names — the only check that catches
   a name mismatch).
4. `[MAXLOGL]` in `noise_search`: rounds and the plateau verdict. This stage's
   output is the reference for two whole stages. ⚠ `MAXLOGL_TOL=20` is GLOBAL to
   `JointMaxLogLSearch` and was tuned for the gb_search riders, so it also
   governs this stage — **kept at 20 (user ruling 2026-09-26)**. With
   `NOISE_SEARCH_CHECKS=5` (the default, pinned explicitly for the record) the
   plateau needs five consecutive flat rounds, which is what makes 20 acceptable
   on a 2-parameter fit whose cold logL is ~1e8. Both are pinned in the file so
   the trade is visible rather than inherited.
5. The psd handover: the `gb_search_1` opening sensitivity must equal the
   `noise_search` maxlogL point. Nothing copies it — it holds because
   `gb_search_1` carries no psd move — so it is worth confirming once.
6. `[GB_IMGROUP]` and `[GB_IMCONV]` calibration lines (first exposure of the
   group rule at 3 months; the pass distribution has never been read anywhere).
7. `[peaks]` / `[stageA]` — with `GB_FSTAT_REFIT_EVERY=1` the peak count must
   start falling by the third epoch or the cadence is a fixed per-iteration bill.
8. `[GB_TRUST]` and the cold `ll AUDIT vs exact` median, against the 6mo
   numbers in §8.
9. GPU memory in `gpu_util_*.csv` — this is the aggressive sig-het sizing on a
   2-GPU node.
10. `[MIDIT_CKPT]` lines — spot partition.

---

## 11. Open items

* **Measure id5's in-band SNR on the cluster** (§6) before committing a long
  allocation. If it comes back well above the ≤63 bound, arm `MBHB_IDS=5` with
  `MBH_MERGER_TIME_BUFFER=20`.
* **The 3-5 mHz yield in `gb_search_1/2`** is the read on whether fk ×1.10 was
  too conservative. Fallback: fk ×1.06 (`2.685950551877e-03`), no other change.
* **`NWALKERS > NGPUS` is unexercised in production.** The WP7 transport gates
  passed and the 6mo campaign is the statistical read at 1 walker/rank; 2
  walkers/rank on one node is new. A `GF_LAYOUT_DRY_RUN=1` pass costs nothing.
* **`GB_FSTAT_REFIT_EVERY=1` is still an untested +22% bet** that the yield decay
  is grid staleness rather than saturation. The handoff names a 3mo run with a
  cheaper fit as the right place to A/B it — not done here, deliberately, to keep
  this run a single-variable change against 6mo.
