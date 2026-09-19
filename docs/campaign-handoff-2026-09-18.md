# 4-GPU campaign handoff — 2026-09-18

Status of the two production global-fit runs, how to assess a snapshot, and
what is still open. Written to be picked up on a different machine: every
path here is either in the repo or on the cluster, and nothing depends on
the originating laptop's local state.

Read [`conventions.md`](conventions.md) and the repo `CLAUDE.md` first for
the project-wide rules. This document is campaign state, not conventions.

---

## 1. The two runs

| | 3-month | 6-month |
|---|---|---|
| submit script | `scripts/fstat_proposal/submit_gf_3mo_v8_4gpu.sh` | `scripts/fstat_proposal/submit_gf_6mo_v8.sh` |
| store base | `gf_prod_3mo_testing` | `gf_prod_6mo_testing` |
| run dir | `gf_prod_3mo_v8_4gpu` | `gf_prod_6mo_v8_4gpu` |
| `TOBS_TARGET` | 7 776 000 s (90 d) | 15 552 000 s (180 d) |
| branches | NOISE, GB, VGB | + MBH, EMRI, SOBHB |
| warm start | none (it is the SOURCE of components) | yes (`rj_warm_search`) |
| GPUs | 4 (2 nodes x 2), one walker per rank | same |
| walkers | 4 | 4 |

Both run one walker per compute rank (`B = 1`), which is load-bearing for
several behaviours below.

### Last known state

**3-month, snapshot (4), store iteration 210.** Healthy, in `full_pe`
since ~iteration 188.

- GB leaves per walker `791 / 767 / 757 / 756` (3071 total); VGB `55/55`
  alive on every walker.
- cold logL plateaued at ~`5.2671e7`, walker spread ~`3e2`.
- **noise converged on all four walkers**: `Soms_d` x1.0005-1.0023,
  `Sa_a` x1.007-1.026 against injection.
- galfor amp `1.56-1.94e-44` on all four, all live.
- 8 F-stat epochs; wall ~140 s/iteration in `gb_search`, ~63 s in `full_pe`.

**6-month, snapshot (13), store iteration 47.** Was healthy, then **OOM'd**
(see §5).

- GB leaves `1090 / 1067 / 1075 / 1056` (4288); VGB `55/55`.
- cold logL `1.0427e8`, monotone increasing.
- **noise converged**: `Soms_d` x1.005-1.008, `Sa_a` x0.977-1.151 — it was
  x2.07-3.08 at iteration 13, so this resolved during the run.
- 3 F-stat epochs; baseline `gb_search` iteration 304.7 s median.

**I do not know the state after the OOM relaunch.** The memory fix is on
`dev` and is the default in the submit script; whether it was relaunched,
and how far it has got, has to be checked on the cluster.

---

## 2. The headline physics result

**The galfor/Sa_a degeneracy resolves itself as GB starts subtracting.**

Early in each run the noise model has to absorb the entire unmodelled
galaxy (at 3mo iteration 23 the GB branch held ~281 leaves out of
15 539 324 catalogue binaries). It can put that power either in the
galactic-foreground model or in the instrument's acceleration noise, and
those two are degenerate at low frequency. Early snapshots showed three of
four 3-month walkers resolving it the WRONG way: galfor driven to
identically zero (`alpha` 10-18 with `f_1` far below the band makes
`exp(-(f/f_1)^alpha)` kill it) and `Sa_a` inflated ~3.5x to compensate.

The decisive diagnostic, measured across walkers over 0.25-25 mHz:

- spread of the **instrument** noise alone: **9.41x**
- spread of the **total** (instrument + galfor): **1.58x**

The total budget was agreed to within ~1.6x. Only the *split* was
degenerate. **That is the number to compute if this reappears** — it
separates "the noise model is wrong" from "the decomposition is
ambiguous".

By 3mo iteration 210 and 6mo iteration 47 both runs had converged all four
walkers. Nothing was done to make that happen; the chains found it.

**Refuted, do not re-investigate:** the excess is not an unmodelled MBHB.
(a) The 3-month run does not load the MBHB brick at all —
`SOURCE_TYPES=NOISE,GB,VGB`, and `COMBINED` (the pre-summed all-class
stream) is not among them. (b) The first MBHB merges **1.96 days after**
the 3-month window closes and its in-window support stops at 0.166 mHz,
below the 0.25 mHz band floor — in-band SNR is exactly 0. The excess also
fell monotonically with frequency, the f^-4 signature of `Sa_a`, not a
chirp.

---

## 3. Assessing a snapshot

### The command

```sh
INTERLOCK=1 scripts/diagnostics/snapshot_to_html.sh \
    "gf_prod_6mo_v8_4gpu_snapshot.tar (13).gz" ~/monitor_6mo.html
```

Positional: tarball, output HTML, optional work dir. Env knobs:

| knob | meaning |
|---|---|
| `INTERLOCK=1` | serialise against other pythons. **Default OFF for humans; an unattended caller should always set it.** |
| `ITERATION=` | row to build truth at (default `iteration - 2`) |
| `CATALOGUE=` | GB catalogue hdf5 **or its directory**; also `MOJITO_CAT`, `MOJITO_CACHE_DIR` |
| `FLO=` / `FHI=` | band in Hz |
| `SKIP_TRUTH=1` | reuse an existing truth npz — **see the trap in §4** |
| `PY=` | interpreter |

It extracts the tar, finds the run dir and live store, builds the truth
set, and renders the page. Underneath it is two positional-arg scripts:

```sh
python scripts/diagnostics/build_truth.py STORE.h5 \
    --iteration <iter-2> --out "$RUN"/gb_truth_3to21.npz \
    --kappa-out "$RUN"/kappa_grid.npz
python scripts/diagnostics/gf_monitor_gen.py "$RUN" out.html
```

`gf_monitor_gen.py` has **no `--out` flag** — `argv[1]` is the run dir,
`argv[2]` the output path.

### What to look at, in order

1. **`missing=K`** on the generated page, and every MISSING reason. Silent
   section drops have produced wrong analyses before. `missing=3` is the
   normal floor for a snapshot tar (hi-f census, F-stat peak caches, the
   extract keep-window note).
2. **Noise**: `Soms_d` and `Sa_a` as ratios to injection
   (`noise_soms_d=1.4961821164690655e-11`,
   `noise_sa_a=2.9824117392856994e-15`), per walker, at the last row where
   `log_like`, psd AND galfor are all valid. Then galfor amp per walker —
   if any is ~0 while `Sa_a` is inflated, that is the degeneracy of §2.
3. **GB leaf counts** per walker and the trend.
4. **cold `log_like`** per walker: monotone? walker spread narrowing?
5. **Acceptance**: psd/galfor in-model per rung, swap per adjacent pair.
6. **`non-positive eigenvalue`** counts across `globalfit_run.log` and every
   rank log, with the worst `lambda/lambda_max`.
7. **Timing**: `[GB_TIMING]` per-move totals; `[GF_TIMING]` if present.

### Reference values

| quantity | healthy | what it means otherwise |
|---|---|---|
| `Soms_d` / injection | ~1.00 | converged in both runs |
| `Sa_a` / injection | ~1.0 | >2 with galfor ~0 = the §2 degeneracy |
| galfor `alpha` | 0.1-2 | at the `GALFOR_ALPHA_MAX=20` rail = slope excursion |
| PT swap acceptance | 0.2-0.3 | ~1.0 = ladder collapsed, those rungs buy nothing |
| psd swap, top pair | 0.0000 | **expected** — the beta=0 rung never exchanges |
| `missing=` | 3 | more = check every reason |

---

## 4. Traps

Every one of these produced a wrong analysis or a crash at least once.

**Snapshot handling**

- **Stale archive.** Check the tar's mtime FIRST.
- **Two h5 in a run dir.** A tar refresh cannot DELETE, so an older-format
  store survives beside the live one, plus ~800-byte stubs. Pick by size
  and the `iteration` attr, and say which you picked.
- **`*_extract.h5` reduced stores.** Chain COORDS outside the last
  `--keep` rows are ZERO-FILLED. `inds`, `log_like`, band/cap and **all
  noise chains are valid for every row**.
- **Mid-flush lag.** Sub-backends lag the main backend by a row. The
  3-month snapshot had psd at 211 rows and galfor at 210. Use the last row
  where all three are valid.

**Store layout**

- **`noise_model_identity` is an HDF5 GROUP** at
  `global_fit/noise_model_identity`, one attribute per key — **not** a JSON
  string attribute. A reader assuming JSON finds nothing, falls back to its
  defaults, and silently reinterprets the store. Use
  `lisatools.globalfit.stock.erebor.noise.read_noise_model_identity()`.
- **`iteration` is on the `global_fit` group**, not the file root. The root
  carries only extract bookkeeping (`extract_torn_reads`). Assume
  `global_fit` for this format.

**Truth sets**

- **`build_truth.py --iteration` defaults to 78.** On a young store that
  reads an UNWRITTEN row and yields all-inf SNRs **with no error**. Always
  pass `iteration - 2`.
- **`--kappa-out` defaults to the CWD**, where it clobbers the LAT-root
  3-month kappa cache. Send both outputs into the run dir.
- **The monitor's truth filename is HARDCODED** to `gb_truth_3to21.npz`,
  looked up in the run dir then the CWD, with no override flag. A truth set
  written under any other `--out` name is silently not found — this is the
  usual cause of "the truth crosses are missing".
- **A stale truth npz loads fine and is silently wrong.** The `tobs` guard
  catches a 3-month set on a 6-month run, but a set built at a different
  iteration or with a narrower `--flo/--fhi` passes every check and just
  yields fewer crosses. The band default widened from 3 mHz to 0.5556 mHz
  on 2026-09-18, so any older npz shows only sources above 3 mHz.
  **`SKIP_TRUTH=1` is a trap in exactly this situation.**
- **`galfor_log_sampling`.** Under it the four log columns are stored as
  `log10` and `alpha` stays linear. Handing the raw row to the foreground
  model makes `amp` and `f_1` negative, so `(f/f_1)**alpha` is NaN — and
  NaN does not raise. Route through `galfor_params_to_physical()`, keyed
  off the STORE's flag, never an env var.

**Generator**

- **CWD matters.** `gf_monitor_gen.py` reads AND REWRITES
  `gf_arm_<tag>.npz` in the working directory. Render from a scratch dir or
  you overwrite the repo-root arm caches.
- **Arm caches drop out** when their source count does not match the current
  truth set. Expected, not a fault — it accounts for 5 MISSING entries on a
  page built against a rebuilt truth set.

**Machine (if running on a laptop)**

- **One python process at a time.** Concurrent interpreters have hard-crashed
  an 8 GB box. Guard with
  `until ! pgrep -x python >/dev/null && ! pgrep -x python3.12 >/dev/null; do sleep 20; done`.
- **NEVER `pgrep -f` with an interpreter path.** It matches the waiting
  shell's own command line and deadlocks forever. Twice in production.
- **Never run or import `tests/test_gbspecial_flow.py`** — 10-26 GB.
- Pin every thread pool to 1.
- **Worktrees**: the scikit-build-core editable finder hard-maps `lisatools`
  to the main checkout and beats `PYTHONPATH`. Use the `.wtenv` shim
  (`LAT_WORKTREE_SRC=<wt>/src PYTHONPATH=<wt>/.wtenv`) or you will silently
  test the wrong tree.

---

## 5. The 6-month OOM (fixed on `dev`, verify on relaunch)

Died in `gb_search` — **not** `full_pe` — at ~4288 leaves:

```
setup_in_model -> gbsignalhetcomputations.py:1074 bin_fold_real
 -> signal_het.py:119  En = c0[...,:,None,:,:] * iC * c0[...,None,:,:,:]
OutOfMemoryError: 2,782,742,528 bytes (89,586,791,936 allocated)
```

`gbsignalhetcomputations.py:1047` sizes each fold chunk to **fill** the byte
cap, so an 8 GiB cap builds an ~8 GiB transient by design. The **code
default is 1 GiB**; 8 GiB was the script's override, with one-block staging
(`SETUP_BATCH=0`) on top.

The submit script's own comment block prescribed this revert, but only at
the `full_pe` handoff. The 6-month run reaches the danger zone earlier.
Now the default:

```sh
GB_SIGHET_FOLD_MAX_BYTES=1073741824   # 8 GiB -> 1 GiB
GB_INMODEL_SETUP_BATCH=2048           # was 0
GB_INFOMAT_MEMPOOL_FREE=1             # was 0
GB_INMODEL_BATCH_MEMPOOL_FREE=1       # was 0
```

All four are transient/scheduling knobs — no stored number changes, so a
mid-store resume is safe. All are env-overridable. Next lever if still
tight: `GB_N_SUBBANDS` 8192 -> 4096. **Never `SIGHET_NT_LAYER`** — not a
mid-store knob, and it multiplies with `GB_N_SUBBANDS`.

The 3-month twin is deliberately NOT reverted (~0.25 MB slots vs the 6mo
~0.5 MB, healthy through iteration 210). That split is declared on the
reversion list in `tests/test_submit_scripts_layout.py`, which refuses an
undeclared divergence between the two scripts.

---

## 6. F-stat refit cadence — changed 2026-09-18

`GB_FSTAT_REFIT_EVERY=50` **used to count branch proposes**, a
stage-dependent multiple of iterations, so the same number meant a
different cadence per stage. Measured from the epoch `DONE.json` ticks
(`Delta(clock)` was exactly 50 at every gap in both runs):

| run | GB moves per `gb_search` iteration | interval | epochs seen |
|---|---|---|---|
| 3-month | **2.00** (`rj_fstat_search`, `rj_prior_removal`) | 25 its | 8 @ it 210 |
| 6-month | **3.00** (+ `rj_warm_search`) | 17 its | 3 @ it 47 |

**It now counts iterations.** The stage combine is the one object eryn
proposes exactly once per iteration; it mints `gf_iteration` and
`_prepare_child` stamps it down the move tree beside `gf_stage_kind`. The
propose census is untouched — `_temper_cadence_fire` reads it.

**Cost differs enormously between the runs**, which is why this matters:

| run | per fit | share of elapsed |
|---|---|---|
| 3-month | 243-289 s | **8.7%** |
| 6-month | **3420 s, 3764 s** | **28.1%** (two fits) |

The 6-month peak counts are ~4x larger (13 513 and 16 176 vs 1945-3791).
So the cadence is a major throughput lever on 6mo and a minor one on 3mo.

**ON THE NEXT RESUME:** existing `clock.json` / `DONE.json` ticks are in the
OLD propose units and read ~2-3x too high, so the first refit is deferred
until the iteration clock catches up. Clear `gb_fstat_fit/*/clock.json` if
an immediate refit is wanted.

**Watch:** band shutoff revives on every new F-stat epoch. A ~3x rarer
cadence makes the submit script's backstop revival load-bearing rather than
a safety net.

---

## 7. Open items

**Watch list**

- **`alpha` rails** at `GALFOR_ALPHA_MAX=20.0` on walker 3 in both runs
  (19.94 at 6mo it 12, ~17.1 at 3mo it 210) while walkers 0-2 sit at
  0.3-2. Amplitudes are healthy, so it is a slope excursion, not the old
  collapse. Unresolved whether the cap is the right value.
- **galfor ladder balance.** At 6mo it 13 the top rungs had swap acceptance
  0.998/1.000/1.000 — collapsed, buying nothing. By it 47 it had inverted:
  cold end 0.011 (too low), hot end 0.888. Neither is the healthy 0.2-0.3.
- **`non-positive eigenvalue`**: 141 (3mo) / 139 (6mo) events, all `1/1
  matrices`. The eigen path is the only inner proposal at `B = 1`.
- **`[GF_TIMING]` does not appear in the 3-month run logs** at all; in the
  6-month it goes to `slurm_stdout_*.log` only. Do not assume it is in
  `globalfit_run.log`.

**Unpushed work, local branches only**

| branch | commit | what |
|---|---|---|
| `noise-ensemble-search` | `f2506667` | tiled per-walker ensemble search for psd/galfor. Gives each walker an inner ensemble of 10 tiled copies so `StretchMove` has a complement at `B = 1`, taking the information matrix off the SEARCH critical path. Maps onto Eryn's folded `nsamplers` axis, so almost no new sampler code. **Off by default** (`{PSD,GALFOR}_ENSEMBLE_SEARCH=1`). Costs ~10x the likelihood CALLS per round under WDM — gate it on rounds-to-plateau falling. |
| `walker-scaleup-24` | `f8e943a2` | planner for 4 -> 12 -> 24 walkers. Key finding: the GB sub-band buffer cap already binds at `B = 1`, so **GB memory is FLAT above block ~2.5** — 12 walkers probe 24's worst case at half the wall time. Projected 1.47x / 1.76x samples per GPU-hour. `scale_up.py` has a smoke test only, no unit tests. |

Both overlap: at `B >= 3` `_resolve_inner_kind` returns `stretch` anyway, so
the eigen fallback disappears without the ensemble move. **Do not launch
both at once** or the galfor verdict is unattributable.

---

## 8. Recent commits on `dev` worth knowing

- `71588588` `fix(build_truth)` — NameError, parsed args are `a` not `args`.
- `58ffbe84` interlock opt-in (`INTERLOCK=1`), default off.
- `f3c2f43a` GB catalogue path settable.
- galfor log-sampling fix + `read_noise_model_identity()` +
  `galfor_params_to_physical()`, and a partial-DTR guard so the monitor
  degrades to a MISSING entry instead of dying with `KeyError: 'nbins'`.
- F-stat refit clock counts iterations.
- 6-month sig-het memory revert.
- `snapshot_to_html.sh`.
- `text.usetex` pinned off in both page generators (it was inherited from
  whatever matplotlibrc the machine carried; nothing on the pages needs a
  real TeX install).

## 9. Artifacts

| | URL |
|---|---|
| 3-Month Multirank | https://claude.ai/artifact/UHsMz4tz3mfHuAhg7f2x1m |
| 6-Month Multirank | https://claude.ai/artifact/LEQvkEJqAkP1qs9XRHSTVz |

Both at Version 2 (3mo it 210, 6mo it 47), built through the committed
path. To republish, regenerate the page and publish with the artifact's
URL — and **preserve the `<title>`**: the generator names its pages
"LISA Global Fit 3-Month v8", while the artifacts are titled
"3-Month Multirank" / "6-Month Multirank", and a redeploy should keep the
artifact's identity stable.
