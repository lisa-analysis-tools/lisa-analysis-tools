# v9 launch — state of play and remaining work

**Written 2026-09-24 as a pre-compaction handoff.** Everything needed to
resume without the conversation.

> **SUPERSEDED 2026-09-24 (later the same day).** Everything in
> "REMAINING WORK" below is now BUILT. Read
> [`2026-09-24-v9-recipe-walkthrough.md`](2026-09-24-v9-recipe-walkthrough.md)
> for how the finished recipe runs and what to watch in the log. This file
> is kept for the traps section and the decision history.
>
> What changed since it was written:
> * the three-stage restructure, gate 6, the per-stage profile, the F-stat
>   peak override + forced fresh-epoch refit, `rj_replace` two-pass, the
>   noise start pin and the conditional noise stages all landed;
> * the cycle now ENDS on `rj_prior_removal` (user amendment): the third
>   in-model slot moved to between `rj_replace` and the removal judge and
>   was renamed `in_model_replace`;
> * `gb-inmodel-converge` merged to `dev` with zero conflicts;
> * commit `e7e4c37c` was split so the HTML-monitor work is its own commit
>   (the mistake in the section below is FIXED; the tree was verified
>   byte-identical across the rewrite);
> * ⚠ **a real defect was found and fixed**: the in-GROUP knobs resolved as
>   `GB_INMODEL_CONVERGE_GROUP*` while the script and every docstring said
>   `GB_INMODEL_GROUP*`, so V9-6 was entirely dead. The preflight now
>   RESOLVES knobs instead of probing symbols, which is the only check that
>   catches a name mismatch.

## Where the code is


| | |
|---|---|
| `LISAanalysistools/` | branch `dev`, **12 commits ahead of origin/dev, NOT pushed** (user reviews before push). Tracked tree clean. |
| `LISAanalysistools-inmodel-converge/` | branch `gb-inmodel-converge` @ `b7766910`, clean. The in-model convergence work. Merges onto dev with **zero conflicted hunks** (trial-merged, 346 tests green — ⚠ but that trial was against `d11f008c`, before the ctor fix; **re-run it against `e9ae70a0`+**). |
| `LISAanalysistools-cap-per-walker/` | the peer session's tree, same commit as dev. Kept deliberately: if anything points back at that work, having the tree beats reconstructing it. |

## ⚠ A MISTAKE TO CLEAN UP BEFORE PUSHING

Commit `e7e4c37c` ("fstat: the refit clock counts ELAPSED iterations")
**wrongly contains 359 lines of `scripts/diagnostics/gf_monitor_gen.py`**.
Cause: a broad `git add -A src tests scripts` swept in (a) ~61 lines of
pre-existing uncommitted work that were already in the tree at `e9ae70a0`,
and (b) the in-progress edits of a *still-running* background agent.

It is not pushed, so it is fixable. **Do not fix it by reverting the working
tree** — the agent's output is the only copy. The fix is to rewrite that
commit so the monitor changes land in their own properly-attributed commit,
once the monitor agent has finished. Until then the content is safe, just
mis-attributed.

Lesson: never `git add -A <dir>` while a background agent is writing in
that directory; stage explicit paths.

## What is DONE and committed

### On `dev` (12 commits)
- **galfor pin** (`1773705f`): `GALFOR_FIXED_PARAMS` reaches both fiducial
  paths, so the foreground can be frozen while psd and gb stay sampled.
  This is what makes a fixed-noise search stage possible.
- **stage-B goldens** (`711d3230`): a clean dev checkout no longer fails 11
  tests in `test_fstat_parallel_fit`.
- 1yr tiled noise-ensemble arming, 1yr null window, VGB preset grid,
  diagnostics/examples.
- **v9 script + specs** (`e0a39584`), see below.
- **peer merge** (`d11f008c`) + **its ctor fix** (`e9ae70a0`).
- **F-stat clock** (`651de861`, `e7e4c37c`): tests repaired to pin iteration
  semantics; clock switched to ELAPSED iterations; `GB_FSTAT_REFIT_EVERY_PE`
  = 250.
- **drift-guard declarations** (`d4b29576`).

### On `gb-inmodel-converge`
- **Scope 1 — per-ROW newborn convergence** inside the RJ moves' polish.
  W=100, D/2, ladder gate at the coldest half, refill on.
- **Scope 2 — per-(walker, band) in-GROUP convergence** for the pure
  in-model move: repeats its whole pass until every sub-band's cold-chain
  logL converges, shutting each off as it goes. Bounded by `max_passes`.
- **Ladder adaptation from the vertical swaps** when the fancy swaps are off.
- 92 tests in `tests/test_inmodel_converge.py`.

## THE THREE CONVERGENCE SCOPES (they are easy to confuse)

| unit | statistic | clock | lifetime | owner |
|---|---|---|---|---|
| row `(t,w,b)` | accepted `delta_ll` | one in-model REPEAT | one repeat block, **newborns only** | `gb-inmodel-converge` |
| `(w,b)` | sub-band cold-chain logL | one in-model PASS | **one proposal group** | `gb-inmodel-converge` |
| `(w,b)` | sub-band cold-chain logL | one sampler ITERATION | **one recipe stage** | on `dev` (peer) |

Scopes 2 and 3 differ ONLY in clock and lifetime. Naming keeps them apart:
`GB_INMODEL_GROUP_*` / `[GB_IMGROUP]` vs `GB_SEARCH_BAND_SHUTOFF_*` /
`[GB_STAGE]`.

## REMAINING WORK, in dependency order

### 1. The recipe restructure — THE BIG ONE, nothing else blocks on it

Design: `2026-09-24-v9-search-stage-restructure-design.md`. Currently
`run_combined_staged.py::build_fit` builds ONE `gb_search` stage. Needed:

Three stages, each with this per-iteration move order:
```
1 rj_warm_search      (stage 3: every 5 iterations)
2 in_model x N        (group rule decides N; 25 repeats/source/pass)
3 rj_fstat_search
4 in_model x N
5 rj_replace          (see item 2)
6 rj_prior_removal    (remove only)
7 in_model x N
```

| stage | noise | phase max | opt SNR | F-stat peak |
|---|---|---|---|---|
| `gb_search_1` | FIXED (pinned psd+galfor) | on | 8 | 8 |
| `gb_search_2` | FIXED | off | 5 | 6.25 |
| `gb_search_3` | sampled | off | 5 | 6.25 |
| `full_pe` | sampled | off | 5 | 6.25 — unchanged from today |

Plus: `noise_search` / `noise_vgb_search` run ONLY when there is no
psd/foreground estimate from a previous run.

**⚠ GATE 6 IS THE ONE MISSING STOPPING GATE.**
`recipe.band_shutoff_w_pending_total(moves) -> int` (0 == converged) and
`band_shutoff_w_armed(moves) -> bool` exist on dev and work, but **nothing
calls them**. Compose, do not replace:
```python
if stop and band_shutoff_w_armed(getattr(sampler, "moves", None)):
    if band_shutoff_w_pending_total(getattr(sampler, "moves", None)):
        stop = False
```
⚠ The `armed` guard is MANDATORY: `pending_total` returns 0 when the feature
is off, which reads as *converged* and would end every stage at its first
check. Criterion in the user's sanctioned words: **"every OCCUPIED (walker,
band) pair has shut off"** — empty pairs can never shut off, so the literal
"all pairs" is unreachable; the nleaves plateau stays composed in to cover
them.

⚠ **TRAP — the F-stat peak threshold is baked into a cache.** Stage 1 at
peak 8 → stage 2 at 6.25 cannot be done by changing an env var: stage B's
load path recomputes nothing and the loader refuses a `peak_min_F` mismatch.
**User's resolution: make the threshold an updateable value in code and
force an F-stat refit on the first call in a new stage** — no file deletion.

Stage profiles apply at `setup_run` rather than by building three named move
sets (moves are built once and shared; three copies would triple the
F-stat-carrying RJ instances).

### 2. `rj_replace`, modernized

User ruling (reversing an earlier "no rj_replace for now"): bring it back,
mirroring the newest GB setups, into **each** search stage **before** the
prior-removal move. Inside one `propose()` it should do **two internal
iterations: one refit/warmstart, one F-stat.**

⚠ It was RETIRED for problems; the user thinks that may have been the old
GPU setup. So **the logging has to be good enough to tell "working" from
"misbehaving the way it used to"**. Instrument for the recorded failure
signature specifically: phase-max acceptance non-attainable, and an ll-drift
of 1.5–1.9e3 (three orders above every other move). See memory
`project_rj_replace_phase_max_nonattainable`,
`project_replace_forensics_verdict`, `project_rj_replace_reinstated`.

### 3. Warm start + the frozen noise point from the SAME folder

Both must come from the **3mo 10-walker noise-fix** run, and the PSD/galfor
pin must use that run's **maxlogL** point (not a posterior mean).
`GB_WARM_START_COMPONENTS` and the `GALFOR_FIXED_PARAMS` /
`fixed_psd_params` values therefore share one source directory.

### 4. Verification still outstanding
- **`test_gbspecial_flow` — THE hard gate, unrun.** 10–26 GB, SIGKILLs an
  8 GB laptop, needs cluster time. It builds REAL moves and would probably
  have caught the ctor bug that shipped in `d11f008c`. Both features'
  changes are in its blast radius.
- Re-run the branch trial-merge against `e9ae70a0`+.
- `test_submit_scripts_layout::ThreeMonthTwinTest` — 2 pre-existing
  failures, the 3mo twin drifted from the 6mo script. Separate reconcile job.
- The HTML monitor agent's output needs review (and the commit fix above).
- An independent agent review of the whole thing against the user's original
  requests, which the user asked for once the work is complete.

## v9 script knobs (all in `submit_gf_6mo_v9_4gpu.sh`, pinned by `SixMonthV9DeltaTest`)

`V9-1` in-model convergence on (ITERS=100, DLL=4.0, GATE_FRAC=0.5,
STOP_FRAC=0.5, REFILL=1, CLASSES=newborn) · `V9-2` `COARSE_GPU_MODE=off` ·
`V9-3` naming/fresh store · `V9-4` caps OFF (`GB_LEAF_CAP_START=` empty) ·
`V9-5` search flip 1.0 · `V9-6` `GB_INMODEL_GROUP=1` (ITERS=3, DLL=4.0,
scale=flat, MAX_PASSES=20) · `V9-7` `GB_RUN_FANCY_TEMPERING=0` with
`GB_TEMPER_VERTICAL=1` KEPT · `V9-8` `GB_SEARCH_IN_MODEL=1`,
`GB_NUM_REPEAT_PROPOSALS=25` · `V9-9` `GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1`,
`GB_SEARCH_STAGE_PER_WALKER=0` · `V9-10` `GB_FSTAT_REFIT_EVERY_PE=250`.

The script **preflights its own feature support** and refuses rather than
running silently as v8 — unrecognized env vars are ignored, so without that
a wasted allocation would produce plausible output.

## Traps that cost time once and must not again

1. **`_adapt_band_temps` has ONE caller (`run_tempering`).** Turning the
   fancy swaps off freezes the temperature ladder silently. Fixed, auto by
   default.
2. **`GB_TEMPER_EVERY_PROPOSES=0` means "fire ALWAYS", not "never"**
   (`n <= 1` returns True). It cannot disable tempering.
3. **`has_run_rj` is allocated once per SORTER, not per pass.** A multi-pass
   loop must reset it or pass 2 picks nothing and looks converged. ⚠ The
   pre-existing reseed multi-pass path may have this bug; unverified.
4. **The leaf cap's D/2 patience form does not port to a fine clock** — it
   retires mid-climb. Use best-now vs best-W-ago through a ring buffer.
5. **Never build a convergence statistic from `ll_ref`** — it is a sig-het
   value with a 1–17 lnL cold offset that is re-anchored mid-block. Use sums
   of accepted deltas.
6. **Fake-based suites bypass `__init__`.** Any change to
   `GBSpecialStretchMove.__init__` must run a real-construction suite;
   `tests.test_temper_shutoff_bands` is the cheap one (~110 s).
7. **bash resolves exports in order** — a later `export X=1` silently
   re-arms an earlier `X=0`. One authoritative export per knob.
8. **NEVER run `tests.test_gbspecial_flow` locally** (10–26 GB, SIGKILL) and
   never `unittest discover` (it pulls that suite in).
