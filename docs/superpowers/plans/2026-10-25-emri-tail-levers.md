# EMRI plunge tail: auto window length, conditional tail, batch instrumentation -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the 2026-10-25 H100 findings (`LISAanalysistools/emri_wave_test_new.log`): (1) let the plunge-chunk window grow so a row's whole tail fits ONE window instead of two; (2) let a row whose tail is short skip the exact tail entirely (lookup through the end), behind a knob that defaults to today's behaviour, so Mike can measure it; (3) instrument the batched chunk path per phase so the next H100 run shows where the 8-per-call path loses 12-17 % per template against the single call.

**Architecture:** All three live in `src/lisatools/sources/emri/wdm_direct.py` (`EMRIDirectWDM.__init__` knob parsing, `accumulate_harmonic_batch` tail planning, `_chunk_rows_batch` phases) plus the two speed scripts (`scripts/emri/emri_direct_stage_timing.py`, `scripts/emri/emri_speed_durations.sh`) and `docs/emri-direct-wdm.md`. The batched design of 2026-10-24 stays: one dense response, one lookup kernel, one `tail_td(pairs)` call, one `wdm_chunk_of_td` over all pairs, harmonics summed in the kernel, handoff per row. Verification is by the 720-day H100 speed script (Mike runs it); the laptop runs the synthetic and spline-path tests only (its compiled backend lacks the dense kernel).

**Tech Stack:** Python 3.12, numpy/cupy via `lisatools.get_backend`, FastEMRIWaveforms (CPU here), `unittest`.

**Spec:** Mike 2026-10-25: "Investigate and implement 1, 2, 3" of the recommendations in the a3 session, after the H100 log. Findings the plan rests on: at 720 d (EMRI 1) the tail is 69 px at threshold 1e-3 and 80 px at 1e-5 (`[split] ... tail median 69 px, 2 windows`), i.e. every plunging row pays two 128-layer windows = 256 layers of dense samples for a tail whose handoff-to-end span is 37 / 48 px plus the 32-px past-end pad; the batched 8-per-call path costs 181 ms vs 161 single (1e-3) and 327 vs 279 (1e-5); the chunk length 128 / 96 / 64 gives the identical mismatch vs production (3.52e-6 / 3.71e-6) on the dense path; the dense channels kernel falls back to global-memory scratch above a few 10^4 samples per item.

## Global Constraints

- Work ONLY in the worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-emri-batch` (branch `emri-plunge-batch`, == origin/dev e5906c90 at the start). Never touch `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools`. Commit by explicit path, no push, no rebase. Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Pure Python; nothing under `src/lisatools/cutils/`.
- DEFAULTS DO NOT CHANGE BEHAVIOUR: `EMRI_DIRECT_NT_SUB` unset = 128 fixed windows exactly as today; `EMRI_DIRECT_TAIL_SKIP_PX` unset = 0 = never skip; the instrumentation only adds stats and prints. The single-row dense result at the defaults must be unchanged to 1e-12.
- Keep the 2026-10-24 batching contract: exactly one `tail_td(pairs)` call and one `wdm_chunk_of_td` call per `accumulate_harmonic_batch` call; per-row common handoff; `finish()` the single exit of `_batch_chunk`; the stats keys `rows, alone, chunk_rows, subs, failed, lookup_pixels, chunk_pixels, dropped_pixels, windows, tail_layers (list), ended_before_window, pairs` stay; new keys are added, never renamed. The `[EMRI_DIRECT] leaf N:` line format does not change.
- Env knob = the attribute name upper-cased with the `EMRI_DIRECT_` prefix; explicit argument > env > default; a bad value raises `ValueError` naming the knob.
- Laptop rules: ONE python at a time (`pgrep -fl "bin/python" | grep -vE "vscode|Code Helper|pet server|jupyter|ipykernel|ipython"` must print nothing; never kill anything); `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; `PYTHONPATH=<worktree>/.wtenv LAT_WORKTREE_SRC=<worktree>/src`, interpreter `/Users/mkatz/miniconda3/envs/deving/bin/python`; no 720-day L1 windows, no `--modes all` gate runs, no stage-timing runs here (the last one pushed swap to 17.8 GB).
- Tests that must stay green: tests/test_emri_wdm_direct.py, tests/test_emri_direct_wiring.py, tests/test_emri_direct_move.py, tests/test_wdm_chunk_splice.py, tests/test_emri_direct_fanout.py (tests/test_tdi_dense.py's 10 errors and tests/test_emri_sparse_response.py's 8 are pre-existing here; the dense tests skip).

## Review Focus

1. Auto mode with a batch whose rows have DIFFERENT tail spans must pick one length that gives every row exactly one window, and the transform stays one call (Task 1 test `test_auto_window_fits_every_row_in_one_window`).
2. Auto mode must never shrink below the floor or exceed the grid (Task 1 test `test_auto_window_respects_floor_and_grid`).
3. A skipped tail must extend that row's lookup to its last pixel and produce no pairs, while other rows in the same batch keep their tails (Task 2 test `test_tail_skip_is_per_row`).
4. The skip knob at 0 must leave every number and stat identical (Task 2 test `test_tail_skip_zero_is_todays_behaviour`).
5. The phase walls must be present and sum to within 10 % of the total on the laptop spline path, and the script must print them without touching the timing it already reports (Task 3 test `test_chunk_rows_batch_records_phase_walls`).

---

### Task 1: `EMRI_DIRECT_NT_SUB=auto` -- one window per tail

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py` (`__init__` ~1348-1365 knob parsing; `accumulate_harmonic_batch` ~525-571 tail planning; `_chunk_rows_batch` ~2023 passes the choice; module docstring ~38)
- Modify: `docs/emri-direct-wdm.md` (the `EMRI_DIRECT_NT_SUB` bullet and the "Plunge-chunk length" subsection)
- Test: `tests/test_emri_wdm_direct.py` (next to `NtSubKnobTest` and the pairs tests)

**Interfaces:**
- Consumes: `tail_chunk_plan(n_lo, n_hi, Nt, Nt_sub)` (`wdm_het.py:376`; interior per window `Nt_sub - 2*(Nt_sub//4) - 2`), the per-row `n_h` / last pixel / `n_end` arithmetic at `wdm_direct.py:536-548`, `stats["pairs"]`.
- Produces: `EMRIDirectWDM.Nt_sub` is an int (fixed mode) or the string `"auto"`; `EMRIDirectWDM.nt_sub_floor` (int, 128 unless `EMRI_DIRECT_NT_SUB=auto:<floor>`, validated even / >= 8 / <= Nt); `accumulate_harmonic_batch(..., Nt_sub=128, nt_sub_auto=False, nt_sub_floor=128, ...)`: with `nt_sub_auto` the call computes ONE window length for the whole call, `Nt_sub_eff = max(nt_sub_floor, round_up_to_multiple_of_4(2 * (s_max + nt_sub_floor // 4) + 4))` capped at the largest even value `<= Nt - pixel_edge`, where `s_max` = the largest `last_px - n_h + 1` over the rows with a tail; the past-end extent stays `nt_sub_floor // 4` (not `Nt_sub_eff // 4`); every row is then tiled with `tail_chunk_plan(n_h, n_end, Nt, Nt_sub_eff)`; `stats["nt_sub_eff"]` (int) and `stats["windows"]` report it. Everything else (`_chunk_start`'s fine-grid floor, `_fine_grid`) keeps using `nt_sub_floor` in auto mode and `Nt_sub` in fixed mode.

- [ ] **Step 1: Write the failing tests** in `tests/test_emri_wdm_direct.py` using the existing two-row synthetic fixture of `test_common_handoff_is_per_row` (rows with handoff-to-end spans of 37 and 48 px on a grid long enough to hold 200-layer windows):

```python
def test_auto_window_fits_every_row_in_one_window(self):
    acc, stats = self._accumulate_two_rows(..., nt_sub_auto=True, nt_sub_floor=128)
    self.assertEqual(stats["nt_sub_eff"], 164)          # 2*(48+32)+4 = 164, a multiple of 4, >= 128
    self.assertEqual(stats["windows"], 2)                # one per row
    self.assertEqual(sorted(r for (r, *_ ) in stats["pairs"]), [0, 1])
    self.assertEqual(self.transform_calls, 1)            # the wrapped wdm_chunk_of_td counter of the neighbour test
    acc_fixed, stats_fixed = self._accumulate_two_rows(..., Nt_sub=164)
    np.testing.assert_allclose(acc, acc_fixed, rtol=1e-12, atol=1e-14)   # auto == the same fixed length

def test_auto_window_respects_floor_and_grid(self):
    _, stats = self._accumulate_two_rows_with_spans(5, 7, nt_sub_auto=True, nt_sub_floor=128)
    self.assertEqual(stats["nt_sub_eff"], 128)           # tiny tails: the floor
    _, stats = self._accumulate_two_rows_with_spans(300, 310, nt_sub_auto=True, nt_sub_floor=128, Nt=400, pixel_edge=8)
    self.assertLessEqual(stats["nt_sub_eff"], 392)       # capped at the grid, even
    self.assertEqual(stats["nt_sub_eff"] % 2, 0)
    self.assertGreaterEqual(stats["windows"], 2)         # a tail that cannot fit is tiled, still correct
```

and in `NtSubKnobTest`: `EMRI_DIRECT_NT_SUB=auto` -> `Nt_sub == "auto"`, `nt_sub_floor == 128`; `auto:96` -> floor 96; `auto:7` and `auto:x` raise `ValueError` naming the knob; the fixed-mode tests stay as they are.

- [ ] **Step 2: Run them** (`python -m unittest tests.test_emri_wdm_direct -k "auto_window or nt_sub_knob"`) -> FAIL.

- [ ] **Step 3: Implement**: knob parsing in `__init__` (`"auto"` / `"auto:<floor>"`; `self.Nt_sub = "auto"`, `self.nt_sub_floor`); in `accumulate_harmonic_batch` compute `s_max` from the per-row `n_h` and last pixel BEFORE the pairs are built, choose `Nt_sub_eff` by the formula, use it in `tail_chunk_plan` and for the window edges, keep the past-end extent at `nt_sub_floor // 4`; `_chunk_rows_batch` (and the single-row path through it) passes `nt_sub_auto=(self.Nt_sub == "auto")`, `nt_sub_floor=self.nt_sub_floor`, `Nt_sub=(self.nt_sub_floor if auto else self.Nt_sub)`; `_chunk_start` / `_fine_grid` read `self.nt_sub_floor` in auto mode. Module docstring + the class docstring knob text.

- [ ] **Step 4: Run** `python -m unittest tests.test_emri_wdm_direct tests.test_wdm_chunk_splice` -> PASS, and the fixed-mode defaults unchanged (`test_tail_is_one_dense_call_and_one_transform`, `test_batched_tail_matches_single_row_tails` green).

- [ ] **Step 5: Docs**: in `docs/emri-direct-wdm.md` describe `auto` (what it computes, why: the H100 tails of 69 / 80 px at 128 need two windows; with auto 164 layers in one window = 36 % fewer dense samples at 1e-5 and 2*(37+32)+4 = 142 -> 144 at 1e-3 = 44 % fewer), and the H100 command `NT_SUB=auto` for `emri_speed_durations.sh` (the shell script already passes any `NT_SUB` string through `--nt-sub`; make sure `emri_direct_stage_timing.py --nt-sub` accepts `auto` and `auto:<floor>` and prints `nt_sub=auto(eff N)` in its `[split]` lines using `stats["nt_sub_eff"]`).

- [ ] **Step 6: Commit** `EMRI direct: EMRI_DIRECT_NT_SUB=auto -- one plunge window per tail`.

---

### Task 2: `EMRI_DIRECT_TAIL_SKIP_PX` -- lookup through the end for short tails

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py` (`__init__` knob; `accumulate_harmonic_batch` per-row handoff ~525-548; `_chunk_rows_batch` passes it; `_batch_chunk` stats init / `finish()`; docstrings)
- Modify: `scripts/emri/emri_direct_stage_timing.py` (`--tail-skip-px`, printed in `[split]` as `skipped S`, JSON `tail_skipped`), `scripts/emri/emri_speed_durations.sh` (`TAIL_SKIP_PX` passed only when set)
- Modify: `docs/emri-direct-wdm.md`
- Test: `tests/test_emri_wdm_direct.py`, `tests/test_emri_direct_wiring.py`

**Interfaces:**
- Consumes: Task 1's `accumulate_harmonic_batch` signature; the gate script's existing `--no-handoff` (which sets `tracks=None` so `k_h = P` everywhere) as the reference for what "lookup through the end" means.
- Produces: `EMRIDirectWDM.tail_skip_px` (int >= 0, env `EMRI_DIRECT_TAIL_SKIP_PX`, default 0); `accumulate_harmonic_batch(..., tail_skip_px=0)`: after the per-row common handoff, a row whose span `s_r = last_px_r - n_h_r + 1` satisfies `0 < s_r <= tail_skip_px` gets `k_h[subs of r] = P_row[r]` (the lookup runs to the row's last pixel, exactly as a row with no trip) and contributes no pairs; `stats["tail_skipped"]` (int) counts them; `tail_layers` gets no entry for them; `_batch_chunk` carries `tail_skipped` through `finish()`; the move's line is unchanged (the count is visible in `[split]` / JSON of the speed script only).

- [ ] **Step 1: Write the failing tests**: synthetic (two rows, spans 10 and 48 px, `tail_skip_px=16`): row 0 has no pairs, its lookup extends to its last pixel (`stats["handoff_pixel_per_row"][0] == P_row0`), row 1 keeps its tail; `stats["tail_skipped"] == 1`; `stats["windows"]` counts row 1 only (`test_tail_skip_is_per_row`); and `tail_skip_px=0` gives arrays and stats identical to today's run (`test_tail_skip_zero_is_todays_behaviour`, compare against a call without the kwarg). In `tests/test_emri_direct_wiring.py` extend the real-generator spline-path mixed batch: with `EMRI_DIRECT_TAIL_SKIP_PX` large enough to skip the fixture's chunk rows, `last_stats["tail_skipped"] == number of chunk rows`, `chunk_pixels == 0`, and the per-row template equals `gen(*p)` built with the same knob (`test_tail_skip_knob_reaches_the_generator`).

- [ ] **Step 2: Run** -> FAIL (`TypeError: unexpected keyword tail_skip_px`).

- [ ] **Step 3: Implement** as in Interfaces; knob parsing (`int >= 0`, else `ValueError` naming `EMRI_DIRECT_TAIL_SKIP_PX`); the two scripts (`--tail-skip-px` int default 0 -> generator kwarg; `TAIL_SKIP_PX` in the shell script passed only when set, same pattern as `NT_SUB`); `[split]` prints ` skipped S` after `windows`.

- [ ] **Step 4: Run** the five test modules -> PASS.

- [ ] **Step 5: Docs**: the knob bullet; the measurement recipe for Mike: accuracy first with the gate on the production path (`GATE_DT=5 GATE_RESPONSE=dense python scripts/emri/emri_direct_wdm_gate.py --table wdm_lookup_emri_cx_NF720_DT5_TL32_fd8x0p01_nld2.h5 --modes all --thresh 1e-5 --no-handoff` against the same run WITHOUT `--no-handoff`: the `>= 0.99` and `direct-prod` columns), then speed + mismatch in one go: `BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 TAIL_SKIP_PX=100 bash scripts/emri/emri_speed_durations.sh` (its `[summary]` "mm vs production" is the accuracy of the skipped template; the threshold 1e-3 tail span is 37 px and 1e-5 is 48 px, so 100 skips both). State plainly that the default stays 0 until Mike reads those.

- [ ] **Step 6: Commit** `EMRI direct: EMRI_DIRECT_TAIL_SKIP_PX -- lookup through the end for short tails`.

---

### Task 3: phase walls of the batched chunk path + the super-linear-cost investigation

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py` (`_chunk_rows_batch` ~2023-2120; `_dense_td`; `accumulate_harmonic_batch` returns walls)
- Modify: `scripts/emri/emri_direct_stage_timing.py` (`[walls]` line + JSON fields)
- Test: `tests/test_emri_direct_wiring.py`

**Interfaces:**
- Produces: `EMRIDirectWDM.last_stats["walls"]` (dict of seconds for one `templates`/`batch` call: `capture` (the per-row `_chunk_item` loop: FEW, tracks, grids), `response` (`_dense_response` / `run_response`), `lookup` (the fused kernel or the Python lookup), `tail_dense` (`tail_td(pairs)` = the dense TD evaluation incl. its blocks), `transform` (`wdm_chunk_of_td`), `splice`, `total`) -- each phase bracketed by `perf_counter` with a device synchronisation first when `xp` is cupy (`cupy.cuda.get_current_stream().synchronize()`), so GPU time lands in its phase; `n_response_points` (the padded common grid length) and `n_response_points_rows` (the per-row lengths before padding) so the padding cost is visible; the stage-timing script prints after each timed batch and after the single call `[walls] grid=G thr=T nt_sub=N batch of B (C per call): capture a ms/row, response b, lookup c, tail d, transform e, splice f, total g; response points padded P (rows p_min..p_max)` and writes the same into the JSON line. The walls are recorded on every call (the cost is a few synchronisations per call).

- [ ] **Step 1: Write the failing test** `test_chunk_rows_batch_records_phase_walls` (spline path, the mixed-batch fixture): after `gen.batch(rows, chunk_rows=4)`, `set(gen.last_stats["walls"]) >= {"capture", "response", "lookup", "tail_dense", "transform", "splice", "total"}`, every value `>= 0`, and `abs(sum(phases) - total) <= 0.1 * total`.

- [ ] **Step 2: Run** -> FAIL (`KeyError: 'walls'`).

- [ ] **Step 3: Implement** the walls (a small context-manager helper `_phase(name)` on the generator that synchronises the device when `xp is cupy`, accumulates into `self._walls`), `n_response_points*`, and the script's `[walls]` line + JSON fields. Keep the walls dict out of the move's log line.

- [ ] **Step 4: Investigate by reading** (no GPU here): for `n_temp = 8` chunk rows, trace every place whose cost could grow faster than linearly with the number of rows: `pad_grid` and the common `N` of `_dense_response` (what happens when the +3 / -5 h shifted rows have grids of different lengths: how many points are added by halving), the `ExactPhaseTDIOutput` construction (any Python loop over templates or harmonics that allocates per element), the `TDDenseTDIonTheFly` construction in `_dense_kernel` (the `np.pad` to the block's `K`, the host->device uploads per call), the all-pairs transform (`FDSignal.wdmtransform` batched path: one FFT or a Python loop?), the cupy allocations per call (`out_arr` 8 x 3 x 179 x 17280 doubles = 594 MB, the tail series, the transform temporaries) against the pool, and the dense kernel's global-memory scratch fallback threshold (`src/lisatools/cutils/lat_spline_tdi_waveform.cu` around the "scratch ... exceeds device dynamic-shared ceiling" message: what N triggers it, how the scratch scales with `n_temp` and `N`). Write the findings in the report as a ranked list with the evidence line numbers. If one of them is a definite super-linear cost that a pure-Python change removes without changing numbers (e.g. a per-template Python loop that can be vectorised, an allocation that can be reused across blocks), implement it with a test that pins the result unchanged; otherwise leave the fix to the next H100 run's `[walls]` numbers and say so.

- [ ] **Step 5: Run** the five test modules -> PASS. **Step 6: Docs** (`[walls]` line in the script docstring and the doc's "In the global fit" section). **Step 7: Commit** `EMRI direct: phase walls of the batched chunk path (+ [walls] in the speed script)`.

---

## What Mike runs on the H100 after the push (one allocation, from the LAT root after `git pull`)

```
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 bash scripts/emri/emri_speed_durations.sh                      # baseline + [walls]
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 NT_SUB=auto bash scripts/emri/emri_speed_durations.sh          # item 1
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 TAIL_SKIP_PX=100 bash scripts/emri/emri_speed_durations.sh    # item 2 speed + mm
GATE_DT=5 GATE_RESPONSE=dense python scripts/emri/emri_direct_wdm_gate.py --table wdm_lookup_emri_cx_NF720_DT5_TL32_fd8x0p01_nld2.h5 --modes all --thresh 1e-5            # item 2 accuracy, with and without --no-handoff
```

Read: `[summary]` sparse/kernel single vs 8/call and "mm vs production"; `[split]` windows / nt_sub_eff / skipped; `[walls]` per phase for the single vs the 8-per-call batch (the phase whose per-row cost grows from single to batch is the regression).

---

## Handoff status (2026-10-25, end of day)

**Where the code is.** `dev` was returned to the pre-change EMRI code: commit
`18cbd285` reverts `c3116473..e05eec17`, so dev's tree is byte-identical to
`37ea50cd` and the global fits launched from dev have none of this work.  Both
plans (this one and `2026-10-24-emri-plunge-chunk-batching.md`) live on the
`emri-plunge-batch` branch, rebuilt on top of that revert, so
`git diff dev...emri-plunge-batch` is the whole change and a later merge into
dev is a fast-forward.  Work on the branch in its own worktree
(`git worktree add ../LAT-emri-batch emri-plunge-batch`) and run its tests and
scripts with `PYTHONPATH=$PWD/tools/wtenv LAT_WORKTREE_SRC=$PWD/src` so the
production checkout stays on dev.

**Every knob keeps the previous behaviour by default**, on this branch too:
`EMRI_DIRECT_CHUNK_BATCH=0` (one chunk row per call), `EMRI_DIRECT_NT_SUB`
unset (128 layers), `EMRI_DIRECT_TAIL_SKIP_PX=0` (never), `EMRI_DIRECT_WALLS=0`
(no device syncs).  The speed script opts in to batching and walls
(`CHUNK_BATCH=1`, `WALLS=1`); nothing in the launchers sets any of them.

**Verified on the H100 so far** (`emri_wave_test_new.log`, code at
`e5906c90`, batching always on at that commit): batched tail accuracy
unchanged (noise-weighted mm vs production 3.52e-6 at thr 1e-3, 3.71e-6 at
1e-5), single-row vs batched-row parity ≤ 1.4e-10, sparse/kernel 720 d at
thr 1e-3: single 162 ms, 1-per-call 160 ms, 8-per-call 178 ms (production
spline path 362 ms); NT_SUB 128 / 96 / 64 give identical mismatch (the window
length is a weak lever: the tail starts at the first harmonic trip, so a
shorter window only moves the fine-grid segment).

**The 2026-10-25 run (`emri_first.log`) ran a stale checkout** (its header
line and argparse behaviour match ~`e5906c90`: batching unconditional, no
`--chunk-batch` / `--walls`, integer-only `--nt-sub`).  Consequences: the
`lever_prod` and `lever_batch` runs executed the same batched code (there is
still NO production-vs-batched comparison at the branch head), no `[walls]`
lines were possible, and `NT_SUB=auto` was rejected by argparse.  The
durations script now prints `[durations] code: <rev> on <branch>` first; read
that before the numbers.  Still to run at the branch head: `CHUNK_BATCH=0` vs
`CHUNK_BATCH=1` on the same rows, the `[walls]` split of the 8-per-call loss
(the leading hypothesis is the warm-up: the 2-row warm-up then the pool free
puts first-use cudaMalloc / cuFFT plan creation inside the timed region),
`NT_SUB=auto`, and `TAIL_SKIP_PX=100` with the gate's mismatch.

**Unexplained once:** the first python of that session failed with
`BackendUnavailableException: 'cuda13x' backend could not be imported`
(`src/lisatools/cutils/__init__.py`, `cuda13x_module_loader`); the next
processes imported it fine.  The script's failure handler now prints 40 lines
of the failing log so the chained ImportError is visible next time.

**Laptop:** the compiled backend here predates `TDDenseTDIWaveformWrap`; the
dense tests skip on this branch and error on dev.  Run the GPU checks on the
cluster.

### Notes for the finisher (2026-10-26)

**Install / rebuild: none needed.**  This branch, and the revert on dev, are
pure Python, shell and markdown (ten `.py`, one `.sh`, three `.md`): nothing
under `src/lisatools/cutils/`, no CMake or `pyproject.toml` change, and no
other repository (FEW, GBGPU, BBHx, Eryn, GBT untouched).  After `git pull` on
dev, or after checking this branch out, no `pip install` or rebuild is needed:
the editable install picks the Python up at the next process start.  The one
rebuild item is pre-existing and unrelated: the cluster's CPU backend module
predates the dense kernel's `stop_taper` argument (three `PlungeTailTest`
errors on the cluster); the GPU module is current and the global fits use it;
`pip install -e . --no-build-isolation` on the cluster refreshes the CPU
module whenever convenient.

**Which code a job ran.**  The pre-change `[EMRI_DIRECT] leaf N:` line ends
at `... F fallbacks, ...`; this branch's line continues with
`| chunk rows A (tail median L px, W windows ...) lookup px P chunk px C
dropped px D`.  A job log with that continuation ran this work.  A job
(re)launched from a dev checkout between e5906c90 and bd9a3433 (2026-10-25
03:06 to 20:30 PDT) ran the batched tail unconditionally, because the knob
did not exist yet; from bd9a3433 to e05eec17 the default was the old one-row
path; from 18cbd285 dev has none of it.  A running process keeps the code it
loaded at launch.

**Equivalence at the defaults.**  An Opus audit of 37ea50cd vs bd9a3433 in
the production configuration found the same operations on bit-identical
inputs for every EMRI template; only the tail transform's cuFFT batch size
differs (<= 1e-16), and atomicAdd ordering is run-to-run noise at both
commits.  Both-mode tests (`tests/test_emri_direct_wiring.py`,
`tests/test_emri_wdm_direct.py`) pin the knob defaults.

**Follow-ups not started.**  (1) A fixed ~24 px pad on longer windows is the
real tail lever (`docs/emri-direct-wdm.md`, chunk-length sweep); the window
length itself is not.  (2) Pre-existing gap: a row ending inside the data
window with no harmonic handoff trip gets no tail.  (3) Consolidate the three
tail forms and the two Python lookups in `wdm_direct.py` (reviewer minors).
(4) The 8-per-call loss: test the warm-up hypothesis first (time a second
pass, or allocate before the pool free) before hunting inside the kernel.
(5) Batch the FEW trajectory calls across rows (one call per row today).

**Cluster commands.**

```
# production checkout: the previous EMRI configuration
git checkout dev && git pull && git rev-parse --short HEAD        # 18cbd285 or later
# this branch, in its own worktree, without touching the production checkout
git fetch origin && git worktree add ../LAT-emri-batch emri-plunge-batch
cd ../LAT-emri-batch && export PYTHONPATH=$PWD/tools/wtenv LAT_WORKTREE_SRC=$PWD/src
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 CHUNK_BATCH=0 OUT=lever_prod  bash scripts/emri/emri_speed_durations.sh
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 CHUNK_BATCH=1 OUT=lever_batch bash scripts/emri/emri_speed_durations.sh
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 NT_SUB=auto    OUT=lever_auto  bash scripts/emri/emri_speed_durations.sh
BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 TAIL_SKIP_PX=100 OUT=lever_skip bash scripts/emri/emri_speed_durations.sh
```

Read the `[durations] code:` line first: it must name a commit on this branch
(af9b0046 or later).  The 2026-10-25 run did not, and its numbers were not
what their directory names said.
