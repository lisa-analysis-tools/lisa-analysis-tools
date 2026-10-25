# EMRI plunge-chunk telemetry, batching and chunk-length check -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the cost of EMRI direct-to-WDM templates for sources that plunge inside the data window (six of the eight at 24 months) without changing their accuracy, in three deliverables: a per-leaf log line that shows the lookup / plunge-chunk split (pushed first), batching of the plunge-chunk rows that today are built one at a time, and a measured answer on whether a shorter plunge chunk is safe.

**Architecture:** `EMRIDirectWDM` (`src/lisatools/sources/emri/wdm_direct.py`) serves every harmonic from the n_ref lookup table up to a per-template handoff and the rest from an exact dense time-domain "plunge chunk" transformed in `Nt_sub` = 128-layer windows. `_batch_chunk` batches lookup-only rows but sends every row with a chunk through the single-template path (`_call_knots`). The move `EMRIDirectLikeMove` (`src/lisatools/globalfit/moves/emridirectmove.py`) prints one `[EMRI_DIRECT] leaf N:` line per leaf from a stats dict. This plan (1) adds the generator's existing counters to that line, (2) gives `_batch_chunk` a batched chunk path in which EVERYTHING is one operation per `templates` call: one dense response for all chunk rows, one lookup kernel, ONE dense time-domain evaluation of every (row, tail window) pair with all of that row's harmonics built in parallel and summed in the kernel, and ONE batched WDM transform of all those series (Mike 2026-10-24: "everything should be batched there ... the WDM transform as one parallel operation across all modes built in time-domain and transformed"), (3) makes `Nt_sub` an env knob and measures accuracy and speed against 128.

**Tech Stack:** Python 3.12, numpy/cupy via `lisatools.get_backend`, FastEMRIWaveforms (CPU on the laptop), `unittest`. Dev env python: `/Users/mkatz/miniconda3/envs/deving/bin/python`.

**Spec:** Mike 2026-10-24 (session a3): "Batch the chunks across rows. Check the shorter chunk option. It is the very end and may cost minimal accuracy, but check. Add the log line. Do the log line first and push. Then do everything else." Background and prior measurements: `docs/emri-direct-wdm.md` (sections "In-window plunge", "In the global fit", the H100 speed table) and the "Prior work" section below.

## Prior work (read before touching code)

- 2026-10-02/03, H100, `scripts/emri/emri_speed_durations.sh` (CD1L EMRI 1, plunge at 518 d): 720 d direct 551 ms single / 630 ms per row with EVERY row built alone; stage split mode list 158 ms (FEW twice), tracks 51 ms, lookup+tail 405 ms (kernel 1.2 ms, dense TD ~250 ms, ~30 chunk transforms 157 ms). Fixes pushed on dev (`a8f21276`, `3d29bc40`): channels-only dense kernel, common handoff (one transform per window), one FEW call per row, source-frame stop taper. After them: 175 ms/template at 720 d (production 396) but the post-taper 720-d GPU rerun is still "pending" in `docs/emri-direct-wdm.md` (line ~355). The 10-03 note named "batching plunging rows" as the next lever; never built.
- 24mo production (job 827, 2026-10-24/25, per-leaf medians over 25-repeat proposes of ~300 rows): inspiral-only leaves 4, 5 at 0.030-0.039 s/row; plunging leaves 0, 3, 2, 1, 6, 7 at 0.109 / 0.134 / 0.155 / 0.160 / 0.244 / 0.396 s/row; generate is 95 % of it, scoring 1.7 s per leaf. The EMRI block is 390-480 s per propose, 80 % of the source block.
- The chunk length `Nt_sub` was never swept: no script option exists; 128 comes from the edge-contamination measurement in `plunge_chunk_wdm` (`wdm_direct.py:218`: "Nt_sub=128: 0.2 at the edge, 5e-5 at 16 px, 6e-6 at 24 px") and `tests/test_wdm_chunk_splice.py`.
- A trajectory that ends inside the window always starts its chunk at least `Nt_sub` layers before its end (`_chunk_start`, `wdm_direct.py:1449-1452`), whatever the handoff triggers say; the handoff triggers (`handoff_pixel`, tol 0.1 rad, `WDM_HALF_SUPPORT_LAYERS` 4, the table fdot axis) can only move it earlier.

## Global Constraints

- Work on `dev` in the main checkout `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools`; commit ONLY your own hunks (other sessions keep uncommitted edits there; `git add` by path, never `-a`). Push after Task 1 (Mike's standing instruction for this plan); the later tasks are committed, not pushed, until Mike says.
- Laptop rules: ONE python process at a time machine-wide (`pgrep -fl "bin/python" | grep -vE "vscode|Code Helper|pet server"` before every run; Jupyter kernels of other sessions may be idle, do not kill them), `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` exported, no 720-day L1 windows on the laptop (SIGKILL at ~6 GB: use the gate's synthetic plunging source or 60-day equal-arm windows at dt 5 s).
- Pure-Python changes only; nothing in `.cu`/`.hh` (the cluster would need a rebuild). If Task 2 finds the dense channel kernel cannot take several templates in one call, stop and report instead of editing C++.
- Every existing test in `tests/test_emri_wdm_direct.py`, `tests/test_emri_direct_wiring.py`, `tests/test_emri_direct_move.py`, `tests/test_wdm_chunk_splice.py`, `tests/test_tdi_dense.py` must stay green; run them with `python -m unittest <module>` from the LAT root.
- The `[EMRI_DIRECT] leaf N:` line keeps its existing fields and order; new fields are APPENDED (the monitor's log parsers grep its prefix).
- No new settings file; a new knob is an env var named for the attribute it seeds (`EMRI_DIRECT_NT_SUB`), read where the sibling knobs are read (`EMRIDirectWDM.__init__`), explicit argument > env > default.
- Accuracy gates come from the existing scripts (`scripts/emri/emri_direct_wdm_gate.py`, `scripts/emri/emri_direct_stage_timing.py`); extend them with options, do not write new harnesses. Never hand-roll a likelihood or inner product.
- Verification is by SPEED TESTS, not by relaunching a global fit (Mike 2026-10-24: "we should verify these changes with speed tests rather than relaunching the global fit ideally"): on the laptop the CPU timing of the synthetic plunging gate source before/after each change; on the cluster Mike runs `scripts/emri/emri_speed_durations.sh` at `DAYS=720` (one H100). The log line of Task 1 is telemetry for production; the speed scripts must show the same split (Task 3 Step 5) so nothing needs a relaunch to be read.

## Review Focus

1. A batch holding BOTH lookup-only rows and chunk rows must give each row the result it gets alone today (Task 2 test `test_mixed_batch_matches_single_calls`).
2. Two chunk rows plunging at different times must each hand off at THEIR OWN common handoff, never at the batch minimum (Task 2 test `test_common_handoff_is_per_row`).
3. A chunk row FEW refuses (`WaveformDomainError`) inside a batched chunk group must stay zero and be listed in `last_failed_rows` without failing the other rows (Task 2 test `test_refused_chunk_row_stays_zero`).
4. The log line must not raise when a leaf's batch had zero chunk rows or zero rows (division by zero in a median) (Task 1 test `test_stats_line_with_no_chunk_rows`).
5. `EMRI_DIRECT_NT_SUB` values that `tail_chunk_plan` cannot tile (odd, < 8, or larger than `Nt`) must be refused at construction with a message naming the knob (Task 3 test `test_nt_sub_knob_refuses_untileable_values`).

---

### Task 1: the lookup / chunk split on the `[EMRI_DIRECT] leaf N:` line (then push)

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py:1895-1995` (`_batch_chunk`: accumulate the alone rows' counters into the batch stats)
- Modify: `src/lisatools/globalfit/moves/emridirectmove.py:169-173` (`_new_stats`), `:312-340` (`_score_chunk`: pull `adapter.last_stats` after `templates`), `:510-521` (`_flush_stats`: the line)
- Test: `tests/test_emri_direct_wiring.py` (next to `test_alone_rows_reuse_the_first_few_call`), `tests/test_emri_direct_move.py`

**Interfaces:**
- Consumes: `EMRIDirectWDM._batch_chunk` stats dict (`rows, alone, subs, failed, lookup_pixels, chunk_pixels, dropped_pixels`); `_call_knots` sets `self.last_stats` with `chunk_start, lookup_pixels, chunk_pixels, dropped_pixels`; `EMRIDirectWDMSignalGen.templates` copies `direct.last_stats` into `self.last_stats` (`direct_signal_gen.py:228`).
- Produces: `EMRIDirectWDM.last_stats` of a batch call additionally carries `tail_layers` (list of int, one per chunk row: `n_end - n_h` summed over that row's windows, i.e. the pixels the exact chunk filled) and `windows` (int, chunk windows transformed in the call); the move's stats dict gains `chunk_rows, lookup_px, chunk_px, dropped_px, windows, tail_layers` (a list); the log line gains ` | chunk rows A (tail median L px, W windows) lookup px P chunk px C dropped px D`. Task 2 keeps these names.

- [ ] **Step 1: Write the failing generator test** in `tests/test_emri_direct_wiring.py`, modelled on `test_alone_rows_reuse_the_first_few_call` (same fixture: a batch where one row hands off to the chunk):

```python
def test_batch_stats_carry_the_alone_rows_chunk_counters(self):
    gen = self._gen_with_one_chunk_row()          # the fixture the neighbour test uses
    gen.batch(self.rows, chunk_rows=4)
    st = gen.last_stats
    self.assertEqual(st["alone"], 1)
    self.assertGreater(st["chunk_pixels"], 0)      # the alone row's chunk pixels are counted
    self.assertEqual(len(st["tail_layers"]), 1)
    self.assertGreaterEqual(st["tail_layers"][0], gen.Nt_sub // 2)   # at least one kept interior
    self.assertGreaterEqual(st["windows"], 1)
```

- [ ] **Step 2: Run it**: `python -m unittest tests.test_emri_direct_wiring -k batch_stats_carry` -> FAIL (`KeyError: 'tail_layers'`).

- [ ] **Step 3: Implement in `_batch_chunk`** (`wdm_direct.py` after the alone `_call_knots`/`self(...)` call at lines 1935-1940): after `out_arr[r] = ...`, read `self.last_stats` of that single call and add `lookup_pixels`, `chunk_pixels`, `dropped_pixels` into `stats` (initialise the three keys to 0 and `stats["tail_layers"] = []`, `stats["windows"] = 0` at line 1926); append the row's tail length. The single-call path must expose it: in `accumulate_harmonic_batch` (`wdm_direct.py:433-457`) add `stats["windows"] = len(windows)` and `stats["tail_layers"] = int(sum(khi - klo for items in windows.values() for (_, klo, khi) in items))` (for the one-row call `S` subs share one keep range per window, so divide by `len(subs)` inside the group loop: count `(khi - klo)` once per `(klo, khi)` group, not per sub). `_call_knots` passes `totals` straight into `last_stats` (line 1726), so nothing else changes there. At the end of `_batch_chunk` (both returns, lines 1967 and 1994) the batch `stats` already carries the kernel-path `tot` for lookup rows; make sure the alone rows' numbers are ADDED, not overwritten (`stats[k] += ...` after `stats.update(... **tot)` -- move the alone accumulation into a local `alone_tot` dict and add it after `tot`).

- [ ] **Step 4: Run** `python -m unittest tests.test_emri_direct_wiring` -> all PASS (the new test and `test_alone_rows_reuse_the_first_few_call`).

- [ ] **Step 5: Write the failing move test** in `tests/test_emri_direct_move.py`, next to `test_failed_batch_falls_back_to_container_path_loudly` (reuse its fake adapter; give the fake a `last_stats` attribute):

```python
def test_stats_line_reports_the_chunk_split(self):
    move, adapter = self._move_with_fake_adapter()
    adapter.last_stats = dict(alone=2, lookup_pixels=1000, chunk_pixels=50, dropped_pixels=3,
                              windows=2, tail_layers=[20, 30])
    move.compute_like_local(self.coords, self.idx)
    with self.assertLogs("lisatools.globalfit.moves.emridirectmove", level="INFO") as cm:
        move._flush_stats()
    line = [l for l in cm.output if "[EMRI_DIRECT] leaf" in l][0]
    self.assertIn("chunk rows 2 (tail median 25 px, 2 windows)", line)
    self.assertIn("lookup px 1000 chunk px 50 dropped px 3", line)

def test_stats_line_with_no_chunk_rows(self):
    move, adapter = self._move_with_fake_adapter()
    adapter.last_stats = dict(alone=0, lookup_pixels=1000, chunk_pixels=0, dropped_pixels=0,
                              windows=0, tail_layers=[])
    move.compute_like_local(self.coords, self.idx)
    with self.assertLogs("lisatools.globalfit.moves.emridirectmove", level="INFO") as cm:
        move._flush_stats()
    line = [l for l in cm.output if "[EMRI_DIRECT] leaf" in l][0]
    self.assertIn("chunk rows 0 (tail median 0 px, 0 windows)", line)
```

- [ ] **Step 6: Run** `python -m unittest tests.test_emri_direct_move -k stats_line` -> FAIL (`AssertionError: 'chunk rows' not found`).

- [ ] **Step 7: Implement in `emridirectmove.py`**: `_new_stats` adds `chunk_rows=0, lookup_px=0, chunk_px=0, dropped_px=0, windows=0, tail_layers=[]`; in `_score_chunk` right after the successful `adapter.templates(...)` (line 324) read `st = getattr(adapter, "last_stats", {}) or {}` and accumulate `chunk_rows += st.get("alone", 0)`, `lookup_px += st.get("lookup_pixels", 0)`, `chunk_px += st.get("chunk_pixels", 0)`, `dropped_px += st.get("dropped_pixels", 0)`, `windows += st.get("windows", 0)`, `tail_layers.extend(st.get("tail_layers", []))`; in `_flush_stats` append to the format string
  `" | chunk rows %d (tail median %d px, %d windows) lookup px %d chunk px %d dropped px %d"` with `int(np.median(tail_layers)) if tail_layers else 0`. Keep the existing fields and their order.

- [ ] **Step 8: Run** `python -m unittest tests.test_emri_direct_move tests.test_emri_direct_wiring tests.test_emri_wdm_direct` -> all PASS.

- [ ] **Step 9: Update the doc**: in `docs/emri-direct-wdm.md`, section "In the global fit", add one sentence to the bullet that describes the `[EMRI_DIRECT] leaf` line (search "fallbacks") naming the six new fields and that `chunk rows` = rows built through the plunge chunk (built alone until Task 2 lands).

- [ ] **Step 10: Commit and push** (own hunks only):

```bash
git add src/lisatools/sources/emri/wdm_direct.py src/lisatools/globalfit/moves/emridirectmove.py \
        tests/test_emri_direct_wiring.py tests/test_emri_direct_move.py docs/emri-direct-wdm.md
git commit -m "EMRI direct: the lookup / plunge-chunk split on the [EMRI_DIRECT] leaf line"
git fetch origin && git rebase origin/dev   # other sessions push to dev; resolve only if your files conflict
git push origin HEAD:dev
```

Report the commit hash and the exact new line format to Mike. The split is read from the speed scripts (Task 3 Step 5 prints it from `emri_direct_stage_timing.py`), not from a relaunched global fit; the 24mo picks the line up whenever it is next relaunched for its own reasons.

---

### Task 2: batch the plunge-chunk rows

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py:338-457` (`accumulate_harmonic_batch`: per-row common handoff, per-row tail windows, batched tail evaluation; drop the `NotImplementedError` at line 441), `:1588-1633` (`_dense_td` / `_dense_tail_fn`: several items per call), `:1895-1995` (`_batch_chunk`: a chunk-row group instead of the alone loop)
- Test: `tests/test_emri_wdm_direct.py` (next to `test_common_handoff_with_a_summed_tail`), `tests/test_emri_direct_wiring.py`

**Interfaces:**
- Consumes: `_dense_kernel(items, t_abs)` (`wdm_direct.py:1522`; `items` = `[(dense inputs, (psi, lam, beta)), ...]`, `t_abs` `(n_temp, N)`) and `TDDenseTDIonTheFly.channels(params, sum_subs, stop_taper)` (`response/tdionfly.py:1530`); `tail_chunk_plan(n_lo, n_hi, Nt, Nt_sub)` and `wdm_chunk_of_td(td_arr (..., N), start, Nf, Nt_sub, dt, backend)` (`wdm_het.py:376`, `:299`; the transform accepts leading batch axes); `handoff_pixel`; the Task 1 stats names.
- Produces:
  - `accumulate_harmonic_batch(acc, table, tracks, tracer, n_ok, tail_td, *, ..., sub_row=None, num_sub=None, lookup_kernel=None, common_handoff=False, row_t_end=None)`: new keyword `row_t_end` (`(n_rows,)` trajectory end times, absolute clock, or `None`); with `sub_row` given, the common handoff is taken PER ROW (`k_h[sub_row == r] = min over that row's subs`), each row's `n_end` is its own last tracked pixel `+ 1 + Nt_sub // 4` (capped at `Nt - pixel_edge`). The tail is built as `pairs = [(row, n0, klo, khi), ...]` over every row's `tail_chunk_plan`, then exactly ONE call `td_all = tail_td(pairs)` returning `(len(pairs), nch, Nf * Nt_sub)` (every harmonic of every pair built in the same kernel launch and summed there), exactly ONE transform `chunks = wdm_chunk_of_td(td_all, 0, Nf, Nt_sub, dt, backend)` -> `(len(pairs), nch, Nf, Nt_sub)` (the transform accepts leading axes), then a splice per pair `acc[row, :, :hi - m_lo, n0 + klo:n0 + khi] += chunks[i, :, m_lo:hi, klo:khi]`. No per-window or per-row Python loop around a kernel or a transform. `tail_td` keeps the attributes `summed=True` and `taper`. The single-row call (`sub_row=None`) goes through the same code with `row = 0` for every pair.
  - `EMRIDirectWDM._dense_td(self, items, t_abs, sum_subs=False, stop_taper=0.0)`: `items` is a LIST of length `n`, `t_abs` is `(n, N)` absolute sample times (one grid per item, so one (row, window) pair is one item with its own window grid); ONE `_dense_kernel(items, t_abs)` call and ONE `channels(par, sum_subs=True, stop_taper=...)` call per block, the result split per item on the kernel's template axis -> `(n, nch, N)`. Blocks: `EMRI_DIRECT_TAIL_BLOCK_ELEMS` (2e7) bounds `n_block * nch * N`, so blocking is over ITEMS, never over samples (at 24 months `N = 1440 * 128 = 184,320`: 36 pairs per block with 3 channels). The existing single-item, unsummed use (none in production; keep `_dense_td(item, ts)` working by wrapping a bare tuple and a 1-D `ts`).
  - `EMRIDirectWDM._dense_tail_fn(self, chunk_items, t0, Nf, Nt_sub, dt)`: returns `tail_td(pairs)` that maps each `(row, n0, klo, khi)` to `chunk_items[row]` and the window grid `t0 + (n0 * Nf + arange(Nf * Nt_sub)) * dt`, stacks the grids and calls `_dense_td` once.
  - `EMRIDirectWDM._batch_chunk`: chunk rows are collected (holder-derived dense inputs, sky, tracks subset, `t_end`, sparse grid with the dense plunge segment) and built by a new method `_chunk_rows_batch(self, out_arr, chunk_items, m_lo, m_hi, few_kwargs) -> dict` (stats: `rows, windows, tail_layers, lookup_pixels, chunk_pixels, dropped_pixels`); `stats["alone"]` keeps counting rows that still go through `__call__` (only the `feed != "knots"` / fixed-`n_fine` branch), `stats["chunk_rows"]` counts the batched ones; the move's `chunk_rows` field (Task 1) becomes `alone + chunk_rows`.

- [ ] **Step 1: Confirm the dense kernel takes several templates per `channels` call.** Read `response/tdionfly.py:1530-1600` and the `_dense_kernel` padding (`wdm_direct.py:1529-1553`); run on CPU, from a scratch script in the scratchpad (never committed):

```python
# two items from the gate's synthetic plunging source (scripts/emri/emri_direct_wdm_gate.py builds one;
# copy its construction), common ts of 2000 samples at dt 5 s around the plunge
a = gen._dense_td([item0], ts, sum_subs=True, stop_taper=120.0)
b = gen._dense_td([item1], ts, sum_subs=True, stop_taper=120.0)
ab = gen._dense_td([item0, item1], ts, sum_subs=True, stop_taper=120.0)   # after Step 3
```

Expected after Step 3: `ab[0] == a`, `ab[1] == b` to 1e-12 relative. If `channels` returns a shape that cannot be split per template (no leading template axis) and the Python-side fix is not obvious from `sub_temp_host` / `offs`, STOP and report: the plan then needs a C++ change and Mike decides.

- [ ] **Step 2: Write the failing tests** in `tests/test_emri_wdm_direct.py` next to `test_common_handoff_with_a_summed_tail` (reuse its synthetic tracks and `tail_td`; build a second row whose handoff pixel differs):

```python
def test_common_handoff_is_per_row(self):
    # rows 0 and 1 each with 2 subs; row 1 hands off 10 pixels earlier
    acc, stats = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=self._tail_two_rows())
    self.assertEqual(stats["handoff_pixel_per_row"], [self.k_row0, self.k_row1])   # expose in stats
    # row 0's lookup ran up to ITS handoff: pixels between k_row1 and k_row0 are non-zero on row 0
    self.assertTrue(np.any(acc[0, :, :, self.k_row1:self.k_row0] != 0))
    # row 1's first tail window starts before row 0's (its handoff is earlier)
    self.assertLess(min(n0 for (r, n0, _, _) in stats["pairs"] if r == 1),
                    min(n0 for (r, n0, _, _) in stats["pairs"] if r == 0))
    self.assertEqual(stats["windows"], len(stats["pairs"]))

def test_batched_tail_matches_single_row_tails(self):
    acc_b, _ = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=self._tail_two_rows())
    acc_0, _ = self._accumulate_one_row(row=0)
    acc_1, _ = self._accumulate_one_row(row=1)
    np.testing.assert_allclose(acc_b[0], acc_0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(acc_b[1], acc_1, rtol=1e-12, atol=1e-14)

def test_tail_is_one_dense_call_and_one_transform(self):
    # two rows, two windows each -> 4 pairs; the tail function and the transform run ONCE
    tail = self._tail_two_rows(); tail.calls = 0
    with mock.patch("lisatools.wdm_het.wdm_chunk_of_td", wraps=wdm_het.wdm_chunk_of_td) as tr:
        _, stats = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=tail)
    self.assertEqual(tail.calls, 1)
    self.assertEqual(tr.call_count, 1)
    self.assertEqual(tr.call_args[0][0].shape[0], stats["windows"])     # all pairs in one batch
```

and in `tests/test_tdi_dense.py` (next to `test_dense_equals_spline_feed`; the fixture there builds `TDDenseTDIonTheFly` inputs for one template):

```python
def test_dense_td_batches_items_with_their_own_grids(self):
    gen, item0, item1 = self._gen_and_two_items()           # two templates of the fixture's source
    ts0, ts1 = self._window_grid(n0=0), self._window_grid(n0=64)
    a = gen._dense_td([item0], ts0[None, :], sum_subs=True, stop_taper=120.0)
    b = gen._dense_td([item1], ts1[None, :], sum_subs=True, stop_taper=120.0)
    ab = gen._dense_td([item0, item1], np.stack([ts0, ts1]), sum_subs=True, stop_taper=120.0)
    np.testing.assert_allclose(asnumpy(ab[0]), asnumpy(a[0]), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(asnumpy(ab[1]), asnumpy(b[0]), rtol=1e-12, atol=1e-15)
```

and in `tests/test_emri_direct_wiring.py` (the real generator on CPU, the fixture of `test_alone_rows_reuse_the_first_few_call` extended to TWO chunk rows plus one lookup-only row):

```python
def test_mixed_batch_matches_single_calls(self):
    gen = self._gen_with_two_chunk_rows_and_one_lookup_row()
    arr = gen.batch(self.rows, chunk_rows=4)
    for r, p in enumerate(self.rows):
        one = gen(*p).arr[:, gen._band()[0]:gen._band()[1]]
        np.testing.assert_allclose(asnumpy(arr[r]), asnumpy(one), rtol=1e-10, atol=1e-13)
    self.assertEqual(gen.last_stats["chunk_rows"], 2)
    self.assertEqual(gen.last_stats["alone"], 0)

def test_refused_chunk_row_stays_zero(self):
    gen = self._gen_with_two_chunk_rows_and_one_lookup_row()
    rows = list(self.rows); rows[1] = self.row_few_refuses      # a row inside the chunk group
    arr = gen.batch(rows, chunk_rows=4, skip_domain_errors=True)
    self.assertEqual(gen.last_failed_rows, [1])
    self.assertTrue(np.all(asnumpy(arr[1]) == 0))
    np.testing.assert_allclose(asnumpy(arr[0]), asnumpy(gen(*rows[0]).arr[:, gen._band()[0]:gen._band()[1]]), rtol=1e-10, atol=1e-13)
```

- [ ] **Step 3: Run them**: `python -m unittest tests.test_emri_wdm_direct -k "per_row or batched_tail"` and `python -m unittest tests.test_emri_direct_wiring -k "mixed_batch or refused_chunk"` -> FAIL (`NotImplementedError: plunge chunks are not batched across templates`, `KeyError`).

- [ ] **Step 4: Implement `_dense_td` / `_dense_tail_fn` for a list of items with per-item grids** (`wdm_direct.py:1588-1633`): `_dense_kernel(items_block, t_abs_block)` with `t_abs_block` `(n_block, N)` (the kernel already pads per-template knots and takes an `(n_temp, N)` time array), `dense.channels(par, sum_subs=True, stop_taper=...)` split per template on the kernel's template axis (`dense.sub_temp_host` / the `offs` array built in `_dense_kernel`; Step 1 established the shape); blocks over items of size `max(1, EMRI_DIRECT_TAIL_BLOCK_ELEMS // (nch * N))`. `_dense_tail_fn(chunk_items, t0, Nf, Nt_sub, dt)` returns `tail_td(pairs)`: `items = [chunk_items[row] for (row, n0, _, _) in pairs]`, `t_abs = np.stack([t0 + (n0 * Nf + np.arange(Nf * Nt_sub)) * dt for (_, n0, _, _) in pairs])`, one `_dense_td(items, t_abs, sum_subs=True, stop_taper=taper)`; set `tail_td.summed = True`, `tail_td.taper = taper`, and count calls on the function object (`tail_td.calls`) for the test.

- [ ] **Step 5: Implement the per-row, fully batched tail in `accumulate_harmonic_batch`** (`wdm_direct.py:395-457`): `k_h` per row minimum when `sub_row` is given (`np.minimum.at` over `sub_row`); per-row `n_end` from `row_t_end` (pixels with `t_pix <= row_t_end[r]`; the common `n_ok_h[-1]` when `row_t_end is None`); build `pairs = [(row, n0, klo, khi)]` from `tail_chunk_plan(n_h_row, n_end_row, Nt, Nt_sub)` for every row (the single-row call has `row = 0` for all), merge subs that share a row and a range (the summed tail makes them one series); then exactly one `td_all = tail_td(pairs)`, exactly one `chunks = wdm_chunk_of_td(td_all, 0, Nf, Nt_sub, dt, backend=backend)`, then the splice loop over pairs (pure indexing, no kernels); remove the `NotImplementedError`; `stats["windows"] = len(pairs)`, `stats["tail_layers"]` (per row: sum of `khi - klo` over its pairs), `stats["handoff_pixel_per_row"]`. The old `tail_td(ts)` single-window callers (`_call_knots` line 1712-1719: the dense `_dense_tail_fn` AND the spline-response `tail_td(ts, ...)` closure) are converted to the `pairs` form (the spline closure evaluates `out.eval_tdi` per pair grid and stacks). The single-row call (`sub_row=None`) must produce output equal to today's to 1e-12 (`test_common_handoff_with_a_summed_tail`, `test_subs_sharing_a_chunk_range_are_transformed_together` stay green; the harmonic sum now happens for all pairs in one kernel call instead of one per window, which reorders floating-point sums -- hence 1e-12, not byte identity).

- [ ] **Step 6: Implement `_chunk_rows_batch` and wire `_batch_chunk`** (`wdm_direct.py:1927-1941`): in the row loop, when `chunk_start is not None` and the knots/dense path is active, capture per row at once (the next row's FEW call overwrites `_last_holder` / `_last_tracks`): `H`, `dense_inputs_from_holder(H, integ, a=p[2], xI0=p[5])`, the sky `(psi, lam, beta)`, `t_end = float(H.t_arr[-1])`, the response grid `self._sparse_grid(H, chunk_start)` cut at `t_end + 2*600 + 2*self.fine_dt_plunge` (as `_call_knots` does, line 1676-1680), and the tracks subset at the pixels (`_maybe_tracks` / `subset_track`, as `_call_knots` lines 1693-1697). After the lookup rows are done, `_chunk_rows_batch` runs ONE `_dense_response(items, grids)`, builds `n_ok` from the common response span and `min(t_end)` is NOT used (each row ends at its own pixel: pass `row_t_end`), one `_lookup_fn` kernel with `n_stop` per sub from the per-row handoffs, and the batched tail; `out_arr[rows]` receives the result. Keep the `self(*p, **few_kwargs)` alone path only for `feed != "knots"` or a fixed `n_fine` (count it in `stats["alone"]`).

- [ ] **Step 6b: Design note the implementer must keep** -- the harmonics are SUMMED inside the dense kernel before the transform (`sum_subs=True`). The WDM transform is linear, so one series per (row, window) transformed once equals the per-harmonic transforms added up, at 1/modes of the FFT work; the parallelism over modes is inside the kernel launch, and the transform's parallelism is over the (row, window) pairs in the one batched call. Do not transform per harmonic.

- [ ] **Step 7: Run the six new tests and the whole EMRI set**: `python -m unittest tests.test_emri_wdm_direct tests.test_emri_direct_wiring tests.test_emri_direct_move tests.test_wdm_chunk_splice tests.test_tdi_dense tests.test_emri_direct_fanout` -> all PASS. Note the wall of `test_emri_direct_wiring` before and after (the fixture builds real FEW templates; it must not grow more than ~2x).

- [ ] **Step 8: Measure on the laptop** (CPU, the gate's synthetic plunging source, dt 5 s, 8 rows per batch -- the production `EMRI_BATCH_MAX_SIZE`): a scratchpad script timing `gen.batch(rows, chunk_rows=8)` before (git stash of the src changes, or `git show HEAD:...` into a temp module) and after, 3 repeats; report seconds per row for 8 chunk rows and for 8 lookup-only rows. Expected: the chunk rows' per-row cost drops toward the lookup rows' plus one tail window per row; if it does not drop at all, profile with `python -X importtime`-free `cProfile` on the batch call and report the top 10 before touching anything else.

- [ ] **Step 9: Update `docs/emri-direct-wdm.md`** ("In the global fit" bullet on alone rows, and the "At 720 d every row hands off to the plunge chunk and is built alone" paragraph): say chunk rows are now batched per `templates` call (one dense response, one lookup kernel, one tail evaluation and one transform per window), and what `chunk rows` on the log line means now.

- [ ] **Step 10: Commit** (no push: Mike's word):

```bash
git add src/lisatools/sources/emri/wdm_direct.py tests/test_emri_wdm_direct.py tests/test_emri_direct_wiring.py docs/emri-direct-wdm.md
git commit -m "EMRI direct: batch the plunge-chunk rows (one response, one lookup, one tail per window)"
```

- [ ] **Step 11: Hand Mike the GPU check** (he runs it; it is also the post-taper 720-d rerun that `docs/emri-direct-wdm.md` lists as pending): on an H100 allocation from the LAT root, `BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32 bash scripts/emri/emri_speed_durations.sh`, before and after the commit (two checkouts or `git stash`), and `scripts/emri/emri_direct_fit_wiring_check.py --backend cuda13x --days 180 --rows 8` for the fit wiring. Pass: mm vs production unchanged to the printed digits, ms per template at 8 per call below the single-template number, no `[EMRI_DIRECT] ... direct batch failed` fallback.

---

### Task 3: the shorter plunge chunk -- knob, accuracy gate, speed

**Files:**
- Modify: `src/lisatools/sources/emri/wdm_direct.py:1205-1212` (`EMRIDirectWDM.__init__`: `Nt_sub=None` -> env `EMRI_DIRECT_NT_SUB` -> 128, validated), `:1-40` (module docstring knob list)
- Modify: `scripts/emri/emri_direct_wdm_gate.py:40-44` (`--nt-sub`), `scripts/emri/emri_direct_stage_timing.py:48-76` (`--nt-sub`, and the JSON line), `scripts/emri/emri_speed_durations.sh:44-54` (`NT_SUB` env passed through)
- Test: `tests/test_emri_wdm_direct.py`, `tests/test_wdm_chunk_splice.py`

**Interfaces:**
- Consumes: `tail_chunk_plan` / `chunk_start_for_pixels` (`wdm_het.py:336-406`; `n_pad = Nt_sub // 4`, interior `Nt_sub - 2 n_pad - 2 >= 1`), `plunge_chunk_wdm` (`wdm_direct.py:203`), the Task 2 stats (`windows`, `tail_layers`).
- Produces: `EMRIDirectWDM(..., Nt_sub=None)`; `self.Nt_sub` = explicit int > `int(os.environ.get("EMRI_DIRECT_NT_SUB", 128))`; `ValueError("EMRI_DIRECT_NT_SUB=%d: the plunge chunk must be an even number of layers >= 8 (n_pad = Nt_sub // 4 on each side) and <= Nt")` otherwise. The gate script prints one block per region (`< 0.9`, `0.9-0.99`, `>= 0.99` of t_plunge) and per `--nt-sub` value: mismatch direct vs TOF and vs production, and the 0.25-1 mHz band SNR of the direct template (the taper leak metric of `docs/emri-direct-wdm.md` "In-window plunge").

- [ ] **Step 1: Write the failing knob tests** in `tests/test_emri_wdm_direct.py` (the module has a cheap `EMRIDirectWDM` construction in `test_handoff_is_first_trip_along_the_track`'s fixture; reuse it):

```python
def test_nt_sub_knob_reads_the_env(self):
    with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "64"}):
        self.assertEqual(self._make_gen().Nt_sub, 64)
    with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "64"}):
        self.assertEqual(self._make_gen(Nt_sub=96).Nt_sub, 96)     # explicit wins
    self.assertEqual(self._make_gen().Nt_sub, 128)

def test_nt_sub_knob_refuses_untileable_values(self):
    for bad in ("7", "6", "0", "-128", "100000"):
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": bad}):
            with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_NT_SUB"):
                self._make_gen()
```

- [ ] **Step 2: Run**: `python -m unittest tests.test_emri_wdm_direct -k nt_sub_knob` -> FAIL.

- [ ] **Step 3: Implement** in `__init__` (`wdm_direct.py:1205-1212`): signature `Nt_sub=None`; resolve; validate `Nt_sub % 2 == 0`, `Nt_sub >= 8`, `Nt_sub <= wdm_set.Nt`; add the knob to the module docstring list (line ~34) and to `docs/emri-direct-wdm.md` next to `EMRI_DIRECT_PLUNGE_TAPER_S`.

- [ ] **Step 4: Run** `python -m unittest tests.test_emri_wdm_direct tests.test_wdm_chunk_splice` -> PASS.

- [ ] **Step 5: Add `--nt-sub` to the two scripts**: `emri_direct_wdm_gate.py` (`ap.add_argument("--nt-sub", default="128", help="comma list of plunge-chunk lengths in layers")`, loop over the values, build the direct generator with `Nt_sub=v`, print the per-region table once per value with a `nt_sub=` column, plus the direct template's 0.25-1 mHz SNR under the script's noise) and `emri_direct_stage_timing.py` (`--nt-sub` int, passed to the generator, written into the JSON line; AND after each timed batch call print the generator's `last_stats` split in the log line's wording -- `chunk rows A (tail median L px, W windows) lookup px P chunk px C dropped px D` -- and write `chunk_rows, windows, lookup_pixels, chunk_pixels, dropped_pixels` into the JSON line, so the speed test shows the lookup / chunk split without a global-fit run). `emri_speed_durations.sh`: `NT_SUB=${NT_SUB:-128}` exported and passed as `--nt-sub "$NT_SUB"` to the stage-timing call.

- [ ] **Step 6: Run the accuracy sweep on the laptop** (one python at a time; the gate source is synthetic and fits in memory; the 60-day dt 5 s equal-arm window is the proven laptop configuration):

```bash
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/emri/emri_direct_wdm_gate.py --table /Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF720_DT5_TL32_fd8x0p01_nld2.h5 \
    --modes all --thresh 1e-5 --nt-sub 128,96,64,48,32 > /private/tmp/claude-501/-Users-mkatz-Research-lisa-sprint-2026/37c344bb-ab4a-4b4b-a83b-4777ca464d91/scratchpad/nt_sub_sweep.txt
```

(the NF720_DT5 table is the one the 10-01 plunge gate used; if it is missing, build it with the gate's own `--table` path through `wdm_lookup_store.ensure_lookup_table` on the laptop, ~minutes). Expected reading: `>= 0.99` region mismatch within 2x of the 128 value and the 0.25-1 mHz SNR within 2x of the 128 value for the candidates that pass; the edge-contamination numbers of `wdm_direct.py:218` predict 96 and 64 pass and 48 / 32 fail.

- [ ] **Step 7: Write the result into `docs/emri-direct-wdm.md`** as a table (`nt_sub | < 0.9 | 0.9-0.99 | >= 0.99 | 0.25-1 mHz SNR | kept interior px`) under "In-window plunge", with the recommendation and the sentence "default stays 128 until the H100 timing below is in".

- [ ] **Step 8: Commit** (no push):

```bash
git add src/lisatools/sources/emri/wdm_direct.py tests/test_emri_wdm_direct.py scripts/emri/emri_direct_wdm_gate.py \
        scripts/emri/emri_direct_stage_timing.py scripts/emri/emri_speed_durations.sh docs/emri-direct-wdm.md
git commit -m "EMRI direct: EMRI_DIRECT_NT_SUB knob + the plunge-chunk length sweep"
```

- [ ] **Step 9: Hand Mike the GPU timing** (he runs it): `BACKEND=cuda13x DAYS="720" CHUNKS="8" ROWS=32 NT_SUB=128 bash scripts/emri/emri_speed_durations.sh` and the same with `NT_SUB=64` (and 96 if 64 failed the gate). Report ms per template and mm vs production side by side. The default changes (launcher export `EMRI_DIRECT_NT_SUB`, level 1 in `scripts/flagship/launcher.sh.in`, then `python scripts/flagship/build.py`) only on Mike's word.

---

## What Mike runs on the cluster (summary)

1. After Task 1's push: nothing is required; the 24mo prints the new `[EMRI_DIRECT] leaf N:` fields whenever it is next relaunched for its own reasons (`git pull` first).
2. After Task 2: the 720-d `emri_speed_durations.sh` before/after on one H100 (`BACKEND=cuda13x DAYS="720" CHUNKS="1,8" ROWS=32`; also closes the "rerun pending" item of 10-03) and the fit wiring check -- this is the verification, no global-fit relaunch.
3. After Task 3: the `NT_SUB` timing pair with the same script; its stage-timing output now carries the lookup / chunk split per call.
