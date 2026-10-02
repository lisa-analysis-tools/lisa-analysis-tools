# SOBBH direct-to-WDM lookup scorer: status and measurements

Branch `sobbh-wdm-lookup` (LAT). Spec `docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md`,
plan `docs/superpowers/plans/2026-09-30-sobbh-wdm-lookup.md`. The SOBBH twin of the EMRI
direct-to-WDM template (`docs/emri-direct-wdm.md`), vectorized over proposal rows and slotted into
`SOBBHChunkedLikeMove` as a comp.

## What exists

| Piece | Where |
|---|---|
| Vectorized n_ref table evaluator (quarter-turn rule, linear / Keys cubic) | `lisatools/wdm_lookup_eval.py` `WDMLookupEvaluator` |
| Batched 3.5PN, batched TDI-on-the-fly, tracer, sparse template, inner products, fill | `lisatools/sources/sobbh/wdm_direct.py` |
| The comp for the move / engine (`get_ll_wdm`, `fill_global_wdm`) | `SOBBHLookupComputations` (same file) |
| Stock knob | `SOBBH_LIKELIHOOD=lookup` + `SOBBH_LOOKUP_TABLE_PATH` (and `SOBBH_LOOKUP_{NUM_M_LAYERS,EVAL_DT,INTERP,ROW_BATCH}`) |
| Gate script | `scripts/sobbh/sobbh_lookup_gate.py` |

## The table

The n_ref table depends on the layer duration only: two tables with `layer_dt = 3600 s` built at
(Nf=64, dt=56.25) and (Nf=128, dt=28.125) agree entry by entry to 4e-9 (test
`TablePortabilityTest`). The laptop EMRI table `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`
(layer 3600 s, offsets [-3, 3) layers, fdot +-8 layer units in steps of 0.01, 30 MB) therefore
serves the production grid (Nf=1440, dt=2.5). SOBBH chirp rates are tiny in layer units
(|fdot| <= 0.002 for catalogue-like sources; the 6-mo catalogue's worst chirper ~0.005), so
`num_m_layers=2` (5 layers per pixel) is *expected* to be converged by analogy with the EMRI
direct-to-WDM measurement (a chirp at fdot 0.25 layer units keeps ~1.3% of its power two layers
out; SOBBH's fdot is at most ~0.005, two orders of magnitude smaller) — this has not been
separately measured for SOBBH in this gate. No plunge chunk is needed; pixels past the fdot axis
(sources merging in band) are dropped and counted.

## Results (laptop CPU, EqualArmlengthOrbits, 2nd-generation TDI, scirdv1)

Both runs below are `scripts/sobbh/sobbh_lookup_gate.py --rows 8`, laptop table
`wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`, full printed output pasted verbatim (a
nanobind "leaked instances/types/functions" block printed by every run at interpreter shutdown is
omitted below — it is a known benign refcount-cleanup artifact at process exit, not a correctness
signal).

### What each column measures

- `mm` = flat `1 - Re(O)` per channel, no maximisation, computed on the **interior** pixels only
  (`EDGE=24` grid-end time pixels excluded on each side of the time axis, because the dense TD->WDM
  transform's grid-end pixels are unreliable edge artifacts, not a lookup defect).
- `ratio` = the norm ratio `sqrt(<h|h>/<truth|truth>)` on that same interior.
- `mm_w`, `dlogL`, `snr` = the scirdv1 XYZ noise-weighted mismatch / log-likelihood difference /
  SNR, computed over **all** pixels, including the dense transform's unreliable grid-end pixels.
- `mm_w_int` (new in this revision) = the same noise-weighted mismatch as `mm_w`, but with both
  `truth` and the lookup template zeroed outside `[EDGE, nt-EDGE)` first — i.e. `mm_w` restricted
  to the interior, added specifically to separate the two effects folded into `mm_w`. Measured
  (from `docs/sobbh_lookup_gate.jsonl`): in every one of the 12 source-rows below, `mm_w_int` is
  within about an order of magnitude of the interior flat `mm` (same pixels, same order), while
  the all-pixel `mm_w` is `69x`-`20,700x` above `mm_w_int` (about 1.8-4.3 orders of magnitude).
  This shows the `mm`-vs-`mm_w` gap is driven almost entirely by the dense TD->WDM transform's own
  unreliable grid-end pixels (present in `mm_w`, excluded from both `mm` and `mm_w_int`), not by
  an interior lookup-vs-truth defect.
- In the timing block, `exact` is the dense TD->WDM transform of the lookup's **own** batched
  response (not the production `SOBBHTDIonFly`, not an independent ground truth) — so
  `lookup_vs_exact_max_abs` isolates the lookup table / sparse-template approximation error
  specifically, while `chunked_vs_exact_max_abs` contains **all** of the chunked comp's
  approximations at once (the `m_band_half_width=3` band truncation, its `Nt_sub`/`N_sparse`
  chunking, and its own edge handling). How much of the chunked number comes from band truncation
  alone versus its other approximations is **unmeasured** by this script.
- Timing keys: `*_warm` is the FIRST call on the batch (includes any one-time setup/caching); the
  unsuffixed key (e.g. `lookup_get_ll`) is the SECOND call on the same batch. Both are single
  wall-clock samples, CPU-only, not medians. The two comps are not scoring an identical band: the
  lookup comp's band is `num_m_layers=2` -> 5 layers per pixel, the chunked comp's is
  `m_band_half_width=3` -> 7 layers. By the time the timing block runs, the lookup comp's `direct`
  object has already been exercised by the six `dense()` calls above, while the chunked comp
  (`ch`) is freshly constructed — so the two comps' `_warm`/unsuffixed pairs are not a clean
  apples-to-apples cold-start comparison between the two comps.

### `--nt 1024` (42.7 days)

```
grid Nf=180 Nt=1024 dt=20.0 layer_dt=3600.0 s; table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
src 0 f_low 0.0045 snr 0.1  mm X/Y/Z 5.26e-08/2.28e-06/5.86e-07  ratio 0.99998/0.99997/0.99998  mm_w 4.56e-04  mm_w_int 6.59e-06  dlogL -8.796e-06  tof-dense 7.4s lookup 4.4s
src 1 f_low 0.0080 snr 0.3  mm X/Y/Z 5.31e-08/8.17e-08/4.77e-08  ratio 1.00000/1.00000/1.00000  mm_w 3.31e-04  mm_w_int 9.06e-08  dlogL -3.840e-05  tof-dense 7.8s lookup 6.6s
src 2 f_low 0.0120 snr 0.3  mm X/Y/Z 1.78e-07/4.95e-08/7.88e-07  ratio 1.00001/1.00001/1.00010  mm_w 1.47e-03  mm_w_int 3.16e-07  dlogL -1.721e-04  tof-dense 5.8s lookup 4.9s
src 3 f_low 0.0150 snr 0.8  mm X/Y/Z 2.76e-07/1.93e-07/1.62e-07  ratio 1.00000/0.99998/1.00000  mm_w 9.32e-04  mm_w_int 3.12e-07  dlogL -5.352e-04  tof-dense 6.0s lookup 5.9s
src 4 f_low 0.0060 snr 0.1  mm X/Y/Z 6.66e-08/5.70e-08/6.08e-08  ratio 1.00000/1.00000/1.00000  mm_w 1.11e-03  mm_w_int 5.39e-08  dlogL -6.334e-06  tof-dense 4.9s lookup 5.2s
src 5 f_low 0.0180 snr 0.5  mm X/Y/Z 1.93e-07/2.60e-07/2.10e-07  ratio 0.99999/1.00000/1.00000  mm_w 1.38e-03  mm_w_int 2.29e-07  dlogL -3.747e-04  tof-dense 4.5s lookup 4.5s
timing: {
 "lookup_get_ll_warm": 6.661789873032831,
 "lookup_get_ll": 5.772116897045635,
 "lookup_fill": 5.717818813980557,
 "lookup_vs_exact_max_abs": 6.449406441786688e-05,
 "chunked_get_ll_warm": 15.221764751011506,
 "chunked_get_ll": 14.695631613954902,
 "chunked_vs_exact_max_abs": 0.0001748772653341578,
 "rows": 8,
 "nt": 1024
}
```

`dropped_pixels == 0` and `merged_rows == 0` for every one of the 6 sources (checked against
`docs/sobbh_lookup_gate.jsonl`). All per-channel `mm` <= 2.28e-6 (well under the 1e-3 gate); all
norm ratios within **1.03e-4** of 1 (recomputed directly from the JSONL: worst `|ratio-1| =
1.0279e-4`, src 2 channel Z; well under the 2e-3 gate). Source SNRs at this duration span
0.075-0.76 (see "SNR context" below). `lookup_vs_exact_max_abs` is 6.45e-5; `chunked_vs_exact_max_abs`
is 1.75e-4 — per the legend above, these are not a clean apples-to-apples split of the same error
source.

### `--nt 4320` (6 months, 20 s)

```
grid Nf=180 Nt=4320 dt=20.0 layer_dt=3600.0 s; table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
src 0 f_low 0.0045 snr 0.6  mm X/Y/Z 1.52e-07/2.06e-07/2.14e-07  ratio 0.99999/0.99999/0.99999  mm_w 7.57e-05  mm_w_int 6.92e-07  dlogL -2.296e-05  tof-dense 10.0s lookup 4.5s
src 1 f_low 0.0080 snr 0.7  mm X/Y/Z 3.03e-08/4.04e-08/3.00e-08  ratio 1.00000/1.00000/1.00000  mm_w 5.05e-05  mm_w_int 4.07e-08  dlogL -2.767e-05  tof-dense 5.3s lookup 5.6s
src 2 f_low 0.0120 snr 0.6  mm X/Y/Z 1.46e-07/3.49e-08/1.36e-07  ratio 1.00001/1.00000/1.00001  mm_w 2.40e-04  mm_w_int 1.28e-07  dlogL -8.599e-05  tof-dense 7.2s lookup 5.7s
src 3 f_low 0.0150 snr 1.5  mm X/Y/Z 3.99e-07/2.18e-07/2.15e-07  ratio 0.99999/1.00000/1.00000  mm_w 5.34e-04  mm_w_int 3.46e-07  dlogL -1.256e-03  tof-dense 23.6s lookup 13.5s
src 4 f_low 0.0060 snr 0.2  mm X/Y/Z 1.14e-07/1.11e-07/1.11e-07  ratio 1.00001/1.00001/1.00001  mm_w 1.42e-04  mm_w_int 1.00e-07  dlogL -7.685e-06  tof-dense 15.1s lookup 14.0s
src 5 f_low 0.0180 snr 1.2  mm X/Y/Z 1.53e-07/1.66e-07/1.57e-07  ratio 1.00000/0.99999/1.00000  mm_w 1.39e-04  mm_w_int 1.66e-07  dlogL -2.085e-04  tof-dense 14.8s lookup 8.8s
timing: {
 "lookup_get_ll_warm": 10.2268609800376,
 "lookup_get_ll": 9.328296154970303,
 "lookup_fill": 10.599311379948631,
 "lookup_vs_exact_max_abs": 5.2925551063925935e-05,
 "chunked_get_ll_warm": 75.62815951905213,
 "chunked_get_ll": 61.12050140206702,
 "chunked_vs_exact_max_abs": 0.0008285584205466046,
 "rows": 8,
 "nt": 4320
}
```

`dropped_pixels == 0` and `merged_rows == 0` for every source again. All per-channel `mm` <= 4.0e-7;
all norm ratios within **1.44e-5** of 1 (recomputed from the JSONL: worst `|ratio-1| = 1.437e-5`).
The lookup comp's `get_ll_wdm` second call for 8 rows on the 6-month grid is 9.33 s vs the chunked
comp's 61.12 s (~6.5x faster at this `row_batch`/`Nt_sub` pairing, a single-sample wall-time ratio
on a shared laptop at these specific knob settings, not a controlled benchmark or an asymptotic
speed-up claim). `fill_global_wdm` for 8 rows is 10.60 s.

### SNR context

The six catalogue-like gate sources span SNR 0.075-1.53 across both durations (measured range
from `docs/sobbh_lookup_gate.jsonl`) for these masses/frequencies at Gpc-scale distances over
these windows. lnL errors and `dlogL` scale roughly as SNR^2 x mismatch, so the move-level anchor
measured separately in `test_sobbh_lookup_move` — lookup-vs-slow lnL max |diff| 0.081 at SNR^2 =
10.7 (SNR ~3.3) — is the measured anchor at a non-trivial SNR, well above the SNR ~0.1-1.5 sources
scored directly by this gate script. The two gate durations mainly differ in SNR, not in any
duration-dependent accuracy effect: going from 42.7 d to 6 months raises each source's integrated
SNR (more cycles observed within the window), and once an error is read relative to signal power
rather than as a raw lnL-unit number, both comps' agreement with `exact` improves rather than
degrades with the longer duration (the lookup's `lookup_vs_exact_max_abs` even decreases in raw
terms, 6.45e-5 -> 5.29e-5, despite the higher-SNR batch; the chunked number's raw growth,
1.75e-4 -> 8.29e-4, is 4.7x while SNR^2 grew ~15.7x; normalised by `h_h`, both comps improve from
42.7 d to 6 months).

### Measured in the test suite

- Table portability: two tables built at different (Nf, dt) but the same layer duration agree
  entry by entry to 4e-9 (`TablePortabilityTest`).
- Batched response vs production `SOBBHTDIonFly` per row/channel: mismatch ~1e-16 (machine
  precision) at `eval_dt = 600 s` — control: a one-day reference-epoch error gives 1.92
  (`BatchedTOFTest`).
- Lookup template vs the response's own dense TD->WDM, on the tiny `eps_freq=0.01` toy table:
  mismatch 1.4e-8..1.7e-4 per channel, norm ratios within 1.34e-4 of 1 — control without the
  quarter-turn parity rule: 0.50 (`DirectWDMTest`; per-channel numbers below).
- Move-level lookup-vs-slow lnL max |diff| 0.081 (median 0.021) at SNR^2 = 10.7, against the
  0.537 bound — control (no parity turn) 5.39 (`test_sobbh_lookup_move`).
- fdot mid-node error 0.108 on the coarse `eps_fdot=0.1` toy table, for the reference
  `get_wdm_coeffs` and the evaluator alike (table resolution: the production table has
  `eps_fdot=0.01` and SOBBH's `|fdot| <= 0.005` layer units sits in its first cell).
- The TDI-on-the-fly amplitude is **signed** (liveness is `amp != 0`, never `amp > 0`).
- The cubic-spline ringing past merger that the tracer masks measured 9e-23.

### Per-channel numbers, `test_sobbh_wdm_direct.py::DirectWDMTest::test_dense_matches_tof_own_transform`

The test itself only asserts (`mm < 1e-3`, `|ratio - 1| < 2e-3` per row/channel) and does not print;
the numbers below were obtained by re-running the test's exact `setUpClass`/body (same tiny table
from `build_tiny_table`, same `ROWS`, same `td_mismatch`) with a `print` added in a throwaway
driver script (no test or library file was modified):

```
row 0 ch 0: mm 1.378e-08 ratio 1.000001
row 0 ch 1: mm 1.362e-08 ratio 1.000000
row 0 ch 2: mm 1.400e-08 ratio 1.000000
row 1 ch 0: mm 3.118e-06 ratio 0.999949
row 1 ch 1: mm 6.496e-06 ratio 0.999947
row 1 ch 2: mm 2.614e-06 ratio 0.999926
row 2 ch 0: mm 1.647e-04 ratio 0.999884
row 2 ch 1: mm 1.639e-04 ratio 0.999866
row 2 ch 2: mm 1.722e-04 ratio 0.999908
last_stats: {'rows': 3, 'pixels': 256, 'lookup_pixels': 768, 'dropped_pixels': 0, 'merged_rows': 0}
worst mm = 1.722e-04; worst |ratio-1| = 1.341e-04
```

Range: mismatch `1.4e-8..1.7e-4` per row/channel, norm ratios within `1.34e-4` of 1 on this tiny
test table. This table is **not** a scaled-down version of the gate's production-layer table in a
"same accuracy, finer grid" sense — its `eps_freq=0.01` / `eps_fdot=0.1` node spacing is 2x
coarser in the offset axis (`eps_freq` 0.01 vs 0.005) and 10x in fdot (`eps_fdot` 0.1 vs 0.01)
than the production table's (see
the fdot mid-node-error fact above), so these numbers are a resolution-limited upper bound on this
toy table, not a smaller- or larger-scale version of the gate's own 42.7 d / 6-month measurement.

## Conventions pinned by tests

- Chunked-basis rows `(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)`; the
  response feed is `phase = gw_phase + pi` with the intrinsic amplitude (`SOBBHTDIonFly`).
- Pixel centres `t_n = t_obs_start + n * layer_dt`, `n` the absolute grid index; the quarter-turn
  rule uses the absolute `(m + n)` parity.
- Inner products carry NO `4 * differential_component` factor (the chunked kernel convention);
  `-0.5 (h_h - 2 d_h)` is the container's source term.
- Batched response vs production `SOBBHTDIonFly` per row/channel: mismatch ~1e-16 (machine
  precision) at `eval_dt = 600 s` (`BatchedTOFTest`; the test's own assertion threshold is
  <= 1e-8, looser than what is actually measured); a one-day reference-epoch error is caught as a
  control (mismatch 1.92).

## Known limits

- Python, `xp`-vectorized (numpy / cupy); no C++/CUDA kernel yet (see follow-ups).
- A source merging inside the window keeps evaluating up to `tc` with the shared node grid
  (production `SOBBHTDIonFly` zeros the last `buffer_time` before merger instead); its pixels past
  the table's fdot axis are dropped (counted in `last_stats["dropped_pixels"]`, one warning per
  call).
- The lookup comp is SINGLE-DEVICE: the stock getter refuses a multi-GPU run (`ValueError`) and
  the comp raises when called off its build device; per-device replicas (the GB
  `_RoutedBandEngine` pattern) are a follow-up. Use `SOBBH_LIKELIHOOD=chunked` for multi-GPU
  walker shards.
- cubic interpolation is Keys cubic convolution, not scipy's global cubic spline.

## Follow-ups

1. **CUDA/C++ kernel** in the chunked-het family: per-pixel `SOBBHTDIonTheFly::get_tdi` amplitude /
   phase at the pixel centres, the table resident on the device, the same quarter-turn rule and
   the same `(d_h, h_h)` accumulation; this Python path is its reference. GPU leads per the sprint
   rule; the comp swap is then a comp swap, not a move change.
2. **CD1L data gate**: per-source mismatch / dlogL of the lookup template vs the mojito L1 SOBBH
   streams (ids 0-5 of the 6-mo run) on the production grid — the bricks are not on this laptop.
   Recipe: `L1ProcessingStep(source_types=["sobbh"], source_ids=dict(sobbh=[id]), orbits_class=L1Orbits,
   frame="icrs")` as `scripts/emri/emri_cd1l_campaign.py` does for EMRIs, the stock SOBBH wave wrap as
   the production template, `SOBBHDirectWDM.dense` as the fast one, report `1 - Re(O)` per channel,
   norm ratio, dlogL, SNR.
3. **Per-device replicas** of the lookup comp (table + orbits/tdi wraps per device, routed per
   shard as the GB `_RoutedBandEngine` does) to lift the single-device restriction.
4. JAX mirror; the fused phase-max quadrature; F-stat / gradient methods on the comp.

## One-GPU speed test (cluster runbook)

`scripts/sobbh/sobbh_lookup_speed_gpu.py` times the lookup comp and the chunked comp head to head
on ONE device at the production 6-month grid (Nf=1440, Nt=4320, dt=2.5): `get_ll_wdm` first call /
warm (median of `--repeats`; the lookup comp is already exercised by the setup fill, so its first
call is not a from-scratch cold start), the fill, the lookup's template/inner split (device-
synchronised spans, taken from the last timed call), the memory pool per comp (`lk_pool` /
`ch_pool` = pool total GB after that comp's calls, freed between the comps; the JSON also holds
the in-use figures) and the per-row lnL difference of the two comps on the same residual.
Device-synchronised timings. Each JSON line records the grid, `m_band`, `fill_band`,
`num_m_layers`, `nt_sub` and a timestamp.

    # cluster, one GPU, the stack's python, this branch checked out
    SOBBH_LOOKUP_TABLE_PATH=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda12x --rows 4,8,32,96,288 \
        --out sobbh_lookup_speed_gpu.jsonl

Reading the table: `ch/look` is warm chunked time over warm lookup time, so `ch/look < 1` means
the chunked comp was FASTER. Expect (a GPU EXPECTATION from the chunked kernel's one-block-per-row
launch, not a measurement) the chunked wall to be roughly flat in the batch size up to the
device's block capacity; the interesting columns are how `look_warm` scales with `rows` and the
`resp` (batched response) / `trac` (tracer) / `look` (lookup) vs `inner` (gathers + reductions)
split. `--row-batch` bounds
the response-spline memory (default 32); raise it on a large-memory device and watch the pool
columns. `resp`/`trac`/`look`/`inner` are not the warm median but the spans of the last timed call.

GPU results: NOT MEASURED here (the laptop has no CUDA backend); the GPU numbers come from the
cluster run. The CPU smoke below (`--backend cpu --laptop --nt 256 --rows 2,4 --repeats 1`) has
the lookup SLOWER than the chunked comp, and it must not be read as a GPU result: the smoke grid
is tiny (256 layers), so the lookup's fixed per-call template build (~4-7 s here, independent of
rows and of the grid: it was also ~7 s on the production grid) dominates; another session's Python
process was running during the original smoke, so the CPU numbers are contended (the Task 7 gate
measured 4.6 s per 8-row lookup call on a 1024-layer grid on the same laptop); and none of this
says anything about the GPU, where the per-row work is parallel. The CPU smoke scales with rows
(chunked 1.10 s at 2 rows, 2.19 s at 4), unlike the GPU expectation above. The cluster run is the
measurement.

CPU smoke, laptop preset:

    backend cpu  grid Nf=180 Nt=256 dt=20.0 layer_dt=3600 s  active 178 x 256  table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5  comp build 0.1 s
    chunked comp: Nt_sub=32 n_chunks=11 build 4.5 s
     rows  look_first  look_warm     resp     trac     look    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
        2       7.570      4.495    4.463    0.003    0.007    0.001      4.448     1.135     1.204     1.669     0.27  3.317e-05      nan      nan

(re-run after the span split: the lookup call's cost is the batched response build, `resp`; the
tracer and the lookup itself are milliseconds on this grid. The 4-row smoke row and the
production-grid table below predate the column rename: `tmpl` there is `resp + trac + look`.)

Production grid on CPU, 2 rows, lookup only (`--backend cpu --rows 2 --repeats 1 --no-chunked`;
completed, no watchdog exit; proves the Nf=1440, Nt=4320, dt=2.5 grid runs):

    backend cpu  grid Nf=1440 Nt=4320 dt=2.5 layer_dt=3600 s  active 179 x 4320  table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5  comp build 0.1 s
     rows  look_first  look_warm     tmpl    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
        2       6.507      5.443    7.170    0.006      9.528       nan       nan       nan      nan        nan      nan      nan
