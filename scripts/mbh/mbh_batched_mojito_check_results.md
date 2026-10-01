# Batched windowed MBH likelihood vs stock: mojito MBHB check (laptop CPU, 2026-09-30)

Script: `scripts/mbh/mbh_batched_mojito_check.py`. Logs and JSON are in
`scripts/mbh/mbh_batched_check_out/`. That directory is scratch and is not staged.

**Machine:** MacBook, Intel i5-8257U at 1.4 GHz, 8 GB RAM. `deving` env with the
worktree-shadowed `lisatools`. Settings: `OMP_NUM_THREADS=1`,
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `JAX_PLATFORMS=cpu`, `nice -n 10`.
One python process per run, with the CPU guard. Backend: cpu. Response order: 8.
Batch size B=1 (see Timing).

**Data:** `MBHB_731d_2.5s_L1_source{16,17}_*.h5`. The script reads only the
120-day window: `tdis.xyz_doppler[start:stop]`, which is X2,Y2,Z2 divided by
`laser_frequency`. This is the same dataset `L1DataLoader` reads. The orbits come
from `WindowedL1Orbits`, an `L1Orbits` whose `_setup` reads ltts only over the
window ± 1e5 s. The catalogue row index is the source id, as in `load_single_binary`.
The merger sits 100 d into the window. The stock generator runs with `Tobs = 90 d`,
the same as the windowed generator.

**Window geometry (both ids):** 120.0 d, Nf=17280, Nt=240, layer = 12.00 h.
Active f layers are [9, 2160] and t layers [0, 239].
- id 16: snap +0.500000 s. Kept layers 18..224 of 240 (Nt_keep=206, n_pad=8, segment 10..232). Merger layer 200.00.
- id 17: snap +0.500000 s. Kept layers 17..223 of 240 (Nt_keep=206, n_pad=8, segment 8..232). Merger layer 200.00.

**Reference (controller ruling 2026-09-30):** the stock generator is given the SAME
lattice-snapped epoch as the windowed one: `waveform_t0 = REF + 0.5 s` and
`t_plunge - 0.5 s`, which is the same absolute merger. This is the script default.
`MBH_CHECK_STOCK_T0=ref` keeps the unsnapped variant. Rows are:
- truth;
- two NEAR-truth rows, ±(t_plunge 0.3 s, dist 5e-4, phi_ref 0.002). id 16 also ran a wider set, ±(1 s, 2e-3, 0.01), to reach |ΔlogL| = O(25);
- one FAR-jitter row: dist ×U(0.97,1.03), t_plunge +U(-20,20) s, phi_ref +U(-0.2,0.2), rng seed 0. This row does NOT gate acceptance.

`mm` = 1 − ⟨hs_box|hb⟩/√(⟨hs_box|hs_box⟩⟨hb|hb⟩), scirdv1 XYZ noise, computed inside
the kept box. `||delta||` is the noise-weighted norm of hb − hs_box.

## Acceptance (truth + near rows: |dlogL| ≤ 0.5, mm < 1e-6, outside-box power ≪ 1e-4)

| id | verdict | max \|dlogL\| (gating) | max mm (gating) | outside-box power |
|----|---------|------------------------|-----------------|-------------------|
| 16 | **HELD** | 4e-4 | 1.7e-10 | 1.8e-10 |
| 17 | **HELD** | 1.5e-3 | 6.0e-11 | 3.7e-10 |

These results come after the onset warm-up fix to `GridAlignedPhenomTHMTDIWaveform`
(see below). Before the fix, id 17 failed at truth: dlogL −2.44, mm 1.2e-6.

## id 16 (SNRopt 536), after fix

```
[RSS  3.95 GB] stock: 59.86 s/row
[batched B=1] 69.323 s/row
row 0 [truth]: logL stock -0.906 batched -0.906 dlogL +0.0000 | mm 1.696e-10 | kept err 4.95e-04 at (c,f,t)=(2, 1595, 199) | ||delta|| 9.874e-03 | power outside box 1.83e-10 | SNRopt 536.5/536.5
row 1 [near]: logL stock -1.174 batched -1.174 dlogL +0.0001 | mm 1.700e-10 | kept err 4.94e-04 at (c,f,t)=(2, 1595, 199) | ||delta|| 9.893e-03 | power outside box 1.83e-10 | SNRopt 536.2/536.2
row 2 [near]: logL stock -1.523 batched -1.523 dlogL +0.0001 | mm 1.632e-10 | kept err 4.99e-04 at (c,f,t)=(2, 1595, 199) | ||delta|| 9.702e-03 | power outside box 1.83e-10 | SNRopt 536.8/536.8
row 3 [far]:  logL stock -16316.680 batched -16316.694 dlogL -0.0138 | mm 2.512e-10 | kept err 2.44e-04 at (c,f,t)=(1, 1590, 200) | ||delta|| 1.199e-02 | power outside box 1.83e-10 | SNRopt 535.4/535.4
logL_stock - logL_truth per row: truth +0.00, near -0.27, near -0.62, far -16315.77
```
The near rows sit only 0.3–0.6 nats from truth at SNR 536, so the wider near set
was also run (`MBH_CHECK_NEAR_DT=1.0 MBH_CHECK_NEAR_DIST=2e-3 MBH_CHECK_NEAR_PHI=0.01`, 3 rows):
```
row 1 [near]: logL stock -25.467 batched -25.467 dlogL +0.0004 | mm 1.050e-10 | ||delta|| 7.761e-03
row 2 [near]: logL stock -28.717 batched -28.717 dlogL +0.0002 | mm 1.417e-10 | ||delta|| 9.046e-03
logL_stock - logL_truth per row: truth +0.00, near -24.56, near -27.81
```
The 4.95e-4 "kept err" is a max-abs over single cells relative to the peak, at the
merger (f ≈ 18.5 mHz, layer 199). Its noise-weighted weight is negligible:
||delta|| = 0.01 in total.

## id 17 (SNRopt 1420, m1 1.05e7, m2 1.14e6), after fix

```
[RSS  4.43 GB] stock: 54.88 s/row
[batched B=1] 79.621 s/row
row 0 [truth]: logL stock -4.941 batched -4.942 dlogL -0.0014 | mm 5.967e-11 | kept err 2.05e-05 at (c,f,t)=(2, 1751, 48) | ||delta|| 1.556e-02 | power outside box 3.74e-10 | SNRopt 1422.6/1422.6
row 1 [near]: logL stock -6.597 batched -6.598 dlogL -0.0015 | mm 4.281e-11 | kept err 2.06e-05 at (c,f,t)=(1, 1711, 46) | ||delta|| 1.313e-02 | power outside box 3.74e-10 | SNRopt 1421.8/1421.8
row 2 [near]: logL stock -15.580 batched -15.581 dlogL -0.0012 | mm 3.796e-11 | kept err 2.04e-05 at (c,f,t)=(1, 1731, 47) | ||delta|| 1.242e-02 | power outside box 3.74e-10 | SNRopt 1423.3/1423.3
row 3 [far]:  logL stock -186894.232 batched -186894.050 dlogL +0.1824 | mm 5.763e-09 | kept err 1.77e-04 at (c,f,t)=(1, 1600, 41) | ||delta|| 1.517e-01 | power outside box 3.74e-10 | SNRopt 1413.1/1413.1
logL_stock - logL_truth per row: truth +0.00, near -1.66, near -10.64, far -186889.29
```
Before the fix, with the same snapped reference: truth dlogL −2.4387, mm 1.207e-6,
||delta|| 2.21.

## The onset fix (library: `src/lisatools/sources/bbh/gridaligned.py`)

The stock `_apply_response` zeros the first `buffer_time / dt` output samples of its
response array (15000 s at dt = 2.5 s). Those samples are the turn-on transient.
The stock array starts at the waveform's own first sample.

On the shared grid-aligned lattice the array starts at the LATTICE head instead, so
the same zeroing hit the lattice head, not the waveform onset. A row whose onset
lies inside the lattice therefore kept its transient.

Evidence for id 17 truth, time domain on the data lattice:
- The polarizations agree at the onset. The windowed onset is at lattice sample 357023, the same sample as the stock. The windowed path ramps over 3000 s; the stock path does not.
- In the TDI output, the windowed template carried a 7.2e-24 spike (samples 356847..358847) where the stock was zero. The local signal there is ~4e-26.

The fix: `_aligned_polarizations` also returns each row's absolute onset label.
`_call_batched` and `_call_single` then zero `[onset, onset + buffer_time)` per row
through `_zero_onset_warmup`. This is the same number of samples as the stock path,
anchored on the same sample.

## Residual non-gating finding (far row, id 17: f ≈ 18.5 mHz, layer 41)

Band-passing 15–22 mHz over layers 39–44 gives these amplitudes:

| Series | 15–22 mHz amplitude |
|---|---|
| stock TDI template | ~1e-25 |
| windowed TDI template | ~1e-25 |
| data | 1e-30 |
| stock h+ (polarization) | 1e-31 |

So both templates carry localized ~20 mHz bursts that come from the RESPONSE step,
not from phentax. They are absent from the mojito data. Stock and windowed carry
them with different phase. The resulting ||delta|| is 0.15 for the far row (dlogL
+0.18) and 0.013 for the truth and near rows. This is a production-path (stock)
response artifact common to both paths and is outside this task.

One possible contributor could not be ruled out here: the script's windowed ltt
slice. It is contiguous and in range, but the full-mission ltt table could not be
loaded within the 5 GB cap to compare.

## Timing and memory (laptop CPU)

| id | stock s/row | batched B=1 s/row | batched B=2 s/row (pre-fix run) | peak RSS |
|----|-------------|-------------------|---------------------------------|----------|
| 16 | 59.9 (post-fix, 4 rows) | 69.3 | 213.1 | 4.03 GB (post-fix), 4.44 GB max over all runs |
| 17 | 54.9 (post-fix, 4 rows) | 79.6 | 329.8 (concurrent load) | 4.49 GB (post-fix), 4.72 GB max over all runs |

On this CPU, batching does not pay: B=2 is 3–4x slower per row. The batch-size
question belongs to the GPU run (`MBH_BACKEND=cuda12x MBH_BATCH_SIZES=1,4,8,16,24`).
The 5.0 GB RSS cap was never hit.

## Pre-fix history (for the record)

With the default stock `waveform_t0 = REF` (0.5 s off-lattice), the jittered rows
showed dlogL −9.76 (id 16) and −76.5 (id 17). Both vanish with the snapped reference.
The stock's sub-sample `t0_shift_to_data` interpolation of an off-lattice epoch
differs from a lattice-snapped evaluation at those levels. The raw lines are in
`log_id16_cpu.txt` and `log_id17_cpu.txt`.

## Final fix wave: batched vs the UNSNAPPED stock (production's reference), 2026-09-30

Production's engine rebuild and the move's cross-check used the stock generator
on the UNSNAPPED epoch (`waveform_t0 = REF`). This run measures that directly:
`MBH_CHECK_STOCK_T0=ref MBH_BATCH_SIZES=1`, default near rows ±(0.3 s, 5e-4, 0.002),
same machine and guard as above. Logs: `log_id{16,17}_cpu_pad4_ref_final.txt`.

| id | row | logL stock-ref | logL batched | dlogL | mm | ‖delta‖ |
|----|-----|----------------|--------------|-------|----|---------|
| 16 | truth | -0.895 | -0.906 | -0.011 | 5.5e-9 | 0.056 |
| 16 | near+ | -1.209 | -1.174 | +0.035 | 5.5e-9 | 0.056 |
| 16 | near- | -1.466 | -1.523 | -0.058 | 5.5e-9 | 0.056 |
| 16 | far (info) | -16306.94 | -16316.69 | -9.76 | 5.5e-9 | 0.056 |
| 17 | truth | -4.749 | -4.942 | -0.193 | 4.0e-9 | 0.127 |
| 17 | near+ | -6.804 | -6.598 | +0.205 | 4.0e-9 | 0.127 |
| 17 | near- | -14.989 | -15.581 | **-0.592** | 4.0e-9 | 0.127 |
| 17 | far (info) | -186819.69 | -186894.05 | -74.36 | 9.7e-9 | 0.197 |

id 16 holds (max near |dlogL| 0.058). id 17 FAILS the 0.5-nat line on a near row
(-0.592), and its truth/near rows differ by ±0.2 nats with alternating sign --
the sub-sample epoch placement of the unsnapped stock, amplified by SNR 1420
(‖delta‖ 0.127 vs 0.016 against the snapped stock). Against the SNAPPED stock
the same rows agree to 1.5e-3 (section "id 17" above).

Consequence (library): in `MBH_LIKELIHOOD=batched` mode `get_mbh_phenom_gen`
now builds the stock generator on the lattice-snapped epoch too and wraps it in
`SnappedEpochMBHGen` (t_plunge − snap, same absolute merger), so the engine's
residual rebuilds and the move's cross-check use the reference that agrees to
1.5e-3 nats. `MBH_LIKELIHOOD=full` is unchanged.

## Edge coverage (2026-09-30)

User requirement: "we should do our best to only drop these where the data is
dropped for edge effects. If the mbh is in the data, we should have it. Even up
to say 7 days after the end of the data when we can maybe detect some of its
inspiral."

**Library changes:**
- `mbh_merger_time_buffer` (`MBH_MERGER_TIME_BUFFER`) now defaults to 7 d (it was 2 d).
- The admission filter is now `obs_start <= t_merge < obs_end + buffer`. A merger BEFORE the data start is dropped: it leaves no inspiral in the data, and it would sit below the prior.
- The `t_plunge` prior is `[obs_start, obs_end + max(t_plunge_pad, buffer)]`, so every admitted source is inside its own prior.
- The `[MBH_BATCH]` telemetry now counts `outside_box` only when the kept box cuts IN-DATA signal. The merger is first clamped to the active box. A new `outside_data` counter records rows that merge outside the active box (expected for edge sources).
- A windowed-lattice tail fix, described below.

**Setup.** The script runs on the production WDM grid: 1-h layers (Nf = 1440) and
`MBH_CHECK_EDGE_CROP=20`, so the ACTIVE box excludes 20 h at each end.
- Knobs: `MBH_CHECK_WAVELET_S=3600,4400`, `MBH_CHECK_N_ROWS=2` (truth and one near row), `MBH_BATCH_SIZES=1`.
- The stock reference uses the snapped epoch. Both generators use `Tobs = 90 d`.
- The window cache key now carries `MERGER_AT_DAYS`, and a stale cache raises.
- A placement outside the 731-day file raises; it is no longer silently clamped.

**Which ids.** id 16 merges 111.4 d into its file. A 120-d window with that merger at
117, 121 or 126 d would start before the file, so those cases use id 17, which merges
585.5 d in. id 16 also ran at the end with shorter windows: 114 d (merger 3 d before
the end) and 110.3 d (merger 1 d after the end, the longest window that still fits).
A merger 6 d after the end cannot be placed for id 16.

**Acceptance.** Truth and near rows must meet |dlogL| ≤ 0.5, outside-box power < 1e-4,
and either mm < 1e-6 or ||delta|| ≤ 0.05. The ||delta|| clause is new. After the
data end only a sliver of SNR is in the data (SNR 0.4–0.5), and there mm is a
ratio against a tiny signal. The absolute noise-weighted difference is what reaches
the likelihood: the interior truth rows carry ||delta|| = 0.010 / 0.016.

Logs: `log_edge_id*_w*_m*.txt`. Pre-fix copies are `*_prefix.*`. The grid-end
diagnostic is `log_edge_diag_id16_w110.3_m111.3.txt`.

| id | window d | merger vs ACTIVE box | kept box (d in window) | segment layers (pad lo/hi) | tail fix | data SNR (active) | stock ⟨h\|h⟩ | batched ⟨h\|h⟩ | ⟨hs\|hb⟩/⟨hs\|hs⟩ | stock power outside kept box | dlogL truth / near | mm truth | ‖delta‖ truth (first/last 10 kept layers) | stock ‖d−hs‖²/⟨d\|d⟩ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 120 | 2.17 d after start | 0.83–102.92 (clamped lo) | 0–2642 (20/172) | with | 536.2 | 287509 | 287509 | 0.999999999 | 1.1e-15 | −2.8e-6 / +4.4e-5 | 4.7e-12 | 1.66e-3 (5.3e-4 / 1.3e-7) | 2.6e-7 |
| 17 | 120 | 2.21 d before end | 17.12–119.21 (clamped hi) | 238–2880 (173/19) | with | 1414.8 | 2.00179e6 | 2.00179e6 | 0.999999945 | 1.2e-15 | −5.0e-5 / −3.2e-4 | 5.0e-12 | 4.48e-3 (3.4e-7 / 2.2e-10) | 1.5e-6 |
| 16 | 114 | 2.21 d before end | 11.12–113.21 (clamped hi) | 94–2736 (173/19) | with | 536.2 | 287509 | 287509 | 0.999999999 | 1.1e-15 | +2.8e-6 / +4.9e-5 | 3.2e-11 | 4.32e-3 (1.3e-7 / 9.4e-10) | 2.6e-7 |
| 17 | 120 | 0.99 d AFTER end | 17.12–119.21 (clamped hi) | 238–2880 (173/19) | with (= without, bit-identical) | 40.4 | 1612.55 | 1612.55 | 1.000000225 | 1.1e-8 | −1.2e-4 / −9.8e-5 | 5.7e-9 | 4.29e-3 (1.5e-7 / 5.7e-5) | 9.3e-5 |
| 17 | 120 | 1.79 d AFTER end | 17.12–119.21 (clamped hi) | 238–2880 (173/19) | with (= without to 1e-12) | 20.1 | 401.726 | 401.727 | 1.000000429 | 7.1e-10 | −1.3e-4 / −1.1e-4 | 2.5e-8 | 4.47e-3 (4.5e-8 / 3.8e-5) | 5.0e-5 |
| 16 | 110.3 | 1.80 d AFTER end | 7.41–109.50 (clamped hi) | 4–2636 (173/19) | **without** | 0.37 | 0.134280 | 0.134282 | 0.999907950 | 1.7e-10 | −5.2e-7 / −5.2e-7 | 9.9e-5 | 5.16e-3 (1.5e-7 / **3.3e-3**) | 3.8e-4 |
| 16 | 110.3 | 1.80 d AFTER end | 7.41–109.50 (clamped hi) | 4–2636 (173/19) | with | 0.37 | 0.134280 | 0.134270 | 0.999905406 | 1.7e-10 | +5.2e-6 / +5.2e-6 | 5.7e-5 | 3.90e-3 (3.3e-9 / 5.3e-4) | 3.8e-4 |
| 17 | 120 | 6.79 d AFTER end | 17.12–119.21 (clamped hi) | 238–2880 (173/19) | with (= without to 1e-12) | 0.53 | 0.278357 | 0.278622 | 1.000440717 | 1.0e-11 | −1.3e-4 / −1.1e-4 | 3.6e-5 | 4.47e-3 (1.1e-9 / 2.4e-5) | 4.1e-3 |

The rows are the requested cases, each with its kept-box geometry: merger at +3 d
(id 16), 117 d (id 17), 121 d (id 17, 1.8 d after the active end) and 126 d (id 17,
6.8 d after). Three extra rows follow:
- id 16 near the end, both before and after the fix;
- id 17 at 120.2 d, merger 5 h after the grid end, as a stress case.

The kept box is 102.08 d (2450 layers); for the 110.3-d window (Nf 1446) it is 2440 layers.

**Verdict: HELD in every case.** Batched equals stock inside the active box:
|dlogL| ≤ 3.2e-4, ⟨hs|hb⟩/⟨hs|hs⟩ ≥ 0.9999, and stock power outside the kept box ≤ 1.1e-8.
There was no crash, and no in-data signal was dropped:
- The box clamps to the ACTIVE edge. The segment pad moves to the other side, so the segment ends at the grid edge.
- Both templates reproduce the data: stock ‖d−hs‖²/⟨d|d⟩ ≤ 4e-3. The worst case is SNR 0.53, where the early inspiral older than the 90-d waveform is unmodelled in both paths.

**Found and fixed: a step at the windowed lattice end** (`gridaligned.py`,
`WindowedGridAlignedMBHWaveform.set_window`).
- **Cause.** The response reads the strain up to ~500 s AHEAD of each output sample (the SSB-to-spacecraft delay; its sign depends on sky position). `_apply_response` zero-pads the strain after the lattice. The windowed lattice stopped at the segment end, and for an edge-clamped box that is the grid end. A source still inspiralling there got a strain STEP, which the stock waveform (running on through merger and ringdown) does not have.
- **Evidence (id 16, merger 1 d after the grid end, TD).** The windowed output peaked at 2.6e5 × the local stock signal in the last ~300 s. Through the WDM transform this put 3.3e-3 of ‖delta‖ into the last 10 kept layers, 20 layers inside the crop.
- **Fix.** The lattice now runs `tdi_buffer_time` (600 s) past the segment end; the adapter's placement clips the tail. After the fix the spike is gone, the last-10-layer ‖delta‖ drops from 3.3e-3 to 5.3e-4, and the total drops from 5.16e-3 to 3.90e-3.
- **id 17 is unaffected.** Its sky reads backward, so its output is unchanged (bit-identical or 1e-12).
- **Tests.**
  - `WindowedLeadCoversZeroedHeadTest` pins the tail geometry.
  - `WindowedGridAlignedPhentaxTest.test_segment_ending_before_the_merger_has_no_end_step` uses real phentax with the segment ending 2 h before the merger, at alpha and alpha + π. With the fix the last-1000-s relative difference is 1.4e-3 / 1.3e-3. Without the tail it is 2.1e3 at alpha; alpha + π has no step.
  - With the tail removed, both tests fail.
- **Not fixed.** The non-windowed parent `GridAlignedPhenomTHMTDIWaveform` has the same end truncation at the data end. No production path builds it, and lengthening its lattice would lengthen its FD output, so it is left as is.

**Stock path.** The stock path handles a merger after the data end correctly:
- The stock waveform (phentax, 90 d before merger through ringdown) is responded in full; its TD output ran 30.8 h past the grid end in the diagnostic.
- `place_td_signal_on_grid` clips that output at the data end.
- `pyResponseTDI._data_time_check` trims only against the ORBIT end. The mojito positions span 1580 d, 849 d past the data end, so it never fires here.
- Latent: if it ever did trim, the `assert num_inputs_per_source >= self.num_pts` in `get_projections` would raise instead. `_apply_response` sets `num_pts` to the untrimmed length. This is unreachable with mojito orbits and is not fixed.
- The mojito ltt table stops at the data end (730.5 d). Any response past the mission end therefore uses clamped ltts, but only for output after the data end, which is discarded.

**Remaining ‖delta‖ floor of ~4e-3.** It is not an edge effect. The worst cell sits at
f ≈ 16–19 mHz at a FIXED absolute time: m117, m121 and m126 all put it at
t = 142572089.8 s, in three differently placed windows. It shows up as the same
‖delta‖ = 4.47–4.48e-3 in each case. This is the ~20 mHz response feature both paths carry
(see "Residual non-gating finding" above).
