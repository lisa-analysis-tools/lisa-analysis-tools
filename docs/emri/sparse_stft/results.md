# Sparse-track STFT EMRI templates: results

Measurements behind [`README.md`](README.md), taken 2026-11-16 to 2026-11-23 on the laptop CPU
(Apple silicon, 24 GB, single thread for the Python parts), branch `feat-emri-sparse-stft`. No GPU
number exists yet.

## Method

Common settings unless stated: data cadence 10 s, STFT segments of 6 h (2160 samples, bins of
46.3 uHz), symmetric Tukey window 0.5, band 1e-4 - 3e-2 Hz (646 bins), TDI 2nd generation, equal-arm
orbits, waveform starting 1 day before the data (production zeroes its first 10000 s after
`waveform_t0`), FEW `mode_selection_threshold` 1e-3, sky `(ra, dec) = (1.1, 0.4)`, `qK = 0.5`,
`phiK = 2.0`, `Phi = (0.3, 0, 1.2)`, `x0 = 1`, `dist = 1 Gpc`.

| Preset | m1 | m2 | a | p0 | e0 | Data | Plunge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non-plunging | 1e6 | 10 | 0.9 | 9.5 | 0.3 | 8 d | none |
| plunging | 1e6 | 100 | 0.9 | 7.0 | 0.4 | 36 d | data day 33.35 (segment 133) |
| short plunger (tests) | 1e6 | 400 | 0.9 | 6.0 | 0.4 | 5 d | 4.62 d after the waveform start |

"Production" is `EMRITDIWaveform.__call__` on the same settings (FEW dense h+, hx ->
`pyResponseTDI` -> window + rfft). Mismatch is `1 - <a|b> / sqrt(<a|a><b|b>)` over all segments
and bins: **white** over X, Y, Z; **noise-weighted** over A, E, T with the analytic TDI-2 PSDs and
the STFT pixel variance `(S/2) T mean(w^2)`. Timings are the best of 2-3 calls after a warm-up.
Comparisons against production use `scripts/emri/compare_sparse_stft_vs_dense.py`.

## Against production

### No plunge (8 d)

| | |
| --- | --- |
| Mismatch (white) | **2.14e-8**; per segment at most 4.0e-8 |
| Amplitude ratio `|h_sparse| / |h_prod|` | 1.00000017 |
| Fill | 32 of 32 segments on tracks |
| Time (CPU) | sparse 0.12 s, production 0.34 s |

With the periodic window the accumulator used before `bce0201a`, the same comparison gave 4.9e-7.

### With a plunge (36 d): the two dense fallbacks

`cubic_tol` 0.03: 10 segments zero (after the plunge), 109 on tracks, 25 dense: segments
109 - 133 (the last 6 days before the plunge). Before the response nulls were fixed
(2026-11-23, section "Response nulls in the tracks") six more were dense, 96 - 101, with
mismatches 1.7e-6 / 7.6e-6 and `SNR^2 x mismatch` 7.3: those segments now carry the linear-chirp
error of a track segment. The times below were measured before that fix.

| | TDI-on-the-fly fallback (default) | production-response fallback |
| --- | --- | --- |
| Mismatch white / noise-weighted | 5.1e-5 / 4.5e-5 | **2.1e-6 / 9.1e-6** |
| Over the 25 dense segments (white) | 5.8e-5 (worst 5.2e-4, segment 133, the plunge) | **1e-16** |
| Over the 109 track segments (white / noise-weighted) | 1.2e-5 / 1.5e-5 | 1.2e-5 / 1.5e-5 |
| Worst segment (white) | 5.2e-4 at 133 | 3.0e-5 at 108 (a track segment) |
| Share of `<h|h>` in dense segments (white / noise-weighted) | 0.84 / 0.41 | 0.84 / 0.41 |
| Time (CPU; production 1.5 s) | 6.2 s | **1.0 s** |

SNR of the source about 980; with the production-response fallback `SNR^2 x mismatch` = 8.7,
the order of the log-likelihood error at the true parameters.

### `cubic_tol` with the production-response fallback (36 d)

| `cubic_tol` | tracks / dense segments | Time (production 1.55 s) | Mismatch white / noise-weighted | Over track segments (noise-weighted) | `SNR^2 x mismatch` |
| --- | --- | --- | --- | --- | --- |
| 0.03 | 109 / 25 | 1.02 s (with 103 / 31) | 2.1e-6 / 9.1e-6 | 1.5e-5 | 8.7 |
| 0.01 | 89 / 45 | 1.08 s | 4.9e-7 / 3.6e-6 | 8.8e-6 | 3.5 |
| 0.003 | 53 / 81 | 1.27 s | 9.5e-8 / 1.0e-6 | 5.3e-6 | 1.0 |

The remaining error is in the track segments; the dense ones are exact.

### Short plungers, every segment dense

| Source | Production | TDI-on-the-fly fallback | Production-response fallback |
| --- | --- | --- | --- |
| m2 400, p0 6.0, 5 d (15 dense, 5 zero) | 0.36 s | 1.43 s, mismatch 1.9e-4 (5.8e-4 in the plunge segment) | 0.31 s, mismatch 4e-16 |
| m2 300, p0 6.0, 7 d (21 dense, 7 zero) | 0.41 s | 1.96 s, mismatch 2.2e-3 (6.4e-3 in the plunge segment) | 0.34 s, mismatch 6e-16 |

## Against an exact dense reference (prototype stage)

Before the drop-in class, tracks were compared with the exact dense series of the same kernel
(not production), with the periodic window (`scripts/emri/sparse_stft_fresnel_prototype.py`):

| Source | Segments | Mismatch Tukey 0.1 | Mismatch Tukey 0.5 |
| --- | --- | --- | --- |
| Non-plunging 24 d, 40 harmonics | 96 on tracks | 2.0e-6 | 9e-9 |
| Plunging 36 d, 60 harmonics, `cubic_tol` 0.1 / 0.03 / 0.01 | 12 / 29 / 41 dense of 130 | 2.3e-5 / 3.6e-6 / 1.0e-6 | 1.1e-5 / 1.7e-6 / 4.8e-7 |

Costs then: FEW holder + inputs 34 ms, sparse response 4 ms (warm), accumulation about 220 ms.

## The symmetric window

The periodic window (ending at the first sample of the next segment) differs from the data's
symmetric Tukey window by about 0.5/N in amplitude: 2.3e-4 at N = 2160, 2.9e-5 at N = 17280. With
`window_samples = N` the offset is 2.5e-7 at N = 2160 (Tukey 0.5) and 1.9e-10 at N = 17280. The
2e-4 amplitude bias of the first prototype was this, not the kernel.

## The production-response fallback, sample by sample

`_fallback_channels` against production `compute_tdi_channels` on the data grid (max |difference|
/ max |production|; zero patterns compared sample by sample):

| Case | First segment | Data end | Gaps / shuffled | Whole data | Zeros |
| --- | --- | --- | --- | --- | --- |
| Non-plunging 6 d | 9.7e-12 | 7.3e-11 | 7.3e-11 / 5.6e-11 | 7.5e-11 | equal |
| Non-plunging 8 d | 9.6e-12 | 7.4e-11 | 7.4e-11 / 5.9e-11 | 7.5e-11 | equal |
| Short plunger 5 d | 3.1e-13 | 0 (after the plunge) | 6.7e-13 / 1.8e-12 | 1.8e-12 | equal (11845) |

Cost: about 10 ms per 6 h segment (4 us per sample, margins included), 0.2-0.3 s for 6-8 days.

### Edge cases found by the review, before and after the fix

| Case | Before (old sample counts) | After |
| --- | --- | --- |
| FEW `pad_output=False`: production stops at the plunge, the window added a response tail after it | 0.88 | **1.8e-12** |
| A sample lost in the years round trip (lead 8010 samples, 2 d): production one sample short at the data end | 0.12 | **2.6e-11** |
| Sub-sample shift (`data_t0 - waveform_t0 = 1996.3 s`) and a window at the waveform start (`k0 = 0`), the windowed response fed with production's own h | | **0 / 2.3e-13 / 4.1e-13** (start / middle / end) |
| Same, end to end | 3.0e-8 | 3.0e-8: h itself (production's 2-d trajectory is cut at T after 4 knots, the holder's runs on; different last DOPR853 step) |

`_production_samples` reproduces FEW's trajectory end bit for bit: 604790.0 (6 d), 777589.9999999999
(8 d), 2678389.9999999995 (30 d); in the last two FEW's end falls one ulp below T and the sample at
T is not summed. FEW's sample count `int(T YRSID_SI / dt)` loses one sample for 2.5% of the
lengths between 5e4 and 2.5e5 at dt = 10 s.

### Mode sum and windowed response, separately (agents of the first session)

* `polarizations_from_item` against production `wave_gen`: 1e-11 (non-plunging and plunging,
  through the plunge), 0.33 us per sample for 48 harmonics.
* Same, production cut at T at day 20 / 33 of the plunging source: 1.1e-6 / 2.4e-5 relative, equal
  to a literal transcription of the C++ to 5e-12: the holder's trajectory runs past T, so the last
  DOPR853 steps differ (integrator tolerance).
* `response_on_window` fed production's h: 2.4e-13 (segment 0) to 2.8e-11 (segment 100).

## CPU profile (180 d, no plunge, one thread)

| `mode_selection_threshold` | Components | FEW holder + inputs | Sparse response + tracks | Fresnel fill | Full-grid `vdot` |
| --- | --- | --- | --- | --- | --- |
| 1e-3 | 42 | 31 ms | 33 ms | **2242 ms** | 1 ms |
| 1e-5 | 148 | 32 ms | 123 ms | **7822 ms** | 1 ms |

720 segments x 646 bins x 3 channels = 1.4e6 pixels. Mean band 63 bins; the 42 components touch
1.9e6 (component, segment, bin) cells per channel, 5.7e6 Fresnel evaluations, about four times the
grid per channel: the harmonic bands overlap heavily, and `<h|h>` carries the cross terms.

## The heterodyne study (2026-11-20)

Question: heterodyne each harmonic against the same harmonic of a densely computed reference, to
drop the dense end steps of fast-chirping signals and to speed up the likelihood.

Fastest sweep in the plunging preset and the plunge-time offset a reference heterodyne would
tolerate (the proposal's harmonic within 0.1 bin of the reference's, `Delta f = fdot Delta t_p`):

| Hours before the plunge | Fastest sweep [bins / 6 h] | Tolerated `Delta t_p` |
| --- | --- | --- |
| 198 | 4.2 | 510 s |
| 150 | 5.6 | 390 s |
| 102 | 8.0 | 270 s |
| 54 | 13.9 | 160 s |
| 30 | 23.2 | 93 s |
| 18 | 33.3 | 65 s |
| 6 | 79.4 | 27 s |

The tracks summed densely against the exact channels, in the dense segments of the plunging
preset (relative windowed L2 error; mismatch at most err^2/2). Measured before the response
nulls were fixed (section "Response nulls in the tracks" below): every segment is within 7e-5
now.

| Segment | Fill | Error |
| --- | --- | --- |
| 100 | dense | 9.2e-4 |
| 103 | tracks | 1.4e-4 |
| 106 | tracks | 2.4e-6 |
| 109 - 124 | dense | 5.8e-6 - 2.4e-5 |
| 127 / 130 / 131 / 132 | dense | 1.7e-4 / 9.4e-4 / 4.9e-4 / 1.4e-3 |
| 133 (plunge) | dense | not covered by tracks |

Conclusions:

* **Waveform heterodyne of the dense end steps: not worth it.** Those segments are exact with the
  production-response fallback, and cheap (`cubic_tol` table above); a reference heterodyne holds
  only within the tolerances above and can only match or lose against exact; sampling the tracks
  instead would leave up to 1e-6 mismatch in the last 1.5 days.
* **The tracks are more accurate than the Fresnel model built on them** (6e-6 - 2e-5 against
  about 1e-5 mismatch in Fresnel segments): the accuracy lever is the per-segment model, not the
  dense steps.
* **The speed lever is the Fresnel fill and the inner products**, on the CPU. An inner-product
  heterodyne (per (harmonic, channel, segment) two precomputed sums of data times reference
  basis, `<h|h>` from data-free sums including harmonic pairs, binned in time) would cut a
  likelihood to about 1e4 - 1e5 multiply-adds plus FEW and the sparse response, but its data sums
  must be rebuilt whenever a walker's residual changes, and it holds only near the reference. To
  be decided with the GPU profile.

## Response nulls in the tracks (2026-11-23)

The track errors of the table above (9e-4 at segment 100, up to 1.4e-3 in the last 1.5 days)
were nulls of the TDI transfer function. As a harmonic sweeps through one, the channel's complex
amplitude goes through zero: a sign change of the amplitude, or equally a step of pi in the
phase. The kernel picks between the two from three samples of the modulus, with second
differences that assume even spacing (`LISATDIonTheFly::new_extract_amplitude_and_phase`). On
the response grid of the 36-d plunging preset (118 points, neighbouring intervals differing by
up to a factor 435) it flags 82 of 180 local minima and misses 62 crossings: 20 only because of
the even-spacing formula, 24 on the 10% linearity tolerance, 12 not at a strict minimum, 6
flagged with a step of pi left. Its output is exact at the response times (1e-11); between them
the splines cannot follow a step of pi.

`signed_amplitude_through_nulls` (`sparse_tracks.py`, in `SplinedTDIOutput.__init__`) reduces
every step of the kernel's `tdi_phase` by its nearest multiple of pi and moves an odd multiple
into the sign of the amplitude. `tdi_phase` is the phase to decide on: the kernel's reference
phase is the carrier at the retarded time of spacecraft 1, so it moves by less than 0.1 rad per
response step for a harmonic above 5% of its peak amplitude (the residual splined against the
carrier without that delay moves by up to 3.4 rad per step near the plunge). The steps further
than 0.5 rad from a multiple of pi (42 of 28080) all lie in the last 40 minutes of the
trajectory, on harmonics at 1e-5 to 7e-2 of their peak.

Track signal against the exact per-harmonic series, relative error per segment, worst of the 133
segments the tracks cover (reference and two proposals one step away in `a` and `p0`):

| Orbits, response grid | Before | After | Segments above 1e-4 before / after |
| --- | --- | --- | --- |
| equal arms, 8 h | 1.7e-3 - 1.8e-3 | 5.9e-5 - 6.9e-5 | 15 - 22 / 0 |
| ESA, 8 h | 1.8e-3 - 1.9e-3 | 6.1e-5 - 7.1e-5 | 15 - 24 / 0 |
| ESA, 1 h | 7.5e-4 - 9.4e-4 | 2.2e-5 - 3.1e-5 | 3 / 0 |
| equal arms, 1 h | 6.9e-4 - 9.9e-4 | 1.3e-4 - 1.6e-4 | 2 / 1 (segment 131) |

The 8-d preset has no null and is unchanged. Not explained: the one segment left at 1.5e-4 with
equal arms on the 1 h grid. Rejected: unwrapping the splined residual with the median slope of
the neighbouring intervals (4e-3 at segment 130 with ESA orbits).

Effect on the template (36 d, `cubic_tol` 0.03, production-response fallback): segments 96 - 101
were filled from the exact series only because the glitched `fdot` tripped the cubic-phase
check; they are track segments now (103 / 31 -> 109 / 25 tracks / dense). Where both templates
use tracks they agree to a mismatch of 1.1e-10. A proposal one step away in `p0` flipped the
fill of 4 segments before, of none now.

Scripts and logs: `scripts/emri/het_spike_out/` (`diag_null_detection.py`,
`diag_tdi_phase_steps.py`, `diag_unwrap_fix.py`, `diag_fix_effect_on_template.py`).
Test: `tests/test_emri_sparse_response.py::NullCrossingSparseResponseTest` (5.6e-2 before, 4e-5
after, relative to the largest amplitude).

## Memory

One FEW `GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux")` holds about 5.5 GB while alive
(two at once: 11.2 GB); building the first `EMRITDIWaveform` peaks at 8.3 GB. Three FEW-heavy
processes at once were killed (exit 137), and so was the waveform test file while its classes
kept their waveforms; with the waveforms released between classes the full test set peaks at
9.3 GB.

## Tests

`tests/test_stft_track_sum.py tests/*emri*.py tests/*tdi*.py tests/test_orbits_config_cache.py
tests/test_fresnel.py tests/test_gb_likelihood_engine.py`: **98 passed, 27 skipped, 95 subtests**
in 241 s. Every skip is GPU-only ("no (usable) GPU backend", "requires a usable CUDA backend",
"requires gbgpu + cupy"). The fallback tests (`tests/test_emri_sparse_stft_waveform.py`):
`PolarizationFallbackTest` (5-d short plunger with `cubic_tol` 0.3: 7 track, 8 dense, 5 zero
segments), `ProductionEdgeCasesTest` (unpadded production, the lost sample, the sub-sample shift),
and the cut-at-T case in `SparseSTFTWaveformTest`; each asserts `_production_samples` against
FEW's `num_pts` / `num_pts_pad` and production's last nonzero sample.

## Reproduce

Environment (the worktree has its own venv; LAT editable from the worktree, GBGPU non-editable
from its `l2d-dev` checkout, FEW `gpu_backend` from `erebor_org_setup/wt-few-gpu-backend`):

```sh
cd erebor_org_setup/wt-emri-sparse-stft
export FEW_FILE_ALLOW_DOWNLOAD=0 FEW_FILE_EXTRA_PATHS="$HOME/Library/Application Support/few/v2.1.0/download"
# after C++ edits: rebuild LAT, then GBGPU (uninstall first or uv reinstalls a cached build)
uv pip uninstall lisaanalysistools gbgpu --python .venv/bin/python
VIRTUAL_ENV=$PWD/.venv PKG_CONFIG_PATH=/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig \
  uv pip install -e $PWD --no-build-isolation --no-deps --no-cache
```

Comparisons against production (one at a time: each holds two FEW generators, about 11 GB):

```sh
MPLBACKEND=Agg .venv/bin/python scripts/emri/compare_sparse_stft_vs_dense.py --preset nonplunging --psd --no-show
MPLBACKEND=Agg .venv/bin/python scripts/emri/compare_sparse_stft_vs_dense.py --preset plunging --psd --no-show \
    --fallback-via-polarizations --cubic-tol 0.003
```

The script prints the mismatches (white and, with `--psd`, noise-weighted), the fill of every
segment and the timings, and saves `spectrograms.png`, `segments.png` and `slice.png`.

Tests:

```sh
.venv/bin/python -m pytest tests/test_stft_track_sum.py tests/*emri*.py tests/*tdi*.py \
    tests/test_orbits_config_cache.py tests/test_fresnel.py tests/test_gb_likelihood_engine.py -q
```

The CPU profile, the heterodyne tables and the track-accuracy table came from short scripts
outside the repository: the profile times `template_inputs`, `EMRITrackSource(...)`,
`stft_templates_from_tracks` and a `vdot` separately; the sweep table evaluates `eval_tracks` at
the segment midpoints of the plunging preset; the track-accuracy table sums
`amp * cos(phase)` from `eval_tracks` at every sample of a segment and compares it with
`_fallback_channels` (production) after the window.
