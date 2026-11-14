# Task: overlapping STFT segments — Round 1 (Python grid, numerical STFT, normalisation, noise term, guards)

Plan: `~/.claude/plans/nifty-weaving-storm.md` (approved 2026-11-14). Branch `feat/stft-overlap`.
Hann window at hop = T_seg / overlap (overlap = 3), `dt` = hop, `overlap = 1` bit-identical to today.

## Checklist
- [x] Baseline: existing STFT-related tests on the unchanged branch (per file, worktree shim)
- [x] `STFTSettings`: `overlap` kwarg + validation; `seg_duration`, `window_power`, `redundancy`,
      `segment_window`, `check_segment_window`, `get_nhop`; `get_nperseg` = segment samples;
      `differential_component`; `kwargs`, `__eq__`, `__repr__`, `get_slice`, `compute_slice_indices`
- [x] `DomainSettingsBase.redundancy` default 1.0
- [x] `get_stft_settings` / `make_factory`: `overlap`, NT = floor((N − nperseg)/nhop) + 1
- [x] `TDSignal.stft`: overlap > 1 frame gather (`_stft_overlapping`), default Hann window, window checks
- [x] `noise_likelihood_term` × redundancy
- [x] Guards on overlap > 1: STFTComputationGroup, batched PSD likelihood (`_log_like_batch`: C++,
      numba and window-convolution paths), `_apply_stft_add` (templates on another grid — covers
      waveformbase, which only builds contiguous grids), stft_band_structure, GWTF
- [x] `tests/test_stft_overlap.py` (19 tests, one GPU)
- [x] Regression: same tests as baseline, identical results
- [x] Commit + push branch (c248fad4)
- [x] Follow-up: name the overlapping window "Tukey alpha = 1 (Hann)" in docs and messages

## Review
- New tests: 19 passed (CPU + GPU 5). Contiguous `TDSignal.stft` is bit-equal to the pre-change
  reshape + rfft; overlapping frames are exact; white-noise Parseval with `differential_component`
  (df / (9/8)) matches dt Σx² to 1e-12; noise term at overlap 3 is exactly 1/3; GPU = CPU to 1e-12.
- Regression (9 files, worktree shim, GPU 5 visible): the per-file pass/fail/skip counts and the
  failure set are identical before and after. 24 failures predate the change: CUDA-class tests of
  stft_cross_terms / noise_layout / store_windows / index_width, sensitivity (CuPy → NumPy implicit
  conversion with a GPU visible), domain_backend_threading (backend-name strings).
  test_stft_multi_gpu skips (one GPU visible).
- Design notes: `overlap` travels in `kwargs` only when ≠ 1, so contiguous grids keep their recorded
  kwargs/hdf attrs; the class-level default `overlap = 1` covers settings objects pickled before.
- Round 2 lifts the guards (C++ geometry, PSD-kernel redundancy, window ownership).

---

# Round 2a — C++ STFT grid geometry for the GB likelihood path (2026-11-14)

The C++ `dt` split into the hop (`dt`, unchanged), the window length (`seg_duration`) and the
inner-product weight (`diff_comp`). Plan: `~/.claude/plans/nifty-weaving-storm.md`.

## Checklist
- [x] `STFTSettings` (+ `STFTDomain`, `STFTFresnel`): trailing `seg_duration`, `diff_comp`
      (negative = contiguous value); `taper_duration` from the resolved `seg_duration`
- [x] Bindings: `seg_duration` / `diff_comp` kwargs on `STFTDomainWrap`, `seg_duration` on
      `STFTFresnelWrap`; read-only properties for the tests
- [x] `STFTFresnel` window `[t0, t0 + seg_duration]`, midpoint anchor, right-ramp rotation
- [x] `lat_stft_kernels.hh`: midpoint anchor + slope clamp from `fresnel->seg_duration` (tile, Fisher)
- [x] DFFR: `segment_duration = seg_duration`; table spacing checked against the hop (argument,
      not a device-struct field); the grid-match check includes `seg_duration`
- [x] ABI 1 → 2 (`STFTSettings` listed)
- [x] `STFTComputationGroup`: kwargs to both wraps (by keyword), kinematics at t0 + seg/2 step dt,
      guard lifted, `window_alpha == 1` required on overlap > 1
- [x] Audit: every inner-product weight goes through `diff_comp` (domains.cu, stft kernels, DFFR,
      cross terms); bare `df` only for bin geometry
- [x] Scratch builds (LAT cpu+cuda12x, GBGPU) for base and new
- [x] Contiguous bit-identity on CPU: base vs new, both responses (templates, ll, swap, grad, Fisher)
- [ ] Same on GPU (GPU 5 was full on 2026-11-14)
- [x] Oracle `scripts/dffr/stft_overlap_oracle.py` on CPU
- [ ] Oracle on GPU
- [x] `tests/test_stft_overlap.py` + regression suite, CPU-only (failure sets vs baseline)
- [x] GBGPU STFT/DFFR suites on the new build, CPU-only
- [ ] LAT + GBGPU suites with the GPU visible
- [ ] Commit + push; hand over the venv reinstall commands (LAT + GBGPU together)

## Review (CPU, 2026-11-14)
- Contiguous grid: all 20 outputs (templates, d_h, h_h, 5 swap terms, gradient, Fisher; both responses,
  production Tukey alpha, real STFTComputationGroup) bit-identical between the base and new scratch builds.
- Oracle (overlap 3, 24 production sources, 8 days): templates equal the offset grids bit for bit for
  both responses; d_h, h_h, swap, Fisher = 8/9 sum over offset grids to <= 1.8e-14; gradient to 1.4e-6
  (central-difference rounding floor, tolerance 1e-5).
- LAT suite (12 files) and GBGPU STFT suites (4 files): same results and failure sets as base (9
  pre-existing test_sensitivity failures, CuPy-free run).
- Harness lesson: KinematicsTableWrap keeps raw pointers; a shim must keep the field arrays alive
  (the Phase 0 script did not, and only got away with it).
