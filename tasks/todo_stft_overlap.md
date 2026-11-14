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
- [ ] Commit + push branch

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
