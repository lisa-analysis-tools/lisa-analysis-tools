# Window convolution in `XYZSensitivityBackend`: changes and effects

Summary of the 2026-10-28 → 10-30 work on `compute_log_like(convolve_window=True)`.
Developed on `perf/window-conv-basis` (worktree `.worktrees/lat-wconv`, 5 commits on
`673152b2`), merged into `perf-window-convolution` and cleaned up there
(`sensitivity_window.py` → `convolution_helpers.py`).

## 1. The problem

On a 2-year dataset at 5 s cadence (N = 12.6M samples, ~5.7M active frequency bins),
`convolve_window=True` was ~7× slower than `convolve_window=False` and could run out of GPU
memory. Every likelihood call:

1. wrote all 6 covariance entries for every walker on the full grid (~450 MB per walker);
2. ran 12 length-N FFTs per walker (an irfft + rfft pair for each entry);
3. made several extra copies (`res`, `xp.real`, the f[0] guard, 6× `.flatten()`).

The memory pool also grew without bound. `fft_batch_size=512` put all walkers in one FFT
batch, and `min(fft_batch_size, n_rows)` meant every new walker count created a new cuFFT
plan with an N-sized workspace.

## 2. The idea

The noise covariance (`PSD.cu`, `get_noise_covariance_single`) has this structure:

```
C_k(f, t) = Soms² · T^oms_k(f,t) s_oms(f) 10^spl_oms(f)
          + Sa²   · T^tm_k(f,t)  s_tm(f)  10^spl_tm(f)
          + R_k(t) · G_w(f; Amp, α, f_1, kn, f_2)
(× noise_normalization, + alias copies of every term)
```

The window convolution is linear, and that gives an exact or approximate method per component:

| Component | Method | Exact? | Cost per call |
|---|---|---|---|
| Instrument noise (OMS, TM), incl. aliasing and filters | convolved bases `B_b` built once; `conv(C) = Σ_b coeff_b B_b`, with `coeff = (Soms², Sa²)` (2 bases, 12 if asymmetric) | **exact** | no FFTs |
| Galactic foreground | all 6 entries are `R_k(t)·G_w(f)`, so `conv = R_k · conv(G_w)`: one real row per walker, convolved on a short circle with a truncated kernel | **exact** (round-off) | 1 short real FFT pair per walker |
| Splines | `10^spl(f)` multiplies the transfer functions per walker and does not separate | — | `"exact"` mode: full legacy path (bounded memory); `"factorized"` mode: `conv(10^spl T) ≈ 10^spl conv(T)`, **approximate**, no FFTs |

A fused kernel then builds each bin's covariance on the fly from the bases, the
coefficients and the galaxy row, inverts the 3×3 matrix and accumulates the likelihood.
No per-walker covariance arrays are ever stored.

**Why it is faster than `convolve_window=False`:** that path recomputes the transfer
functions and noise shapes (several `sin`/`cos`/`pow` per bin) for every walker. The bases
cache them once, so the kernel only streams the precomputed values and the data, close to
the memory-bandwidth limit.

### Galaxy details (`get_galaxy_setup`, `galaxy_convolved_rows`)

- **Setup, once:**
  - the ratios `R_k/R_00`, and `rho(t)` when there are several time segments;
  - the legacy f[0]-guard decision;
  - kernel truncation `K` (tail mass ≤ 1e-17);
  - support `n_sup`: the whole band, or the bins below `galaxy_fmax`;
  - FFT length `M ≥ 2(lo + n_sup + K) + 1`, smooth and fixed.
- **Per call:**
  - rows from the C++ `get_galactic_row_wrap`, bitwise equal to the XX galaxy entry;
  - legacy edge padding reproduced exactly: the constant above the band is added as
    `c·Σkernel`, and the rest is convolved on the length-`M` circle;
  - walkers in fixed-size batches, so cuFFT plans are reused.
- **`galaxy_fmax` (optional):** each call checks a rigorous analytic tail bound per walker,
  including aliases. If any walker fails, it warns and falls back to the full band, so a
  `galaxy_fmax` that is too low costs speed, never accuracy.

## 3. What changed

### Python (`src/lisatools/sensitivity.py`, `src/lisatools/convolution_helpers.py`)

- **New `XYZSensitivityBackend` kwargs:**
  - `window_convolution_mode`: `"exact"` (default) | `"factorized"` | `"legacy"`;
  - `galaxy_fmax: Optional[float] = None`;
  - `psd_batch_size: Optional[int] = None`, walker chunking in `compute_log_like` (exact, PSDs are independent).
- **`fft_batch_size`:** default is now `"auto"`, sized from free GPU memory, power-of-two
  batches, few plans. **An integer is now an upper bound** (`min(bs, next_pow2(n_rows), budget)`, padded).
- **Dispatch:** `_fast_window_convolution`, `_window_matrix_elements` (used by
  `compute_sensitivity_matrix` / `set_sensitivity_matrix` / `__call__`), `_window_galaxy_rows`,
  `_window_spline_weights` and `_log_like_batch`.
- **Shared caches:** walker copies (`__deepcopy__`, `__call__`) share every `_wconv_*`
  cache by reference: bases 0.8 GB (symmetric) / 4.9 GB (asymmetric) at 2 yr.
- **Legacy path hardening:**
  - fixed-size FFT batches;
  - no `res` / `xp.real` copies;
  - a non-mutating f[0] guard;
  - `.ravel()` instead of `.flatten()`;
  - a shared zero spline buffer;
  - duplicate `fft_batch_size` property removed.
- **Bases memory:** they are built one at a time, so the one-off build peak stays around 1–2 GB.

### C++ (`cutils/PSD.cu`, `PSD.hpp`, `binding_detector.{cxx,hpp}`)

- **`XYZSensitivityMatrix::psd_likelihood_basis_wrap`:**
  - fused basis-combination likelihood, with optional galaxy rows and multiplicative spline weights (any strides);
  - CPU and GPU from one source;
  - 64-bit offsets;
  - explicit `num_data`.
- **`XYZSensitivityMatrix::get_galactic_row_wrap`:** per-walker galaxy rows. It shares a
  new `galaxy_shape_tdi` device helper with `get_noise_covariance_single`, so there is no
  duplicated physics.
- **`PSD_FORCE_INLINE`** on `invert_3x3_hermitian` / `quadratic_form` (CPU). Without it,
  GCC's inlining and FMA contraction changed and the legacy CPU kernel lost bitwise reproducibility.
- **Python fallbacks removed:** after the reinstall, the cupy / numpy copies of these
  kernels (and the Python copy of the galaxy formula) were deleted. A binary without the
  methods raises "rebuild lisatools".

### Bug fixes (existing code)

- **`psd_likelihood_xyz_kernel`:** `int base_idx = data_index*3*total_tf_pairs` overflowed
  int32 at 2 yr for `data_index ≥ 126`. The old binary crashed with an illegal memory
  access in a synthetic test. It is now 64-bit, as are the external-matrix and spline offsets.
- **`get_noise_covariance_kernel`:** 64-bit spline offset.
- **CPU binding length checks** (`get_noise_covariance_wrap`, `psd_likelihood_wrap` spline
  arrays, `get_noise_tfs_mosa_wrap`): the products were formed in `int`, so correctly sized
  arrays were rejected (expected length 1.8e19). They now use the `(N, multiplier)` split.
  `size_t` product.

## 4. Effects

### Speed and memory: 2 yr, dt = 5 s, A100, `compute_log_like`, 8 / 32 walkers

| Setup | Before (`convolve_window=True`) | After |
|---|---|---|
| no galaxy, symmetric | 1.1 s / 4.4 s | **4.2 ms / 17 ms** (exact) |
| galaxy, symmetric | 1.10 s / 4.44 s, +8.4 / +22.6 GB | **34 ms / 130 ms**, +5.8 GB (exact) |
| galaxy, asymmetric | 0.68 s / 2.75 s | **41 ms / 158 ms** (exact) |
| galaxy, `galaxy_fmax=0.03` | | **16 ms / 62 ms**, +2.3 GB (exact) |
| galaxy + splines | 0.58 s / 2.37 s, +22 GB | **38 ms / 142 ms** (factorized) |
| reference: `convolve_window=False` | 0.16 s / 0.64 s | |

- Legacy path alone (e.g. splines in exact mode) with `psd_batch_size=4`: peak memory flat
  at ~4 GB, whatever the walker count.
- With the galaxy on, most of the remaining time is the per-walker galaxy FFT
  (~29 of 34 ms at 8 walkers).

### Accuracy (vs the unmodified implementation, CPU and GPU)

Tested on 10 fixtures: galaxy on/off × splines on/off × symmetric/asymmetric, aliasing,
filters, a band including f = 0, and a narrow 1–1.5 mHz band.

- **Covariance matrices** (exact mode vs legacy): ≤ ~1e-15, band-normalised.
- **Log-likelihood:** |Δll| 0.01–0.13 on ll ≈ 4e6, within the legacy path's own CPU-vs-GPU
  spread. The legacy convolved likelihood is only reproducible to ~1e-8 relative. The
  f = 0 fixture differs by up to 2.3 against a legacy spread of 1.6.
- **Fused C++ kernel vs legacy compiled kernel** on the same covariance: ~1e-14 relative.
- **Legacy mode:** bitwise unchanged on CPU (and on GPU without galaxy).
- **Factorized splines** (approximate):
  - 2-day fixtures: |Δll| 0.02–0.4;
  - 2 yr: |Δll| ≲ 0.5 above 1 mHz;
  - **not accurate for few-day segments** with strongly varying splines (Δll up to 1–60).
  - See `agent5_error_budget.md`.

## 5. Caveats and known limitations

- **STFT / WDM domains are untested** with the new path. All fixtures are frequency-domain
  (`num_times = 1`). The per-time galaxy scaling (`rho`) and the multi-segment bases are
  implemented but not yet validated; run new-vs-legacy on an STFT config before trusting it.
- **Low-frequency round-off floor.** Below ~1e-4 Hz (and below ~1 mHz with splines) FFT
  round-off (~eps·max|C|) is comparable to the smallest eigenvalue of the nearly singular
  XYZ covariance. The legacy convolved likelihood does not reproduce itself there: ~25 in ll
  at 2 yr without splines, and 100 or more with them. This affects every mode equally.
  Improving it, e.g. with a better-conditioned basis, is separate work.
- **CPU bitwise reproducibility** of the legacy kernel depends on inlining (see
  `PSD_FORCE_INLINE`). Re-run `test_native.py` / `test_agent4_legacy.py` after rebuilding
  with different compiler flags.
- **Bases cache invalidation.** The bases are built once per backend. They are not rebuilt
  if the window, grid, orbits or filters change on an existing instance (construct a new backend).
- **Factorized mode:** opt-in only; use it for long observations.
- **GPU length checks:** the GPU build of the bindings never checks array lengths (unchanged).

## 6. Usage

```python
XYZSensitivityBackend(
    ..., window_values=w, convolve_window=True,
    window_convolution_mode="exact",   # default; "factorized" for splines on long data
    galaxy_fmax=0.03,                  # optional speed-up when fitting the galaxy
    psd_batch_size=None,               # bound memory of the legacy path if needed
    fft_batch_size="auto",             # integer = upper bound
)
```

No config change is needed to get the speed-up: `"exact"` is the default.

## 7. Tests and tools (`scripts/window_conv/`)

| File | What it checks |
|---|---|
| `test_integration.py` | exact/factorized vs legacy on the same instance; chunking; cache sharing in copies |
| `test_native.py` (`--overflow`) | fused C++ vs legacy kernel; galaxy rows vs covariance kernel; legacy goldens; int32-overflow test |
| `test_agent2_instrument.py` | bases and fused likelihood vs legacy |
| `test_agent3_galaxy.py` | galaxy convolution vs legacy (224 checks), short circle, tail bound, fallback |
| `test_agent4_legacy.py` | hardened legacy path vs the reference commit `673152b2` (CPU bitwise) |
| `test_length_checks.py` | CPU binding length checks above 2^31 elements (zero-stride views) |
| `bench_2yr.py` | time and peak pool memory per mode |
| `fixtures.py`, `compare.py`, `make_golden.py`, `golden/` | fixtures and reference outputs from `673152b2` (`golden/` is git-ignored) |
| `build_native.sh`, `_use_worktree.py` | isolated native build (`WCONV_NATIVE=1`), no shared-venv install |
| `agent5_error_budget.md` | factorized-mode error study (study code in commit `6bebe8fe`) |

Run: `cd scripts/window_conv && CUDA_VISIBLE_DEVICES=<free> <venv>/bin/python test_integration.py`.

## 8. Commits (`perf/window-conv-basis`)

| Commit | Content |
|---|---|
| `6565999f` | exact fast window convolution + bounded-memory legacy path |
| `6bebe8fe` | opt-in factorized window convolution for splines |
| `1b40cd6e` | isolated native build tooling |
| `43992de6` | C++ basis likelihood + galactic rows; int64 data offsets |
| `500e83b3` | size_t length checks; 64-bit spline index |

Then merged into `perf-window-convolution`, and cleaned up there (uncommitted at the time of
writing): rename to `convolution_helpers.py`, Python kernel fallbacks and factorized study
code removed, tests updated.
