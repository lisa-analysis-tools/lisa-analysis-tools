# Factorized window convolution: error budget (agent 5)

> The study code (`agent5_error_budget.py`, `agent5_summarize.py`, the moment bases and
> `factorized_galaxy`) was removed after the study; it is in commit `6bebe8fe`. Only the
> order-0 spline factorization was kept (`convolution_helpers.factorized_spline_weights`).

**Question.** Is `window_convolution_mode="factorized"` accurate enough to offer as an opt-in?
It uses zero per-walker FFTs and relies on `conv(m·T) ≈ m·conv(T)`. Is Δlog L ≪ 1, and for
which components?

**Short answer.**

| Component | Verdict |
|---|---|
| Galaxy | **Yes.** Use the default `split="powerlaw"`, `order=2`. Agent 3's exact galaxy is cheap anyway, so this is a fallback, not a need. |
| Splines | **Yes for long datasets** (≥ ~1 month): the error is at or below the legacy path's own numerical reproducibility. **No for short segments** (days) with strongly varying splines: Δll is O(1–10) there. |
| Second-order spline correction | **Never.** It diverges below ~1 mHz. |

Raw outputs are in `agent5_out/`:

- `res_*.json`
- `log_*.txt`
- `tables.md`: every row, with medians.

Code: `agent5_error_budget.py`, `agent5_summarize.py`, `agent5_diag_outliers.py`.

## What is compared

All comparisons are made on the **same backend instance**, GPU (`cuda12x`):

- **exact**: legacy `_compute_matrix_elements(..., convolve_window=True)`, i.e. 12 length-N FFTs per walker.
- **factorized**: unit bases × `10^spl(f)`, plus factorized galaxy rows.
- Δll: both covariances go through the same compiled kernel (`psd_likelihood_wrap` with the
  external matrix). So Δll isolates the approximation.

**Legacy floor.** This is the legacy path run on the CPU twin vs on the GPU, with the same
draws and the same data (`--floor`). It measures how reproducible the "exact" reference is. The
twin also differs by the ~1e-9 relative non-determinism of the GPU galactic grid. So the floor is
also the likelihood's sensitivity to perturbations of C at the 1e-9 level.

**Metrics**, with dip-masked bins excluded:

- **entry**: per-bin relative error of the 6 entries. Diagonals are divided by C_ii; off-diagonals
  by sqrt(C_ii·C_jj). The max is over walkers and entries.
- **white**: the whitened error sqrt(tr[(C⁻¹ΔC)²]) = ‖L⁻¹ΔC L⁻ᴴ‖_F. This is what drives Δll.
- **Δll**: |ll_fact − ll_exact| per walker, as max or median over walkers. It is also split by
  band: f < 1e-4 Hz / 1e-4–1e-3 Hz / ≥ 1e-3 Hz. The band sums reproduce the kernel ll to
  ≤ 1e-6 (`bandchk`).

**Draws.** There are 16 walkers per family at 2 days and 30 days, and 2 walkers at 730 days.

| Family | Galaxy parameters | Knee |
|---|---|---|
| `prior` | fixture prior; f_2 ∈ [4e-4, 1.6e-3] Hz, kn ∈ [1.6, 6.3] mHz | — |
| `sharp` | f_2 ∈ [1e-4, 4e-4] Hz, kn ∈ [1, 10] mHz | sharp knee inside the band |

| Suffix | Spline knot amplitudes | Meaning |
|---|---|---|
| `_spl01` | σ = 0.1 dex | fixture |
| `_spl03` | σ = 0.3 dex | strongly varying |
| `_spl05` | σ = 0.5 dex | strongly varying |

Spline knots are 8, at uniform log-f positions. Draws with knot gaps < 0.05 dex are rejected,
because nearly coincident knots make Akima overshoot to 10^24. That draw made even the legacy C
non-positive-definite in 7.7k bins (`agent5_diag_outliers.py`).

**Cases.**

- 2 days at 5 s: all of `gal1_spl0_{sym,asym}`, `narrow_gal1_spl0_sym`, `gal0_spl1_{sym,asym}`
  and `gal1_spl1_{sym,asym}`.
- 30 days at 5 s: `gal1_spl1_sym`, `gal1_spl0_sym`, `gal0_spl1_asym`.
- 730 days at 5 s: `gal1_spl1_sym` on GPU, with nf = 5.67M.

## Building-block validation

| Check | Result |
|---|---|
| Python mirror of `get_galactic_foreground` × `galaxy_common_tdi_factor` × norm (+ alias), against a galaxy-only compiled evaluation (band-normalised) | 4e-16 (2 d), 3e-15 (730 d) |
| Mirrored galaxy convolved by `_fact_convolve` vs the legacy galaxy-only convolution | ≤ 5e-12. This is cancellation in the legacy `total − instrument`; 6e-16 in the narrow band |
| `_fact_convolve(moment 0)` on the raw instrument bases vs `get_instrument_bases` (convolved) | 0.0 (bitwise) |
| CPU (numpy) path vs the exact galaxy, `order=2` | 6e-7 (narrow) / 6e-5 (wide, band-normalised, sharp knee) |
| `fused_log_like(basis_coeffs, gal_conv, gal_ratios, spline_weights)` fed with these outputs, vs the compiled kernel on the same assembled C | 1e-9 (narrow) to 1e-2 typical, up to ~10 at ill-conditioned low-f bins (agent 2's numerics, not the approximation) |

## 1. Galaxy: choice of split and order

Max Δll over walkers, given per band as f < 1e-4 / 1e-4–1e-3 / ≥ 1e-3 Hz. The instrument part
is exact.

| Case | Legacy floor | `full`, o0 | `powerlaw`, o0 | **`powerlaw`, o2** |
|---|---|---|---|---|
| 2 d sym, prior | 3e-2 / 1e-2 / 8e-6 | 1.7 / 1.7 / 0.66 | 2e-2 / 0.11 / 0.25 | **3e-2 / 3e-3 / 2e-3** |
| 2 d sym, sharp | 2e-2 / 3e-3 / 1e-5 | 1.8 / 0.9 / 2.6 | 2e-2 / 6e-2 / 2.6 | **1e-2 / 2e-3 / 4.5e-2** |
| 2 d asym, prior | 8e-2 / 2e-2 / 1e-5 | 1.7 / 1.3 / 1.0 | 6e-2 / 0.16 / 0.59 | **7e-2 / 3e-3 / 3e-3** |
| 2 d asym, sharp | 5e-2 / 1e-2 / 1e-5 | 1.3 / 1.3 / 5.3 | 2e-2 / 4e-2 / 5.5 | **2e-2 / 7e-3 / 0.11** |
| 2 d narrow [1, 1.5] mHz, prior | – / – / 3e-8 | 0.16 | 4e-2 | **1.7e-3** |
| 2 d narrow, sharp | – / – / 3e-8 | 0.10 | 2e-2 | **7e-4** |
| 2 d gal+spl asym, sharp, σ = 0.5 (worst) | 5e-2 / 0.13 / 2e-3 | 3.7 / 3.1 / 23 | 6e-2 / 0.11 / 25 | **6e-2 / 6e-2 / 0.79** |
| 30 d sym, prior | 1.5 / 0.17 / 1e-4 | 38 / 0.66 / 0.65 | 0.22 / 0.11 / 0.51 | **0.23 / 5e-2 / 3e-3** |
| 30 d sym, sharp | 1.5 / 0.24 / 2e-4 | 48 / 0.58 / 1.7 | 8e-2 / 2e-2 / 1.7 | **8e-2 / 6e-2 / 1.6e-2** |
| 730 d gal+spl, prior σ = 0.1 | 230 / 51 / 3.5e-2 | – | 1.1 / 2.5e-2 / 0.17 | **1.1 / 8.5e-2 / 9e-4** |
| 730 d gal+spl, sharp σ = 0.3 | 110 / 28 / 8.2 | – | 1.8 / 0.48 / 5.0 | **1.8 / 0.57 / 0.14** |

Entry errors for `powerlaw`, o2 (p99 / max), tightening with T:

| Duration | p99 | max |
|---|---|---|
| 2 d | ≤ 7e-7 | 2e-4 (at the knee, 1–4 mHz) |
| 30 d | ≤ 4e-8 | 7e-6 |
| 730 d | ≤ 7e-10 | 2e-7 |

The other variants have much larger entry errors:

- `powerlaw`, o0: p99 1e-4 at 2 d.
- `full`, o0: max 3e-2, at the low-f end where f^{-7/3} varies by tens of % per bin.

**Chosen split: `powerlaw`.** The walker-independent f^{-7/3} goes into T. The per-walker
factor m = Amp·exp(−(f/f₁)^α)·½(1+tanh((kn−f)/f₂)) is smooth wherever it matters.

**Order 2 is worth it.** The correction is +m'·conv(δK,T) + ½m''·conv(δ²K,T). The derivatives
come from exact evaluations of m at f ± df, and the moment bases are precomputed. This cuts the
≥ 1 mHz (knee) error by 10–100× and makes Δll ≪ 1 in every case.

**Remaining Δll is at or below the floor.** What is left sits in f < 1e-4 Hz. There the
whitened error is identical for o0 and o2, and comparable to the legacy floor. It is round-off
amplified by the near-singular low-f C, not the approximation.

**Edge handling is exact.** The legacy path pads out-of-band bins with the edge value.
`factorized_galaxy` factorizes only the in-band part and adds the edge contribution exactly,
through `_fact_edge_rows`. Without this, the narrow band had entry errors of 2e-4 at every bin and
Δll up to 0.13. With it, Δll ≤ 2e-3.

**Aliases.** Each alias term gets its own basis, B[j][a]. For the fixtures (|f_a| ≥ 0.11 Hz)
the galaxy aliases are numerically zero.

## 2. Splines: factorized `10^spl` (order 0)

The galaxy part is exact. **fusedif** is what `fused_log_like(spline_weights=10^spl)` computes
today: the legacy edge-padded bases × w(f).

Max Δll (median), with the band split f < 1e-4 / 1e-4–1e-3 / ≥ 1e-3 Hz:

| Case | Legacy floor | **spline_fact (o0, exact edges)** | fusedif | spline o2 |
|---|---|---|---|---|
| 2 d sym, σ = 0.1 | 0.17 (5e-2) | 0.45 (4e-2): 0.48 / 0.11 / 5e-3 | 0.46 | 1.3e3 |
| 2 d sym, σ = 0.3 | 0.26 (3e-2) | 3.1 (0.2): 2.7 / 0.63 / 0.19 | 3.1 | 3.7e2 |
| 2 d sym, σ = 0.5 | 0.76 (0.17) | 12 (1.0): 10 / 2.2 / 0.75 | 12 | 2.5e3 |
| 2 d asym, σ = 0.5 | 1.6 (9e-2) | 64 (1.3): 64 / 9.2 / 2.8 | 64 | 3.8e2 |
| 2 d gal+spl sym, σ = 0.1 / 0.3 / 0.5 | 0.12 / 5e-2 / 0.14 | 0.13 / 2.8 / 1.3 | – | ≥ 3.6e2 |
| 2 d gal+spl asym, sharp σ = 0.5 | 0.14 | 9.8: 0.76 / 2.8 / 6.3 | – | 1.3e3 |
| 30 d gal+spl sym, σ = 0.1 | 0.46 | 0.28: 0.21 / 4e-2 / 3e-2 | – | – |
| 30 d gal+spl sym, σ = 0.3 | 2.7 | 2.6: 2.4 / 0.29 / 2.5e-2 | – | – |
| 30 d gal+spl sym, σ = 0.5 | 40 | 40: 35 / 4.7 / 0.18 | – | – |
| 30 d gal+spl sym, sharp σ = 0.5 | 3.3 | 5.1: 2.6 / 2.5 / 4.2 | – | – |
| 30 d asym, σ = 0.3 | 220 | 100: 51 / 49 / 4.7e-2 | – | – |
| 30 d asym, σ = 0.5 | 8.2e3 | 9.9e3 (both in f < 1e-4) | – | – |
| 730 d gal+spl, σ = 0.1 | 290: 230 / 51 / 3.5e-2 | **16: 14 / 2.0 / 1.4e-3** | – | – |
| 730 d gal+spl, σ = 0.3 | 130: 110 / 23 / 7e-2 | **4.7: 3.7 / 1.1 / 6e-3** | – | – |
| 730 d gal+spl, sharp σ = 0.3 | 140: 110 / 28 / 8.2 | **5.9: 5.7 / 0.37 / 0.52** | – | – |

Entry error p99 for spline_fact:

| Duration | p99 |
|---|---|
| 2 d | 4e-6 to 9e-5 |
| 30 d | 2e-7 to 7e-6 |
| 730 d | ≈ 1e-8 |

The max sits at the lowest bins, 50–150 µHz, at 2e-5 to 0.1. The scale works out as follows:

- The relative error is ~ (μ₂/f²)·(log-slope of the spline)².
- μ₂ ≈ 2 f_min·df/π² is the kernel's second moment in Hz².
- So the error shrinks ∝ df, i.e. ∝ 1/T, except within ~2 f_min of the band start. That is
  because the Tukey taper `1/min_freq` makes the kernel tails reach ~f_min in Hz whatever T is.

**Second-order splines (`spline_fact_o2`) are worse, not better.**

- The median entry error drops to 1e-13.
- But below ~1 mHz, w = 10^spl(log f) is not smooth on the kernel scale. The Taylor series
  diverges: Δll reaches 10²–10³.

**The edge-exact assembly and today's fusedif are indistinguishable for splines** in wide bands.
So agent 2's `spline_weights` interface already gives the o0 accuracy above.

## 3. Finding about the reference itself

**With splines, the legacy convolved ll is not reproducible below ~1 mHz.**

| Duration | Legacy CPU vs GPU, sym | Asym, σ = 0.5 |
|---|---|---|
| 2 d | Δll 0.02–3 | 1.6 |
| 30 d | Δll 0.5–40 | 8e3 |
| 730 d | Δll 130–290 | – |

It is concentrated in f < 1e-3 Hz:

- The unit bases have a 1e11 dynamic range across the band.
- So the FFT round-off floor eps·max_f|C| is comparable to the smallest eigenvalue of the
  near-singular low-f XYZ covariance. The whitened error at 50 µHz is 0.1–1 for the legacy path
  against itself.

Above 1 mHz the legacy floor is 1e-5 to 8. The factorized-spline error at 730 d is below the
floor in every band except ≥ 1 mHz, where it is comparable (1e-3 to 0.5, against a floor of
4e-2 to 8).

## Recommendation

1. **Galaxy.** `factorized_galaxy(..., split="powerlaw", order=2)` is a sensible opt-in:
   - Δll ≤ 0.1 in the well-conditioned band for all draws, 0.8 in the worst 2-day sharp-knee
     case.
   - It is ≤ 1e-1 at 30 d and 730 d above 1e-4 Hz.
   - It is below the legacy floor below 1e-4 Hz.

   It is, however, unnecessary when agent 3's exact galaxy (one short real FFT pair per walker)
   is available. Prefer exact as the default and keep factorized as an FFT-free fallback.
2. **Splines.** Order-0 factorization is the only zero-FFT option.
   - For realistic durations (≥ 1 month, and especially 1–2 yr) it is a sensible opt-in. Its
     Δll is far below the legacy path's own CPU/GPU irreproducibility. In the clean ≥ 1 mHz band
     it is ≤ 0.5 at 2 yr.
   - For few-day segments with strongly varying splines (σ ≥ 0.3 dex) Δll is O(1–60). Use exact
     there.
   - Do not offer an order-2 spline correction.
3. **Not ≪ 1 everywhere.** In f < 1e-4 Hz no variant can be validated against the legacy path,
   because the legacy path itself is not reproducible there. If those bins matter, the
   convolution needs better conditioning (e.g. convolve in a whitened/AET basis or with a
   per-band scale), whatever the mode. This is a separate work item for the coordinator.

(Edge terms: exact for the galaxy (real rows) and for the real parts of the instrument
entries; the imaginary edge padding of the complex cross entries is approximated by the
even indicator rows, measured residual 5e-13 band-normalised.)

## Interface for the fused kernel

- **Galaxy.** `factorized_galaxy` returns `(gal_conv, gal_ratios)` in exactly the
  `fused_log_like` form. The o2 correction, the edge terms and the alias sum are folded into
  `gal_conv` in Python, so no kernel change is needed. `gal_conv` spans the whole active band
  unless `n_gal` is passed. Cost per walker: (1 + n_alias) × (order + 1) fused multiply-adds per
  bin.
- **Splines.** `factorized_spline_weights(..., order=0)` gives `(num_psds, 2, num_freqs)`, which
  is `spline_weights` as is. The exact-edge variant would need extra per-walker complex edge
  coefficients × the fixed rows `P_lo` / `P_hi`. It is not worth it: indistinguishable from
  fusedif above.

## Added to `src/lisatools/sensitivity_window.py` (agent-5 section)

**Public functions**

- `factorized_galaxy`
- `factorized_spline_weights`
- `factorized_instrument_moment_bases` (study / o2 only)
- `FACT_GALAXY_SPLITS`

**Private helpers**

- `_fact_galaxy_shape`
- `_fact_common_tdi`
- `_fact_alias_grids`
- `_fact_guard_f0`
- `_fact_zero_knots`
- `_fact_convolve` / `_fact_convolve_full` (legacy-identical convolution, any δ^j moment, edge or zero padding)
- `_fact_edge_rows`
- `_fact_galaxy_R`
- `_fact_galaxy_setup`

**Cached on the backend** (add to the `__deepcopy__` share list):

- `_wconv_fact_gal`
- `_wconv_fact_edge`
- `_wconv_fact_inst_moments`
