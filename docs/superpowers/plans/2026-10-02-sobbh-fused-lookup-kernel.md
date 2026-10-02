# SOBBH fused lookup kernel: response splines -> (d_h, h_h) or the fill, in one launch

Status 2026-10-02 (night): IMPLEMENTED and CPU-tested on the worktree branch
`sobbh-wdm-lookup` (uncommitted), built against the EMRI session's `wdm_lookup_kernels.hh` as
it stands in their worktree (identical copy; their kernel is CPU-tested but not yet pushed --
both land together on Mike's word). CUDA build not compiled yet (no nvcc on the laptop): the
cluster `pip install -e . --no-build-isolation` is its first compile; the launcher preflight
and `GPUKernelParityTest` prove it against the CPU build. Measured below (section "Result").
Originally: PLANNED, waiting on the EMRI fused-kernel helpers. Agreed with the EMRI
direct-WDM session (lisa-sprint-2026-d9): they own `lat_spline_tdi_waveform.{hh,cu}`,
`binding_lat_spline_tdi.hpp`, `binding_flr.cxx`, `cutils/__init__.py` and the new header
`src/lisatools/cutils/wdm_lookup_kernels.hh` until their kernel is pushed; this kernel is added
SEPARATELY afterwards and includes that header's `CUDA_DEVICE` helpers (cubic B-spline 2-D in
mirror mode with the inside check, GBT spline evaluation with derivatives 0..2,
`dense_phase_derivs`, the per-layer quarter-turn rule). Their kernel only WRITES templates
(EMRI needs cross-harmonic terms in `<h|h>`); the accumulate mode below is SOBBH-specific
(one sub per row, so `w^T invC w` is thread-local). Their write mode has a has-carrier switch
(carrier = 0: phase = the residual spline alone), which is the SOBBH case.

## Why (cluster, one H100, cuda13x, production grid, sparse 12-h response, row batch 32)

| months | rows | look_warm | resp | trac | look | inner | chunked |
|---|---|---|---|---|---|---|---|
| 6 | 8 | 0.051 | 0.033 | 0.010 | 0.007 | 0.002 | 1.697 |
| 6 | 288 | 0.569 | 0.366 | 0.113 | 0.073 | 0.015 | 3.721 |
| 24 | 8 | 0.059 | 0.040 | 0.010 | 0.007 | 0.002 | 6.828 |
| 24 | 288 | 0.757 | 0.433 | 0.107 | 0.197 | 0.022 | 14.875 |

- `resp` at 288 rows was nine response builds of ~0.04 s each (the fixed cost per build):
  FIXED in Python on 2026-10-02 -- one response per call, `row_batch` only blocks the
  tracer / lookup (`tests/test_sobbh_lookup_batching.py`).
- `trac + look + inner` is ~20 ms per 32-row block at 6 months (7 spline evaluations with host
  syncs, 16-tap gathers for cos and sin at 3 channels x 5 layers x pixels, the index
  gathers of the inner products) and ~17 MB of temporaries per row (70 MB at 24 months). The
  fused kernel replaces all three with one launch and no per-pixel global memory.

## Semantics (must equal the Python path bit-for-bit up to roundoff)

Reference: `SOBBHDirectWDM.sparse_from` + `sparse_inner_products` (get_ll) and `scatter_add`
(fill), with `WDMLookupEvaluator(interp="spline")` (the table's uniform cubic B-spline,
mirror boundaries, prefiltered coefficients; == `scipy.ndimage.map_coordinates(order=3,
mode="mirror", prefilter=False)`; `tests/test_wdm_lookup_eval.py` pins it).

Per thread = one (row r, pixel n), n in [ind_min_t, ind_max_t], t = t_obs_start + n layer_dt:

1. For each channel c: `amp = A_spl[r, c](t)` (0 if t >= tc[r]); `phi = P_spl[r, c](t)`,
   `f = P'(t) / 2pi`, `fdot = P''(t) / 2pi` where `P = tdi_phase + phase_ref` (ONE combined
   spline per (r, c), built once per call in Python from the response output; GBT flats
   `x, y, c1, c2, c3`, index `(r * nch + c) * Neval + j`, segment `searchsorted(x, t,
   'right') - 1` clamped to `Neval - 2`). Mirror `f < 0`: `phi, f, fdot -> -phi, -f, -fdot`.
2. `live_c = amp != 0` (the TDI amplitude is SIGNED). Common window: `f_ref = mean of f over
   live channels`; `m0 = floor(f_ref / layer_df)`; layers `m = m0 - L .. m0 + L` (L =
   `num_m_layers` = 2).
3. For each layer m and channel c: `delta = f_c - m layer_df`; in support iff `delta in
   [f_min, f_max]` and `fdot_c in [fdot_min, fdot_max]` (nfdot > 1); `c_tab, s_tab` =
   B-spline at `(fdot_c, delta)` of the prefiltered cos and sin-unbaked tables; if `(m_ref +
   n_ref)` odd: `c_tab = -c_tab`; if `(m + n)` odd: `(c_tab, s_tab) -> (s_tab, -c_tab)`;
   `w_c = amp_c (c_tab cos phi_c - s_tab sin phi_c)`; valid iff in support, live and
   `ind_min_f <= m <= ind_max_f`; else `w_c = 0`.
4. ACCUMULATE mode: `ma = m - ind_min_f`, `na = n - ind_min_t`; with `d = data[di[r]]`,
   `C = invC[ni[r]]` (XYZ: full 3x3 per pixel, `(nch, nch, Nf_active, Nt_active)`; AET / AE:
   the diagonal, `(nch, Nf_active, Nt_active)`): `d_h += sum_{c,c'} d_c(ma, na) C_cc'(ma, na)
   w_c'` and `h_h += sum_{c,c'} w_c C_cc'(ma, na) w_c'` over the 5 layers. Block reduction per
   row (grid `(n_pixel_blocks, rows)`), then one `atomicAdd` per block into `d_h[r]`,
   `h_h[r]`. Counters (atomicAdd, optional): dropped channel-pixels (live, fdot out of axis).
5. FILL mode: `atomicAdd(buf[di[r], c, ma, na], factor[r] * w_c)` for every valid (m, c).

## Interface

`SOBBHLookupKernelWrap` (CPU / GPU aliased names, `#define ... CPU/GPU` as the EMRI wrap) or a
free function; arrays by pointer (device on GPU), table coefficients uploaded once per comp
(cached on the comp). Python: `SOBBHLookupComputations(lookup="kernel" | "python")`, env knob
`SOBBH_LOOKUP_KERNEL` -> attribute `lookup_kernel` (default `kernel` when the backend module has
it, else python with one INFO line); `get_ll_wdm` / `fill_global_wdm` call it after
`direct.response(p)` and one combined phase-spline fit.

## Tests (`tests/test_sobbh_lookup_kernel.py`)

- kernel (CPU build) == Python path: `d_h`, `h_h` per row to 1e-12 relative on the toy grid
  (ROWS + the chirpy in-band row + a merging row), XYZ full and AET diagonal, per-walker
  `data_index` / `noise_index` routing (two different slabs, swapped indices);
- fill: kernel buffer == `scatter_add` buffer to 1e-12, factors and slab routing;
- edges: pixel at the B-spline edge nodes, `f < 0` mirror, out-of-support layers, rows past
  merger, `nfdot == 1` table;
- negative controls / mutations: no quarter turn, `(m_ref + n_ref)` parity sign, channel-mean
  window replaced by per-channel windows (must break the XYZ `h_h`), dropping the `tc` mask;
- GPU == CPU parity (skips without a GPU; `lisatools.get_backend("gpu")`).

## Build / run

Laptop CPU (private build): as the EMRI plan (`cmake -S <worktree> -B scratchpad/latbuild
-DLISATOOLS_WITH_GPU=OFF ...`, prepend the build dir to `PYTHONPATH`). Cluster:
`pip install -e . --no-build-isolation`, then `python -m unittest tests.test_sobbh_lookup_kernel
-v` and the speed script at 6 / 12 / 24 months (`--lookup kernel,python` axis).

## Expected

`trac + look + inner` -> a few ms per call at 288 rows (one launch over ~1.2M (row, pixel)
threads); per-row memory -> ~0; then the call is the response build (~0.03 s fixed at 8 rows:
the PN at 2048 nodes, uploads, input / output spline fits, kernel ~2 ms) -- the `pn` span the
speed script now prints decides whether the PN goes on the device next.

## Result (2026-10-02, laptop, private CPU build)

- `tests/test_sobbh_lookup_kernel.py`: kernel == Python lookup to 1e-10 relative for `<d|h>`,
  `<h|h>`, lnL (XYZ full invC, swapped data / noise routing, 5 rows incl. a chirpy and a merging
  row), the AET diagonal, the fill, the stats (lookup pixels, dropped pixels, merged rows); the
  Python path without the quarter turn differs at O(1); `kernel` without the B-spline table is
  refused. Mutations each caught (kernel rebuilt per mutation): no quarter turn (4 tests fail),
  no merger mask (4), XYZ diagonal only (1), f without the reference phase (4).
- All SOBBH lookup suites pass with the kernel (82 tests) and without it (installed backend).
- Laptop CPU, 6 months (`--laptop --nt 4320`), the lookup step (Python: tracer + lookup + inner)
  vs the fused kernel (serial on the CPU): 0.214 -> 0.046 s at 4 rows, 0.538 -> 0.091 at 8,
  2.35 -> 0.365 at 32; whole call 0.313 -> 0.127, 0.779 -> 0.242, 2.94 -> 0.939 s.
- Launch geometry: grid (pixel blocks, rows), 256 threads, rows chunked over the 65535 grid.y
  limit; one block reduction + atomicAdd per (block, row).
